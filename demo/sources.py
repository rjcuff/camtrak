"""
sources - read test footage in the formats tracking benchmarks use.

Supported inputs for --source:
  * a video file (mp4, avi, mov, ...)
  * a folder of images (sorted by name), or a folder containing img/
    - OTB:     Skiing/img/0001.jpg     + Skiing/groundtruth_rect.txt
    - LaSOT:   bird-1/img/00000001.jpg + bird-1/groundtruth.txt
               (+ full_occlusion.txt, out_of_view.txt: target absent flags)
    - UAV123:  data_seq/UAV123/bird1/000001.jpg  + anno/UAV123/bird1.txt (pass with --gt)

Ground-truth files hold one box per line, "x,y,w,h" separated by commas,
tabs or spaces. A line of NaN (UAV123) means the target is not visible.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np

from camtrak.detect import CutDetector  # noqa: F401  (re-exported for the demos)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
GT_NAMES = ("groundtruth_rect.txt", "groundtruth.txt", "groundtruth_rect.1.txt")
Box = tuple[float, float, float, float]


def _image_dir(path: Path) -> Path:
    return path / "img" if (path / "img").is_dir() else path


def list_images(folder: Path) -> list[Path]:
    images = sorted(p for p in _image_dir(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"no images found in {_image_dir(folder)}")
    return images


class FrameSource:
    """
    Frames by index from a video file or an image folder. Reading the next
    frame in order is fast; jumping (for the frame slider) seeks.
    Image folders are assumed to be 30 fps.
    """

    def __init__(self, source: Path) -> None:
        self.source = source
        self._images: list[Path] | None = None
        self._cap = None
        self._next = 0
        if source.is_dir():
            self._images = list_images(source)
            self.count, self.fps = len(self._images), 30.0
        else:
            self._cap = cv2.VideoCapture(str(source))
            if not self._cap.isOpened():
                raise OSError(f"cannot open video {source}")
            self.count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self, index: int) -> np.ndarray | None:
        """BGR frame `index`, or None past the end."""
        if index < 0 or index >= self.count:
            return None
        if self._images is not None:
            img = cv2.imread(str(self._images[index]), cv2.IMREAD_COLOR)
            if img is None:
                raise OSError(f"cannot read image {self._images[index]}")
            return img
        if index != self._next:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = self._cap.read()
        self._next = index + 1
        return img if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


def find_groundtruth(source: Path) -> Path | None:
    if source.is_dir():
        for name in GT_NAMES:
            if (source / name).is_file():
                return source / name
    return None


def load_groundtruth(path: Path) -> list[Box | None]:
    """One entry per frame; None where the target is absent (NaN or zero-size)."""
    boxes: list[Box | None] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            values = [float(v) for v in re.split(r"[,\s]+", line.strip())[:4]]
        except ValueError:
            continue  # header or comment line
        if len(values) < 4 or any(math.isnan(v) for v in values) or values[2] <= 0 or values[3] <= 0:
            boxes.append(None)
        else:
            boxes.append((values[0], values[1], values[2], values[3]))
    return boxes


def load_absent_flags(source: Path, n: int) -> list[bool]:
    """LaSOT full_occlusion.txt / out_of_view.txt, OR-ed. All False if missing."""
    absent = [False] * n
    if not source.is_dir():
        return absent
    for name in ("full_occlusion.txt", "out_of_view.txt"):
        f = source / name
        if f.is_file():
            flags = [v.strip() == "1" for v in re.split(r"[,\s]+", f.read_text().strip())]
            for i, flag in enumerate(flags[:n]):
                absent[i] = absent[i] or flag
    return absent
