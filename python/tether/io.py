"""
tether.io - frames from wherever they come from.

    src = open_source("clip.mp4")        # video file
    src = open_source("frames/")         # folder of images (sorted by name)
    src = open_source("0")               # webcam 0
    src = open_source("rtsp://camera/stream")   # network stream (rtsp, http, udp ...)

    frame = src.read(i)    # files and folders: frame i (seeking if needed)
    frame = src.next()     # any source: the next frame, None at the end

Live sources (cameras, streams) cannot seek: `live` is True, `count` is 0,
and read(i) simply returns the next frame.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class Source:
    def __init__(self, spec: str | int | Path) -> None:
        self.spec = str(spec)
        self.live = False
        self._images: list[Path] | None = None
        self._cap = None
        self._next = 0
        path = Path(self.spec)
        if self.spec.isdigit():
            self._cap = cv2.VideoCapture(int(self.spec))
            self.live = True
        elif "://" in self.spec:
            self._cap = cv2.VideoCapture(self.spec)
            self.live = True
        elif path.is_dir():
            d = path / "img" if (path / "img").is_dir() else path
            self._images = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not self._images:
                raise FileNotFoundError(f"no images in {d}")
        elif path.is_file():
            self._cap = cv2.VideoCapture(str(path))
        else:
            raise FileNotFoundError(f"no such video, folder, camera or stream: {self.spec}")
        if self._cap is not None and not self._cap.isOpened():
            raise OSError(f"cannot open {self.spec}")
        if self._images is not None:
            self.count, self.fps = len(self._images), 30.0
        else:
            self.count = 0 if self.live else int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps and 1 <= fps <= 240 else 30.0

    @property
    def seekable(self) -> bool:
        return not self.live

    def next(self) -> np.ndarray | None:
        return self.read(self._next)

    def read(self, index: int) -> np.ndarray | None:
        if self.live:
            ok, img = self._cap.read()
            self._next += 1
            return img if ok else None
        if index < 0 or (self.count and index >= self.count):
            return None
        if self._images is not None:
            self._next = index + 1
            return cv2.imread(str(self._images[index]), cv2.IMREAD_COLOR)
        if index != self._next:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = self._cap.read()
        self._next = index + 1
        return img if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "Source":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_source(spec: str | int | Path) -> Source:
    return Source(spec)
