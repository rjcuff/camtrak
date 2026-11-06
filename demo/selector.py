"""
selector - pick the target: scrub to any frame, draw a box, then move and resize it.

  mouse   drag on empty space   draw a new box
          drag inside the box   move it
          drag a corner/edge    resize it
  keys    Enter or space        start tracking with this box
          a / d                 previous / next frame   (A / D: 10 frames)
          slider at the top     jump anywhere in the video
          Esc                   cancel
"""
from __future__ import annotations

import cv2
import numpy as np

from sources import FrameSource

Box = tuple[float, float, float, float]
HANDLE = 8            # px (on screen) around an edge that counts as "grab the edge"
MIN_SIZE = 4          # px (in the image)
MAX_VIEW = (1280, 720)
HELP = "drag: draw | inside: move | edge/corner: resize | a/d: frame | Enter: track | Esc: cancel"


class _State:
    def __init__(self, box: Box | None, view_scale: float) -> None:
        self.box = box               # image coordinates
        self.s = view_scale          # screen px per image px
        self.mode: str | None = None  # "new", "move" or a set of edges "lrtb"
        self.anchor = (0.0, 0.0)
        self.start_box: Box | None = None

    def _hit(self, sx: float, sy: float) -> str | None:
        """Which part of the box is under the screen point: edges, 'move', or None."""
        if self.box is None:
            return None
        x, y, w, h = (v * self.s for v in self.box)
        edges = ""
        near_x = x - HANDLE <= sx <= x + w + HANDLE
        near_y = y - HANDLE <= sy <= y + h + HANDLE
        if near_y and abs(sx - x) <= HANDLE:
            edges += "l"
        if near_y and abs(sx - (x + w)) <= HANDLE:
            edges += "r"
        if near_x and abs(sy - y) <= HANDLE:
            edges += "t"
        if near_x and abs(sy - (y + h)) <= HANDLE:
            edges += "b"
        if edges:
            return edges
        if x < sx < x + w and y < sy < y + h:
            return "move"
        return None

    def on_mouse(self, event: int, sx: int, sy: int, _flags: int, _param) -> None:
        ix, iy = sx / self.s, sy / self.s
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self._hit(sx, sy)
            self.mode = hit or "new"
            self.anchor = (ix, iy)
            self.start_box = self.box
            if self.mode == "new":
                self.box = (ix, iy, 0.0, 0.0)
        elif event == cv2.EVENT_MOUSEMOVE and self.mode:
            self._drag(ix, iy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mode = None
            if self.box and (self.box[2] < MIN_SIZE or self.box[3] < MIN_SIZE):
                self.box = self.start_box if self.start_box else None

    def _drag(self, ix: float, iy: float) -> None:
        ax, ay = self.anchor
        if self.mode == "new":
            self.box = (min(ax, ix), min(ay, iy), abs(ix - ax), abs(iy - ay))
            return
        x, y, w, h = self.start_box  # type: ignore[misc]
        dx, dy = ix - ax, iy - ay
        if self.mode == "move":
            self.box = (x + dx, y + dy, w, h)
            return
        x0, y0, x1, y1 = x, y, x + w, y + h
        if "l" in self.mode:
            x0 = min(x0 + dx, x1 - MIN_SIZE)
        if "r" in self.mode:
            x1 = max(x1 + dx, x0 + MIN_SIZE)
        if "t" in self.mode:
            y0 = min(y0 + dy, y1 - MIN_SIZE)
        if "b" in self.mode:
            y1 = max(y1 + dy, y0 + MIN_SIZE)
        self.box = (x0, y0, x1 - x0, y1 - y0)


def _render(img: np.ndarray, st: _State, index: int, count: int) -> np.ndarray:
    view = cv2.resize(img, None, fx=st.s, fy=st.s, interpolation=cv2.INTER_AREA) if st.s != 1 else img.copy()
    if st.box:
        x, y, w, h = (round(v * st.s) for v in st.box)
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 255), 2)
        for cx, cy in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
            cv2.rectangle(view, (cx - 4, cy - 4), (cx + 4, cy + 4), (0, 255, 255), -1)
        size = f"  box {round(st.box[2])}x{round(st.box[3])}"
    else:
        size = ""
    for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(view, f"frame {index}/{count - 1}{size}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, thick, cv2.LINE_AA)
        cv2.putText(view, HELP, (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    colour, thick, cv2.LINE_AA)
    return view


def select_target(src: FrameSource, start: int = 0,
                  box: Box | None = None) -> tuple[int, Box] | None:
    """Let the user choose a frame and a box. Returns (frame index, box) or None."""
    window = "Tether - select target"
    try:
        return _select(src, window, start, box)
    finally:
        cv2.destroyWindow(window)


def _select(src: FrameSource, window: str, start: int, box: Box | None) -> tuple[int, Box] | None:
    index = max(0, min(start, src.count - 1))
    img = src.read(index)
    if img is None:
        return None
    h, w = img.shape[:2]
    st = _State(box, min(1.0, MAX_VIEW[0] / w, MAX_VIEW[1] / h))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, st.on_mouse)
    slider = {"pos": index}
    if src.count > 1:
        cv2.createTrackbar("frame", window, index, src.count - 1,
                           lambda v: slider.__setitem__("pos", v))

    while True:
        if slider["pos"] != index:
            frame = src.read(slider["pos"])
            if frame is not None:
                index, img = slider["pos"], frame
        cv2.imshow(window, _render(img, st, index, src.count))
        key = cv2.waitKey(20) & 0xFF
        step = {ord("a"): -1, ord("d"): 1, ord("A"): -10, ord("D"): 10}.get(key)
        if step and src.count > 1:
            slider["pos"] = max(0, min(src.count - 1, index + step))
            cv2.setTrackbarPos("frame", window, slider["pos"])
        elif key in (13, 10, ord(" ")) and st.box:
            x, y, bw, bh = st.box
            x, y = max(0.0, x), max(0.0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw >= MIN_SIZE and bh >= MIN_SIZE:
                return index, (x, y, bw, bh)
        elif key == 27:
            return None
