"""
hud - draw tracked targets as a targeting overlay.

Per target: corner brackets, a centre dot, and a label panel
    TRK ID: 3 [LOCKED]
    POS: 1134, 578
    SPD: 106 px/s
Plus a magnified close-up of the focused target in the bottom-right corner,
and a one-line status strip at the top.

Colours by state: red = LOCKED, amber = COASTING, grey = SEARCHING (lost).
Sizes scale with the frame height, so 720p and 4K look the same.
"""
from __future__ import annotations

import cv2
import numpy as np

import tether as tt

COLOURS = {tt.TRACKING: (60, 60, 255), tt.COASTING: (0, 170, 255), tt.LOST: (150, 150, 150)}
TEXT = (235, 235, 235)
PANEL = (55, 55, 55)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Hud:
    def __init__(self, frame_shape, *, labels: bool = True, inset: bool = True,
                 trails: bool = False) -> None:
        H, W = frame_shape[:2]
        self.u = H / 1080.0                    # 1.0 at 1080p
        self.font = max(0.38, 0.48 * self.u)
        self.thick = max(1, round(1.2 * self.u))
        self.line_h = max(14, round(19 * self.u))
        self.labels = labels
        self.inset = inset
        self.trails = trails
        self.W, self.H = W, H

    # ------------------------------------------------------------- one target
    def brackets(self, img, box, colour, focused: bool) -> None:
        x, y, w, h = box
        x0, y0, x1, y1 = round(x), round(y), round(x + w), round(y + h)
        L = max(4, round(min(w, h) * 0.28))
        t = self.thick + (1 if focused else 0)
        for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * L, cy), colour, t, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * L), colour, t, cv2.LINE_AA)
        cx, cy = round(x + w / 2), round(y + h / 2)
        cv2.circle(img, (cx, cy), max(1, round(2 * self.u)), colour, -1, cv2.LINE_AA)

    def label(self, img, t: tt.Target, colour) -> None:
        x, y, w, h = t.box
        cx, cy = t.center
        lines = [(f"TRK ID: {t.id} [{t.label}]", colour),
                 (f"POS: {round(cx)}, {round(cy)}", TEXT),
                 (f"SPD: {round(t.speed)} px/s", TEXT)]
        widths = [cv2.getTextSize(s, FONT, self.font, self.thick)[0][0] for s, _ in lines]
        pw = max(widths) + round(12 * self.u)
        ph = self.line_h * len(lines) + round(8 * self.u)
        px = round(x + w + 4 * self.u)
        py = round(y)
        if px + pw > self.W:                   # no room on the right: put it on the left
            px = round(x - 4 * self.u - pw)
        px = max(0, min(self.W - pw, px))
        py = max(0, min(self.H - ph, py))
        roi = img[py:py + ph, px:px + pw]
        if roi.size:
            roi[:] = (roi.astype(np.float32) * 0.35 + np.array(PANEL, np.float32) * 0.65).astype(np.uint8)
        for i, (s, c) in enumerate(lines):
            org = (px + round(6 * self.u), py + round(4 * self.u) + self.line_h * (i + 1) - round(5 * self.u))
            cv2.putText(img, s, org, FONT, self.font, c, self.thick, cv2.LINE_AA)

    def trail(self, img, t: tt.Target, colour) -> None:
        pts = list(t.trail)
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), colour, 1, cv2.LINE_AA)

    # ------------------------------------------------------------- close-up
    def closeup(self, img, clean, t: tt.Target) -> None:
        x, y, w, h = t.box
        cx, cy = t.center
        side = max(w, h) * 1.6 + 8
        size = round(self.H * 0.25)
        x0, y0 = round(cx - side / 2), round(cy - side / 2)
        x1, y1 = x0 + round(side), y0 + round(side)
        H, W = clean.shape[:2]
        crop = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if sx1 <= sx0 or sy1 <= sy0:
            return
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = clean[sy0:sy1, sx0:sx1]
        zoom = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
        m = round(12 * self.u)
        ox, oy = self.W - size - m, self.H - size - m
        img[oy:oy + size, ox:ox + size] = zoom
        colour = COLOURS.get(t.status, TEXT)
        cv2.rectangle(img, (ox - 1, oy - 1), (ox + size, oy + size), colour, max(2, self.thick + 1))
        self.text_panel(img, f"TRK {t.id}  x{size / side:.1f}", (ox, oy), colour)

    # ------------------------------------------------------------- frame
    def draw(self, img, targets: list[tt.Target], focus_id: int | None, status: str = "",
             clean: np.ndarray | None = None) -> None:
        """Draw onto `img` in place. `clean` (the undrawn frame) feeds the close-up."""
        clean = img.copy() if clean is None and self.inset else clean
        for t in targets:
            colour = COLOURS.get(t.status, TEXT)
            if self.trails:
                self.trail(img, t, colour)
            self.brackets(img, t.box, colour, t.id == focus_id)
            if self.labels:
                self.label(img, t, colour)
        focus = next((t for t in targets if t.id == focus_id), None)
        if self.inset and focus is not None:
            self.closeup(img, clean, focus)
        if status:
            self.text_panel(img, status, (round(10 * self.u), round(10 * self.u)), TEXT)

    def text_panel(self, img, text: str, org, colour) -> None:
        """One line of text on a dark translucent panel whose top-left is `org`.
        (An outline drawn with a thicker stroke does not line up: OpenCV's
        letter spacing depends on the stroke thickness.)"""
        (tw, th), base = cv2.getTextSize(text, FONT, self.font, self.thick)
        pad = round(5 * self.u) + 2
        x0, y0 = org
        x1, y1 = min(self.W, x0 + tw + 2 * pad), min(self.H, y0 + th + base + 2 * pad)
        roi = img[y0:y1, x0:x1]
        if roi.size:
            roi[:] = (roi.astype(np.float32) * 0.35 + np.array(PANEL, np.float32) * 0.65).astype(np.uint8)
        cv2.putText(img, text, (x0 + pad, y0 + pad + th), FONT, self.font, colour, self.thick, cv2.LINE_AA)
