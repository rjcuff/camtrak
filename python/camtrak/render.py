# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md.
"""
camtrak.render - Camtrak's on-screen look.

    r = Renderer(frame.shape, theme="camtrak")
    r.draw(img, targets, focus_id=3, status="6 tracked / auto / 60 fps", clean=frame)

Design language
  * Every target has its own color (stable per ID), so many targets stay
    distinguishable at a glance.
  * Box: a thin rounded outline with a small ring at the center. Solid =
    locked, dashed = coasting, gray dashed = searching. The focused target
    gets four short "lock" ticks at the middle of each side.
  * Tag: a short leader line (the link) from the box corner to a pill with
    the ID chip in the target's color and its speed. Tags are placed so they
    avoid each other and the boxes; in a crowd they shrink to just the chip.
  * Trail: fading dots along the recent path.
  * Focus card: the focused target magnified, with a reticle, its speed and
    a confidence bar, in whichever corner is free of targets.
  * Status pill: the camtrak mark, wordmark and run stats, bottom left.

Themes: "camtrak" (default), "mono", "contrast", "light". Add your own with
register_theme(). Sizes scale with the frame height.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from . import COASTING, LOST, TRACKING

FONT = cv2.FONT_HERSHEY_SIMPLEX
WORDMARK_FONT = cv2.FONT_HERSHEY_DUPLEX


def hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def bgr_hex(c) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(c[2]), int(c[1]), int(c[0]))


@dataclass(frozen=True)
class Theme:
    name: str
    palette: tuple            # target colors, BGR; cycled by target ID
    accent: tuple             # brand color: wordmark, mark, hints
    panel: tuple              # card / pill background
    panel_alpha: float
    text: tuple
    muted: tuple              # secondary text, searching targets
    chip_text: tuple          # text on a colored ID chip
    trails: bool = True


CAMTRAK_PALETTE = tuple(hex_bgr(h) for h in (
    "#2EE6C5",   # camtrak teal
    "#9B7BFF",   # violet
    "#FFB547",   # amber
    "#4CC3FF",   # sky
    "#FF6FA8",   # pink
    "#B6F35C",   # lime
    "#FF8A65",   # coral
    "#E9E46B",   # sand
))

THEMES: dict[str, Theme] = {
    "camtrak": Theme("camtrak", CAMTRAK_PALETTE, hex_bgr("#2EE6C5"), hex_bgr("#0E1320"), 0.72,
                    hex_bgr("#E8EEF7"), hex_bgr("#8A96AB"), hex_bgr("#0E1320")),
    "mono": Theme("mono", (hex_bgr("#2EE6C5"),), hex_bgr("#2EE6C5"), hex_bgr("#0E1320"), 0.65,
                  hex_bgr("#E8EEF7"), hex_bgr("#8A96AB"), hex_bgr("#0E1320"), trails=False),
    "contrast": Theme("contrast", (hex_bgr("#FFE600"), hex_bgr("#00E5FF"), hex_bgr("#FF4FD8")),
                      hex_bgr("#FFE600"), (0, 0, 0), 0.88, (255, 255, 255), hex_bgr("#B0B0B0"), (0, 0, 0)),
    "light": Theme("light", tuple(hex_bgr(h) for h in ("#008F7A", "#5B3FD9", "#C77700", "#0077B6",
                                                       "#C2185B", "#4C8C00")),
                   hex_bgr("#008F7A"), hex_bgr("#F5F7FA"), 0.85, hex_bgr("#121826"),
                   hex_bgr("#5A6475"), (255, 255, 255)),
}


def register_theme(theme: Theme) -> None:
    THEMES[theme.name] = theme


# --------------------------------------------------------------------- drawing primitives

def _blend(img, x0, y0, x1, y1, color, alpha, radius=0) -> None:
    """Fill a (rounded) rectangle with `color` at `alpha`, clipped to the image."""
    H, W = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    roi = img[cy0:cy1, cx0:cx1]
    fill = np.empty_like(roi)
    fill[:] = color
    if radius > 0:
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        r = min(radius, (x1 - x0) // 2, (y1 - y0) // 2)
        cv2.rectangle(mask, (r, 0), (x1 - x0 - r - 1, y1 - y0 - 1), 255, -1)
        cv2.rectangle(mask, (0, r), (x1 - x0 - 1, y1 - y0 - r - 1), 255, -1)
        for cx, cy in ((r, r), (x1 - x0 - r - 1, r), (r, y1 - y0 - r - 1), (x1 - x0 - r - 1, y1 - y0 - r - 1)):
            cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
        m = mask[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].astype(np.float32)[..., None] * (alpha / 255.0)
        roi[:] = (roi * (1 - m) + fill * m).astype(np.uint8)
    else:
        cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, dst=roi)


def _dashed_line(img, p0, p1, color, thick, dash, gap) -> None:
    length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    if length < 1:
        return
    dx, dy = (p1[0] - p0[0]) / length, (p1[1] - p0[1]) / length
    s = 0.0
    while s < length:
        e = min(length, s + dash)
        cv2.line(img, (round(p0[0] + dx * s), round(p0[1] + dy * s)),
                 (round(p0[0] + dx * e), round(p0[1] + dy * e)), color, thick, cv2.LINE_AA)
        s = e + gap


def _rounded_outline(img, x0, y0, x1, y1, color, thick, radius, dashed=False, dash=6) -> None:
    r = int(max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2)))
    edges = (((x0 + r, y0), (x1 - r, y0)), ((x1, y0 + r), (x1, y1 - r)),
             ((x1 - r, y1), (x0 + r, y1)), ((x0, y1 - r), (x0, y0 + r)))
    for a, b in edges:
        if dashed:
            _dashed_line(img, a, b, color, thick, dash, dash * 0.8)
        else:
            cv2.line(img, a, b, color, thick, cv2.LINE_AA)
    if r > 0:
        for (cx, cy), ang in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270),
                              ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), ang, 0, 90, color, thick, cv2.LINE_AA)


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# --------------------------------------------------------------------- renderer

class Renderer:
    def __init__(self, frame_shape, theme: str | Theme = "camtrak", *, labels: bool = True,
                 focus_card: bool = True, trails: bool | None = None, status: bool = True) -> None:
        H, W = frame_shape[:2]
        self.W, self.H = W, H
        self.u = max(0.85, H / 1080.0)       # below ~900 px tall, keep text readable
        self.theme = THEMES[theme] if isinstance(theme, str) else theme
        if trails is not None:
            self.theme = replace(self.theme, trails=trails)
        self.labels = labels
        self.focus_card = focus_card
        self.show_status = status
        self.font = 0.46 * self.u
        self.thick = max(1, round(1.3 * self.u))

    # ----------------------------------------------------------------- helpers
    def color(self, target_id: int) -> tuple:
        p = self.theme.palette
        return p[(target_id - 1) % len(p)]

    def color_hex(self, target_id: int) -> str:
        return bgr_hex(self.color(target_id))

    def _px(self, v: float) -> int:
        return max(1, round(v * self.u))

    def _text(self, img, s, org, color, scale=1.0, font=FONT, thick=None) -> None:
        cv2.putText(img, s, (round(org[0]), round(org[1])), font, self.font * scale, color,
                    thick or self.thick, cv2.LINE_AA)

    def _text_size(self, s, scale=1.0, font=FONT, thick=None):
        (w, h), base = cv2.getTextSize(s, font, self.font * scale, thick or self.thick)
        return w, h, base

    # ----------------------------------------------------------------- target
    def _box(self, img, t, color, focused: bool) -> None:
        x, y, w, h = t.box
        x0, y0, x1, y1 = round(x), round(y), round(x + w), round(y + h)
        r = max(2, min(self._px(7), round(min(w, h) / 4)))
        thick = self.thick + (1 if focused else 0)
        if t.status == LOST:
            _rounded_outline(img, x0, y0, x1, y1, self.theme.muted, max(1, thick - 1), r, dashed=True,
                             dash=self._px(4))
        else:
            _rounded_outline(img, x0, y0, x1, y1, color, thick, r, dashed=t.status == COASTING,
                             dash=self._px(6))
        cx, cy = round(x + w / 2), round(y + h / 2)
        cv2.circle(img, (cx, cy), self._px(2.5), color if t.status != LOST else self.theme.muted,
                   max(1, self.thick - 1), cv2.LINE_AA)
        if focused and t.status != LOST:
            k = self._px(6)
            for (px, py, dx, dy) in ((cx, y0, 0, -1), (cx, y1, 0, 1), (x0, cy, -1, 0), (x1, cy, 1, 0)):
                cv2.line(img, (px, py), (px + dx * k, py + dy * k), color, thick, cv2.LINE_AA)

    def _trail(self, img, t, color) -> None:
        pts = list(t.trail)[:-1]                       # newest last; skip the current point
        for i, (px, py) in enumerate(pts[::-3][:12]):  # every 3rd, newest first, shrinking
            rad = max(1, round(self._px(2.2) * (1 - i / 14)))
            cv2.circle(img, (round(px), round(py)), rad, color, -1, cv2.LINE_AA)

    def _tag(self, img, t, color, taken: list, compact: bool) -> None:
        x, y, w, h = t.box
        chip = f"{t.id:02d}"
        info = "" if compact else (f"{round(t.speed)} px/s" if t.status == TRACKING
                                   else ("coasting" if t.status == COASTING else "searching"))
        pad = self._px(5)
        cw, ch, _ = self._text_size(chip, 1.0, thick=self.thick + 1)
        iw = self._text_size(info)[0] if info else 0
        th = ch + 2 * pad
        chip_w = cw + 2 * pad
        tw = chip_w + (iw + 2 * pad if info else 0)
        lead = self._px(10)
        # candidate spots: above-right, above-left, below-right, below-left, right, left
        corners = [
            ((x + w, y), (x + w + lead, y - lead - th), 1),
            ((x, y), (x - lead - tw, y - lead - th), -1),
            ((x + w, y + h), (x + w + lead, y + h + lead), 1),
            ((x, y + h), (x - lead - tw, y + h + lead), -1),
            ((x + w, y + h / 2), (x + w + lead, y + h / 2 - th / 2), 1),
            ((x, y + h / 2), (x - lead - tw, y + h / 2 - th / 2), -1),
        ]
        best = None
        for anchor, (tx, ty), side in corners:
            rect = (round(tx), round(ty), round(tx + tw), round(ty + th))
            inside = rect[0] >= 0 and rect[1] >= 0 and rect[2] <= self.W and rect[3] <= self.H
            if inside and not any(_overlap(rect, o) for o in taken):
                best = (anchor, rect, side)
                break
        if best is None:              # everything is crowded: first spot that fits the frame
            anchor, (tx, ty), side = corners[0]
            tx = min(max(0, tx), self.W - tw)
            ty = min(max(0, ty), self.H - th)
            best = (anchor, (round(tx), round(ty), round(tx + tw), round(ty + th)), side)
        (ax, ay), rect, side = best
        taken.append(rect)
        x0, y0, x1, y1 = rect
        line_c = color if t.status != LOST else self.theme.muted
        # leader: from the box corner to the near edge of the tag
        ex = x0 if side > 0 else x1
        ey = (y0 + y1) // 2
        cv2.line(img, (round(ax), round(ay)), (ex, ey), line_c, max(1, self.thick - 1), cv2.LINE_AA)
        rad = th // 2
        _blend(img, x0, y0, x1, y1, self.theme.panel, self.theme.panel_alpha, rad)
        _blend(img, x0, y0, x0 + chip_w, y1, line_c, 1.0, rad)
        self._text(img, chip, (x0 + pad, y1 - pad), self.theme.chip_text, 1.0, thick=self.thick + 1)
        if info:
            self._text(img, info, (x0 + chip_w + pad, y1 - pad),
                       self.theme.text if t.status == TRACKING else self.theme.muted)

    # ----------------------------------------------------------------- focus card
    def _card(self, img, clean, t, targets) -> None:
        color = self.color(t.id)
        size = round(self.H * 0.24)
        head = self._px(26)
        foot = self._px(26)
        m = self._px(14)
        cw, chh = size, head + size + foot
        spots = [(self.W - cw - m, m), (m, m), (self.W - cw - m, self.H - chh - m - self._px(40)),
                 (m, self.H - chh - m - self._px(40))]

        def crowd(spot):
            sx, sy = spot
            return sum(1 for o in targets
                       if sx - 20 <= o.center[0] <= sx + cw + 20 and sy - 20 <= o.center[1] <= sy + chh + 20)
        ox, oy = min(spots, key=lambda s: (crowd(s), spots.index(s)))
        _blend(img, ox, oy, ox + cw, oy + chh, self.theme.panel, min(0.92, self.theme.panel_alpha + 0.15),
               self._px(10))
        # header: wordmark + ID chip
        self._wordmark(img, (ox + self._px(10), oy + head - self._px(8)), scale=0.95)
        chip = f"{t.id:02d}"
        w, h, _ = self._text_size(chip, thick=self.thick + 1)
        pad = self._px(5)
        _blend(img, ox + cw - w - 3 * pad, oy + self._px(5), ox + cw - pad, oy + head - self._px(5), color,
               1.0, self._px(8))
        self._text(img, chip, (ox + cw - w - 2 * pad, oy + head - self._px(9)), self.theme.chip_text,
                   thick=self.thick + 1)
        # magnified view
        x, y, bw, bh = t.box
        cx, cy = t.center
        side = max(bw, bh) * 1.8 + 8
        x0, y0 = round(cx - side / 2), round(cy - side / 2)
        x1, y1 = x0 + round(side), y0 + round(side)
        H, W = clean.shape[:2]
        crop = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = clean[sy0:sy1, sx0:sx1]
        zoom = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
        zy = oy + head
        img[zy:zy + size, ox:ox + size] = zoom
        c = (ox + size // 2, zy + size // 2)
        rr = round(size * 0.34)
        gap = round(size * 0.08)
        cv2.circle(img, c, rr, color, self.thick, cv2.LINE_AA)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(img, (c[0] + dx * gap, c[1] + dy * gap), (c[0] + dx * (rr + self._px(8)),
                     c[1] + dy * (rr + self._px(8))), color, self.thick, cv2.LINE_AA)
        # footer: speed, state, confidence bar
        fy = zy + size
        state = {TRACKING: "locked", COASTING: "coasting", LOST: "searching"}.get(t.status, "")
        self._text(img, f"{round(t.speed)} px/s", (ox + self._px(10), fy + foot - self._px(9)), self.theme.text)
        bx0, bx1 = ox + cw // 2 + self._px(4), ox + cw - self._px(10)
        by0, by1 = fy + foot // 2 - self._px(3), fy + foot // 2 + self._px(3)
        _blend(img, bx0, by0, bx1, by1, self.theme.muted, 0.5, self._px(3))
        fill = bx0 + round((bx1 - bx0) * max(0.0, min(1.0, t.confidence)))
        if fill > bx0 + 2:
            _blend(img, bx0, by0, fill, by1, color, 1.0, self._px(3))
        sw = self._text_size(state, 0.8)[0]
        self._text(img, state, (bx0 - sw - self._px(8), fy + foot - self._px(9)), self.theme.muted, 0.8)

    # ----------------------------------------------------------------- brand
    def _mark(self, img, center, r) -> None:
        """The camtrak mark: a ring with a short line to a small node."""
        a = self.theme.accent
        cv2.circle(img, center, r, a, max(1, self.thick), cv2.LINE_AA)
        cv2.circle(img, center, max(1, r // 3), a, -1, cv2.LINE_AA)
        n = (center[0] + round(r * 1.6), center[1] - round(r * 1.1))
        cv2.line(img, (center[0] + round(r * 0.7), center[1] - round(r * 0.5)), n, a, max(1, self.thick - 1),
                 cv2.LINE_AA)
        cv2.circle(img, n, max(1, r // 3), a, -1, cv2.LINE_AA)

    def _wordmark(self, img, org, scale=1.0) -> int:
        r = self._px(6 * scale)
        self._mark(img, (org[0] + r, org[1] - r), r)
        tx = org[0] + round(r * 3.2)
        self._text(img, "camtrak", (tx, org[1]), self.theme.accent, 1.15 * scale, font=WORDMARK_FONT)
        return tx + self._text_size("camtrak", 1.15 * scale, font=WORDMARK_FONT)[0]

    def status(self, img, text: str) -> None:
        m = self._px(14)
        h = self._px(30)
        y1 = self.H - m
        y0 = y1 - h
        tw = self._text_size(text)[0]
        mark_w = self._px(14) * 3 + self._text_size("camtrak", 1.15, font=WORDMARK_FONT)[0]
        x0, x1 = m, m + self._px(12) + mark_w + self._px(14) + tw + self._px(14)
        _blend(img, x0, y0, x1, y1, self.theme.panel, self.theme.panel_alpha, h // 2)
        end = self._wordmark(img, (x0 + self._px(12), y1 - self._px(10)))
        self._text(img, text, (end + self._px(14), y1 - self._px(10)), self.theme.muted)

    def hint(self, img, text: str) -> None:
        """A one-line instruction pill, top center."""
        tw, th, _ = self._text_size(text)
        pad = self._px(10)
        x0 = (self.W - tw) // 2 - pad
        y0 = self._px(14)
        _blend(img, x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad, self.theme.panel, self.theme.panel_alpha,
               (th + 2 * pad) // 2)
        self._text(img, text, (x0 + pad, y0 + pad + th), self.theme.accent)

    # ----------------------------------------------------------------- frame
    def draw(self, img, targets, focus_id: int | None = None, status: str = "",
             clean: np.ndarray | None = None) -> np.ndarray:
        """Draw onto `img` in place (and return it). `clean` (the undrawn
        frame) feeds the focus card; without it a copy of `img` is taken."""
        if clean is None and self.focus_card and focus_id is not None:
            clean = img.copy()
        if self.theme.trails:
            for t in targets:
                if t.status != LOST:
                    self._trail(img, t, self.color(t.id))
        for t in targets:
            self._box(img, t, self.color(t.id), t.id == focus_id)
        if self.labels:
            taken = [(round(t.box[0]), round(t.box[1]), round(t.box[0] + t.box[2]), round(t.box[1] + t.box[3]))
                     for t in targets]
            compact = len(targets) > 20
            # focused and locked targets get the best spots
            order = sorted(targets, key=lambda t: (t.id != focus_id, t.status, t.id))
            for t in order:
                self._tag(img, t, self.color(t.id), taken, compact)
        focus = next((t for t in targets if t.id == focus_id), None)
        if self.focus_card and focus is not None:
            self._card(img, clean, focus, targets)
        if self.show_status and status:
            self.status(img, status)
        return img
