# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md.
"""
camtrak.extent - where exactly is the object, and how big is it?

The correlation tracker is excellent at *where* but estimates *how big* only
from the box's own contents, so a box that starts loose (or picks up some
background) can grow into that background. ExtentModel adds a second,
independent opinion based on color:

  * at lock, it learns which colors belong to the target (inside the box) and
    which to its surroundings (a ring around the box): a likelihood ratio per
    color bin, like a back-projection.
  * each frame, it finds the target-colored blob at the tracked point and
    returns its box (fit), or reports that there is none there (presence).

Used to tighten a loose starting box, keep the box size honest while
tracking, and catch a tracker that drifted onto background while still
claiming a confident match.

Needs OpenCV + numpy.
"""
from __future__ import annotations

import cv2
import numpy as np

Box = tuple[float, float, float, float]

_LB, _AB, _BB = 8, 12, 12            # Lab histogram bins
_NBINS = _LB * _AB * _BB
_MAX_SIDE = 160                      # analysis window is resampled to at most this many px


def _bins(lab: np.ndarray) -> np.ndarray:
    L = (lab[..., 0].astype(np.int32) * _LB) >> 8
    a = (lab[..., 1].astype(np.int32) * _AB) >> 8
    b = (lab[..., 2].astype(np.int32) * _BB) >> 8
    return (L * _AB + a) * _BB + b


def _window(frame, cx, cy, half_w, half_h):
    H, W = frame.shape[:2]
    x0, y0 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    x1, y1 = int(min(W, cx + half_w)), int(min(H, cy + half_h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    s = min(1.0, _MAX_SIDE / max(x1 - x0, y1 - y0))
    roi = frame[y0:y1, x0:x1]
    if s < 1.0:
        roi = cv2.resize(roi, (max(4, round((x1 - x0) * s)), max(4, round((y1 - y0) * s))),
                         interpolation=cv2.INTER_AREA)
    return roi, (x0, y0, x1, y1), s


class ExtentModel:
    """Color model of one target. `ok` is False when the target's colors do
    not stand out from its surroundings; then fit() and presence are not used."""

    RATIO = 2.0         # a color is "target" when it is this much likelier inside the box
    MIN_SEPARABLE = 0.3  # share of the target's pixels that must be in such colors

    def __init__(self, frame: np.ndarray, box: Box) -> None:
        self.ok = False
        self.lut = np.zeros(_NBINS, np.float32)
        self.separability = 0.0
        if frame.ndim != 3:
            return
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        win = _window(frame, cx, cy, 1.3 * w, 1.3 * h)        # ring out to 2.6x the box
        if win is None:
            return
        roi, (x0, y0, x1, y1), s = win
        idx = _bins(cv2.cvtColor(roi, cv2.COLOR_BGR2LAB))
        hh, ww = idx.shape
        ys, xs = np.mgrid[0:hh, 0:ww]
        px = x0 + (xs + 0.5) / s
        py = y0 + (ys + 0.5) / s
        inside = (np.abs(px - cx) <= w / 2) & (np.abs(py - cy) <= h / 2)
        ring = (np.abs(px - cx) > 0.65 * w) | (np.abs(py - cy) > 0.65 * h)
        if inside.sum() < 16 or ring.sum() < 64:
            return
        # center-weighted (Epanechnikov kernel): the object is usually in the
        # middle of the box, background at its edges, especially in a loose box
        kern = np.clip(1.0 - ((px - cx) / (w / 2)) ** 2 - ((py - cy) / (h / 2)) ** 2, 0.0, None)
        ht = np.bincount(idx[inside], weights=kern[inside], minlength=_NBINS).astype(np.float32)
        hb = np.bincount(idx[ring], minlength=_NBINS).astype(np.float32)
        ht /= ht.sum()
        hb /= hb.sum()
        ratio = (ht + 1e-4) / (hb + 1e-4)
        self.lut = np.log(ratio).astype(np.float32)
        self.separability = float(ht[ratio >= self.RATIO].sum())
        self.ok = self.separability >= self.MIN_SEPARABLE

    def likelihood(self, frame: np.ndarray, cx: float, cy: float, half_w: float, half_h: float):
        win = _window(frame, cx, cy, half_w, half_h)
        if win is None:
            return None
        roi, rect, s = win
        return self.lut[_bins(cv2.cvtColor(roi, cv2.COLOR_BGR2LAB))], rect, s

    def fit(self, frame: np.ndarray, box: Box, search: float = 1.25) -> tuple[Box | None, bool]:
        """(box of the target-colored blob at the box's center or None, present).
        present is False when nothing target-colored is near the center."""
        if not self.ok:
            return None, True
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        got = self.likelihood(frame, cx, cy, search * w + 4, search * h + 4)
        if got is None:
            return None, True
        L, (x0, y0, x1, y1), s = got
        mask = (L >= np.log(self.RATIO)).astype(np.uint8)
        k = max(3, int(0.08 * min(w, h) * s) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        hh, ww = mask.shape
        pcx, pcy = int((cx - x0) * s), int((cy - y0) * s)
        pick = int(labels[min(max(pcy, 0), hh - 1), min(max(pcx, 0), ww - 1)])
        if pick == 0:
            best, best_d = 0, 1e9
            reach = 0.35 * max(w, h) * s + 2
            for i in range(1, n):
                bx, by, bw, bh, area = stats[i]
                if area < 4:
                    continue
                d = float(np.hypot(max(bx - pcx, 0, pcx - (bx + bw)), max(by - pcy, 0, pcy - (by + bh))))
                if d < best_d:
                    best, best_d = i, d
            if best == 0 or best_d > reach:
                return None, False
            pick = best
        bx, by, bw, bh, area = stats[pick]
        if area < 4:
            return None, False
        # touching the window on opposite sides: the blob is background leaking in
        if (bx <= 0 and bx + bw >= ww) or (by <= 0 and by + bh >= hh):
            return None, True
        return (x0 + bx / s, y0 + by / s, bw / s, bh / s), True


def tighten(frame: np.ndarray, box: Box) -> Box:
    """A loose starting box, tightened onto the target-colored blob inside it.
    Returns the box unchanged when there is no clear blob, or it is not
    clearly smaller."""
    m = ExtentModel(frame, box)
    if not m.ok:
        return box
    fit, _ = m.fit(frame, box, search=0.5)
    if fit is None:
        return box
    x, y, w, h = box
    fx, fy, fw, fh = fit
    share = fw * fh / max(1.0, w * h)
    if share > 0.85 or share < 0.05:
        return box
    pad = 0.06
    return (fx - fw * pad, fy - fh * pad, fw * (1 + 2 * pad), fh * (1 + 2 * pad))
