# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md.
"""
camtrak.detect - find objects without the user drawing a box.

snap_box(frame, x, y)
    Click-to-lock. Finds the object under (or next to) a click and returns a
    box around it: the pixels that differ from the local background (color
    distance from the median color around the click), the connected blob
    nearest the click. Works for a jet in the sky, a skier on snow, a car on
    tarmac. Falls back to a fixed square if nothing stands out.

BlobDetector
    Auto mode. Finds small, compact objects that stand out against a plain
    background (birds, aircraft, drones against sky; people on snow) with
    morphological top-hat / black-hat filters, and rejects busy areas (tree
    lines, cloud edges, crowds) where everything stands out. A detection only
    becomes a target after it is seen again, close by, on a later pass.

CutDetector
    Flags hard cuts in edited video (color histogram jump).

GlobalMotion / MotionDetector / AutoDetector
    Camera-motion estimate (features + RANSAC affine), and a detector for
    objects that move relative to the scene on busy backgrounds (vehicles on
    terrain, people on a pitch): frames are aligned to cancel the camera,
    then the current frame is compared with two earlier ones; what differs
    from both is a mover. AutoDetector combines both detectors and remembers
    places that turned out to be scenery.

Needs OpenCV + numpy (the C core does not).
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

Box = tuple[float, float, float, float]


def _pad(box: Box, frac: float, min_side: float, shape) -> Box:
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    w, h = max(min_side, w * (1 + 2 * frac)), max(min_side, h * (1 + 2 * frac))
    H, W = shape[:2]
    x0, y0 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x1, y1 = min(float(W), cx + w / 2), min(float(H), cy + h / 2)
    return (x0, y0, x1 - x0, y1 - y0)


def _ring_colors(win: np.ndarray, k: int = 5) -> np.ndarray:
    """The main colors on the window's border (the local background), k-means."""
    ring = np.concatenate([win[0], win[-1], win[:, 0], win[:, -1]]).reshape(-1, win.shape[2])
    ring = np.ascontiguousarray(ring, np.float32)
    k = min(k, len(ring))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(ring, k, None, crit, 1, cv2.KMEANS_PP_CENTERS)
    return centers


def snap_box(frame: np.ndarray, x: float, y: float, *, min_side: float = 12,
             fallback: float = 40, max_half: int = 0) -> Box:
    """
    Box around the object at (x, y). Never fails; worst case a fixed square.

    A square window around the click grows until the object fits inside it.
    Its border is the local background, summarized as a few colors (so
    tarmac, lane paint and shadow all count as background); pixels unlike
    every one of them are the object. The blob under the click wins, else
    the nearest one.
    """
    H, W = frame.shape[:2]
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) if frame.ndim == 3 else frame[:, :, None]
    lab = lab.astype(np.float32)
    best_box, best_area = None, 0
    if max_half <= 0:   # default: big enough for an object filling most of the frame height
        max_half = max(320, int(0.5 * min(H, W)))
    half = 20.0
    while half <= max_half:
        x0, y0 = int(max(0, x - half)), int(max(0, y - half))
        x1, y1 = int(min(W, x + half)), int(min(H, y + half))
        half *= 1.5
        ww, wh = x1 - x0, y1 - y0
        if ww < 8 or wh < 8:
            break
        win = lab[y0:y1, x0:x1]
        centers = _ring_colors(win)
        dist = np.min(np.stack([np.sqrt(((win - c) ** 2).sum(axis=2)) for c in centers]), axis=0)
        ring_d = np.concatenate([dist[0], dist[-1], dist[:, 0], dist[:, -1]])
        thr = max(14.0, float(np.percentile(ring_d, 98)) + 10.0)
        mask = (dist > thr).astype(np.uint8)
        # join the parts of one object (helmet, body, wheels); gap grows with the window
        k = max(3, int(0.07 * max(ww, wh)) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            continue
        px, py = int(x - x0), int(y - y0)
        pick = int(labels[min(py, wh - 1), min(px, ww - 1)])
        if pick == 0 or stats[pick][4] < 6:
            pick, pick_d = -1, 1e9
            for i in range(1, n):
                bx, by, bw, bh, area = stats[i]
                if area < 6:
                    continue
                d = float(np.hypot(max(bx - px, 0, px - (bx + bw)), max(by - py, 0, py - (by + bh))))
                if d < pick_d:
                    pick, pick_d = i, d
            if pick < 0 or pick_d > 12:
                continue
        bx, by, bw, bh, area = stats[pick]
        touches = (bx <= 0 and x0 > 0) or (by <= 0 and y0 > 0) or \
                  (bx + bw >= ww and x1 < W) or (by + bh >= wh and y1 < H)
        if touches or bw * bh > 0.4 * ww * wh:
            continue                       # not contained at this scale: look wider
        # the object is the largest well-contained blob seen at any scale; if a
        # scale finds a much smaller one, we have stepped past it
        if area >= best_area:
            best_box, best_area = (float(x0 + bx), float(y0 + by), float(bw), float(bh)), area
    if best_box is not None:
        return _pad(best_box, 0.0, min_side, frame.shape)
    return _pad((x - fallback / 2, y - fallback / 2, fallback, fallback), 0.0, min_side, frame.shape)


def refine_box(frame: np.ndarray, box: Box, *, pad: float = 0.1) -> Box:
    """
    Tighten a hand-drawn box onto the object inside it. A loose box carries
    background into the tracker's model, and the size estimate can then creep
    outwards into that background. Returns the original box when no clear
    single object is found inside it.
    """
    x, y, w, h = box
    s = snap_box(frame, x + w / 2, y + h / 2, max_half=int(max(w, h)) + 10)
    sx0, sy0, sx1, sy1 = s[0], s[1], s[0] + s[2], s[1] + s[3]
    ix = max(0.0, min(sx1, x + w) - max(sx0, x))
    iy = max(0.0, min(sy1, y + h) - max(sy0, y))
    inside = ix * iy / max(1.0, s[2] * s[3])          # share of the snapped box inside the drawn one
    share = s[2] * s[3] / max(1.0, w * h)             # snapped area relative to the drawn one
    if inside >= 0.85 and 0.15 <= share <= 0.8:
        return _pad(s, pad, 8, frame.shape)
    return box


class BlobDetector:
    """
    Small-object detector for auto mode.

    detect(frame) -> boxes of compact blobs that contrast with a plain
    surrounding. confirm(frame) -> the subset that were also seen on the
    previous pass within `max_jump` px (objects, not one-frame noise).
    """

    def __init__(self, *, work_width: int = 960, min_size: int = 3, max_size: int = 60,
                 sensitivity: float = 6.0, max_clutter: float = 0.04,
                 max_jump: float = 60.0) -> None:
        self.work_width = work_width
        self.min_size = min_size          # px, at work resolution
        self.max_size = max_size          # px, at work resolution
        self.sensitivity = sensitivity    # threshold = sensitivity x robust noise level
        self.max_clutter = max_clutter    # max fraction of busy pixels around a blob
        self.max_jump = max_jump          # px (full resolution) between two passes
        self._prev: list[tuple[float, float, float]] = []

    def reset(self) -> None:
        self._prev = []

    def detect(self, frame: np.ndarray) -> list[Box]:
        H, W = frame.shape[:2]
        s = min(1.0, self.work_width / W)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
        k = max(7, int(self.max_size * 0.6) | 1)
        # background = median over a window larger than the objects; unlike a
        # morphological closing it does not bridge neighboring objects
        bg = cv2.medianBlur(small, k)
        resp = cv2.absdiff(small, bg).astype(np.float32)
        sub = resp[::3, ::3]                     # a subsample estimates the noise level just as well
        med = float(np.median(sub))
        mad = float(np.median(np.abs(sub - med))) * 1.4826 + 1.0
        thr = max(14.0, med + self.sensitivity * mad)
        mask = (resp > thr).astype(np.uint8)
        # busy-ness map: share of strong-ish pixels in a neighborhood
        busy = cv2.blur((resp > thr * 0.5).astype(np.float32), (k * 3, k * 3))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes: list[Box] = []
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if max(bw, bh) < self.min_size or max(bw, bh) > self.max_size:
                continue
            if area < 0.2 * bw * bh and max(bw, bh) > 8:
                continue                   # thin streak (cloud edge, contrail, power line)
            cx, cy = cents[i]
            own = area / float((k * 3) ** 2)
            if busy[int(cy), int(cx)] - own > self.max_clutter:
                continue                   # surrounded by other stuff: texture, not an object
            boxes.append(_pad((bx / s, by / s, bw / s, bh / s), 0.15, 12, frame.shape))
        return boxes

    def confirm(self, frame: np.ndarray, camera: np.ndarray | None = None) -> list[Box]:
        """detect(), keeping only blobs also seen on the previous pass. `camera`
        (2x3, previous pass -> now) moves the previous detections first."""
        boxes = self.detect(frame)
        now = [(b[0] + b[2] / 2, b[1] + b[3] / 2, max(b[2], b[3])) for b in boxes]
        prev = self._prev
        if camera is not None and prev:
            pts = cv2.transform(np.array([[(p[0], p[1]) for p in prev]], np.float32), camera)[0]
            prev = [(float(x), float(y), p[2]) for (x, y), p in zip(pts, prev)]
        keep = []
        for b, (cx, cy, size) in zip(boxes, now):
            for px, py, ps in prev:
                if np.hypot(cx - px, cy - py) <= self.max_jump and 0.5 <= size / ps <= 2.0:
                    keep.append(b)
                    break
        self._prev = now
        return keep


# ---------------------------------------------------------------------------
# camera motion and movers


def _h(M: np.ndarray) -> np.ndarray:
    return np.vstack([M, [0.0, 0.0, 1.0]])


def compose(*Ms: np.ndarray) -> np.ndarray:
    """compose(A, B) = apply A, then B (all 2x3 affine)."""
    out = np.eye(3)
    for M in Ms:
        out = _h(M) @ out
    return out[:2]


class GlobalMotion:
    """
    How the camera moved between consecutive frames, as a 2x3 affine
    (previous frame -> this frame, full-resolution pixels). None when the
    picture has too little texture to tell (plain sky) or the estimate is
    unreliable (most features disagree).
    """

    def __init__(self, work_width: int = 640, min_features: int = 40,
                 min_inlier_ratio: float = 0.6) -> None:
        self.work_width = work_width
        self.min_features = min_features
        self.min_inlier_ratio = min_inlier_ratio
        self.scale = 1.0
        self.gray: np.ndarray | None = None      # last frame, small gray
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        H, W = frame.shape[:2]
        s = min(1.0, self.work_width / W)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g.copy()
        self.scale, self.gray = s, g
        prev, self._prev = self._prev, g
        if prev is None or prev.shape != g.shape:
            return None
        pts = cv2.goodFeaturesToTrack(prev, 400, 0.01, 10, blockSize=7)
        if pts is None or len(pts) < self.min_features:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < self.min_features:
            return None
        M, inl = cv2.estimateAffine2D(pts[ok], nxt[ok], method=cv2.RANSAC,
                                      ransacReprojThreshold=1.0, maxIters=1000, confidence=0.995)
        if M is None or inl is None or inl.mean() < self.min_inlier_ratio \
                or inl.sum() < self.min_features:
            return None
        M = M.astype(np.float64)
        M[:, 2] /= s                      # small -> full resolution (uniform scale)
        return M


class MotionDetector:
    """
    Finds objects that move relative to the scene, with the camera's own
    motion canceled. Feed every frame with push(); call detect() any time.

    The current frame is compared with the frames `gap` and `2*gap` back,
    each first warped onto the current view. A mover differs from both at its
    current position; its old positions ("ghosts") differ in only one, so
    taking the minimum of the two differences removes them. Differences are
    measured against the 3x3 min/max of the warped frame, so 1 px alignment
    error on edges does not count as motion.
    """

    def __init__(self, *, gap: int = 4, threshold: float = 18.0, min_size: int = 3,
                 max_frac: float = 0.15, max_moving: float = 0.03) -> None:
        self.gap = gap
        self.threshold = threshold        # gray levels
        self.min_size = min_size          # px at work resolution
        self.max_frac = max_frac          # largest blob, fraction of the frame side
        self.max_moving = max_moving      # more of the frame "moving" than this = bad alignment
        self._hist: deque = deque(maxlen=2 * gap + 1)   # (small gray, camera from previous, small)

    def reset(self) -> None:
        self._hist.clear()

    def push(self, gray_small: np.ndarray, camera_small: np.ndarray | None) -> None:
        if camera_small is None and self._hist:
            self._hist.clear()            # chain broken: start again
        self._hist.append((gray_small, camera_small))

    def _diff(self, cur: np.ndarray, old: np.ndarray, M: np.ndarray) -> np.ndarray:
        h, w = cur.shape
        warped = cv2.warpAffine(old, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpAffine(np.full_like(old, 255), M, (w, h), flags=cv2.INTER_NEAREST)
        k = np.ones((3, 3), np.uint8)
        d = cv2.max(cv2.subtract(cur, cv2.dilate(warped, k)), cv2.subtract(cv2.erode(warped, k), cur))
        d[cv2.erode(valid, np.ones((7, 7), np.uint8)) == 0] = 0
        return d

    def detect(self, scale: float, shape) -> list[Box]:
        if len(self._hist) < self._hist.maxlen:
            return []
        frames = list(self._hist)
        cur = frames[-1][0]
        g = self.gap
        M1 = compose(*[m for _, m in frames[-g:]])            # t-g  -> t
        M2 = compose(*[m for _, m in frames[-2 * g:]])        # t-2g -> t
        cur_b = cv2.GaussianBlur(cur, (3, 3), 0)
        d1 = self._diff(cur_b, cv2.GaussianBlur(frames[-1 - g][0], (3, 3), 0), M1)
        d2 = self._diff(cur_b, cv2.GaussianBlur(frames[0][0], (3, 3), 0), M2)
        both = cv2.min(d1, d2)
        mask = (both > self.threshold).astype(np.uint8)
        if mask.mean() > self.max_moving:
            return []                      # alignment failed (parallax, zoom blur): trust nothing
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        h, w = cur.shape
        boxes: list[Box] = []
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if max(bw, bh) < self.min_size or area < 6 or max(bw, bh) > self.max_frac * max(w, h):
                continue
            boxes.append(_pad((bx / scale, by / scale, bw / scale, bh / scale), 0.3, 14, shape))
        return boxes


class AutoDetector:
    """
    Auto mode's candidate source. push(frame) every frame (returns the camera
    motion, or None); candidates(frame) every few frames.

      * objects that stand out against a plain surrounding (BlobDetector)
      * objects that move relative to the scene (MotionDetector), when the
        scene has enough texture to measure camera motion
    Both must be seen on two passes in a row. Places reported as scenery via
    forget() are skipped for a while (they move with the camera).
    """

    def __init__(self, *, sensitivity: float = 6.0, motion: bool = True,
                 forget_frames: int = 120) -> None:
        self.blobs = BlobDetector(sensitivity=sensitivity)
        self.motion = MotionDetector() if motion else None
        self.camera = GlobalMotion()
        self.forget_frames = forget_frames
        self._since_pass: np.ndarray | None = np.eye(3)[:2]   # camera motion since last pass
        self._prev_motion: list[tuple[float, float, float]] = []
        self._scenery: list[list] = []      # [x, y, radius, frames left]
        self.textured = False               # camera motion measurable on the last frame

    def reset(self) -> None:
        self.blobs.reset()
        self.camera.reset()
        if self.motion:
            self.motion.reset()
        self._since_pass = np.eye(3)[:2]
        self._prev_motion = []
        self._scenery = []

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        M = self.camera.update(frame)
        self.textured = M is not None
        if self.motion:
            Ms = None
            if M is not None:
                Ms = M.copy()
                Ms[:, 2] *= self.camera.scale
            self.motion.push(self.camera.gray, Ms)
        self._since_pass = compose(self._since_pass, M) if (M is not None and self._since_pass is not None) else None
        if self._scenery:
            if M is not None:
                pts = cv2.transform(np.array([[(p[0], p[1]) for p in self._scenery]], np.float32), M)[0]
                for p, (x, y) in zip(self._scenery, pts):
                    p[0], p[1] = float(x), float(y)
            for p in self._scenery:
                p[3] -= 1
            self._scenery = [p for p in self._scenery if p[3] > 0]
        return M

    def forget(self, box: Box) -> None:
        """This box was scenery (did not move with respect to the scene)."""
        x, y, w, h = box
        self._scenery.append([x + w / 2, y + h / 2, max(w, h), self.forget_frames])

    def _is_scenery(self, box: Box) -> bool:
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        return any(np.hypot(cx - p[0], cy - p[1]) < max(p[2], 12) for p in self._scenery)

    @staticmethod
    def _center(frame: np.ndarray, box: Box) -> Box:
        """A motion blob mostly covers the leading edge of a mover. Re-box the
        object found at the blob's center, if that is a sensible object box."""
        x, y, w, h = box
        s = snap_box(frame, x + w / 2, y + h / 2, max_half=int(max(w, h) * 2) + 20)
        cx, cy = x + w / 2, y + h / 2
        covers = s[0] <= cx <= s[0] + s[2] and s[1] <= cy <= s[1] + s[3]
        if covers and s[2] * s[3] <= 6.0 * w * h and min(s[2], s[3]) >= 0.3 * min(w, h):
            return s
        return box

    def candidates(self, frame: np.ndarray, skip=None) -> list[Box]:
        """New-object boxes. skip(box) -> True for boxes already tracked: they are
        dropped before the (costly) re-boxing of motion blobs."""
        cam = self._since_pass
        out = self.blobs.confirm(frame, cam)
        if self.motion:
            boxes = self.motion.detect(self.camera.scale, frame.shape)
            now = [(b[0] + b[2] / 2, b[1] + b[3] / 2, max(b[2], b[3])) for b in boxes]
            prev = self._prev_motion
            if cam is not None and prev:
                pts = cv2.transform(np.array([[(p[0], p[1]) for p in prev]], np.float32), cam)[0]
                prev = [(float(x), float(y), p[2]) for (x, y), p in zip(pts, prev)]
            elif cam is None:
                prev = []
            for b, (cx, cy, size) in zip(boxes, now):
                if any(np.hypot(cx - px, cy - py) <= max(40.0, size) for px, py, _ in prev):
                    if skip is None or not skip(b):
                        out.append(self._center(frame, b))
            self._prev_motion = now
        self._since_pass = np.eye(3)[:2]
        # merge overlapping candidates from the two detectors, drop known scenery
        merged: list[Box] = []
        for b in sorted(out, key=lambda b: -b[2] * b[3]):
            if self._is_scenery(b):
                continue
            cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
            if any(m[0] <= cx <= m[0] + m[2] and m[1] <= cy <= m[1] + m[3] for m in merged):
                continue
            merged.append(b)
        return merged


class CutDetector:
    """
    Flags hard cuts in edited video: a big jump in the color histogram between
    consecutive frames. Works on a 160 px wide thumbnail, so it is cheap at any
    resolution. Threshold tuned on drone ski footage (cuts scored 0.27-0.84).
    """

    def __init__(self, threshold: float = 0.25, min_gap: int = 5) -> None:
        self.threshold = threshold
        self.min_gap = min_gap
        self._prev = None
        self._since = min_gap

    def reset(self) -> None:
        self._prev = None
        self._since = self.min_gap

    def is_cut(self, img: np.ndarray) -> bool:
        thumb = cv2.resize(img, (160, max(1, 160 * img.shape[0] // img.shape[1])))
        hist = cv2.calcHist([cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)], [0, 1], None,
                            [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        cut = False
        self._since += 1
        if self._prev is not None and self._since >= self.min_gap:
            cut = cv2.compareHist(self._prev, hist, cv2.HISTCMP_BHATTACHARYYA) > self.threshold
            if cut:
                self._since = 0
        self._prev = hist
        return cut
