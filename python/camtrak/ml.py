# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md.
"""
camtrak.ml - Camtrak + a small neural tracker as a "second opinion".

    import camtrak as tt
    trk = tt.MLTracker()                 # uses models/vittrack.onnx
    trk.lock(frame, (x, y, w, h))
    r = trk.update(frame)                # same Result as tt.Tracker
    trk.last_info                        # what the ML side did this frame

How the two cooperate:
  * Camtrak (correlation filter, ~0.6 ms) is the primary tracker and the judge
    of honesty: it knows when it has lost the target.
  * VitTrack (~4 ms, OpenCV Zoo, Apache 2.0) runs alongside. It recognizes the
    target from deep features, so it copes better with big appearance changes
    and scene cuts, but it never admits it is lost and can drift.
  * While Camtrak is confident, it keeps VitTrack on the right object
    (re-initializing it when the two disagree, and periodically so VitTrack
    knows the latest look of the target).
  * While Camtrak is COASTING or LOST, VitTrack's box is a candidate. It is
    accepted only if it is plausible (sane size, inside the frame) and Camtrak's
    own appearance models agree with it (camtrak_verify PSR >= verify_psr).
    Then Camtrak is moved there (camtrak_reacquire) and carries on.

Requires OpenCV >= 4.9 (cv2.TrackerVit). The pure-C core does not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import LOST, TRACKING, Result, Tracker, CamtrakError

DEFAULT_MODEL = Path(__file__).resolve().parent.parent.parent / "models" / "vittrack.onnx"


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class MLTracker:
    def __init__(self, model_path: str | Path | None = None, *, verify_psr: float = 7.0,
                 ml_min_score: float = 0.3, resync_every: int = 15, max_rejections: int = 10,
                 **params: float) -> None:
        import cv2  # only the ML path needs OpenCV

        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.is_file():
            raise CamtrakError(f"ML model not found: {path}")
        self._cv2 = cv2
        self._model = str(path)
        self.verify_psr = verify_psr
        self.ml_min_score = ml_min_score
        self.resync_every = resync_every
        self.max_rejections = max_rejections
        self.tracker = Tracker(**params)
        self._vit = None
        self._since_sync = 0
        self._rejections = 0
        self._good: tuple[np.ndarray, tuple] | None = None   # last view Camtrak confirmed
        self._good_crop = None
        self._sweep_i = 0
        self.last_info: dict = {}

    def _new_vit(self, frame: np.ndarray, box):
        p = self._cv2.TrackerVit_Params()
        p.net = self._model
        vit = self._cv2.TrackerVit_create(p)
        vit.init(frame, tuple(int(round(v)) for v in box))
        return vit

    def _init_vit(self, frame: np.ndarray, box) -> None:
        self._vit = self._new_vit(frame, box)
        self._since_sync = 0

    def _remember_good(self, frame: np.ndarray, box) -> None:
        """Keep the last confirmed view: the frame (for resets) and a crop of
        the target with context (for the whole-frame sweep)."""
        self._good = (frame.copy(), tuple(box))
        h, w = frame.shape[:2]
        x, y, bw, bh = box
        x0, y0 = max(0, int(x - bw)), max(0, int(y - bh))
        x1, y1 = min(w, int(x + 2 * bw)), min(h, int(y + 2 * bh))
        self._good_crop = (frame[y0:y1, x0:x1].copy(), (x - x0, y - y0, bw, bh))

    def _sweep_probe(self, frame: np.ndarray):
        """
        Whole-frame search with the ML tracker, one spot per call. VitTrack only
        searches around where its template was taken, so we paste the last
        confirmed view of the target onto a blank canvas at the next grid spot,
        start a tracker from that, and let it look around that spot in the
        real frame. Returns (ok, box, score, tracker).
        """
        if self._good_crop is None:
            return False, None, 0.0, None
        crop, (ox, oy, bw, bh) = self._good_crop
        h, w = frame.shape[:2]
        step = max(bw, bh) * 2.5
        xs = np.arange(step / 2, w, step)
        ys = np.arange(step / 2, h, step)
        spots = [(cx, cy) for cy in ys for cx in xs]
        cx, cy = spots[self._sweep_i % len(spots)]
        self._sweep_i += 1
        canvas = np.zeros_like(frame)
        px, py = int(cx - bw / 2 - ox), int(cy - bh / 2 - oy)
        ch, cw = crop.shape[:2]
        sx0, sy0 = max(0, -px), max(0, -py)
        dx0, dy0 = max(0, px), max(0, py)
        dx1, dy1 = min(w, px + cw), min(h, py + ch)
        if dx1 <= dx0 or dy1 <= dy0:
            return False, None, 0.0, None
        canvas[dy0:dy1, dx0:dx1] = crop[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
        probe = self._new_vit(canvas, (cx - bw / 2, cy - bh / 2, bw, bh))
        ok, b = probe.update(frame)
        return ok, tuple(float(v) for v in b), float(probe.getTrackingScore()), probe

    def _camtrak_shaped(self, box) -> tuple:
        """The ML box's position and area, but the shape of the user's box,
        exactly as camtrak_reacquire sizes it, so the box never changes shape."""
        x, y, w, h = box
        area = w * h
        bw = (area * self._aspect) ** 0.5
        bh = bw / self._aspect
        return (x + w / 2 - bw / 2, y + h / 2 - bh / 2, bw, bh)

    def lock(self, frame: np.ndarray, bbox) -> None:
        self._aspect = float(bbox[2]) / float(bbox[3])
        self.tracker.lock(frame, bbox)
        self._init_vit(frame, bbox)
        self._remember_good(frame, bbox)
        self._rejections = 0

    def scene_cut(self) -> None:
        self.tracker.scene_cut()

    def _plausible(self, box, ref, frame: np.ndarray) -> bool:
        h, w = frame.shape[:2]
        x, y, bw, bh = box
        if bw < 4 or bh < 4 or x < -bw / 2 or y < -bh / 2 or x + bw / 2 > w or y + bh / 2 > h:
            return False
        if bw * bh > 0.25 * w * h:       # VitTrack sometimes balloons to the whole frame
            return False
        ratio = (bw * bh) / max(1.0, ref[2] * ref[3])
        return 1 / 9 <= ratio <= 9

    def update(self, frame: np.ndarray) -> Result:
        r = self.tracker.update(frame)
        ok, vbox = self._vit.update(frame)
        vbox = tuple(float(v) for v in vbox)
        vscore = float(self._vit.getTrackingScore())
        self._since_sync += 1
        info = {"ml_score": round(vscore, 3), "ml_box": [round(v, 1) for v in vbox],
                "verify_psr": None, "action": ""}

        if r.status == TRACKING and r.confidence >= 0.8:
            # Camtrak is sure: keep the ML tracker on the same object.
            self._rejections = 0
            if not ok or _iou(vbox, r.box) < 0.3 or self._since_sync >= self.resync_every:
                info["action"] = "ml_resync" if self._since_sync < self.resync_every else "ml_refresh"
                self._init_vit(frame, r.box)
                self._remember_good(frame, r.box)
        elif r.status != TRACKING and ok and vscore >= self.ml_min_score \
                and self._plausible(vbox, r.box, frame):
            # Camtrak is unsure: does the ML candidate look like our target?
            v = self.tracker.verify(frame, vbox)
            info["verify_psr"] = round(v["psr"], 2)
            info["verify_color"] = round(v["color"], 3)
            if v["accept"]:
                box = self._camtrak_shaped(vbox)
                self.tracker.reacquire(frame, box)
                info["action"] = "ml_reacquire"
                conf = min(1.0, max(0.0, (v["psr"] - 5.0) / 15.0))
                r = Result(box, conf, v["psr"], TRACKING)
            else:
                info["action"] = "ml_rejected"
                self._rejections += 1
                if self._rejections >= self.max_rejections and self._good is not None:
                    # The ML tracker keeps proposing the wrong object (e.g. the
                    # other skier): restart it from the last confirmed view.
                    self._init_vit(*self._good)
                    self._rejections = 0
                    info["action"] = "ml_reset"

        if r.status == LOST and info["action"] != "ml_reacquire":
            # Nothing near the last position: sweep the ML tracker over the frame.
            pok, pbox, pscore, probe = self._sweep_probe(frame)
            if pok and pscore >= self.ml_min_score and self._plausible(pbox, r.box, frame):
                v = self.tracker.verify(frame, pbox)
                info.update(verify_psr=round(v["psr"], 2), verify_color=round(v["color"], 3))
                if v["accept"]:
                    box = self._camtrak_shaped(pbox)
                    self.tracker.reacquire(frame, box)
                    self._vit, self._since_sync = probe, 0
                    info["action"] = "ml_sweep_reacquire"
                    conf = min(1.0, max(0.0, (v["psr"] - 5.0) / 15.0))
                    r = Result(box, conf, v["psr"], TRACKING)
                else:
                    info["action"] = "ml_sweep_rejected"
        self.last_info = info
        return r

    def debug(self) -> dict:
        return {**self.tracker.debug(), **{k: v for k, v in self.last_info.items() if k != "ml_box"}}

    def close(self) -> None:
        self.tracker.close()
        self._vit = None

    def __enter__(self) -> "MLTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
