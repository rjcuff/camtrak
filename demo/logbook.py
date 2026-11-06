"""
logbook - every demo run saves a folder you can send for debugging.

  logs/<date>-<time>_<video name>/
    frames.csv      one row per frame: result + the tracker's internal state
    events.jsonl    what the user did (pick, re-pick, pause, quit) and scene cuts
    snapshots/      annotated frames at every status change, pick and cut
                    (plus one every 5 s), so the failure can be *seen*
    summary.json    totals, settings, timing
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

MAX_SNAPSHOTS = 400
SNAPSHOT_WIDTH = 960
DEBUG_FIELDS = ["pred_x", "pred_y", "vel_x", "vel_y", "scale", "scale_factor", "misses",
                "lost_frames", "search_rings", "used_anchor", "learned", "windows_tried",
                "ml_score", "verify_psr", "verify_color", "action"]   # last four: MLTracker only


class Logbook:
    def __init__(self, root: Path, source: Path, fps: float) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = root / f"{stamp}_{source.stem}"
        (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.dir / "frames.csv", "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["frame", "time_s", "status", "confidence", "psr", "bbox_x", "bbox_y",
                            "bbox_w", "bbox_h", "scene_cut", "update_ms", *DEBUG_FIELDS])
        self._events = open(self.dir / "events.jsonl", "w")
        self._fps = fps or 30.0
        self._last_status: str | None = None
        self._last_snapshot = -10**9
        self.snapshots = 0
        self.started = time.time()

    def event(self, frame: int, kind: str, **data) -> None:
        record = {"frame": frame, "time_s": round(frame / self._fps, 2), "event": kind, **data}
        self._events.write(json.dumps(record) + "\n")
        self._events.flush()

    def frame(self, index: int, r, debug: dict, cut: bool, ms: float, annotated: np.ndarray) -> None:
        self._csv.writerow([index, round(index / self._fps, 2), r.status_name, f"{r.confidence:.3f}",
                            f"{r.psr:.2f}", *(f"{v:.1f}" for v in r.box), int(cut), f"{ms:.3f}",
                            *(f"{v:.2f}" if isinstance(v, float) else ("" if v is None else v)
                              for v in (debug.get(k) for k in DEBUG_FIELDS))])
        if debug.get("action") in ("ml_reacquire", "ml_rejected", "ml_sweep_reacquire"):
            self.event(index, debug["action"], verify_psr=debug.get("verify_psr"),
                       verify_color=debug.get("verify_color"), ml_score=debug.get("ml_score"))
            if debug["action"].endswith("reacquire"):
                self.snapshot(index, annotated, "ml_reacquire")
        changed = r.status_name != self._last_status
        if changed:
            self.event(index, "status", status=r.status_name, from_status=self._last_status,
                       psr=round(r.psr, 2))
        if changed or cut or index - self._last_snapshot >= 5 * self._fps:
            reason = "cut" if cut else (r.status_name.lower() if changed else "periodic")
            self.snapshot(index, annotated, reason)
        self._last_status = r.status_name

    def snapshot(self, index: int, image: np.ndarray, reason: str) -> None:
        if self.snapshots >= MAX_SNAPSHOTS:
            return
        h, w = image.shape[:2]
        if w > SNAPSHOT_WIDTH:
            image = cv2.resize(image, (SNAPSHOT_WIDTH, h * SNAPSHOT_WIDTH // w), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(self.dir / "snapshots" / f"{index:06d}_{reason}.jpg"), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.snapshots += 1
        self._last_snapshot = index

    def close(self, summary: dict) -> None:
        summary = {**summary, "wall_time_s": round(time.time() - self.started, 1),
                   "snapshots": self.snapshots}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self._csv_file.close()
        self._events.close()
