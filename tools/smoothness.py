"""
smoothness - how smooth was a run? Reads a log folder's frames.csv.

    python tools/smoothness.py logs/20261105-230732_videoplayback_trimmed

Within each shot (scene cuts split the video), over TRACKING frames:
  jitter      center acceleration: how far the box moved compared with the
              move before (a steady glide scores ~0, a shaky box scores high), px
  size jumps  frames where width changed by more than 8% from the previous frame
  jumps       frames where the center moved more than half the box size at once
  flicker     status changes per 100 frames (TRACKING <-> COASTING <-> LOST)
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    rows = list(csv.DictReader(open(Path(sys.argv[1]) / "frames.csv")))
    accel, size_jumps, jumps, changes, tracked = [], 0, 0, 0, 0
    prev = prev2 = None
    for i, r in enumerate(rows):
        if i and r["status"] != rows[i - 1]["status"]:
            changes += 1
        if r["scene_cut"] == "1" or r["status"] != "TRACKING":
            prev = prev2 = None
            continue
        tracked += 1
        x, y = float(r["bbox_x"]) + float(r["bbox_w"]) / 2, float(r["bbox_y"]) + float(r["bbox_h"]) / 2
        w, h = float(r["bbox_w"]), float(r["bbox_h"])
        if prev:
            if abs(w / prev[2] - 1) > 0.08:
                size_jumps += 1
            if math.hypot(x - prev[0], y - prev[1]) > 0.5 * min(w, h):
                jumps += 1
            if prev2:
                accel.append(math.hypot(x - 2 * prev[0] + prev2[0], y - 2 * prev[1] + prev2[1]))
        prev2, prev = prev, (x, y, w, h)
    accel.sort()
    n = len(rows)
    print(f"frames {n}, tracking {tracked} ({100 * tracked / n:.0f}%)")
    if accel:
        print(f"jitter      median {accel[len(accel) // 2]:.2f} px, 95th pct {accel[int(0.95 * len(accel))]:.2f} px")
    print(f"size jumps  {size_jumps}   center jumps {jumps}   flicker {100 * changes / n:.1f} per 100 frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
