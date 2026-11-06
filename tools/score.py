"""
score - compare a tether_demo CSV against a make_clip ground-truth CSV.

    python tools/score.py clips/track.csv clips/synthetic.truth.csv

Splits the clip into before / during / after the occlusion and reports
centre error, precision at 20 px, the status mix and PSR range in each part.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter


def load(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser(description="Score a Tether run against ground truth.")
    ap.add_argument("track")
    ap.add_argument("truth")
    args = ap.parse_args()

    truth = {int(r["frame"]): r for r in load(args.truth)}
    occluded = [f for f, r in truth.items() if r["occluded"] == "1"]
    first, last = (min(occluded), max(occluded)) if occluded else (10**9, 10**9)
    parts = {"before": (1, first - 1), "occluded": (first, last), "after": (last + 1, 10**9)}

    rows = load(args.track)
    for name, (lo, hi) in parts.items():
        errs, psrs, states = [], [], Counter()
        for r in rows:
            f = int(r["frame"])
            if not lo <= f <= hi:
                continue
            g = truth[f]
            gx = float(g["bbox_x"]) + float(g["bbox_w"]) / 2
            gy = float(g["bbox_y"]) + float(g["bbox_h"]) / 2
            errs.append(math.hypot(float(r["center_x"]) - gx, float(r["center_y"]) - gy))
            psrs.append(float(r.get("psr", "nan")))
            states[r["status"]] += 1
        if not errs:
            continue
        print(f"{name:9s} frames {lo:3d}-{min(hi, max(int(r['frame']) for r in rows)):3d}  "
              f"mean err {sum(errs)/len(errs):6.1f}px  within20 {100*sum(e <= 20 for e in errs)/len(errs):5.1f}%  "
              f"psr {min(psrs):5.1f}..{max(psrs):5.1f}  {dict(states)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
