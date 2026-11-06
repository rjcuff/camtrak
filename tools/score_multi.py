"""
score_multi - score a camtrak_hud --csv run against a multi-object truth CSV.

    python tools/score_multi.py clips/sky.truth.csv clips/sky_run.csv

Per frame, every truth object is matched to the nearest LOCKED/COASTING track
whose center is within `--radius` px (greedy, closest first). Reports:
  coverage     share of truth object-frames that have a matching track
  false        share of track-frames not near any truth object (wrong things tracked)
  id_switches  times a truth object's matched track ID changed
  ids/object   average distinct track IDs per truth object (1.0 = perfectly stable)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np


def load(path, id_col="id"):
    per = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            f = int(r["frame"])
            x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
            per[f].append((r[id_col], x + w / 2, y + h / 2, r.get("status", "LOCKED")))
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("run")
    ap.add_argument("--radius", type=float, default=15.0)
    a = ap.parse_args()
    truth, run = load(a.truth), load(a.run)
    frames = sorted(set(truth) & set(run)) or sorted(truth)
    covered = total = false = tracks = switches = 0
    last: dict[str, str] = {}
    ids: dict[str, set] = defaultdict(set)
    for f in frames:
        tr = [t for t in run.get(f, []) if t[3] != "SEARCHING"]
        gt = truth[f]
        pairs = sorted((np.hypot(g[1] - t[1], g[2] - t[2]), gi, ti)
                       for gi, g in enumerate(gt) for ti, t in enumerate(tr))
        used_g, used_t = set(), set()
        for d, gi, ti in pairs:
            if d > a.radius or gi in used_g or ti in used_t:
                continue
            used_g.add(gi); used_t.add(ti)
            gid, tid = gt[gi][0], tr[ti][0]
            if gid in last and last[gid] != tid:
                switches += 1
            last[gid] = tid
            ids[gid].add(tid)
        covered += len(used_g); total += len(gt)
        tracks += len(tr)
        # a track is false if it is not near any truth object at all
        false += sum(1 for t in tr if all(np.hypot(g[1] - t[1], g[2] - t[2]) > a.radius * 2 for g in gt))
    print(f"frames {len(frames)}  objects {len({g[0] for f in truth for g in truth[f]})}")
    print(f"coverage     {100 * covered / max(1, total):.1f}%  ({covered}/{total} object-frames)")
    print(f"false        {100 * false / max(1, tracks):.1f}%  ({false}/{tracks} track-frames)")
    print(f"id_switches  {switches}")
    print(f"ids/object   {np.mean([len(v) for v in ids.values()]) if ids else 0:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
