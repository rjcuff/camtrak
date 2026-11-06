"""
bench_folder - run Tether on every benchmark sequence under a folder and print a table.

    python tools/bench_folder.py D:/LaSOT/bird          (bird-1 ... bird-20)
    python tools/bench_folder.py D:/OTB --csv otb.csv   (Skiing, Bird1, ...)

A sequence is any folder containing groundtruth.txt or groundtruth_rect.txt.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent / "demo"
sys.path.insert(0, str(DEMO))
import tether_demo  # noqa: E402
from sources import find_groundtruth  # noqa: E402

COLUMNS = ["sequence", "frames", "precision_20px", "success_auc", "mean_centre_error_px",
           "honest_when_absent", "confidence_when_correct", "confidence_when_wrong",
           "avg_update_ms", "LOST"]


def run_one(seq: Path, max_frames: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run.json"
        argv = ["--source", str(seq), "--json", str(out)]
        if max_frames:
            argv += ["--max-frames", str(max_frames)]
        with contextlib.redirect_stdout(io.StringIO()):
            code = tether_demo.main(argv)
        if code != 0 or not out.exists():
            return {"sequence": seq.name, "error": f"exit {code}"}
        s = json.loads(out.read_text())
    scores = s.get("scores", {})
    return {"sequence": seq.name, "frames": s["frames"], **{k: scores.get(k) for k in COLUMNS[2:8]},
            "avg_update_ms": s["avg_update_ms"], "LOST": s["status_counts"]["LOST"]}


def fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Tether over a folder of benchmark sequences.")
    ap.add_argument("root", type=Path)
    ap.add_argument("--csv", type=Path, help="also save the table as CSV")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    seqs = sorted({p for p in args.root.rglob("*") if p.is_dir() and find_groundtruth(p)})
    if find_groundtruth(args.root):
        seqs.insert(0, args.root)
    if not seqs:
        print(f"no sequences with groundtruth found under {args.root}", file=sys.stderr)
        return 1

    results = []
    print("  ".join(f"{c[:14]:>14}" for c in COLUMNS))
    for seq in seqs:
        r = run_one(seq, args.max_frames)
        results.append(r)
        print("  ".join(f"{fmt(r.get(c)):>14}" for c in COLUMNS) if "error" not in r
              else f"{seq.name:>14}  {r['error']}")

    ok = [r for r in results if "error" not in r and r.get("precision_20px") is not None]
    if ok:
        mean = lambda k: sum(r[k] for r in ok) / len(ok)  # noqa: E731
        print(f"\nmean over {len(ok)} sequences: precision@20 {mean('precision_20px'):.3f}  "
              f"success AUC {mean('success_auc'):.3f}  {mean('avg_update_ms'):.2f} ms/frame")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS + ["error"], extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
