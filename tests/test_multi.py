"""
test_multi - regression gate for multi-target tracking, auto-detect and click-to-lock.

    python tests/test_multi.py          (needs numpy + opencv-python and a built library)

Renders the synthetic sky clip (10 aircraft/birds, clouds, tree line, pan)
into a temp folder and checks:
  * auto mode finds and holds every object, almost nothing false, stable IDs
  * auto mode finds moving vehicles on busy terrain under a moving camera
  * click-to-lock boxes each object tightly from a single click
  * MultiTracker add / remove / at / duplicate pruning behave
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "demo")]
import tether as tt  # noqa: E402
from detect import snap_box  # noqa: E402


def run(*args: str) -> str:
    out = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(args)}\n{out.stdout}\n{out.stderr}")
    return out.stdout


def score(truth: Path, run_csv: Path) -> dict:
    text = run("tools/score_multi.py", str(truth), str(run_csv))
    vals = {}
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] in ("coverage", "false", "id_switches", "ids/object"):
            vals[parts[0]] = float(parts[1].rstrip("%"))
    return vals


def main() -> int:
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        ok &= cond
        print(f"{'ok  ' if cond else 'FAIL'} {name}: {detail}")

    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "sky.mp4"
        truth = clip.with_suffix(".truth.csv")
        run("tools/make_sky_clip.py", str(clip))

        # auto mode
        auto_csv = Path(tmp) / "auto.csv"
        run("demo/tether_hud.py", "--source", str(clip), "--auto", "--csv", str(auto_csv))
        s = score(truth, auto_csv)
        check("auto coverage", s["coverage"] >= 95, f"{s['coverage']}% of object-frames (>= 95)")
        check("auto false", s["false"] <= 3, f"{s['false']}% of track-frames (<= 3)")
        check("auto ids", s["id_switches"] == 0 and s["ids/object"] <= 1.05,
              f"{s['id_switches']:.0f} switches, {s['ids/object']} ids/object")

        # auto mode on busy terrain: vehicles found by camera-compensated motion
        ground = Path(tmp) / "ground.mp4"
        run("tools/make_ground_clip.py", str(ground))
        g_csv = Path(tmp) / "ground_run.csv"
        run("demo/tether_hud.py", "--source", str(ground), "--auto", "--csv", str(g_csv))
        g = score(ground.with_suffix(".truth.csv"), g_csv)
        check("ground coverage", g["coverage"] >= 75, f"{g['coverage']}% of object-frames (>= 75)")
        check("ground false", g["false"] <= 40, f"{g['false']}% of track-frames (<= 40)")

        # click-to-lock on frame 0
        rows = [r for r in csv.DictReader(open(truth)) if r["frame"] == "0"]
        cap = cv2.VideoCapture(str(clip))
        _, frame = cap.read()
        cap.release()
        errs, ratios = [], []
        for r in rows:
            x, y, w, h = (float(r[k]) for k in "xywh")
            b = snap_box(frame, x + w / 2 + 2, y + h / 2 + 1)      # slightly off-centre click
            errs.append(np.hypot(b[0] + b[2] / 2 - x - w / 2, b[1] + b[3] / 2 - y - h / 2))
            ratios.append(b[2] * b[3] / (w * h))
        check("snap centre", max(errs) <= 3, f"worst centre error {max(errs):.1f} px (<= 3)")
        check("snap size", 0.8 <= min(ratios) and max(ratios) <= 3.5,
              f"box area / object area {min(ratios):.2f}..{max(ratios):.2f}")

        pts = [f"{float(r['x']) + float(r['w']) / 2:.0f},{float(r['y']) + float(r['h']) / 2:.0f}" for r in rows]
        pts_csv = Path(tmp) / "pts.csv"
        run("demo/tether_hud.py", "--source", str(clip), "--no-ml", "--csv", str(pts_csv),
            *[a for p in pts for a in ("--point", p)])
        s = score(truth, pts_csv)
        held = s["coverage"] / (100 * len(rows) / 10)   # 10 objects in the clip, len(rows) clicked
        check("click tracking", held >= 0.95 and s["id_switches"] == 0,
              f"{100 * held:.0f}% of clicked objects' frames held, {s['id_switches']:.0f} switches")

    # MultiTracker API
    img = np.full((240, 320, 3), 200, np.uint8)
    cv2.rectangle(img, (50, 50), (70, 70), (20, 20, 20), -1)
    cv2.rectangle(img, (200, 120), (224, 140), (20, 20, 160), -1)
    mt = tt.MultiTracker(fps=30, ml=False)
    a = mt.add(img, (46, 46, 28, 28))
    b = mt.add(img, (196, 116, 32, 28))
    c = mt.add(img, (47, 47, 28, 28), source="auto")      # duplicate of a
    mt.update(img)
    check("api duplicate", mt.get(c) is None and mt.get(a) is not None, f"removed {mt.removed}")
    check("api at", mt.at(210, 130).id == b, "point lookup finds the right target")
    check("api speed", all(t.speed < 5 for t in mt.targets), "static targets have ~0 px/s")
    mt.remove(a)
    check("api remove", len(mt) == 1, f"{len(mt)} left")
    mt.close()

    print("test_multi: OK" if ok else "test_multi: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
