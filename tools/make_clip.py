"""
make_clip - write a synthetic test video with known ground truth.

    python tools/make_clip.py clips/synthetic.mp4

A small dark "skier" moves across a panning, textured snow-and-trees
background, with sensor noise and one occlusion (a tree trunk sweeps
in front of it around frame 150). A ground-truth CSV is written next to the
video, and the first-frame box is printed so you can pass it to --bbox.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECONDS = 1280, 720, 30, 10
TARGET = 22          # target size, px
PAN = 400            # background is this much wider than the frame


def make_background(rng: np.random.Generator) -> np.ndarray:
    snow = rng.integers(170, 235, size=(H, W + PAN), dtype=np.uint8)
    snow = cv2.GaussianBlur(snow, (0, 0), 4)
    # a few darker "trees" for clutter
    for _ in range(40):
        x = int(rng.integers(0, W + PAN))
        y = int(rng.integers(0, H))
        cv2.circle(snow, (x, y), int(rng.integers(8, 30)), int(rng.integers(60, 120)), -1)
    snow = cv2.GaussianBlur(snow, (0, 0), 1.5)
    return cv2.cvtColor(snow, cv2.COLOR_GRAY2BGR)


def make_target(rng: np.random.Generator) -> np.ndarray:
    t = np.full((TARGET, TARGET, 3), (40, 40, 160), np.uint8)       # red-ish jacket
    t[: TARGET // 3] = (30, 30, 30)                                # dark helmet
    t[TARGET * 2 // 3 :] = (90, 60, 20)                            # blue trousers
    noise = rng.integers(-15, 16, size=t.shape)
    return np.clip(t.astype(int) + noise, 0, 255).astype(np.uint8)


def target_box(i: int) -> tuple[int, int]:
    t = i / FPS
    cx = W * (0.15 + 0.7 * t / SECONDS) + 60 * math.sin(2.1 * t)
    cy = H * (0.30 + 0.35 * t / SECONDS) + 25 * math.sin(3.3 * t)
    return round(cx - TARGET / 2), round(cy - TARGET / 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("out", type=Path, help="output .mp4")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    bg = make_background(rng)
    target = make_target(rng)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"cannot write {args.out}")

    truth_path = args.out.with_suffix(".truth.csv")
    # Same boxes in the OTB format (x,y,w,h per line) so tether_demo --gt can score it.
    gt_path = args.out.with_suffix(".groundtruth.txt")
    with open(truth_path, "w", newline="") as fh, open(gt_path, "w") as gt:
        rows = csv.writer(fh)
        rows.writerow(["frame", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "occluded"])
        for i in range(FPS * SECONDS):
            pan = int(PAN / 2 + (PAN / 2 - 10) * math.sin(0.4 * i / FPS))
            frame = bg[:, pan : pan + W].copy()
            x, y = target_box(i)
            frame[y : y + TARGET, x : x + TARGET] = target

            # occluder: a tree trunk sweeping across the target around frame 150
            occluded = 0
            if 135 <= i <= 170:
                tx = x - 40 + (i - 135) * 3
                cv2.rectangle(frame, (tx, 0), (tx + 30, H), (35, 45, 40), -1)
                occluded = int(tx < x + TARGET and tx + 30 > x)

            noise = rng.integers(-8, 9, size=frame.shape)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            writer.write(frame)
            rows.writerow([i, x, y, TARGET, TARGET, occluded])
            gt.write(f"{x},{y},{TARGET},{TARGET}\n")
    writer.release()

    x0, y0 = target_box(0)
    print(f"wrote {args.out} ({W}x{H}, {FPS * SECONDS} frames) and {truth_path}")
    print(f"first-frame box: --bbox {x0},{y0},{TARGET},{TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
