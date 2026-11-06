"""
make_sky_clip - synthetic multi-target test video with known ground truth.

    python tools/make_sky_clip.py clips/sky.mp4

A formation of small aircraft and some birds cross a hazy, cloudy sky
above a tree line; the camera pans slowly; there is sensor noise and two
objects cross paths. Writes clips/sky.truth.csv (frame,id,x,y,w,h) so
tools/score_multi.py can score a tether_hud run.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECONDS = 1280, 720, 30, 12
PAN = 300


def background(rng) -> np.ndarray:
    Wb = W + PAN
    sky = np.zeros((H, Wb, 3), np.float32)
    top, bottom = np.array([170, 120, 60], np.float32), np.array([235, 215, 190], np.float32)
    for y in range(H):
        sky[y] = top + (bottom - top) * (y / H) ** 0.8
    clouds = np.zeros((H, Wb), np.float32)
    for _ in range(60):
        cx, cy = rng.integers(0, Wb), rng.integers(0, int(H * 0.6))
        cv2.ellipse(clouds, (int(cx), int(cy)), (int(rng.integers(40, 160)), int(rng.integers(15, 50))),
                    0, 0, 360, float(rng.uniform(0.3, 0.8)), -1)
    clouds = cv2.GaussianBlur(clouds, (0, 0), 25)
    sky = sky * (1 - clouds[..., None] * 0.6) + 245 * clouds[..., None] * 0.6
    # tree line: busy dark texture at the bottom
    treeline = int(H * 0.82)
    for x in range(0, Wb, 3):
        top_y = treeline - int(abs(rng.normal(0, 18)))
        sky[top_y:, x:x + 3] = rng.uniform(25, 70, size=3)
    noise = rng.normal(0, 18, size=(H - treeline + 40, Wb, 3)).astype(np.float32)
    sky[treeline - 40:] = np.clip(sky[treeline - 40:] + noise * (sky[treeline - 40:] < 90), 0, 255)
    return np.clip(sky, 0, 255).astype(np.uint8)


def plane(size: int, colour) -> np.ndarray:
    """A small swept-wing silhouette, pointing right; returns BGRA."""
    s = size
    img = np.zeros((s, s, 4), np.uint8)
    body = np.array([[0.05, 0.5], [0.95, 0.45], [1.0, 0.5], [0.95, 0.55]]) * s
    wing = np.array([[0.35, 0.5], [0.6, 0.5], [0.3, 0.05], [0.2, 0.05]]) * s
    wing2 = wing.copy(); wing2[:, 1] = s - wing2[:, 1]
    tail = np.array([[0.0, 0.5], [0.15, 0.5], [0.05, 0.25]]) * s
    for poly in (body, wing, wing2, tail):
        cv2.fillPoly(img, [poly.astype(np.int32)], (*colour, 255), cv2.LINE_AA)
    return img


def bird(size: int, phase: float) -> np.ndarray:
    s = size
    img = np.zeros((s, s, 4), np.uint8)
    flap = 0.25 + 0.2 * math.sin(phase)
    pts = (np.array([[0.0, 0.5 - flap], [0.5, 0.55], [1.0, 0.5 - flap]]) * s).astype(np.int32)
    cv2.polylines(img, [pts], False, (30, 30, 35, 255), max(2, s // 6), cv2.LINE_AA)
    return img


def paste(frame, sprite, cx, cy):
    s = sprite.shape[0]
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    if x0 < 0 or y0 < 0 or x0 + s > frame.shape[1] or y0 + s > frame.shape[0]:
        return None
    a = sprite[..., 3:4].astype(np.float32) / 255
    roi = frame[y0:y0 + s, x0:x0 + s].astype(np.float32)
    frame[y0:y0 + s, x0:x0 + s] = (roi * (1 - a) + sprite[..., :3] * a).astype(np.uint8)
    ys, xs = np.nonzero(sprite[..., 3] > 60)
    return (x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    bg = background(rng)

    objs = []   # (id, kind, size, path(t)->(x,y) in background coords)
    for k in range(5):        # formation, left to right, slight climb
        objs.append((k + 1, "plane", 22 - 2 * k,
                     lambda t, k=k: (150 + 85 * t + 40 * k, 330 - 12 * t + 28 * k)))
    objs.append((6, "plane", 26, lambda t: (1400 - 110 * t, 140 + 30 * math.sin(0.6 * t))))  # crosses
    for k in range(4):        # birds drifting
        objs.append((7 + k, "bird", 14 + 2 * k,
                     lambda t, k=k: (500 + 220 * k + 30 * math.sin(0.5 * t + k), 200 + 60 * k + 15 * math.sin(0.9 * t + 2 * k))))

    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    with open(args.out.with_suffix(".truth.csv"), "w", newline="") as fh:
        rows = csv.writer(fh)
        rows.writerow(["frame", "id", "x", "y", "w", "h"])
        for i in range(FPS * SECONDS):
            t = i / FPS
            pan = int(PAN / 2 + (PAN / 2 - 5) * math.sin(0.25 * t))
            canvas = bg.copy()
            boxes = []
            for oid, kind, size, path in objs:
                bx, by = path(t)
                spr = plane(size, (45, 40, 40)) if kind == "plane" else bird(size, 9 * t + oid)
                b = paste(canvas, spr, bx, by)
                if b is not None:
                    boxes.append((oid, b))
            frame = canvas[:, pan:pan + W].copy()
            frame = np.clip(frame.astype(np.int16) + rng.normal(0, 3, frame.shape).astype(np.int16),
                            0, 255).astype(np.uint8)
            writer.write(frame)
            for oid, (x, y, w, h) in boxes:
                x -= pan
                if 0 <= x and x + w <= W:
                    rows.writerow([i, oid, x, y, w, h])
    writer.release()
    print(f"wrote {args.out} ({FPS * SECONDS} frames) and {args.out.with_suffix('.truth.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
