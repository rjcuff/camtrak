"""
make_ground_clip - synthetic aerial test video: vehicles on busy terrain.

    python tools/make_ground_clip.py clips/ground.mp4

A drone looks down on fields, roads, buildings and trees (textured: a
"stands out from a plain background" detector cannot work here). The camera
pans, slowly rotates and zooms. Eight vehicles drive along different headings
at different speeds; some are low contrast (grey car on grey road), one
stops halfway. Writes clips/ground.truth.csv (frame,id,x,y,w,h) for
tools/score_multi.py.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECONDS = 1280, 720, 30, 12
WW, WH = 2400, 1700          # world (terrain) size


def terrain(rng) -> np.ndarray:
    def noise(scale, amp):
        small = rng.normal(0, 1, (WH // scale + 2, WW // scale + 2)).astype(np.float32)
        return cv2.resize(small, (WW, WH), interpolation=cv2.INTER_CUBIC) * amp
    base = 110 + noise(200, 25) + noise(40, 12) + noise(6, 8)
    img = np.stack([base * 0.8, base * 1.05, base * 0.85], axis=2)          # greenish fields
    for _ in range(25):                                                      # field patches
        x, y = rng.integers(0, WW), rng.integers(0, WH)
        col = rng.uniform(60, 170, 3)
        poly = np.array([[x, y], [x + rng.integers(150, 400), y + rng.integers(-60, 60)],
                         [x + rng.integers(150, 400), y + rng.integers(150, 350)],
                         [x + rng.integers(-60, 60), y + rng.integers(150, 350)]], np.int32)
        cv2.fillPoly(img, [poly], col.tolist())
    img += noise(5, 10)[..., None]
    for k in range(6):                                                       # roads
        p0 = (int(rng.integers(0, WW)), 0) if k % 2 else (0, int(rng.integers(0, WH)))
        p1 = (int(rng.integers(0, WW)), WH) if k % 2 else (WW, int(rng.integers(0, WH)))
        cv2.line(img, p0, p1, (95, 95, 98), int(rng.integers(26, 40)), cv2.LINE_AA)
    for _ in range(140):                                                     # buildings
        x, y = int(rng.integers(0, WW)), int(rng.integers(0, WH))
        w, h = int(rng.integers(14, 60)), int(rng.integers(14, 60))
        cv2.rectangle(img, (x, y), (x + w, y + h), rng.uniform(120, 230, 3).tolist(), -1)
        cv2.rectangle(img, (x + 4, y + h), (x + w + 4, y + h + 5), (40, 45, 40), -1)   # shadow
    for _ in range(900):                                                     # trees
        x, y = int(rng.integers(0, WW)), int(rng.integers(0, WH))
        cv2.circle(img, (x, y), int(rng.integers(4, 13)), (30, int(rng.integers(50, 90)), 35), -1)
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def vehicle_poly(cx, cy, heading, length, width):
    c, s = math.cos(heading), math.sin(heading)
    pts = [(-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2), (-length / 2, width / 2)]
    return np.array([(cx + c * x - s * y, cy + s * x + c * y) for x, y in pts], np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    world = terrain(rng)

    # id, start (world), heading, speed px/s, length, width, colour BGR, stop time
    cars = [
        (1, (700, 600), 0.10, 70, 26, 12, (40, 40, 200), None),
        (2, (1500, 500), 2.90, 55, 22, 11, (230, 230, 230), None),
        (3, (900, 1100), -0.60, 90, 24, 11, (30, 30, 30), None),
        (4, (1300, 800), 1.70, 45, 20, 10, (100, 100, 104), None),     # grey on grey: low contrast
        (5, (1100, 650), -2.40, 60, 28, 13, (200, 120, 30), 6.0),      # stops at 6 s
        (6, (1600, 1000), -1.30, 80, 18, 9, (40, 190, 220), None),
        (7, (850, 850), 0.80, 35, 22, 10, (180, 60, 150), None),
        (8, (1400, 1150), 3.60, 100, 30, 13, (60, 140, 60), None),     # green on green
    ]
    writer = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    with open(a.out.with_suffix(".truth.csv"), "w", newline="") as fh:
        rows = csv.writer(fh)
        rows.writerow(["frame", "id", "x", "y", "w", "h"])
        for i in range(FPS * SECONDS):
            t = i / FPS
            canvas = world.copy()
            polys = []
            for cid, (sx, sy), hd, v, L, Wd, col, stop in cars:
                tt = min(t, stop) if stop is not None else t
                cx, cy = sx + math.cos(hd) * v * tt, sy + math.sin(hd) * v * tt
                shadow = vehicle_poly(cx + 3, cy + 4, hd, L, Wd)
                cv2.fillPoly(canvas, [shadow.astype(np.int32)], (35, 40, 35), cv2.LINE_AA)
                poly = vehicle_poly(cx, cy, hd, L, Wd)
                cv2.fillPoly(canvas, [poly.astype(np.int32)], col, cv2.LINE_AA)
                polys.append((cid, poly))
            # camera: centre drifts, slow rotation and zoom
            cam_x = 1200 + 260 * math.sin(0.20 * t)
            cam_y = 850 + 160 * math.sin(0.27 * t + 1)
            ang = 4.0 * math.sin(0.3 * t)
            zoom = 1.0 + 0.12 * math.sin(0.22 * t)
            M = cv2.getRotationMatrix2D((cam_x, cam_y), ang, zoom)
            M[0, 2] += W / 2 - cam_x
            M[1, 2] += H / 2 - cam_y
            frame = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR)
            frame = np.clip(frame.astype(np.int16) + rng.normal(0, 3, frame.shape).astype(np.int16),
                            0, 255).astype(np.uint8)
            writer.write(frame)
            for cid, poly in polys:
                p = cv2.transform(poly[None], M)[0]
                x0, y0 = p.min(axis=0)
                x1, y1 = p.max(axis=0)
                if x0 >= 0 and y0 >= 0 and x1 < W and y1 < H:
                    rows.writerow([i, cid, f"{x0:.1f}", f"{y0:.1f}", f"{x1 - x0:.1f}", f"{y1 - y0:.1f}"])
    writer.release()
    print(f"wrote {a.out} ({FPS * SECONDS} frames) and {a.out.with_suffix('.truth.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
