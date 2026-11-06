# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md.
"""
camtrak.cli - the Camtrak app: track objects in any video, camera or stream.

  camtrak --source clip.mp4 --display                 pick targets (click / drag), then play
  camtrak --source clip.mp4 --auto --display          find objects automatically
  camtrak --source 0 --auto --display                 webcam
  camtrak --source rtsp://cam/stream --auto --jsonl -  live stream, results as JSON lines on stdout
  camtrak --source clip.mp4 --auto --output out.mp4 --csv out.csv --jsonl out.jsonl

(`camtrak` once installed with pip; otherwise `python demo/camtrak_hud.py ...`.)

In the window:
  left click   lock onto the object under the cursor (box sized automatically);
               click a target to magnify it
  left drag    lock onto exactly the box you draw
  right click  drop the target under the cursor
  Tab          next target in the focus card       t  next color theme
  x            auto-detect on / off                 c  drop all targets
  space        pause / resume (start, in the picker)
  a / d        picker only: previous / next frame (A / D: 10 frames)
  q / Esc      quit
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import version
from .io import open_source
from .pipeline import Pipeline
from .render import THEMES

WINDOW = "camtrak"
MAX_VIEW = (1280, 720)
DRAG_MIN = 5          # screen px: less than this between press and release is a click


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    try:
        x, y, w, h = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("bbox must be x,y,w,h (e.g. 469,409,26,38)")
    if w < 4 or h < 4:
        raise argparse.ArgumentTypeError("bbox width and height must be at least 4 px")
    return x, y, w, h


def parse_point(text: str) -> tuple[float, float]:
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("point must be x,y (e.g. 640,360)")
    return x, y


def parse_args(argv):
    p = argparse.ArgumentParser(prog="camtrak", description="Track many objects in any video, camera or stream.")
    p.add_argument("--source", required=True,
                   help="video file, image folder, camera number (0) or stream URL (rtsp://...)")
    p.add_argument("--bbox", type=parse_bbox, action="append", default=[],
                   help="start with this box x,y,w,h (repeatable)")
    p.add_argument("--point", type=parse_point, action="append", default=[],
                   help="start with the object at x,y; box found automatically (repeatable)")
    p.add_argument("--start", type=int, default=0, help="frame to start on (files only)")
    p.add_argument("--width", type=int, default=0,
                   help="scale wider video down to this width first (e.g. 1920 for 4K input; faster)")
    p.add_argument("--auto", action="store_true", help="find objects automatically")
    p.add_argument("--auto-every", type=int, default=3, help="auto: look for new objects every N frames")
    p.add_argument("--max-targets", type=int, default=60, help="auto: never track more than this")
    p.add_argument("--sensitivity", type=float, default=6.0,
                   help="auto: lower finds fainter objects (and more false ones)")
    p.add_argument("--no-motion", action="store_true",
                   help="auto: only the stands-out-from-plain-background detector")
    p.add_argument("--theme", default="camtrak", choices=sorted(THEMES), help="color theme")
    p.add_argument("--display", action="store_true", help="show the video window")
    p.add_argument("--output", type=Path, help="write the annotated video (mp4)")
    p.add_argument("--csv", type=Path, help="per-frame, per-target results as CSV")
    p.add_argument("--jsonl", help="per-frame results as JSON lines (a file, or - for stdout)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    p.add_argument("--no-ml", action="store_true", help="no neural second opinion")
    p.add_argument("--no-labels", action="store_true", help="boxes only, no tags")
    p.add_argument("--no-card", "--no-inset", dest="no_card", action="store_true",
                   help="no magnified focus card")
    p.add_argument("--no-trails", action="store_true", help="do not draw recent paths")
    p.add_argument("--no-cut-detect", action="store_true", help="do not look for scene cuts")
    p.add_argument("--no-fit", action="store_true",
                   help="do not fit boxes to the object with the color model (box size from the tracker alone)")
    p.add_argument("--exact-boxes", action="store_true",
                   help="use drawn / --bbox boxes as given (default: tighten loose boxes onto the object)")
    return p.parse_args(argv)


class Mouse:
    """Collects clicks / drags / right clicks in image coordinates."""

    def __init__(self, scale: float) -> None:
        self.s = scale
        self.down = None
        self.drag = None
        self.events: list[tuple] = []

    def __call__(self, event, sx, sy, _flags, _param) -> None:
        ix, iy = sx / self.s, sy / self.s
        if event == cv2.EVENT_LBUTTONDOWN:
            self.down = (sx, sy, ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and self.down:
            _, _, ax, ay = self.down
            self.drag = (min(ax, ix), min(ay, iy), abs(ix - ax), abs(iy - ay))
        elif event == cv2.EVENT_LBUTTONUP and self.down:
            dsx, dsy, ax, ay = self.down
            if abs(sx - dsx) + abs(sy - dsy) < DRAG_MIN:
                self.events.append(("click", ix, iy))
            else:
                box = (min(ax, ix), min(ay, iy), abs(ix - ax), abs(iy - ay))
                if box[2] >= 4 and box[3] >= 4:
                    self.events.append(("box", *box))
            self.down, self.drag = None, None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.events.append(("drop", ix, iy))

    def take(self) -> list[tuple]:
        ev, self.events = self.events, []
        return ev


class App:
    def __init__(self, args) -> None:
        self.a = args
        self.src = open_source(args.source, args.width)
        self.pipe = Pipeline(self.src.fps, auto=args.auto, ml=not args.no_ml,
                             cut_detect=not args.no_cut_detect, max_targets=args.max_targets,
                             auto_every=args.auto_every, sensitivity=args.sensitivity,
                             motion=not args.no_motion, theme=args.theme, labels=not args.no_labels,
                             focus_card=not args.no_card, trails=False if args.no_trails else None,
                             extent=not args.no_fit)
        self.mouse: Mouse | None = None
        self.scale = 1.0
        self.live_fps = 0.0

    # -------------------------------------------------------------- input
    def apply_mouse(self, frame: np.ndarray) -> bool:
        changed = False
        for ev in self.mouse.take() if self.mouse else []:
            if ev[0] == "click":
                hit = self.pipe.target_at(ev[1], ev[2])
                if hit:
                    self.pipe.set_focus(hit.id)
                else:
                    self.pipe.add(frame, point=(ev[1], ev[2]))
            elif ev[0] == "box":
                self.pipe.add(frame, box=ev[1:], refine=not self.a.exact_boxes)
            elif ev[0] == "drop":
                hit = self.pipe.target_at(ev[1], ev[2])
                if hit:
                    self.pipe.remove(hit.id)
            changed = True
        return changed

    def common_key(self, key: int) -> bool:
        """Keys that work in the picker and while playing. True if handled."""
        if key == 9:                                    # Tab
            self.pipe.cycle_focus()
        elif key == ord("t"):
            names = sorted(THEMES)
            self.pipe.theme = names[(names.index(self.pipe.theme) + 1) % len(names)]
        elif key == ord("x"):
            self.pipe.set_auto(not self.pipe.auto)
        elif key == ord("c"):
            self.pipe.clear()
        else:
            return False
        return True

    # -------------------------------------------------------------- output
    def view(self, frame: np.ndarray, hint: str = "") -> np.ndarray:
        out = self.pipe.render(frame, fps=self.live_fps or None)
        if hint:
            self.pipe.renderer(frame.shape).hint(out, hint)
        if self.mouse and self.mouse.drag:
            x, y, w, h = (round(v) for v in self.mouse.drag)
            color = self.pipe.renderer(frame.shape).theme.accent
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
        return out

    def show(self, img: np.ndarray) -> None:
        s = self.scale
        cv2.imshow(WINDOW, cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img)

    # -------------------------------------------------------------- picker
    def pick(self, index: int, frame: np.ndarray):
        hint = ("click an object to lock it  |  drag a box  |  x: auto  |  t: theme  |  space: go"
                if self.src.seekable else "click an object to lock it  |  x: auto  |  space: go")
        while True:
            self.apply_mouse(frame)
            self.show(self.view(frame, hint + ("  |  a/d: frame" if self.src.seekable else "")))
            key = cv2.waitKey(20) & 0xFF
            step = {ord("a"): -1, ord("d"): 1, ord("A"): -10, ord("D"): 10}.get(key)
            if step and self.src.seekable:
                i = max(0, min(self.src.count - 1, index + step))
                nxt = self.src.read(i)
                if nxt is not None:
                    index, frame = i, nxt
                    for t in self.pipe.tracks.targets:   # targets follow the frame you settle on
                        t.tracker.lock(frame, t.box)
            elif self.common_key(key):
                pass
            elif key in (13, 10, ord(" ")) and (len(self.pipe.tracks) or self.pipe.auto):
                return index, frame
            elif key in (27, ord("q")):
                return None

    # -------------------------------------------------------------- run
    def run(self) -> int:
        a = self.a
        index = max(0, a.start) if self.src.seekable else 0
        frame = self.src.read(index)
        if frame is None:
            print(f"error: cannot read a frame from {a.source}", file=sys.stderr)
            return 1
        H, W = frame.shape[:2]
        self.scale = min(1.0, MAX_VIEW[0] / W, MAX_VIEW[1] / H)
        for box in a.bbox:
            self.pipe.add(frame, box=box, focus=False, refine=not a.exact_boxes)
        for pt in a.point:
            self.pipe.add(frame, point=pt, focus=False)
        has_targets = bool(a.bbox or a.point)

        if a.display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            self.mouse = Mouse(self.scale)
            cv2.setMouseCallback(WINDOW, self.mouse)
            if not has_targets and not a.auto:
                picked = self.pick(index, frame)
                if picked is None:
                    return 1
                index, frame = picked
        elif not has_targets and not a.auto:
            print("Nothing to track: pass --bbox / --point, --auto, or --display to pick.", file=sys.stderr)
            return 1

        log = sys.stderr if a.jsonl == "-" else sys.stdout
        print(f"camtrak {version()}: {W}x{H} @ {self.src.fps:.0f} fps, "
              f"{'live' if self.src.live else f'{self.src.count} frames'}, theme {a.theme}"
              f"{', auto' if a.auto else ''}", file=log)
        writer = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*"mp4v"), self.src.fps, (W, H)) \
            if a.output else None
        csv_file = open(a.csv, "w", newline="") if a.csv else None
        rows = csv.writer(csv_file) if csv_file else None
        if rows:
            rows.writerow(["frame", "id", "source", "status", "confidence", "x", "y", "w", "h",
                           "center_x", "center_y", "speed_px_s", "vx_px_s", "vy_px_s"])
        jl = (sys.stdout if a.jsonl == "-" else open(a.jsonl, "w")) if a.jsonl else None

        times: list[float] = []
        processed, paused = 0, False
        first = True
        tick = time.perf_counter()
        shown = self.view(frame)
        try:
            while True:
                if not paused:
                    if a.max_frames and processed >= a.max_frames:
                        break
                    if not first:
                        nxt = self.src.read(index + 1)
                        if nxt is None:
                            break
                        index, frame = index + 1, nxt
                    first = False
                    if self.mouse:
                        self.apply_mouse(frame)
                    res = self.pipe.process(frame)
                    processed += 1
                    times.append(res.ms)
                    now = time.perf_counter()
                    inst = 1.0 / max(1e-6, now - tick)
                    self.live_fps = inst if not self.live_fps else 0.9 * self.live_fps + 0.1 * inst
                    tick = now
                    shown = self.view(frame)
                    if writer:
                        writer.write(shown)
                    if rows:
                        for t in res.targets:
                            cx, cy = t.center
                            rows.writerow([index, t.id, t.source, t.label, f"{t.confidence:.3f}",
                                           *(f"{v:.1f}" for v in t.box), f"{cx:.1f}", f"{cy:.1f}",
                                           f"{t.speed:.1f}", f"{t.velocity[0]:.1f}", f"{t.velocity[1]:.1f}"])
                    if jl:
                        d = res.to_dict()
                        d["frame"] = index
                        jl.write(json.dumps(d) + "\n")
                    if processed % 100 == 0 and not a.display and a.jsonl != "-":
                        print(f"  frame {index}: {len(res.targets)} targets, "
                              f"{np.mean(times[-100:]):.1f} ms/frame", file=log)
                if a.display:
                    if paused and self.apply_mouse(frame):
                        shown = self.view(frame)
                    self.show(shown if not (self.mouse and self.mouse.drag) else self.view(frame))
                    key = cv2.waitKey(1 if not paused else 20) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord(" "):
                        paused = not paused
                    elif self.common_key(key):
                        shown = self.view(frame)
        finally:
            self.pipe.close()
            self.src.close()
            if writer:
                writer.release()
            if csv_file:
                csv_file.close()
            if jl and jl is not sys.stdout:
                jl.close()
            if a.display:
                cv2.destroyAllWindows()

        avg = float(np.mean(times)) if times else 0.0
        print(f"{processed} frames, {avg:.2f} ms/frame (tracking + detection), scene cuts: "
              f"{self.pipe.cut_count}, auto targets spawned: {self.pipe.spawned}", file=log)
        if a.output:
            print(f"annotated video: {a.output}", file=log)
        return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:   # OpenCV's DNN backend prints a harmless warning per neural-model init
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    try:
        return App(args).run()
    except (OSError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
