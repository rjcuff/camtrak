"""
tether_hud - track many objects at once with a targeting overlay.

  Pick targets yourself (click an object to lock it, or drag a box):
      python demo/tether_hud.py --source clip.mp4 --display

  Let it find small objects against sky / snow by itself:
      python demo/tether_hud.py --source airshow.mp4 --auto --display

  Unattended, save the annotated video and per-target CSV:
      python demo/tether_hud.py --source clip.mp4 --auto --output out.mp4 --csv out.csv
      python demo/tether_hud.py --source clip.mp4 --bbox 469,409,26,38 --bbox 800,300,30,30 --output out.mp4

Picking (before playback) and while playing, in the video window:
  left click        lock onto the object under the cursor (box sized automatically)
  left drag         lock onto exactly the box you draw
  right click       drop the target under the cursor
  Tab               cycle which target the close-up shows
  x                 auto-detect on / off          c  drop all targets
  space             pause / resume (start, in the picker)
  a / d             picker only: previous / next frame (A / D: 10 frames)
  q / Esc           quit

Every target gets an ID, a state (LOCKED / COASTING / SEARCHING), its
position and its image-plane speed in px/s. The focused target is shown
magnified in the bottom-right corner.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE))
import tether as tt  # noqa: E402
from detect import AutoDetector, snap_box  # noqa: E402
from hud import Hud  # noqa: E402
from sources import CutDetector, FrameSource  # noqa: E402
from tether_demo import parse_bbox  # noqa: E402

WINDOW = "Tether HUD"
MAX_VIEW = (1280, 720)
DRAG_MIN = 5          # screen px: less than this between press and release is a click


def parse_args(argv):
    p = argparse.ArgumentParser(description="Track many objects with a targeting HUD.")
    p.add_argument("--source", required=True, type=Path, help="video file or image folder")
    p.add_argument("--bbox", type=parse_bbox, action="append", default=[],
                   help="initial box x,y,w,h (repeat for several targets)")
    p.add_argument("--point", action="append", default=[],
                   help="initial click x,y: lock onto the object there (repeatable)")
    p.add_argument("--start", type=int, default=0, help="frame to start on")
    p.add_argument("--auto", action="store_true", help="detect small objects automatically")
    p.add_argument("--auto-every", type=int, default=3, help="auto: run the detector every N frames")
    p.add_argument("--max-targets", type=int, default=60, help="auto: never track more than this")
    p.add_argument("--sensitivity", type=float, default=6.0,
                   help="auto: lower finds fainter objects (and more false ones)")
    p.add_argument("--no-motion", action="store_true",
                   help="auto: only the stands-out-from-plain-background detector")
    p.add_argument("--display", action="store_true", help="show the video window")
    p.add_argument("--output", type=Path, help="write the annotated video (mp4)")
    p.add_argument("--csv", type=Path, help="per-frame, per-target results")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    p.add_argument("--no-ml", action="store_true", help="no neural second opinion")
    p.add_argument("--no-labels", action="store_true", help="brackets only, no text panels")
    p.add_argument("--no-inset", action="store_true", help="no magnified close-up")
    p.add_argument("--trails", action="store_true", help="draw each target's recent path")
    p.add_argument("--no-cut-detect", action="store_true", help="do not look for scene cuts")
    return p.parse_args(argv)


class Mouse:
    """Collects clicks / drags / right clicks in image coordinates."""

    def __init__(self, scale: float) -> None:
        self.s = scale
        self.down = None
        self.drag = None              # current drag rectangle (image coords), for drawing
        self.events: list[tuple] = []  # ("click", x, y) | ("box", x, y, w, h) | ("drop", x, y)

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
        self.args = args
        self.src = FrameSource(args.source)
        self.fps = self.src.fps
        self.mt = tt.MultiTracker(fps=self.fps, ml=not args.no_ml)
        self.det = AutoDetector(sensitivity=args.sensitivity, motion=not args.no_motion)
        self.auto = args.auto
        self.cuts = None if args.no_cut_detect else CutDetector()
        self.focus: int | None = None
        self.focus_pinned = False      # True once the user chose the close-up target
        self.hud: Hud | None = None
        self.mouse: Mouse | None = None
        self.view_scale = 1.0
        self.n_cuts = 0
        self.spawned = 0
        self.track_ms: list[float] = []

    # ----------------------------------------------------------- targets
    def apply_mouse(self, frame: np.ndarray) -> bool:
        changed = False
        for ev in self.mouse.take() if self.mouse else []:
            if ev[0] == "click":
                hit = self.mt.at(ev[1], ev[2])
                if hit:                                    # click on a target = focus it
                    self.focus = hit.id
                else:
                    self.focus = self.mt.add(frame, snap_box(frame, ev[1], ev[2]))
                self.focus_pinned = True
            elif ev[0] == "box":
                self.focus = self.mt.add(frame, ev[1:])
                self.focus_pinned = True
            elif ev[0] == "drop":
                hit = self.mt.at(ev[1], ev[2])
                if hit:
                    self.mt.remove(hit.id)
            changed = True
        return changed

    def refocus(self) -> None:
        """Pick the close-up target. A target the user chose keeps the close-up
        while it exists; otherwise follow the best-established LOCKED target and
        move on when it is lost."""
        cur = self.mt.get(self.focus) if self.focus is not None else None
        live = self.mt.visible
        locked = [t for t in live if t.status == tt.TRACKING]
        if cur and (self.focus_pinned or cur.miss_frames < 5 or not locked):
            return
        self.focus_pinned = False
        pool = [t for t in locked if t.source == "user"] or locked or live
        self.focus = max(pool, key=lambda t: (t.tracked_frames * t.confidence, -t.id)).id if pool else None

    def cycle_focus(self) -> None:
        ids = sorted(t.id for t in self.mt.visible)
        if not ids:
            self.focus = None
            return
        later = [i for i in ids if self.focus is None or i > self.focus]
        self.focus = later[0] if later else ids[0]
        self.focus_pinned = True

    def auto_spawn(self, frame: np.ndarray) -> None:
        for box in self.det.candidates(frame):
            if len(self.mt) >= self.args.max_targets:
                break
            if not self.mt.overlaps(box):
                self.mt.add(frame, box, source="auto", tentative=True)
                self.spawned += 1

    # ----------------------------------------------------------- drawing
    def render(self, frame: np.ndarray, index: int, live_fps: float) -> np.ndarray:
        out = frame.copy()
        shown = self.mt.visible
        n_lock = sum(1 for t in shown if t.status == tt.TRACKING)
        status = (f"TETHER  frame {index}  targets {len(shown)} ({n_lock} locked)"
                  f"{'  AUTO' if self.auto else ''}  {live_fps:.0f} fps")
        self.hud.draw(out, shown, self.focus, status, clean=frame)
        if self.mouse and self.mouse.drag:
            x, y, w, h = (round(v) for v in self.mouse.drag)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 1)
        return out

    def show(self, img: np.ndarray) -> None:
        s = self.view_scale
        cv2.imshow(WINDOW, cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img)

    # ----------------------------------------------------------- picker
    def pick(self, index: int) -> int | None:
        """Choose a start frame and targets before playback. Returns the frame index."""
        frame = self.src.read(index)
        hint = "click: lock object | drag: box | right-click: drop | a/d: frame | x: auto | Enter/space: go"
        while True:
            if self.apply_mouse(frame):
                self.refocus()
            out = self.render(frame, index, 0.0)
            self.hud.text_panel(out, hint, (round(10 * self.hud.u), out.shape[0] - round(44 * self.hud.u) - 8),
                                (0, 255, 255))
            self.show(out)
            key = cv2.waitKey(20) & 0xFF
            step = {ord("a"): -1, ord("d"): 1, ord("A"): -10, ord("D"): 10}.get(key)
            if step:
                nxt = self.src.read(max(0, min(self.src.count - 1, index + step)))
                if nxt is not None:
                    index = max(0, min(self.src.count - 1, index + step))
                    frame = nxt
                    for t in self.mt.targets:             # targets follow the frame you settle on
                        t.tracker.lock(frame, t.box)
            elif key == ord("x"):
                self.auto = not self.auto
            elif key == ord("c"):
                self.mt.clear()
            elif key in (13, 10, ord(" ")) and (len(self.mt) or self.auto):
                return index
            elif key in (27, ord("q")):
                return None

    # ----------------------------------------------------------- main loop
    def run(self) -> int:
        a = self.args
        index = max(0, min(a.start, self.src.count - 1))
        first = self.src.read(index)
        if first is None:
            print(f"error: cannot read frame {index}", file=sys.stderr)
            return 1
        H, W = first.shape[:2]
        self.hud = Hud(first.shape, labels=not a.no_labels, inset=not a.no_inset, trails=a.trails)
        self.view_scale = min(1.0, MAX_VIEW[0] / W, MAX_VIEW[1] / H)
        for box in a.bbox:
            self.mt.add(first, box)
        for pt in a.point:
            x, y = (float(v) for v in pt.split(","))
            self.mt.add(first, snap_box(first, x, y))
        self.refocus()

        if a.display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            self.mouse = Mouse(self.view_scale)
            cv2.setMouseCallback(WINDOW, self.mouse)
            if not a.bbox and not a.point and not a.auto:
                picked = self.pick(index)
                if picked is None:
                    return 1
                if picked != index:
                    index, first = picked, self.src.read(picked)
        elif not a.bbox and not a.point and not a.auto:
            print("Nothing to track: pass --bbox / --point, --auto, or --display to pick.")
            return 1

        print(f"Tether {tt.version()}: {len(self.mt)} target(s), auto={'on' if self.auto else 'off'}, "
              f"{W}x{H} @ {self.fps:.0f} fps, {self.src.count} frames")
        if self.cuts:
            self.cuts.is_cut(first)
        writer = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (W, H)) \
            if a.output else None
        csv_file = open(a.csv, "w", newline="") if a.csv else None
        rows = csv.writer(csv_file) if csv_file else None
        if rows:
            rows.writerow(["frame", "id", "source", "status", "confidence", "x", "y", "w", "h",
                           "center_x", "center_y", "speed_px_s", "vx_px_s", "vy_px_s"])

        frame, clean = first, first
        processed, paused = 0, False
        shown = self.render(first, index, 0.0)
        tick, live_fps = time.perf_counter(), 0.0
        try:
            while True:
                if not paused:
                    if a.max_frames and processed >= a.max_frames:
                        break
                    nxt = self.src.read(index + 1)
                    if nxt is None:
                        break
                    index, frame = index + 1, nxt
                    processed += 1
                    if self.cuts and self.cuts.is_cut(frame):
                        self.n_cuts += 1
                        self.mt.scene_cut()
                        self.det.reset()
                    t0 = time.perf_counter()
                    camera = self.det.push(frame) if self.auto else None
                    self.mt.update(frame, camera)
                    self.track_ms.append((time.perf_counter() - t0) * 1000)
                    for _, why, box in self.mt.removed:
                        if why == "static":
                            self.det.forget(box)
                    if self.auto and index % max(1, a.auto_every) == 0:
                        self.auto_spawn(frame)
                    clean = frame
                    if self.mouse:
                        self.apply_mouse(frame)
                    self.refocus()
                    now = time.perf_counter()
                    live_fps = 0.9 * live_fps + 0.1 / max(1e-6, now - tick) if live_fps else 1 / max(1e-6, now - tick)
                    tick = now
                    shown = self.render(frame, index, live_fps)
                    if writer:
                        writer.write(shown)
                    if rows:
                        for t in self.mt.visible:
                            cx, cy = t.center
                            rows.writerow([index, t.id, t.source, t.label, f"{t.confidence:.3f}",
                                           *(f"{v:.1f}" for v in t.box), f"{cx:.1f}", f"{cy:.1f}",
                                           f"{t.speed:.1f}", f"{t.velocity[0]:.1f}", f"{t.velocity[1]:.1f}"])
                    if processed % 100 == 0 and not a.display:
                        print(f"  frame {index}: {len(self.mt)} targets, "
                              f"{np.mean(self.track_ms[-100:]):.1f} ms/frame tracking")
                if a.display:
                    if paused and self.apply_mouse(clean):      # edits while paused show at once
                        self.refocus()
                        shown = self.render(clean, index, live_fps)
                    self.show(shown if not (self.mouse and self.mouse.drag)
                              else self.render(clean, index, live_fps))
                    key = cv2.waitKey(1 if not paused else 20) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord(" "):
                        paused = not paused
                    elif key == 9:                                # Tab
                        self.cycle_focus()
                        shown = self.render(clean, index, live_fps)
                    elif key == ord("x"):
                        self.auto = not self.auto
                        if not self.auto:
                            self.mt.clear("auto")
                    elif key == ord("c"):
                        self.mt.clear()
                        self.focus = None
                        shown = self.render(clean, index, live_fps)
        finally:
            self.mt.close()
            self.src.close()
            if writer:
                writer.release()
            if csv_file:
                csv_file.close()
            if a.display:
                cv2.destroyAllWindows()

        avg = float(np.mean(self.track_ms)) if self.track_ms else 0.0
        print(f"{processed} frames, {avg:.2f} ms/frame tracking all targets, scene cuts: {self.n_cuts}, "
              f"auto targets spawned: {self.spawned}")
        if a.output:
            print(f"annotated video: {a.output}")
        return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return App(args).run()
    except (OSError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
