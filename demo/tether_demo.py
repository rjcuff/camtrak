"""
tether_demo - track an object through a video or an image sequence.

  Pick the target yourself (scrub to any frame, draw, move, resize):
      python demo/tether_demo.py --source clip.mp4 --display

  Pass the box, save results:
      python demo/tether_demo.py --source clip.mp4 --bbox 469,409,26,38 --csv out.csv --output out.mp4

  Benchmark sequence (OTB / LaSOT folder): starts from the ground-truth box and scores itself:
      python demo/tether_demo.py --source LaSOT/bird/bird-2 --display

Keys while playing:
  space  pause / resume          s  pick the target again (any frame)
  q/Esc  quit

Edited videos: scene cuts are detected automatically. The tracker is told
(so it searches instead of coasting) and playback carries on; press s at any
time to re-pick, or use --pause-on-cut to stop at every cut.

OpenCV is used only here, to read video, let you pick the box, and draw the
result. The tracker core itself does not depend on OpenCV.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE))
import tether as tt  # noqa: E402
from logbook import Logbook  # noqa: E402
from metrics import FrameRecord, describe, summarise  # noqa: E402
from selector import select_target  # noqa: E402
from sources import (CutDetector, FrameSource, find_groundtruth,  # noqa: E402
                     load_absent_flags, load_groundtruth)

STATUS_COLOURS = {tt.TRACKING: (0, 220, 0), tt.COASTING: (0, 200, 255), tt.LOST: (0, 0, 255)}
TRUTH_COLOUR = (255, 200, 0)
WINDOW = "Tether"
MAX_VIEW = (1280, 720)


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    try:
        x, y, w, h = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("bbox must be x,y,w,h (e.g. 469,409,26,38)")
    if w < 4 or h < 4:
        raise argparse.ArgumentTypeError("bbox width and height must be at least 4 px")
    return x, y, w, h


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track one object with Tether.")
    p.add_argument("--source", required=True, type=Path, help="video file or image folder")
    p.add_argument("--bbox", type=parse_bbox, help="initial box x,y,w,h in pixels")
    p.add_argument("--start", type=int, default=0, help="frame index the --bbox refers to")
    p.add_argument("--select-target", action="store_true",
                   help="pick the box interactively (default when no --bbox or ground truth)")
    p.add_argument("--gt", type=Path, help="ground-truth boxes (auto-found in OTB/LaSOT folders)")
    p.add_argument("--display", action="store_true", help="show tracking live")
    p.add_argument("--show-truth", action="store_true", help="also draw the ground-truth box")
    p.add_argument("--no-cut-detect", action="store_true", help="do not look for scene cuts")
    p.add_argument("--pause-on-cut", action="store_true",
                   help="stop at each scene cut so you can re-pick (default: keep playing)")
    p.add_argument("--csv", type=Path, help="write per-frame results to CSV")
    p.add_argument("--json", type=Path, help="write a run summary (and scores) to JSON")
    p.add_argument("--output", type=Path, help="write an annotated video (mp4)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    p.add_argument("--log-dir", type=Path, default=HERE.parent / "logs",
                   help="where each run's log folder goes (default: tether/logs)")
    p.add_argument("--no-log", action="store_true", help="do not save a log folder")
    p.add_argument("--no-ml", action="store_true",
                   help="correlation tracker only (no neural second opinion)")
    return p.parse_args(argv)


def boxed(image: np.ndarray, box, colour=(0, 255, 255), text: str = "") -> np.ndarray:
    out = image.copy()
    x, y, w, h = (round(v) for v in box)
    cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
    if text:
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    return out


def draw(frame: np.ndarray, r: tt.Result, index: int, ms: float, truth, banner: str) -> None:
    x, y, w, h = r.box
    colour = STATUS_COLOURS.get(r.status, (255, 255, 255))
    if truth is not None:
        tx, ty, tw, th = truth
        cv2.rectangle(frame, (round(tx), round(ty)), (round(tx + tw), round(ty + th)), TRUTH_COLOUR, 1)
    thickness = 1 if r.status == tt.LOST else 2
    cv2.rectangle(frame, (round(x), round(y)), (round(x + w), round(y + h)), colour, thickness)

    label = f"#{index}  {r.status_name}  conf {r.confidence:.2f}  {round(w)}x{round(h)}  {ms:.2f} ms"
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    bar_w = 200   # confidence bar
    cv2.rectangle(frame, (10, 40), (10 + bar_w, 52), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, 40), (10 + round(bar_w * r.confidence), 52), colour, -1)
    if banner:
        cv2.putText(frame, banner, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, banner, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


def show(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    s = min(1.0, MAX_VIEW[0] / w, MAX_VIEW[1] / h)
    cv2.imshow(WINDOW, cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame)


def resolve_groundtruth(args: argparse.Namespace) -> tuple[list | None, Path | None]:
    gt_path = args.gt or find_groundtruth(args.source)
    if gt_path is None:
        return None, None
    boxes = load_groundtruth(gt_path)
    absent = load_absent_flags(args.source, len(boxes))
    return [None if a else b for b, a in zip(boxes, absent)], gt_path


def make_tracker(no_ml: bool):
    """Correlation + ML second opinion when the model is available, else correlation only."""
    if not no_ml:
        try:
            return tt.MLTracker()
        except (tt.TetherError, AttributeError) as exc:
            print(f"note: ML second opinion unavailable ({exc}); using correlation only")
    return tt.Tracker()


def initial_target(args, src: FrameSource, truth) -> tuple[int, tuple] | None:
    if args.bbox:
        return args.start, args.bbox
    if not args.select_target and truth and truth[0] is not None:
        return 0, truth[0]
    return select_target(src, start=args.start)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        src = FrameSource(args.source)
    except (OSError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    truth, gt_path = resolve_groundtruth(args)
    picked = initial_target(args, src, truth)
    if picked is None:
        print("No target selected.")
        return 1
    index, bbox = picked
    first = src.read(index)
    if first is None:
        print(f"error: cannot read frame {index}", file=sys.stderr)
        return 1

    tracker = make_tracker(args.no_ml)
    tracker.lock(first, bbox)
    print(f"Tether {tt.version()} [{'correlation + ML' if isinstance(tracker, tt.MLTracker) else 'correlation'}]: locked on {tuple(round(v, 1) for v in bbox)} at frame {index}"
          f" of {args.source}" + (f"  (ground truth: {gt_path})" if gt_path else ""))
    cuts = None if args.no_cut_detect else CutDetector()
    if cuts:
        cuts.is_cut(first)
    log = None if args.no_log else Logbook(args.log_dir, args.source, src.fps)
    if log:
        log.event(index, "pick", box=[round(v, 1) for v in bbox],
                  how="bbox" if args.bbox else ("truth" if truth and not args.select_target else "user"))
        log.snapshot(index, boxed(first, bbox, text=f"#{index} PICKED"), "pick")

    writer = None
    if args.output:
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (w, h))
    csv_file = open(args.csv, "w", newline="") if args.csv else None
    rows = csv.writer(csv_file) if csv_file else None
    if rows:
        rows.writerow(["frame", "status", "confidence", "psr", "bbox_x", "bbox_y", "bbox_w",
                       "bbox_h", "center_x", "center_y", "scene_cut", "update_ms"])

    times: list[float] = []
    records: list[FrameRecord] = []
    counts = {name: 0 for name in ("TRACKING", "COASTING", "LOST")}
    n_cuts, n_relocks, processed = 0, 0, 0
    paused = False
    cut_banner_until = -1
    shown = first
    last_raw, last_box = first, bbox
    try:
        while True:
            if not paused:
                if args.max_frames and processed >= args.max_frames:
                    break
                frame = src.read(index + 1)
                if frame is None:
                    break
                index += 1
                processed += 1
                last_raw = frame.copy()   # clean copy for re-picking (draw() paints on frame)
                cut = bool(cuts and cuts.is_cut(frame))
                if cut:
                    n_cuts += 1
                    tracker.scene_cut()
                    if log:
                        log.event(index, "scene_cut")
                t0 = time.perf_counter()
                r = tracker.update(frame)
                ms = (time.perf_counter() - t0) * 1000.0
                times.append(ms)
                counts[r.status_name] += 1
                last_box = r.box
                gt_box = truth[index] if truth and index < len(truth) else None
                if truth and index < len(truth):
                    records.append(FrameRecord(r.box, r.confidence, r.status_name, gt_box))
                if rows:
                    cx, cy = r.center()
                    rows.writerow([index, r.status_name, f"{r.confidence:.4f}", f"{r.psr:.2f}",
                                   *(f"{v:.2f}" for v in r.box), f"{cx:.2f}", f"{cy:.2f}",
                                   int(cut), f"{ms:.3f}"])
                pause_here = cut and args.display and args.pause_on_cut
                banner = "SCENE CUT - s: pick target again, space: let it search" if pause_here else ""
                if cut and not pause_here:
                    cut_banner_until = index + int(src.fps)   # show a short note, keep playing
                if not banner and index < cut_banner_until:
                    banner = "scene cut - searching (s to re-pick)"
                draw(frame, r, index, ms, gt_box if args.show_truth else None, banner)
                if log:
                    log.frame(index, r, tracker.debug(), cut, ms, frame)
                if writer:
                    writer.write(frame)
                shown = frame
                paused = pause_here
            if args.display:
                show(shown)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key in (ord("q"), 27):
                    if log:
                        log.event(index, "quit")
                    break
                if key == ord(" "):
                    paused = not paused
                    if log:
                        log.event(index, "pause" if paused else "resume")
                if key == ord("s"):
                    if log:
                        log.event(index, "repick_start", tracker_box=[round(v, 1) for v in last_box])
                    picked = select_target(src, start=index, box=last_box)
                    if picked is not None:
                        index, bbox = picked
                        last_raw = src.read(index)
                        tracker.lock(last_raw, bbox)
                        n_relocks += 1
                        if log:
                            log.event(index, "repick", box=[round(v, 1) for v in bbox])
                            log.snapshot(index, boxed(last_raw, bbox, text=f"#{index} RE-PICKED"),
                                         "repick")
                        if cuts:
                            cuts.reset()
                            cuts.is_cut(last_raw)
                        print(f"frame {index}: re-locked on {tuple(round(v, 1) for v in bbox)}")
                        shown = last_raw.copy()
                    paused = False
    finally:
        tracker.close()
        src.close()
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()
        if args.display:
            cv2.destroyAllWindows()

    avg = sum(times) / len(times) if times else 0.0
    summary: dict = {
        "source": str(args.source), "frames": processed, "initial_bbox": list(bbox),
        "avg_update_ms": round(avg, 3), "max_update_ms": round(max(times), 3) if times else None,
        "tracker_fps": round(1000.0 / avg) if avg else None, "status_counts": counts,
        "scene_cuts": n_cuts, "relocks": n_relocks,
    }
    print(f"{processed} frames, {avg:.3f} ms/frame in tracker (~{summary['tracker_fps']} fps), "
          f"status: {counts}, scene cuts: {n_cuts}, re-picks: {n_relocks}")
    if records:
        summary["scores"] = summarise(records)
        print("scores vs ground truth:")
        print(describe(summary["scores"]))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))
    if log:
        log.close({**summary, "tether_version": tt.version(), "command": sys.argv})
        print(f"\nLog saved to: {log.dir}\n(send this folder path to get the run analysed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
