"""
metrics - standard single-object tracking scores, plus Tether-specific honesty checks.

  precision@20  fraction of frames whose box centre is within 20 px of the truth (OTB)
  success AUC   area under the "IoU > threshold" curve, thresholds 0..1 (OTB/LaSOT)
  honesty       when the target is truly absent, how often Tether did NOT claim TRACKING
  calibration   average confidence on correct frames (IoU >= 0.5) vs wrong frames
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameRecord:
    box: Box
    confidence: float
    status: str
    truth: Box | None      # None = target absent in this frame


def iou(a: Box, b: Box) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def centre_error(a: Box, b: Box) -> float:
    return math.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2, a[1] + a[3] / 2 - b[1] - b[3] / 2)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarise(records: list[FrameRecord]) -> dict:
    present = [r for r in records if r.truth is not None]
    absent = [r for r in records if r.truth is None]
    ious = [iou(r.box, r.truth) for r in present]  # type: ignore[arg-type]
    errs = [centre_error(r.box, r.truth) for r in present]  # type: ignore[arg-type]

    thresholds = [i / 20 for i in range(21)]
    success = [sum(v > t for v in ious) / len(ious) for t in thresholds] if ious else []

    correct_conf = [r.confidence for r, v in zip(present, ious) if v >= 0.5]
    wrong_conf = [r.confidence for r, v in zip(present, ious) if v < 0.5]
    false_tracking_absent = sum(r.status == "TRACKING" for r in absent)

    def rnd(x: float | None, nd: int = 3) -> float | None:
        return None if x is None else round(x, nd)

    return {
        "frames_scored": len(present),
        "precision_20px": rnd(sum(e <= 20 for e in errs) / len(errs)) if errs else None,
        "success_auc": rnd(_mean(success)),
        "mean_centre_error_px": rnd(_mean(errs), 1),
        "mean_iou": rnd(_mean(ious)),
        "absent_frames": len(absent),
        "honest_when_absent": rnd(1 - false_tracking_absent / len(absent)) if absent else None,
        "confidence_when_correct": rnd(_mean(correct_conf)),
        "confidence_when_wrong": rnd(_mean(wrong_conf)),
    }


def describe(s: dict) -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.1f}%"

    def num(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.2f}"

    lines = [
        f"  precision@20px     {pct(s['precision_20px'])}",
        f"  success AUC        {num(s['success_auc'])}",
        f"  mean centre error  {s['mean_centre_error_px']} px   mean IoU {num(s['mean_iou'])}",
        f"  confidence         {num(s['confidence_when_correct'])} when correct, "
        f"{num(s['confidence_when_wrong'])} when wrong",
    ]
    if s["absent_frames"]:
        lines.append(f"  honesty            {pct(s['honest_when_absent'])} of {s['absent_frames']} "
                     f"target-absent frames NOT reported as TRACKING")
    return "\n".join(lines)
