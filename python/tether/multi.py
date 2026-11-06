"""
tether.multi - follow many targets at once, each with an ID and a speed.

    import tether as tt
    mt = tt.MultiTracker(fps=30)
    a = mt.add(frame, (x, y, w, h))            # returns the new target's ID
    b = mt.add(frame, box2, source="auto")
    for t in mt.update(next_frame):            # every live target
        print(t.id, t.status_name, t.center, t.speed)   # speed in px/s

Each target has its own Tether tracker (~0.2-0.6 ms per target per frame), so
dozens of targets stay real-time on one CPU core. The first `ml_max` targets
picked by the user also get the neural second opinion (MLTracker, ~4 ms each)
when the model is available.

Targets with source="auto" (from a detector) are pruned automatically: when
they stay LOST for `auto_drop` frames, or when they end up on the same object
as another target. User-picked targets are never removed except by remove().

Tentative targets (add(..., tentative=True)) are on probation: they are not in
`visible` until they prove to be objects. If the caller passes the camera
motion to update() (a 2x3 affine, previous frame -> this frame), a target is
confirmed once it has moved relative to the scene; one that stays put while
the camera moves is scenery (a tree, a building) and is dropped after
`probation` frames with reason "static". Without camera motion (plain sky,
nothing to measure against), a target is confirmed after `confirm_plain`
confident frames.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import COASTING, LOST, TRACKING, Result, TetherError, Tracker

_LABELS = {TRACKING: "LOCKED", COASTING: "COASTING", LOST: "SEARCHING"}


def iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Target:
    id: int
    source: str                        # "user" or "auto"
    tracker: object                    # Tracker or MLTracker
    result: Result
    age: int = 0                       # frames since added
    lost_frames: int = 0               # consecutive LOST frames
    miss_frames: int = 0               # consecutive frames not TRACKING (COASTING or LOST)
    tracked_frames: int = 0            # total TRACKING frames
    velocity: tuple[float, float] = (0.0, 0.0)   # px/s, smoothed
    trail: deque = field(default_factory=lambda: deque(maxlen=48))
    confirmed: bool = True             # False while a tentative target is on probation
    anchor: tuple[float, float] | None = None    # where it would be if it were scenery
    plain_frames: int = 0              # confident frames with no camera motion available
    moving: int = 0                    # consecutive frames away from the scenery anchor

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.result.box

    @property
    def status(self) -> int:
        return self.result.status

    @property
    def status_name(self) -> str:
        return self.result.status_name

    @property
    def label(self) -> str:
        """HUD wording: LOCKED / COASTING / SEARCHING."""
        return _LABELS.get(self.result.status, "?")

    @property
    def confidence(self) -> float:
        return self.result.confidence

    @property
    def center(self) -> tuple[float, float]:
        return self.result.center()

    @property
    def speed(self) -> float:
        """Image-plane speed in px/s (smoothed)."""
        return float(np.hypot(*self.velocity))


class MultiTracker:
    def __init__(self, fps: float = 30.0, *, ml: bool = True, ml_max: int = 4,
                 auto_drop: int | None = None, duplicate_iou: float = 0.5,
                 speed_smoothing: float = 0.25, probation: int | None = None,
                 confirm_plain: int = 6, **params: float) -> None:
        """
        fps            frame rate, for px/s speeds
        ml             give user-picked targets the neural second opinion if available
        ml_max         at most this many targets get it (each costs ~4 ms/frame)
        auto_drop      drop auto targets after this many LOST frames (default: 1 s)
        duplicate_iou  two targets overlapping more than this are on the same object
        probation      tentative targets that have not moved relative to the scene
                       within this many frames are dropped (default: 2 s)
        confirm_plain  without camera motion, confirm after this many confident frames
        params         tracker tuning, passed to every Tracker (see tether.h)
        """
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.ml = ml
        self.ml_max = ml_max
        self.auto_drop = auto_drop if auto_drop is not None else int(round(self.fps))
        self.duplicate_iou = duplicate_iou
        self.alpha = speed_smoothing
        self.probation = probation if probation is not None else int(round(2 * self.fps))
        self.confirm_plain = confirm_plain
        self.params = params
        self._targets: dict[int, Target] = {}
        self._next_id = 1
        self._ml_ok = ml
        self.removed: list[tuple[int, str, tuple]] = []   # (id, reason, box) from the last update()

    # ------------------------------------------------------------------ targets
    @property
    def targets(self) -> list[Target]:
        return list(self._targets.values())

    @property
    def visible(self) -> list[Target]:
        """Targets to show: everything except tentative targets still on probation."""
        return [t for t in self._targets.values() if t.confirmed]

    def __len__(self) -> int:
        return len(self._targets)

    def get(self, target_id: int) -> Target | None:
        return self._targets.get(target_id)

    def _new_tracker(self, source: str):
        n_ml = sum(1 for t in self._targets.values() if type(t.tracker).__name__ == "MLTracker")
        if self._ml_ok and source == "user" and n_ml < self.ml_max:
            try:
                from .ml import MLTracker
                return MLTracker(**self.params)
            except (TetherError, AttributeError, ImportError):
                self._ml_ok = False      # model or OpenCV missing: correlation only
        return Tracker(**self.params)

    def add(self, frame: np.ndarray, box, source: str = "user", tentative: bool = False) -> int:
        """Start tracking `box` on `frame`. Returns the new target's ID.
        tentative=True: on probation until it proves to be a moving object."""
        box = tuple(float(v) for v in box)
        trk = self._new_tracker(source)
        trk.lock(frame, box)
        tid = self._next_id
        self._next_id += 1
        t = Target(tid, source, trk, Result(box, 1.0, 0.0, TRACKING), confirmed=not tentative)
        t.anchor = t.center
        t.trail.append(t.center)
        self._targets[tid] = t
        return tid

    def remove(self, target_id: int) -> bool:
        t = self._targets.pop(target_id, None)
        if t is None:
            return False
        t.tracker.close()
        return True

    def clear(self, source: str | None = None) -> None:
        """Remove every target (or only those from `source`)."""
        for tid in [t.id for t in self._targets.values() if source in (None, t.source)]:
            self.remove(tid)

    def at(self, x: float, y: float, margin: float = 6.0) -> Target | None:
        """The smallest visible target whose box (plus margin) contains the point."""
        hits = [t for t in self.visible
                if t.box[0] - margin <= x <= t.box[0] + t.box[2] + margin
                and t.box[1] - margin <= y <= t.box[1] + t.box[3] + margin]
        return min(hits, key=lambda t: t.box[2] * t.box[3]) if hits else None

    def overlaps(self, box, threshold: float = 0.2, grow: float = 0.7) -> bool:
        """Does `box` cover, or sit right next to, an existing target?"""
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        for t in self._targets.values():
            if iou(box, t.box) > threshold:
                return True
            tx, ty = t.center
            if abs(cx - tx) < grow * (w + t.box[2]) / 2 and abs(cy - ty) < grow * (h + t.box[3]) / 2:
                return True
        return False

    # ------------------------------------------------------------------ frames
    def scene_cut(self) -> None:
        """Next frame is a new shot: auto targets are dropped, user targets search."""
        self.clear("auto")
        for t in self._targets.values():
            t.tracker.scene_cut()
            t.velocity = (0.0, 0.0)
            t.trail.clear()

    def update(self, frame: np.ndarray, camera: np.ndarray | None = None) -> list[Target]:
        """Track every target into `frame`. `camera`: optional 2x3 affine of the
        camera's motion since the previous frame (judges tentative targets)."""
        self.removed = []
        for t in list(self._targets.values()):     # copy: probation may remove targets
            prev = t.center
            t.result = t.tracker.update(frame)
            t.age += 1
            self._update_motion(t, prev)
            t.lost_frames = t.lost_frames + 1 if t.status == LOST else 0
            if t.status == TRACKING:
                t.tracked_frames += 1
                t.miss_frames = 0
            else:
                t.miss_frames += 1
            if not t.confirmed:
                self._probation(t, camera)
        self._prune(frame.shape[1], frame.shape[0])
        return self.targets

    def _update_motion(self, t: Target, prev: tuple[float, float]) -> None:
        if t.status == LOST:
            t.velocity = (t.velocity[0] * 0.8, t.velocity[1] * 0.8)
            return
        cx, cy = t.center
        dx, dy = cx - prev[0], cy - prev[1]
        # a jump bigger than the box itself is a re-acquisition, not motion
        if abs(dx) > max(t.box[2], 8) * 1.5 or abs(dy) > max(t.box[3], 8) * 1.5:
            t.trail.clear()
            t.trail.append((cx, cy))
            return
        a = self.alpha
        vx, vy = dx * self.fps, dy * self.fps
        t.velocity = ((1 - a) * t.velocity[0] + a * vx, (1 - a) * t.velocity[1] + a * vy)
        t.trail.append((cx, cy))

    SETTLE = 4          # frames a new tracker needs to centre itself on the object
    SUSTAIN = 4         # consecutive frames the scene-relative movement must hold

    def _probation(self, t: Target, camera) -> None:
        cx, cy = t.center
        if camera is not None:
            if t.anchor is not None:
                ax, ay = t.anchor
                t.anchor = (camera[0][0] * ax + camera[0][1] * ay + camera[0][2],
                            camera[1][0] * ax + camera[1][1] * ay + camera[1][2])
            if t.age <= self.SETTLE or t.anchor is None:
                t.anchor = (cx, cy)            # measure from where the tracker settled
                t.moving = 0
            else:
                moved = float(np.hypot(cx - t.anchor[0], cy - t.anchor[1]))
                far = moved > max(6.0, 0.5 * max(t.box[2], t.box[3]))
                t.moving = t.moving + 1 if (far and t.status == TRACKING) else 0
                if t.moving >= self.SUSTAIN:
                    t.confirmed = True
        else:
            t.anchor = None                    # cannot tell scenery from objects right now
            if t.status == TRACKING:
                t.plain_frames += 1
                if t.plain_frames >= self.confirm_plain:
                    t.confirmed = True
        if not t.confirmed and t.age >= self.probation:
            self.removed.append((t.id, "static", t.box))
            self.remove(t.id)

    def _prune(self, W: int, H: int) -> None:
        for t in list(self._targets.values()):
            if t.source != "auto":
                continue                       # only the user removes user targets
            x, y, w, h = t.box
            cx, cy = x + w / 2, y + h / 2
            if t.miss_frames >= self.auto_drop:
                self.removed.append((t.id, "lost", t.box))
                self.remove(t.id)
            elif t.miss_frames and (cx < w / 4 or cy < h / 4 or cx > W - w / 4 or cy > H - h / 4):
                # unsure and mostly out of the picture: it flew out
                self.removed.append((t.id, "left_frame", t.box))
                self.remove(t.id)
        # two trackers on one object: keep the user one, then the older one
        live = sorted(self._targets.values(), key=lambda t: (t.source != "user", t.id))
        for i, a in enumerate(live):
            if a.id not in self._targets:
                continue
            for b in live[i + 1:]:
                if b.id in self._targets and b.source == "auto" \
                        and iou(a.box, b.box) > self.duplicate_iou:
                    self.removed.append((b.id, "duplicate", b.box))
                    self.remove(b.id)

    def close(self) -> None:
        self.clear()

    def __enter__(self) -> "MultiTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
