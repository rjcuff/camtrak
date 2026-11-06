# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md.
"""
camtrak.pipeline - everything an application needs, one call per frame.

    import camtrak as tt

    pipe = tt.Pipeline(fps=30, auto=True)          # auto: find objects by itself
    pipe.add(frame, box=(x, y, w, h))              # or point=(x, y): box sized automatically
    for frame in video:
        result = pipe.process(frame)               # track every target
        for t in result.targets:                   # what to use in your app
            print(t.id, t.label, t.center, t.speed)
        send(result.to_dict())                     # JSON-ready: ids, boxes, speeds, colors
        show(pipe.render(frame, result))           # optional: Camtrak's HUD drawn on a copy

The pipeline owns:
  * MultiTracker   - one Camtrak tracker per target, IDs, speeds, pruning
  * AutoDetector   - finds new targets (plain-background blobs, scene-relative movers)
  * CutDetector    - edited video: at a cut, auto targets are dropped and yours search
  * focus          - which target the HUD magnifies (best locked one, or the one you pick)
  * Renderer       - the HUD, themeable

Nothing here opens windows or files: feed it frames from anywhere (camera,
stream, file, a game engine, a robot) and use the results however you like.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import COASTING, LOST, TRACKING
from .detect import AutoDetector, CutDetector, refine_box, snap_box
from .extent import tighten
from .multi import MultiTracker, Target
from .render import THEMES, Renderer, bgr_hex

STATE_NAMES = {TRACKING: "locked", COASTING: "coasting", LOST: "searching"}


@dataclass
class FrameResult:
    index: int                         # frames processed so far (0-based index of this frame)
    targets: list[Target]              # visible targets after this frame
    scene_cut: bool = False
    camera_motion: np.ndarray | None = None   # 2x3 affine prev -> this frame (auto mode), or None
    ms: float = 0.0                    # time spent in process()
    focus: int | None = None
    removed: list = field(default_factory=list)   # (id, reason, box) dropped this frame
    colors: dict = field(default_factory=dict)   # id -> "#RRGGBB" in the current theme

    def to_dict(self) -> dict:
        """Plain dict for JSON / IPC / logging."""
        out = []
        for t in self.targets:
            cx, cy = t.center
            out.append({
                "id": t.id,
                "source": t.source,
                "state": STATE_NAMES.get(t.status, "unknown"),
                "box": [round(v, 1) for v in t.box],
                "center": [round(cx, 1), round(cy, 1)],
                "velocity": [round(t.velocity[0], 1), round(t.velocity[1], 1)],
                "speed": round(t.speed, 1),
                "confidence": round(t.confidence, 3),
                "color": self.colors.get(t.id),
            })
        return {"frame": self.index, "scene_cut": self.scene_cut, "focus": self.focus,
                "ms": round(self.ms, 2), "targets": out,
                "removed": [{"id": i, "reason": r} for i, r, _ in self.removed]}


class Pipeline:
    def __init__(self, fps: float = 30.0, *, auto: bool = False, ml: bool = True,
                 cut_detect: bool = True, max_targets: int = 60, auto_every: int = 3,
                 sensitivity: float = 6.0, motion: bool = True, theme: str = "camtrak",
                 labels: bool = True, focus_card: bool = True, trails: bool | None = None,
                 extent: bool = True, **tracker_params: float) -> None:
        self.fps = fps
        self.auto = auto
        self.max_targets = max_targets
        self.auto_every = max(1, auto_every)
        self.tracks = MultiTracker(fps=fps, ml=ml, extent=extent, **tracker_params)
        self.detector = AutoDetector(sensitivity=sensitivity, motion=motion)
        self.cuts = CutDetector() if cut_detect else None
        self.theme = theme
        self._render_opts = dict(labels=labels, focus_card=focus_card, trails=trails)
        self._renderer: Renderer | None = None
        self.focus: int | None = None
        self.focus_pinned = False
        self.index = -1
        self.spawned = 0
        self.cut_count = 0

    # ------------------------------------------------------------------ targets
    def add(self, frame: np.ndarray, box=None, point=None, focus: bool = True,
            refine: bool = True) -> int:
        """Track a new object: give its box (x, y, w, h), or a point on it and the
        box is found automatically. refine: tighten a loose box onto the object
        inside it (a loose box lets background into the model). Returns the ID."""
        if box is None:
            if point is None:
                raise ValueError("give box=(x, y, w, h) or point=(x, y)")
            box = snap_box(frame, float(point[0]), float(point[1]))
        elif refine:
            box = tuple(float(v) for v in box)
            tight = tighten(frame, box) if frame.ndim == 3 else box
            box = tight if tight != box else refine_box(frame, box)
        tid = self.tracks.add(frame, box)
        if focus:
            self.set_focus(tid)
        return tid

    def remove(self, target_id: int) -> bool:
        return self.tracks.remove(target_id)

    def clear(self) -> None:
        self.tracks.clear()
        self.focus = None

    def target_at(self, x: float, y: float) -> Target | None:
        return self.tracks.at(x, y)

    def set_focus(self, target_id: int | None) -> None:
        """Magnify this target in the HUD (None: choose automatically)."""
        self.focus = target_id
        self.focus_pinned = target_id is not None

    def cycle_focus(self) -> None:
        ids = sorted(t.id for t in self.tracks.visible)
        if ids:
            later = [i for i in ids if self.focus is None or i > self.focus]
            self.set_focus(later[0] if later else ids[0])

    def set_auto(self, on: bool) -> None:
        self.auto = on
        if not on:
            self.tracks.clear("auto")

    @property
    def targets(self) -> list[Target]:
        return self.tracks.visible

    # ------------------------------------------------------------------ frames
    def process(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        self.index += 1
        cut = bool(self.cuts and self.cuts.is_cut(frame))
        if cut:
            self.cut_count += 1
            self.tracks.scene_cut()
            self.detector.reset()
        camera = self.detector.push(frame) if self.auto else None
        self.tracks.update(frame, camera)
        removed = list(self.tracks.removed)
        for _, why, box in removed:
            if why == "static":
                self.detector.forget(box)
        if self.auto and self.index % self.auto_every == 0:
            for box in self.detector.candidates(frame, skip=self.tracks.overlaps):
                if len(self.tracks) >= self.max_targets:
                    break
                if not self.tracks.overlaps(box):
                    self.tracks.add(frame, box, source="auto", tentative=True)
                    self.spawned += 1
        self._refocus()
        visible = self.tracks.visible
        return FrameResult(self.index, visible, cut, camera, (time.perf_counter() - t0) * 1000,
                           self.focus, removed, {t.id: self._color_hex(t.id) for t in visible})

    def _refocus(self) -> None:
        """Keep a target the user chose; else follow the best-established
        locked target, moving on when it is lost."""
        cur = self.tracks.get(self.focus) if self.focus is not None else None
        live = self.tracks.visible
        locked = [t for t in live if t.status == TRACKING]
        if cur and cur.confirmed and (self.focus_pinned or cur.miss_frames < 5 or not locked):
            return
        self.focus_pinned = False
        pool = [t for t in locked if t.source == "user"] or locked or live
        self.focus = max(pool, key=lambda t: (t.tracked_frames * t.confidence, -t.id)).id if pool else None

    # ------------------------------------------------------------------ drawing
    def renderer(self, frame_shape) -> Renderer:
        r = self._renderer
        if r is None or (r.H, r.W) != tuple(frame_shape[:2]) or r.theme.name != self.theme:
            self._renderer = Renderer(frame_shape, self.theme, **self._render_opts)
        return self._renderer

    def _color_hex(self, tid: int) -> str:
        palette = THEMES[self.theme].palette
        return bgr_hex(palette[(tid - 1) % len(palette)])

    def render(self, frame: np.ndarray, result: FrameResult | None = None, status: str | None = None,
               fps: float | None = None) -> np.ndarray:
        """A copy of `frame` with the HUD drawn on it."""
        out = frame.copy()
        targets = result.targets if result else self.tracks.visible
        if status is None:
            parts = [f"{len(targets)} tracked"]
            if self.auto:
                parts.append("auto")
            if fps:
                parts.append(f"{fps:.0f} fps")
            status = "   /   ".join(parts)
        self.renderer(frame.shape).draw(out, targets, self.focus, status, clean=frame)
        return out

    def close(self) -> None:
        self.tracks.close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
