# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md.
"""
camtrak - Python interface to the Camtrak C tracker.

    import camtrak as tt
    trk = tt.Tracker()
    trk.lock(frame, (x, y, w, h))      # numpy uint8, gray (H, W) or BGR (H, W, 3)
    r = trk.update(next_frame)
    print(r.box, r.confidence, r.status)

Higher level (needs OpenCV): camtrak.Pipeline (multi-target tracking,
auto-detect, HUD in one call per frame), camtrak.io.open_source (files,
folders, cameras, streams), camtrak.Renderer and THEMES (the HUD), and the
`camtrak` command (camtrak.cli).

The core binding uses ctypes; it will be replaced by compiled bindings
(pybind11/CFFI) with the same interface.

The shared library is found via the CAMTRAK_LIBRARY environment variable, or
next to this package, or in ../../build.
"""
from __future__ import annotations

import ctypes as C
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Tracker", "MLTracker", "MultiTracker", "Target", "Pipeline", "FrameResult",
           "Renderer", "THEMES", "Result", "TRACKING", "COASTING", "LOST", "CamtrakError"]

TRACKING, COASTING, LOST = 0, 1, 2
_STATUS_NAMES = {TRACKING: "TRACKING", COASTING: "COASTING", LOST: "LOST"}
_ERRORS = {-1: "invalid argument", -2: "out of memory", -3: "tracker not locked"}
_GRAY8, _BGR8 = 1, 3


class CamtrakError(RuntimeError):
    pass


class _Box(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("w", C.c_float), ("h", C.c_float)]


class _Frame(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_uint8)), ("width", C.c_int), ("height", C.c_int),
                ("stride", C.c_int), ("format", C.c_int)]


class _Params(C.Structure):
    # must match camtrak_params in camtrak.h, field for field
    _fields_ = [("padding", C.c_float), ("template_size", C.c_int),
                ("learning_rate", C.c_float), ("sigma_factor", C.c_float),
                ("lambda_", C.c_float), ("psr_update", C.c_float), ("psr_track", C.c_float),
                ("kalman_q", C.c_float), ("kalman_r", C.c_float), ("coast_damping", C.c_float),
                ("lost_after", C.c_int), ("psr_reacquire", C.c_float),
                ("reacquire_budget", C.c_int), ("scale_step", C.c_float),
                ("adaptive_color", C.c_int)]


class _Debug(C.Structure):
    # must match camtrak_debug in camtrak.h
    _fields_ = [("pred_x", C.c_float), ("pred_y", C.c_float), ("vel_x", C.c_float),
                ("vel_y", C.c_float), ("scale", C.c_float), ("scale_factor", C.c_float),
                ("misses", C.c_int), ("lost_frames", C.c_int), ("search_rings", C.c_int),
                ("used_anchor", C.c_int), ("learned", C.c_int), ("windows_tried", C.c_int),
                ("channel_gain", C.c_float)]


class _Verdict(C.Structure):
    _fields_ = [("psr", C.c_float), ("color", C.c_float), ("accept", C.c_int)]


class _Result(C.Structure):
    _fields_ = [("box", _Box), ("confidence", C.c_float), ("psr", C.c_float),
                ("status", C.c_int)]


def _library_name() -> str:
    if sys.platform == "win32":
        return "camtrak_c.dll"
    if sys.platform == "darwin":
        return "libcamtrak_c.dylib"
    return "libcamtrak_c.so"


def _load() -> C.CDLL:
    here = Path(__file__).resolve().parent
    candidates = []
    if os.environ.get("CAMTRAK_LIBRARY"):
        candidates.append(Path(os.environ["CAMTRAK_LIBRARY"]))
    candidates += [here / _library_name(), here.parent.parent / "build" / _library_name()]
    for path in candidates:
        if path.is_file():
            lib = C.CDLL(str(path))
            break
    else:
        raise CamtrakError("Camtrak shared library not found; tried: "
                          + ", ".join(str(p) for p in candidates)
                          + ". Build it first (build.bat or cmake), or set CAMTRAK_LIBRARY.")

    lib.camtrak_default_params.restype = _Params
    lib.camtrak_default_params.argtypes = []
    lib.camtrak_create.restype = C.c_void_p
    lib.camtrak_create.argtypes = [C.POINTER(_Params)]
    lib.camtrak_destroy.restype = None
    lib.camtrak_destroy.argtypes = [C.c_void_p]
    lib.camtrak_lock.restype = C.c_int
    lib.camtrak_lock.argtypes = [C.c_void_p, C.POINTER(_Frame), _Box]
    lib.camtrak_update.restype = C.c_int
    lib.camtrak_update.argtypes = [C.c_void_p, C.POINTER(_Frame), C.POINTER(_Result)]
    lib.camtrak_get_debug.restype = C.c_int
    lib.camtrak_get_debug.argtypes = [C.c_void_p, C.POINTER(_Debug)]
    lib.camtrak_verify.restype = C.c_int
    lib.camtrak_verify.argtypes = [C.c_void_p, C.POINTER(_Frame), _Box, C.POINTER(_Verdict)]
    lib.camtrak_reacquire.restype = C.c_int
    lib.camtrak_reacquire.argtypes = [C.c_void_p, C.POINTER(_Frame), _Box]
    lib.camtrak_scene_cut.restype = C.c_int
    lib.camtrak_scene_cut.argtypes = [C.c_void_p]
    lib.camtrak_version.restype = C.c_char_p
    lib.camtrak_version.argtypes = []
    return lib


_lib = _load()


def version() -> str:
    return _lib.camtrak_version().decode()


@dataclass(frozen=True)
class Result:
    box: tuple[float, float, float, float]  # x, y, w, h
    confidence: float
    psr: float
    status: int

    @property
    def status_name(self) -> str:
        return _STATUS_NAMES.get(self.status, "UNKNOWN")

    @property
    def lost(self) -> bool:
        return self.status == LOST

    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h / 2


def _as_frame(img: np.ndarray) -> tuple[_Frame, np.ndarray]:
    """Wrap a numpy image without copying (unless it is not C-contiguous)."""
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise TypeError("frame must be a numpy uint8 array")
    if img.ndim == 2:
        fmt = _GRAY8
    elif img.ndim == 3 and img.shape[2] == 3:
        fmt = _BGR8
    else:
        raise ValueError(f"frame must be (H, W) gray or (H, W, 3) BGR, got shape {img.shape}")
    img = np.ascontiguousarray(img)
    frame = _Frame(img.ctypes.data_as(C.POINTER(C.c_uint8)), img.shape[1], img.shape[0],
                   img.strides[0], fmt)
    return frame, img  # keep `img` alive while C reads it


def _check(code: int) -> None:
    if code != 0:
        raise CamtrakError(_ERRORS.get(code, f"error {code}"))


class Tracker:
    """Single-target tracker. Not thread-safe; use one per thread."""

    def __init__(self, **params: float) -> None:
        """Optional tuning, e.g. Tracker(psr_reacquire=12, scale_step=0). See camtrak.h."""
        p = _lib.camtrak_default_params()
        for name, value in params.items():
            field = "lambda_" if name == "lambda" else name
            if field not in {f for f, _ in _Params._fields_}:
                raise TypeError(f"unknown tracker parameter {name!r}")
            setattr(p, field, value)
        self._h = _lib.camtrak_create(C.byref(p))
        if not self._h:
            raise CamtrakError("camtrak_create failed (invalid parameters?)")

    def lock(self, frame: np.ndarray, bbox) -> None:
        x, y, w, h = (float(v) for v in bbox)
        f, _keep = _as_frame(frame)
        _check(_lib.camtrak_lock(self._h, C.byref(f), _Box(x, y, w, h)))

    def update(self, frame: np.ndarray) -> Result:
        f, _keep = _as_frame(frame)
        r = _Result()
        _check(_lib.camtrak_update(self._h, C.byref(f), C.byref(r)))
        return Result((r.box.x, r.box.y, r.box.w, r.box.h), r.confidence, r.psr, r.status)

    def debug(self) -> dict:
        """Internal state from the last update(), for logging and tuning."""
        d = _Debug()
        _check(_lib.camtrak_get_debug(self._h, C.byref(d)))
        return {name: getattr(d, name) for name, _ in _Debug._fields_}

    def verify(self, frame: np.ndarray, bbox) -> dict:
        """Does bbox look like the selected target? {'psr', 'color', 'accept'}; no state change."""
        f, _keep = _as_frame(frame)
        v = _Verdict()
        _check(_lib.camtrak_verify(self._h, C.byref(f), _Box(*(float(x) for x in bbox)), C.byref(v)))
        return {"psr": v.psr, "color": v.color, "accept": bool(v.accept)}

    def reacquire(self, frame: np.ndarray, bbox) -> None:
        """Move onto bbox keeping the learned appearance; TRACKING resumes next frame."""
        f, _keep = _as_frame(frame)
        _check(_lib.camtrak_reacquire(self._h, C.byref(f), _Box(*(float(v) for v in bbox))))

    def scene_cut(self) -> None:
        """The next frame is a different shot: skip coasting and start searching."""
        _check(_lib.camtrak_scene_cut(self._h))

    def close(self) -> None:
        if self._h:
            _lib.camtrak_destroy(self._h)
            self._h = None

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


from .ml import MLTracker  # noqa: E402  (needs OpenCV only when used)
from .multi import MultiTracker, Target  # noqa: E402


def __getattr__(name: str):
    """Pipeline / Renderer / THEMES need OpenCV; import them only when used,
    so the core tracker works with numpy alone."""
    if name in ("Pipeline", "FrameResult"):
        from . import pipeline
        return getattr(pipeline, name)
    if name in ("Renderer", "THEMES", "Theme", "register_theme"):
        from . import render
        return getattr(render, name)
    if name in ("io", "detect", "render", "pipeline", "cli"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'camtrak' has no attribute {name!r}")

