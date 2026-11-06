# camtrak

**Fast, CPU-only tracking of many objects at once.** Point at objects (or let
camtrak find them) and it follows each one through video, camera feeds and
streams: an ID, a position, a speed and a confidence per object, every frame,
in about half a millisecond per object.

* **C core**, plain C11, no dependencies, zero allocations per frame.
* **Python API**: one `Pipeline` call per frame gives you plain data (JSON-ready)
  and, if you want it, camtrak's own HUD drawn on the frame.
* **The `camtrak` app**: files, image folders, webcams, RTSP/HTTP streams;
  click to lock, auto-detect, CSV / JSON-lines output.

## Quick start

```sh
./build.sh test                         # build the C core (macOS / Linux) and run its tests
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[vision]"              # the camtrak package + the `camtrak` command

camtrak --source clip.mp4 --display      # click objects to lock them, space to play
camtrak --source clip.mp4 --auto --display          # let camtrak find objects
camtrak --source 0 --auto --display                 # webcam
camtrak --source clip.mp4 --auto --output out.mp4 --jsonl out.jsonl
```

Without installing: `python demo/camtrak_hud.py` takes the same options.

**In the window:** left click = lock the object under the cursor (box sized
automatically; click a target to magnify it), left drag = lock that box,
right click = drop a target, Tab = next target in the focus card, `t` = next
color theme, `x` = auto-detect on/off, `c` = drop all, space = pause (start,
in the picker), `a`/`d` = choose the start frame, `q` = quit.

## The camtrak look

Every target gets its own color, so a crowd stays readable:

* **Box:** thin rounded outline with a center ring. Solid = locked, dashed =
  coasting (briefly unsure), gray dashed = searching (lost). The focused target
  gets four lock ticks.
* **Tag:** a short leader line (the link) to a pill with the ID chip in the
  target's color and its speed. Tags avoid each other and the boxes; in a
  crowd (> 20 targets) they shrink to the chip.
* **Trail:** fading dots along each target's recent path.
* **Focus card:** the focused target magnified with a reticle, its speed and a
  confidence bar, placed in whichever corner has no targets.
* **Status pill:** the camtrak mark and wordmark with live stats.

Themes: `camtrak` (default: teal, violet, amber, sky, pink, lime, coral, sand
on navy), `mono` (one accent, no trails), `contrast` (yellow / cyan / magenta
on black, for bright or busy footage) and `light` (light panels). Pick with
`--theme`, press `t` to cycle, or add your own:

```python
from camtrak.render import Theme, register_theme, hex_bgr
register_theme(Theme("acme", palette=(hex_bgr("#FF5A1F"), hex_bgr("#1F8FFF")), accent=hex_bgr("#FF5A1F"),
                     panel=hex_bgr("#111111"), panel_alpha=0.75, text=hex_bgr("#FFFFFF"),
                     muted=hex_bgr("#999999"), chip_text=hex_bgr("#111111")))
```

## Use camtrak in your application

```python
import camtrak as ct

src = ct.io.open_source("rtsp://camera/stream")   # file, folder, camera number or URL
pipe = ct.Pipeline(fps=src.fps, auto=True)        # auto: find objects by itself
pipe.add(first_frame, point=(640, 360))           # or box=(x, y, w, h); returns the ID

while (frame := src.next()) is not None:
    result = pipe.process(frame)                  # track everything
    for t in result.targets:
        t.id, t.label, t.box, t.center, t.speed, t.velocity, t.confidence
    payload = result.to_dict()                    # JSON-ready, includes each target's color
    hud = pipe.render(frame, result)              # optional: the HUD on a copy of the frame
```

`examples/embed.py` is a complete 25-line app. Other ways in:

* **Any language, no code:** `camtrak --source ... --jsonl -` streams one JSON
  object per frame to stdout (`{"frame", "targets": [{"id", "state", "box",
  "center", "velocity", "speed", "confidence", "color"}], "removed"}`); pipe it
  into Node, Go, a web socket, a database.
* **C / C++ / Rust / anything with a C FFI:** `include/camtrak/camtrak.h`, one
  `camtrak_tracker` per object (see *Usage (C)* below).
* **Lower-level Python:** `ct.Tracker` (one object), `ct.MultiTracker` (many,
  with IDs and speeds), `ct.detect` (click-to-lock `snap_box`, `AutoDetector`),
  `ct.Renderer` (draw the HUD on your own frames).

The core needs only numpy; `Pipeline`, the detectors, the HUD and the app
need OpenCV (`pip install -e ".[vision]"`).

## How it finds and follows objects

* **One tracker per object.** Each target has its own correlation-filter
  tracker (0.2-0.6 ms), so dozens of targets stay real time; the first 4 you
  pick also get a neural second opinion (VitTrack) when the model is present.
* **Click-to-lock** grows a window around the click until the object fits,
  treats the window's border colors as background, and boxes the blob under
  the click. **Drawn boxes** that are clearly looser than the object inside
  are tightened onto it (`--exact-boxes` to turn this off).
* **Auto mode, plain backgrounds** (sky, snow, water): compact blobs that
  stand out from a median-filtered background; clutter (tree lines, crowds,
  cloud edges) is rejected.
* **Auto mode, busy backgrounds** (terrain, streets, pitches): the camera's
  motion is measured (features + RANSAC) and frames are aligned to cancel it;
  what still moves is a candidate. New auto targets are on probation until
  they move relative to the scenery, so trees and buildings are dropped and
  their spots remembered.
* **Color:** at lock, each tracker picks the channel that best separates its
  target from the surroundings: brightness, or a color projection (Fisher
  discriminant) when that is clearly better, e.g. a red car on a brown field
  (`adaptive_color`, default on).
* **Housekeeping:** auto targets are dropped when they leave the frame, stay
  unconfirmed, or duplicate another target; yours are only dropped by you. At
  a scene cut, auto targets are re-detected and yours search.

## Numbers

Synthetic clips with ground truth (`tools/make_sky_clip.py`,
`tools/make_ground_clip.py`), scored with `tools/score_multi.py`; coverage =
object-frames with a track within 15 px, 3 random seeds each:

| Clip | Mode | Coverage | False track-frames | ID switches |
|---|---|---|---|---|
| sky: 10 aircraft and birds, clouds, tree line, panning | `--auto` | 97-98% | 0.4-0.5% | 0 |
| sky | one click per object | 100% of clicked | 0.7% | 0 |
| terrain: 8 vehicles, busy ground, panning/rotating/zooming camera | `--auto` | 73-89% | 12-25% | 1-7 |

Single trackers locked on the exact vehicle boxes: 62-75% of frames within
15 px with brightness only, 87-99% with the adaptive color channel.

Known weak spots: a box that starts much looser than the object can slowly
grow into textured background (tightening drawn boxes helps but does not
cure it; most false track-frames on terrain are such boxes on real
vehicles), and near-invisible objects (a gray car on a gray road) get lost.

Speed: 1080p, 24-cut, 66 s montage in `--auto`: 5-10 ms/frame for camera
motion, detection and all targets (MacBook). `tests/test_multi.py` gates the
numbers above; `tests/test_app.py` drives the app's window with scripted
clicks and keys.

## Build the core

Requires CMake >= 3.16 and a C11 compiler.

Linux / macOS / Raspberry Pi / Jetson:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Without CMake (macOS / Linux):

```sh
./build.sh test
```

Windows (Visual Studio 2022 or Build Tools with the C++ workload):

```bat
build.bat
```

## Single-target demo and benchmarks

Needs Python 3 with `numpy` and `opencv-python`.

```sh
# make a synthetic test clip (prints the first-frame box)
python tools/make_clip.py clips/synthetic.mp4

# pick the target yourself and watch it track
python demo/camtrak_demo.py --source clips/synthetic.mp4 --display

# or pass the box and save results
python demo/camtrak_demo.py --source clips/synthetic.mp4 --bbox 181,205,22,22 \
    --csv clips/track.csv --json clips/run.json --output clips/tracked.mp4
```

**Picking the target:** a window opens. Use the slider (or `a`/`d`) to go to
any frame, drag to draw a box, drag inside it to move it, drag an edge or
corner to resize it, then press Enter.

**While playing:** space pauses, `s` re-picks the target on the current frame,
`q` quits. Box color: green = TRACKING, orange = COASTING, red = LOST; the bar
is the confidence (0..1).

**ML second opinion (default when `models/vittrack.onnx` exists, `--no-ml` to
disable):** a small neural tracker (VitTrack, OpenCV Zoo, Apache 2.0, ~4 ms)
runs alongside. While Camtrak is confident it keeps the neural tracker on the
right object; while Camtrak is lost, the neural tracker's box is a candidate
that must pass Camtrak's checks (`camtrak_verify`: shape, color, mostly inside
the frame, and standing out from its surroundings like the selection did)
before Camtrak moves there (`camtrak_reacquire`, which keeps the user's box
shape so the box never jumps). While lost, the neural tracker also sweeps the
whole frame, one region per frame. On a 48 s, 15-shot drone ski montage with
two skiers this took unattended tracking from 43 to ~1,000 of 1,452 frames,
with no frame on the wrong skier or on background in any checked sample, no
box size jumps, and median jitter 0.2 px (`tools/smoothness.py`).

**Logs:** every run saves `logs/<date>-<time>_<video>/` with per-frame
internal state (`frames.csv`), user actions and cuts (`events.jsonl`),
snapshots at every status change (`snapshots/`) and `summary.json`.

**Edited videos:** scene cuts are detected and the tracker is told
(`camtrak_scene_cut`), so it searches instead of coasting on a stale
prediction. Playback keeps going and shows "scene cut - searching"; press `s`
any time to re-pick. `--pause-on-cut` stops at every cut instead.

### Real footage and benchmark sequences

`--source` also accepts an image folder. OTB and LaSOT sequence folders are
detected automatically: the ground-truth box starts the tracker, and the run
is scored at the end (precision@20 px, success AUC, honesty on target-absent
frames, confidence when right vs wrong).

```sh
python demo/camtrak_demo.py --source D:/LaSOT/bird/bird-2 --display --show-truth
python demo/camtrak_demo.py --source D:/UAV123/data_seq/UAV123/bird1 --gt D:/UAV123/anno/UAV123/bird1.txt
python tools/bench_folder.py D:/LaSOT/bird --csv bird.csv      # every sequence, one table
```

For your own videos (no ground truth), use `--select-target --display`.

## Usage (Python)

```python
import camtrak as ct          # after pip install -e . (or python/ on sys.path)

trk = ct.Tracker()
trk.lock(first_frame, (x, y, w, h))      # numpy uint8, gray or BGR
r = trk.update(frame)
print(r.box, r.confidence, r.status_name)
```

## Usage (C)

```c
#include <camtrak/camtrak.h>

camtrak_tracker *t = camtrak_create(NULL);                  /* default params */

camtrak_frame f = { pixels, width, height, stride, CAMTRAK_FORMAT_BGR8 };
camtrak_box box = { 410, 220, 36, 28 };                    /* x, y, w, h */
camtrak_lock(t, &f, box);

while (next_frame(&f)) {
    camtrak_result r;
    if (camtrak_update(t, &f, &r) == CAMTRAK_OK)
        printf("%.1f %.1f %.1f %.1f  conf %.2f  status %d\n",
               r.box.x, r.box.y, r.box.w, r.box.h, r.confidence, r.status);
}
camtrak_destroy(t);
```

Every function that can fail returns a `camtrak_error` (`CAMTRAK_OK` = 0).
`camtrak_lock` allocates; `camtrak_update` never touches the heap.

## Layout

```
include/camtrak/camtrak.h       public C API (opaque handle)
src/fft.[ch]                  radix-2 1D/2D FFT with precomputed plans
src/patch.[ch]                patch sampling (bilinear, gray/BGR), adaptive color channel,
                              log + normalize + cosine window
src/mosse.[ch], scale.[ch]    correlation filter; DSST-style size filter
src/camtrak.c                  tracker state and the per-frame pipeline
python/camtrak/__init__.py     ctypes binding: Tracker, Result
python/camtrak/multi.py        MultiTracker: many targets, IDs, speeds, probation, pruning
python/camtrak/detect.py       click-to-lock, box tightening, auto-detect, camera motion, scene cuts
python/camtrak/pipeline.py     Pipeline: everything per frame, FrameResult.to_dict()
python/camtrak/render.py       the camtrak HUD and themes
python/camtrak/io.py           files, folders, cameras, streams
python/camtrak/cli.py          the `camtrak` app
examples/embed.py             camtrak inside your own app
demo/                         single-target demo, benchmark helpers, camtrak_hud.py (= `camtrak`)
tools/                        synthetic clips with ground truth, scoring
tests/                        C: FFT, API, synthetic tracking + speed;
                              Python: test_multi.py (accuracy gate), test_app.py (window, scripted)
```

## How the tracker decides TRACKING / COASTING / LOST

* **PSR** (peak-to-sidelobe ratio) = how far the response peak stands above
  the rest of the response, in standard deviations. Clean match: 15-100.
  Occluded or wrong object: 3-11. `confidence` is PSR mapped to 0..1
  (5 -> 0, 20 -> 1).
* **TRACKING**: PSR >= `psr_track` (11). The peak is fed to a constant-velocity
  Kalman filter. If PSR >= `psr_update` (12) the appearance model also learns.
* **COASTING**: PSR too low. The measurement is ignored and the box follows the
  Kalman prediction, with velocity damped 5% per frame. Nothing is learned,
  so an occluder cannot overwrite the target's appearance.
* **LOST**: after `lost_after` (20) coasting frames, or immediately after a
  scene cut. The box stays at the last confident position. Each frame,
  `reacquire_budget` (24) windows are searched in a spiral around it whose
  radius grows with time lost (nearby first, whole frame after ~3 s), at three
  sizes. Frames alternate between the current appearance model and a frozen
  copy from lock time, which an occluder can never have contaminated. A match
  with PSR >= `psr_reacquire` (14) switches back to TRACKING.
* **Size**: once the position is known, a separate scale filter compares the
  target box at 17 sizes (1.04^-8 .. 1.04^8) and picks the best; the box may
  change by at most 2.5% per frame. It samples the box only (no surroundings),
  so a shadow or a tree next to the target does not inflate it.

## How the correlation filter works

1. **Patch.** Cut a window `padding x` the target size around the last known
   center, resampled so its long side is `template_size` (64) pixels; each side
   is a power of two.
2. **Preprocess.** `log(x+1)`, then zero mean / unit variance, then multiply by
   a cosine window so the patch fades to zero at its edges.
3. **Correlate.** In the Fourier domain the learned filter is
   `conj(H) = A / (B + lambda)` with `A = avg(G . conj(F))`, `B = avg(|F|^2)`,
   where `G` is the FFT of a Gaussian peak. Response = `IFFT(conj(H) . Z)`.
4. **Locate.** The response maximum, refined to sub-pixel with a parabola fit,
   is where the target moved.
5. **Adapt.** Cut a patch at the new position and blend it into `A` and `B`
   with learning rate 0.125.

The first frame trains on 8 slightly rotated/scaled copies of the target so
the filter does not just memorize a single image.

## Current numbers (v0.3.0, Windows x64, MSVC, Release)

`ctest` gates (fail the build if they regress):

| Test | Result |
|---|---|
| FFT vs double-precision DFT, up to 64x64; paired real FFT | relative error < 3e-7 |
| Synthetic 320x240, 24 px target, panning background, noise | mean error 0.56 px, 100% within 5 px |
| Full occlusion for 50 frames | 0 false TRACKING frames, LOST reported, re-acquired 10 frames after reappearing, 100% within 5 px after |
| Target grows 24 -> 51 px | size error 1.3%, 100% within 5 px |
| Speed, 1280x720 | 0.63 ms/frame tracking with scale (~1600 fps); LOST frames up to ~7 ms |

Demo clip `tools/make_clip.py` (720p skier, tree passes in front), scored with `tools/score.py`:

| Phase | Before tree | While hidden | After tree |
|---|---|---|---|
| 1 MOSSE only | 1.3 px | 37.5 px | 110 px, never recovers |
| 2 + PSR gating | 1.3 px | 14.5 px | 45 px |
| 3 + Kalman coasting | 0.6 px | 6.6 px | 403 px (coasts away) |
| 4 + LOST / re-acquire | 0.6 px | 6.9 px | **1.6 px, 98% within 20 px** |
