# Tether

Fast, CPU-only visual object tracker. Draw a box around an object in the first
frame; Tether follows it in every frame after that.

The core is plain C11 with no dependencies beyond the C standard library.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Project layout, radix-2 FFT, MOSSE correlation filter | done |
| 2 | Peak-to-sidelobe ratio confidence, gated model update | done |
| 3 | Constant-velocity Kalman filter, COASTING | done |
| 4 | LOST after N misses, spiral re-detection, re-acquire | done |
| 5 | Scale estimation (DSST-style scale filter) | done |
| 5b | Scene-cut handling, clean "anchor" model for re-acquisition | done |
| 6 | Colour model (veto look-alikes), ML second opinion (VitTrack), verify/reacquire API, run logs | done |
| 7 | Multi-target (`MultiTracker`), click-to-lock, auto-detect small objects, targeting HUD with speed + close-up | done |
| later | HOG/colour features, ONNX verifier, Python/Node bindings, OpenCV comparison bench | |

## Build

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

## HUD: track many objects at once

`demo/tether_hud.py` tracks any number of objects and draws a targeting
overlay: corner brackets, a centre dot, and a panel per target with
`TRK ID: n [LOCKED]`, its position and its speed in px/s, plus a magnified
close-up of one target in the bottom-right corner. Red = LOCKED, amber =
COASTING, grey = SEARCHING.

```sh
# pick targets yourself: click an object to lock it (box sized automatically), or drag a box
python demo/tether_hud.py --source clip.mp4 --display

# let it find small objects against sky / snow by itself (aircraft, birds, drones, skiers)
python demo/tether_hud.py --source airshow.mp4 --auto --display

# unattended: save the annotated video and a per-target CSV
python demo/tether_hud.py --source clip.mp4 --auto --output out.mp4 --csv out.csv
python demo/tether_hud.py --source clip.mp4 --point 640,360 --bbox 100,80,40,30 --output out.mp4
```

**In the window** (before playback and while playing): left click = lock the
object under the cursor; left drag = lock exactly that box; right click = drop
a target; Tab = choose which target the close-up shows; `x` = auto-detect
on/off; `c` = drop all; space = pause (and start, in the picker); `a`/`d` =
pick the start frame; `q` = quit. You can add targets at any moment, paused or
not.

**How it works.** Each target has its own Tether tracker (0.2-0.6 ms each),
so dozens of targets stay real time; the first 4 targets you pick also get the
ML second opinion. *Click-to-lock* (`demo/detect.py: snap_box`) grows a window
around the click until the object fits, treats the window's border colours as
background, and boxes the blob under the click. *Auto mode*
(`BlobDetector`) subtracts a median-filtered background, keeps compact
blobs that stand out from a plain surrounding (tree lines, crowds and cloud
edges are rejected as clutter), and spawns a target only when a blob is seen
on two passes in a row. Auto targets are dropped when they leave the frame,
stay unconfirmed for 1 s, or land on an object that is already tracked; your
own targets are never dropped. At a scene cut, auto targets are cleared and
re-detected, and yours start searching.

**Busy backgrounds.** On textured ground (vehicles seen from a drone, people
on a pitch) "stands out" is not enough: every tree does. Auto mode then
measures the camera's motion (features + RANSAC), finds things that move
relative to the scene (`MotionDetector`: frames aligned to cancel the camera,
current frame compared with two earlier ones), and puts every new auto
target on probation: it is only shown once it has moved relative to the
scenery; one that moves with the camera is a tree or a building and is
dropped, and that spot is remembered.

**Colour.** The correlation tracker sees one channel. Brightness alone loses
colour contrast (a red car on a brown field has about the same brightness),
so at lock time Tether compares brightness with the colour projection that
best separates the target from its surroundings (Fisher discriminant) and
tracks in that when it is clearly better (`adaptive_color`, default on).

Synthetic clips with ground truth, scored with `tools/score_multi.py`
(coverage = object-frames with a track within 15 px):

| Clip | Mode | Coverage | False track-frames | ID switches |
|---|---|---|---|---|
| sky, 10 aircraft/birds, 3 seeds (`tools/make_sky_clip.py`) | `--auto` | 97-98% | 0.4-0.5% | 0 |
| sky | one click per object | 100% of clicked | 0.7% | 0 |
| aerial terrain, 8 vehicles, moving camera, 3 seeds (`tools/make_ground_clip.py`) | `--auto` | 73-86% | 12-29% | 1-7 |

Tracking from exact boxes on the terrain clips: 62-75% of frames within
15 px with brightness only, 86-98% with the adaptive colour channel.
Known weak spot: a box that starts looser than the object can slowly grow
into the background (most of the false track-frames on terrain are such
boxes on real vehicles).

On a 66 s, 1080p, 24-cut montage (jets, bikes, F1, skiers, birds) `--auto`
costs 5-10 ms/frame for camera motion + tracking. `python tests/test_multi.py`
gates these numbers.

```python
import tether as tt
mt = tt.MultiTracker(fps=30)
a = mt.add(frame, (x, y, w, h))              # returns the ID
for t in mt.update(next_frame):
    print(t.id, t.label, t.center, t.speed)  # LOCKED / COASTING / SEARCHING, px/s
```

## Demo: track a video

Needs Python 3 with `numpy` and `opencv-python` (OpenCV is used only by the demo).

```sh
# make a synthetic test clip (prints the first-frame box)
python tools/make_clip.py clips/synthetic.mp4

# pick the target yourself and watch it track
python demo/tether_demo.py --source clips/synthetic.mp4 --display

# or pass the box and save results
python demo/tether_demo.py --source clips/synthetic.mp4 --bbox 181,205,22,22 \
    --csv clips/track.csv --json clips/run.json --output clips/tracked.mp4
```

**Picking the target:** a window opens. Use the slider (or `a`/`d`) to go to
any frame, drag to draw a box, drag inside it to move it, drag an edge or
corner to resize it, then press Enter.

**While playing:** space pauses, `s` re-picks the target on the current frame,
`q` quits. Box colour: green = TRACKING, orange = COASTING, red = LOST; the bar
is the confidence (0..1).

**ML second opinion (default when `models/vittrack.onnx` exists, `--no-ml` to
disable):** a small neural tracker (VitTrack, OpenCV Zoo, Apache 2.0, ~4 ms)
runs alongside. While Tether is confident it keeps the neural tracker on the
right object; while Tether is lost, the neural tracker's box is a candidate
that must pass Tether's checks (`tether_verify`: shape, colour, mostly inside
the frame, and standing out from its surroundings like the selection did)
before Tether moves there (`tether_reacquire`, which keeps the user's box
shape so the box never jumps). While lost, the neural tracker also sweeps the
whole frame, one region per frame. On a 48 s, 15-shot drone ski montage with
two skiers this took unattended tracking from 43 to ~1,000 of 1,452 frames,
with no frame on the wrong skier or on background in any checked sample, no
box size jumps, and median jitter 0.2 px (`tools/smoothness.py`).

**Logs:** every run saves `logs/<date>-<time>_<video>/` with per-frame
internal state (`frames.csv`), user actions and cuts (`events.jsonl`),
snapshots at every status change (`snapshots/`) and `summary.json`.

**Edited videos:** scene cuts are detected and the tracker is told
(`tether_scene_cut`), so it searches instead of coasting on a stale
prediction. Playback keeps going and shows "scene cut - searching"; press `s`
any time to re-pick. `--pause-on-cut` stops at every cut instead.

### Real footage and benchmark sequences

`--source` also accepts an image folder. OTB and LaSOT sequence folders are
detected automatically: the ground-truth box starts the tracker, and the run
is scored at the end (precision@20 px, success AUC, honesty on target-absent
frames, confidence when right vs wrong).

```sh
python demo/tether_demo.py --source D:/LaSOT/bird/bird-2 --display --show-truth
python demo/tether_demo.py --source D:/UAV123/data_seq/UAV123/bird1 --gt D:/UAV123/anno/UAV123/bird1.txt
python tools/bench_folder.py D:/LaSOT/bird --csv bird.csv      # every sequence, one table
```

For your own videos (no ground truth), use `--select-target --display`.

## Usage (Python)

```python
import tether as tt          # python/ on sys.path; temporary ctypes binding

trk = tt.Tracker()
trk.lock(first_frame, (x, y, w, h))      # numpy uint8, gray or BGR
r = trk.update(frame)
print(r.box, r.confidence, r.status_name)
```

## Usage (C)

```c
#include <tether/tether.h>

tether_tracker *t = tether_create(NULL);                  /* default params */

tether_frame f = { pixels, width, height, stride, TETHER_FORMAT_BGR8 };
tether_box box = { 410, 220, 36, 28 };                    /* x, y, w, h */
tether_lock(t, &f, box);

while (next_frame(&f)) {
    tether_result r;
    if (tether_update(t, &f, &r) == TETHER_OK)
        printf("%.1f %.1f %.1f %.1f  conf %.2f  status %d\n",
               r.box.x, r.box.y, r.box.w, r.box.h, r.confidence, r.status);
}
tether_destroy(t);
```

Every function that can fail returns a `tether_error` (`TETHER_OK` = 0).
`tether_lock` allocates; `tether_update` never touches the heap.

## Layout

```
include/tether/tether.h   public API (opaque handle)
src/fft.[ch]              radix-2 1D/2D FFT with precomputed plans
src/patch.[ch]            patch sampling (bilinear, gray/BGR), log + normalise + cosine window
src/mosse.[ch]            MOSSE filter: train, update, respond, sub-pixel peak
src/tether.c              tracker state and the per-frame pipeline
tests/                    FFT vs naive DFT, API validation, synthetic tracking + speed,
                          test_multi.py (multi-target / auto / click-to-lock gate)
python/tether/multi.py    MultiTracker: many targets, IDs, speeds, pruning
demo/tether_hud.py        multi-target app with the targeting HUD
demo/hud.py, detect.py    HUD drawing; click-to-lock and auto-detect
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
   centre, resampled so its long side is `template_size` (64) pixels; each side
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
the filter does not just memorise a single image.

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
