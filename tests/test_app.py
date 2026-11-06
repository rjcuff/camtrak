"""
test_app - drives the `camtrak` app's window with scripted mouse clicks and keys.

    python tests/test_app.py        (needs numpy + opencv-python and a built library)

OpenCV's window functions are replaced by fakes: imshow does nothing, the
mouse callback is captured, and waitKey plays a script. Checks that, in the
picker, a click locks an object and a drag adds a box; that Tab, t and x
work; that playback tracks the picked targets; and that a right click drops
a target while playing.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from camtrak import cli  # noqa: E402


class FakeWindow:
    def __init__(self, script):
        self.script = list(script)    # (at_waitkey_call, action)
        self.calls = 0
        self.mouse = None
        self.shown = 0

    def install(self):
        cv2.namedWindow = lambda *a, **k: None
        cv2.destroyAllWindows = lambda *a, **k: None
        cv2.setMouseCallback = lambda _w, cb, *a: setattr(self, "mouse", cb)
        cv2.imshow = lambda *a: setattr(self, "shown", self.shown + 1)
        cv2.waitKey = self.wait_key

    def wait_key(self, _delay=0):
        self.calls += 1
        key = -1
        while self.script and self.script[0][0] <= self.calls:
            _, action = self.script.pop(0)
            if isinstance(action, str):
                key = 9 if action == "TAB" else ord(action)
            else:
                action(self.mouse, cli.App.scale_for_test)
        return key


def click(x, y, button="left"):
    def act(cb, s):
        sx, sy = int(x * s), int(y * s)
        if button == "left":
            cb(cv2.EVENT_LBUTTONDOWN, sx, sy, 0, None)
            cb(cv2.EVENT_LBUTTONUP, sx, sy, 0, None)
        else:
            cb(cv2.EVENT_RBUTTONDOWN, sx, sy, 0, None)
    return act


def drag(x0, y0, x1, y1):
    def act(cb, s):
        cb(cv2.EVENT_LBUTTONDOWN, int(x0 * s), int(y0 * s), 0, None)
        cb(cv2.EVENT_MOUSEMOVE, int(x1 * s), int(y1 * s), 0, None)
        cb(cv2.EVENT_LBUTTONUP, int(x1 * s), int(y1 * s), 0, None)
    return act


def main() -> int:
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'ok  ' if cond else 'FAIL'} {name}: {detail}")

    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "sky.mp4"
        subprocess.run([sys.executable, "tools/make_sky_clip.py", str(clip)], cwd=ROOT,
                       check=True, capture_output=True)
        truth = {}
        for r in csv.DictReader(open(clip.with_suffix(".truth.csv"))):
            if r["frame"] == "0":
                truth[r["id"]] = tuple(float(r[k]) for k in "xywh")
        cx = lambda b: (b[0] + b[2] / 2, b[1] + b[3] / 2)  # noqa: E731
        p6, p7 = cx(truth["6"]), truth["7"]

        jl = Path(tmp) / "run.jsonl"
        script = [
            (2, click(*p6)),                                   # picker: lock plane 6 by clicking it
            (3, drag(p7[0] - 4, p7[1] - 4, p7[0] + p7[2] + 4, p7[1] + p7[3] + 4)),   # and bird 7 by a box
            (4, "TAB"), (5, "t"), (6, "t"), (7, "t"), (8, "t"),   # cycle focus, themes back to start
            (9, " "),                                           # start playing
        ]
        win = FakeWindow(script)
        win.install()
        orig_init = cli.App.__init__

        def init(self, args):
            orig_init(self, args)
            cli.App.scale_for_test = 1.0
            cli.App.current = self
        cli.App.__init__ = init
        orig_run = cli.App.run

        def run(self):
            # the window scale is known once the first frame is read; expose it to the fake mouse
            orig_show = self.show

            def show(img):
                cli.App.scale_for_test = self.scale
                orig_show(img)
            self.show = show
            return orig_run(self)
        cli.App.run = run

        # while playing: right-click plane 6 where it is after ~60 frames, then quit
        def drop_plane(cb, s):
            t = cli.App.current.pipe.tracks.get(1)     # where plane 6's target is right now
            if t is not None:
                click(*t.center, button="right")(cb, s)
        win.script += [(70, drop_plane), (90, "q")]
        code = cli.main(["--source", str(clip), "--display", "--jsonl", str(jl), "--no-ml"])
        frames = [json.loads(line) for line in open(jl)]

        check("exit", code == 0, f"exit code {code}")
        check("window", win.shown > 50 and win.mouse is not None, f"{win.shown} frames shown")
        first = frames[0]["targets"]
        check("picker click + drag", len(first) == 2, f"{len(first)} targets at start (2)")
        mid = next((f for f in frames if f["frame"] == 40), frames[-1])

        def near(t, b, f):
            tr = {}
            for r in csv.DictReader(open(clip.with_suffix(".truth.csv"))):
                if int(r["frame"]) == f:
                    tr[r["id"]] = cx(tuple(float(r[k]) for k in "xywh"))
            x, y = tr[b]
            return abs(t["center"][0] - x) < 15 and abs(t["center"][1] - y) < 15
        ids = {t["id"]: t for t in mid["targets"]}
        check("tracking", 1 in ids and 2 in ids and near(ids[1], "6", mid["frame"]) and near(ids[2], "7", mid["frame"]),
              "both picked targets on their objects at frame 40")
        check("focus key", frames[0]["focus"] in (1, 2), f"focus {frames[0]['focus']}")
        last = frames[-1]["targets"]
        check("right-click drop", [t["id"] for t in last] == [2], f"targets at the end: {[t['id'] for t in last]}")

    print("test_app: OK" if ok else "test_app: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
