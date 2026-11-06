"""
embed.py - Tether inside your own application, in ~25 lines.

    pip install -e ".[vision]"      (once, from the repo; after ./build.sh)
    python examples/embed.py clips/sky.mp4        # or 0 for the webcam

Reads frames, tracks everything Tether finds, and hands you plain data per
frame. Replace `use()` with whatever your app does: drive a gimbal, raise an
alert, send to a web dashboard, count objects crossing a line...
"""
import json
import sys

import cv2

import tether as tt


def use(result: tt.FrameResult) -> None:
    data = result.to_dict()                 # {"frame", "targets": [{"id", "state", "box", "speed", ...}]}
    fast = [t["id"] for t in data["targets"] if t["speed"] > 100]
    if fast:
        print(json.dumps({"frame": data["frame"], "fast_targets": fast}))


def main() -> None:
    source = tt.io.open_source(sys.argv[1] if len(sys.argv) > 1 else "0")
    pipe = tt.Pipeline(fps=source.fps, auto=True, theme="tether")
    while (frame := source.next()) is not None:
        result = pipe.process(frame)
        use(result)
        cv2.imshow("my app", pipe.render(frame, result))     # optional: Tether's HUD
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    source.close()
    pipe.close()


if __name__ == "__main__":
    main()
