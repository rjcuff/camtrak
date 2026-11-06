"""
tether_hud - the Tether app, runnable from a source checkout.

    python demo/tether_hud.py --source clip.mp4 --display
    python demo/tether_hud.py --source clip.mp4 --auto --output out.mp4

Same as the `tether` command after `pip install -e .`; see python/tether/cli.py
for every option and key.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from tether.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
