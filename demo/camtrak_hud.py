"""
camtrak_hud - the Camtrak app, runnable from a source checkout.

    python demo/camtrak_hud.py --source clip.mp4 --display
    python demo/camtrak_hud.py --source clip.mp4 --auto --output out.mp4

Same as the `camtrak` command after `pip install -e .`; see python/camtrak/cli.py
for every option and key.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from camtrak.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
