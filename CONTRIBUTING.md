# Contributing to Camtrak

Thanks for helping! Bug reports, test footage, and pull requests are all welcome.

## Before you start

* **Bugs:** open an issue with the command you ran, what you expected, and what
  happened. If you can, attach a short clip (or a link) and the output of
  `camtrak --source clip.mp4 ... --jsonl out.jsonl`.
* **Features:** open an issue first so we can agree on the approach.

## Contributor License Agreement

Camtrak is dual-licensed (AGPL v3 and a commercial license, see
[LICENSING.md](LICENSING.md)). To keep that possible, every contributor must
sign a Contributor License Agreement before a pull request can be merged. It
lets the project distribute your contribution under both licenses; you keep
the copyright to your work. You will be asked to sign it on your first pull
request.

## Development

```sh
./build.sh test                       # C core: build and run the C tests
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[vision]"
python tests/test_multi.py            # accuracy gate (synthetic clips with ground truth)
python tests/test_app.py              # the app, driven with scripted clicks and keys
```

Guidelines:

* C core: C11, no dependencies beyond libc/libm, no heap allocation in
  `camtrak_update`. Keep `include/camtrak/camtrak.h` stable.
* Accuracy changes: include before/after numbers from `tools/score_multi.py`
  (and `tests/test_multi.py` must still pass).
* American English in code, comments, and docs.
