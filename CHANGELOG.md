# Changelog

## 0.4.0 (unreleased): first public release

* Renamed the project to **Camtrak** (C API `camtrak_*`, Python package `camtrak`, `camtrak` command).
* Multi-object tracking: `MultiTracker` with IDs, speeds, probation, and pruning.
* `Pipeline`: tracking, auto-detect, scene cuts, and drawing in one call per frame; JSON-ready results.
* Auto-detect: small objects against plain backgrounds; movers on busy backgrounds with camera-motion compensation.
* Click-to-lock and loose-box tightening.
* Adaptive color channel in the C core (`adaptive_color`): tracks in the color projection that best separates the target.
* Size filter: slower learning, growth capped harder than shrinking.
* The Camtrak HUD with themes (`camtrak`, `mono`, `contrast`, `light`).
* The `camtrak` app: files, image folders, webcams, streams; CSV and JSON Lines output.
* Licensing: AGPL v3 or commercial license.

## 0.3.0

* Single-object C tracker: MOSSE correlation filter, PSR confidence, Kalman coasting,
  re-detection when lost, DSST-style scale filter, color model, scene-cut handling,
  VitTrack second opinion.
