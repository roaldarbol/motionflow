# Changelog

## Unreleased

### Fixed

- TV-L1 works. It called an OpenCV method that doesn't exist and so always failed; the tests skipped it because OpenCV's contrib modules weren't installed. It now also uses OpenCV's default iteration counts, which converge to the same result about 3.5× faster.

### Changed

- motionflow depends on `opencv-contrib-python-headless` instead of `opencv-python-headless`, so TV-L1 works out of the box. Installing a second OpenCV wheel next to it, as the docs used to advise, broke both.
- `TVL1Flow`'s `iterations` argument is replaced by `inner_iterations` and `outer_iterations`, matching OpenCV.
- Development uses pixi instead of uv (see CONTRIBUTING.md). CI runs on Linux, macOS and Windows, and on Python 3.10–3.14, and checks formatting and linting.
- Package metadata: PEP 639 licence fields, author and maintainer, Python 3.13 and 3.14 classifiers, and a `py.typed` marker. Development dependencies are no longer published as extras (`dev`, `dev-torch`, `docs`).
- The docs no longer describe things that don't exist (`motionflow setup-vendors`, a conda-forge package), and say which methods don't work yet.

## 0.1.0 (2026-03-10)

First release.
