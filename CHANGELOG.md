# Changelog

## Unreleased

### Added

- `motionflow.flow()` replaces `process_video()` and returns a `Result` (output path, frame counts, frame rate, encoder, timing, and the normalisation used). `process_video()` still works but is deprecated (#12).
- A pipeline for any per-frame method: subclass `FrameTransform` (or `Transform`) and pass it to `motionflow.run()`, which handles reading, writing, ranges, scaling, output naming and safety. `FlowTransform.flows()` yields the flow fields themselves (#12).
- Everything needed for custom pipelines is importable from `motionflow` (#12).
- `--start`/`--end` (seconds, clock time or frame numbers), `--scale`, `--crf` and `--overwrite` options, also as `process_video` arguments.
- `VideoReader.frames()` yields frames with their index and timestamp; `motionflow.video.parse_time` parses time positions.

### Fixed

- `-o` pointing at the input no longer destroys the input; motionflow refuses (#10).
- A failed or interrupted run no longer leaves a partial or empty file that looks complete: output is written to a temporary file and moved into place at the end (#10).
- MPEG-TS (`.ts`, `.MTS`) and raw H.264 (`.h264`) inputs work; they used to give an empty output while reporting success (#11).
- `--model-size small` was ignored on the command line, so RAFT always used the large model (#11).
- On macOS, `.mp4` output was silently written as `.avi`, and the path reported was a file that didn't exist (#11).
- Videos with rotation metadata (e.g. from phones) are read the way players show them.
- Odd frame sizes are padded by a pixel instead of cropped (#11).
- Error messages containing square brackets, such as `pip install "motionflow[torch]"`, are shown in full (#11).
- TV-L1 works. It called an OpenCV method that doesn't exist and so always failed; the tests skipped it because OpenCV's contrib modules weren't installed. It now also uses OpenCV's default iteration counts, which converge to the same result about 3.5× faster.

### Changed

- **The output has one frame per input frame**, so it lines up with the input: output frame *i* shows the motion from input frame *i − 1* to *i*, and frame 0 shows none. It used to have one frame fewer (#12).
- **`stride`** estimates flow between frames `stride` apart and divides it by `stride`, so magnitudes stay in pixels per frame; before, they grew with the stride. The output still has one frame per input frame (#12).
- **`smooth`** averages over a window centred on each frame instead of the preceding frames, so smoothing no longer lags behind the motion. `smooth=1` is the same as 0 (#12).
- Progress bars adapt to notebooks and can be turned off (`progress=False`).
- Video is read and written with PyAV (FFmpeg) instead of OpenCV. `.mp4` output is H.264 on every platform, so it plays in browsers, slides and Jupyter; `--codec` takes codec names (`h264`, `hevc`, `vp9`, `av1`, `ffv1`, `mjpeg`, `mpeg4`), and OpenCV FourCC codes still work (#13).
- motionflow won't overwrite an existing output unless asked to (`--overwrite`, `overwrite=True`) (#10).
- Default output names for inputs in containers motionflow can't write (`.ts`, `.h264`, …) end in `.mp4`.
- `on_progress` receives `None` as the total when the input doesn't record its length.
- motionflow depends on `opencv-contrib-python-headless` instead of `opencv-python-headless`, so TV-L1 works out of the box. Installing a second OpenCV wheel next to it, as the docs used to advise, broke both.
- `TVL1Flow`'s `iterations` argument is replaced by `inner_iterations` and `outer_iterations`, matching OpenCV.
- Development uses pixi instead of uv (see CONTRIBUTING.md). CI runs on Linux, macOS and Windows, and on Python 3.10–3.14, and checks formatting and linting.
- Package metadata: PEP 639 licence fields, author and maintainer, Python 3.13 and 3.14 classifiers, and a `py.typed` marker. Development dependencies are no longer published as extras (`dev`, `dev-torch`, `docs`).
- The docs no longer describe things that don't exist (`motionflow setup-vendors`, a conda-forge package), and say which methods don't work yet.

## 0.1.0 (2026-03-10)

First release.
