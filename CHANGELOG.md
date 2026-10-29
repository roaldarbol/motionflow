# Changelog

## Unreleased

### Added

- **Chromatic motion** (`motionflow chromatic`, `motionflow.chromatic()`, `ChromaticMotion`): BehaveAI's motion encoding (Troscianko et al., *PLOS Biology* 2026), with recent movement in blue and older movement in green and red over a dimmed greyscale frame. Written from the method's description, with BehaveAI's options: exponential or sequential references, gains, threshold, tail-only, motion-only and frame skipping (#21).
- `motionflow download` fetches the deep methods' weights ahead of time; `motionflow methods` shows whether they're cached. Weights are checksummed and cached in `~/.cache/motionflow` (`MOTIONFLOW_CACHE_DIR` to move it) (#16).
- SEA-RAFT has `small`, `medium` and `large` sizes; `--device` is checked up front with a clear message (#16).
- **DIS optical flow** (`--method dis`, `DISFlow`), now the default. On 720p and 1080p test sequences with known motion it was 3–4× faster than Farneback and more accurate, including on large motions Farneback couldn't follow (#20).
- **Subcommands**: `motionflow flow VIDEO`, with `motionflow VIDEO` as the short form, so existing commands keep working; and `motionflow methods`, which lists the methods and whether they can run here. Room for the coming commands (#14).
- **Several inputs at once**: files, folders and patterns such as `'trials/*.mp4'`, with `-o` as an output folder. Folders skip earlier outputs (`*_flow.*`); a failing video doesn't stop the rest (#14).
- `-q/--quiet`, `-v/--verbose` (full error details), `-h`, `-V`, and `python -m motionflow` (#14).
- A colour key in the corner of the output (`--legend`, on by default): a colour wheel for `hsv`, a colour bar for `heatmap`, a reference arrow for `quiver`, labelled with the speed at full strength (#15).
- `--min-flow` hides motion slower than a threshold; `--percentile` sets which speed is full strength; `--quiver-scale` sets the arrow length (#15).
- `LucasKanadeFlow(points="grid")` tracks a regular grid of points; `grid_step` was accepted but ignored before (#15).
- `motionflow.flow()` replaces `process_video()` and returns a `Result` (output path, frame counts, frame rate, encoder, timing, and the normalisation used). `process_video()` still works but is deprecated (#12).
- A pipeline for any per-frame method: subclass `FrameTransform` (or `Transform`) and pass it to `motionflow.run()`, which handles reading, writing, ranges, scaling, output naming and safety. `FlowTransform.flows()` yields the flow fields themselves (#12).
- Everything needed for custom pipelines is importable from `motionflow` (#12).
- `--start`/`--end` (seconds, clock time or frame numbers), `--scale`, `--crf` and `--overwrite` options, also as `process_video` arguments.
- `VideoReader.frames()` yields frames with their index and timestamp; `motionflow.video.parse_time` parses time positions.

### Fixed

- **SEA-RAFT and NeuFlow v2 work.** Their setup step downloaded files that no longer exist, wrote into the installed package, and the models were wired up wrongly (wrong output, missing initialisation; NeuFlow's input was scaled twice, giving flow off by ~17 px on a 3 px shift). Their model code is now included at pinned upstream commits, with the fixes; on a known shift, SEA-RAFT is within 0.01 px and NeuFlow within 0.08 px (#16).
- RAFT normalises its input as torchvision intends ([−1, 1], not [0, 1]) and pads by repeating edges (#16).
- Very small frames (under 128 px) no longer crash the deep methods.
- Lucas-Kanade output was almost entirely black, with every style: its tracked points are now drawn as dots or arrows (#15).
- Quiver arrows were drawn at their true length, so ordinary motion of a few pixels was invisible; they're now scaled to the grid, averaged over each cell, and left out where nothing moves (#15).
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

- Weights download on first use without asking, like PyTorch's and torchvision's; the interactive prompts, which broke batch jobs, are gone. The `motionflow setup-vendors` step is gone too (#16).
- **The default method is DIS** instead of Farneback (#20). Use `--method farneback` for results like motionflow 0.1's; `process_video` keeps Farneback.
- The command line prints output paths on stdout, one per line, and everything else on stderr, so `out=$(motionflow video.mp4)` works. Progress bars only appear in an interactive terminal, so batch-job logs stay clean. Exit status 1 means a video failed, 2 a usage error, 130 Ctrl-C (#14).
- Options that have no effect with the other options given (such as `--quiver-step` without `--viz quiver`) are reported instead of silently ignored (#14).
- `-o` without an extension is a folder.
- **Colours use one scale for the whole video by default** (`--normalize auto`): the 99th percentile of all motion, measured exactly on short videos and sampled on long ones. The old default, `per-frame`, stretched each frame to its own largest motion, so a still scene's noise looked like motion. `global` now uses the percentile too, rather than the single largest value, and the scale never drops below 1 px/frame (#15).
- `--overlay` (the flow's opacity over the original frame) replaces `--blend` (the original's opacity) and `--viz blend` (#15). `process_video` still accepts `blend`.
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
