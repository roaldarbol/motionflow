# CLI Reference

## Synopsis

```
motionflow [OPTIONS] INPUT
```

## Arguments

| Argument | Description |
|----------|-------------|
| `INPUT`  | Path to the input video file. |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output PATH` | `<input>_flow.<ext>` | Output file path. See [Output files](#output-files). |
| `-m, --method METHOD` | `farneback` | Flow algorithm. See [Methods](methods.md). |
| `--viz STYLE` | `hsv` | Visualisation style. |
| `--blend FLOAT` | `0.0` | Opacity of original frame over flow (0 = pure flow, 1 = pure original). |
| `--stride INT` | `1` | Compute flow every N frames. |
| `--normalize MODE` | `per-frame` | Magnitude normalisation: `per-frame` or `global`. |
| `--max-flow FLOAT` | — | Manual magnitude cap. Overrides `--normalize`. |
| `--smooth INT` | `0` | Temporal smoothing window in frames. |
| `--codec TEXT` | `auto` | Output codec. See [Output files](#output-files). |
| `--crf INT` | — | Output quality, lower is better and bigger: the CRF for `h264`, `hevc`, `vp9` and `av1`; the quantiser (2–31) for `mjpeg` and `mpeg4`. |
| `--start TIME` | — | Start here: seconds (`12.5`), clock time (`1:30`, `01:02:03.5`) or a frame number (`120f`). |
| `--end TIME` | — | Stop here (exclusive), in the same formats as `--start`. |
| `--scale FLOAT` | `1.0` | Resize before computing flow, e.g. `0.5` for half size. Much faster on large videos. |
| `--overwrite` | off | Replace the output file if it already exists. |
| `--fps FLOAT` | — | Output FPS. Preserves source FPS if not set. |
| `--device TEXT` | `auto` | Device for deep methods: `auto`, `cuda`, `mps`, `cpu`. |
| `--model-size SIZE` | `large` | Model size for RAFT/SEA-RAFT: `small` or `large`. |
| `--quiver-step INT` | `16` | Grid spacing in pixels for quiver arrows. |
| `--version` | — | Print version and exit. |

## Examples

Basic usage with defaults (Farneback, HSV visualisation):

```bash
motionflow recording.mp4
```

TV-L1 method, heatmap visualisation, composited 30% over the original:

```bash
motionflow recording.mp4 --method tvl1 --viz heatmap --blend 0.3
```

RAFT with GPU, every other frame, saved to a specific path:

```bash
motionflow recording.mp4 --method raft --device cuda --stride 2 -o flow.mp4
```

Global normalisation for a clip where most motion is subtle:

```bash
motionflow recording.mp4 --normalize global
```

!!! note
    `--normalize global` reads the video twice — once to compute the maximum flow
    magnitude across all frames, and once to render. For long videos, prefer
    `--max-flow` with a manually estimated cap instead.

Quiver overlay with coarser arrows:

```bash
motionflow recording.mp4 --viz quiver --quiver-step 32
```

One minute from 1:30, at half size:

```bash
motionflow recording.mp4 --start 1:30 --end 2:30 --scale 0.5
```

Lossless output, for analysing the colours later:

```bash
motionflow recording.mp4 -o flow.mkv --codec ffv1
```

## Output files

By default the output goes next to the input as `<input>_flow.<ext>`, keeping the input's extension when motionflow can write it, and using `.mp4` otherwise (for example for `.ts`, `.h264` or `.wmv` inputs).

The extension picks the container, and `--codec auto` picks a codec to match:

| Extension | Codec (`auto`) | Notes |
|-----------|----------------|-------|
| `.mp4`, `.m4v`, `.mov` | H.264 | Plays in browsers, slides and Jupyter |
| `.mkv` | H.264 | |
| `.webm` | VP9 | AV1 where FFmpeg has no VP9 encoder (conda-forge on Windows) |
| `.avi` | MJPEG | Large files; each frame stored separately |

Other codecs: `hevc` (smaller files), `av1`, `ffv1` (lossless; use `.mkv`), `mjpeg`, `mpeg4`, or the name of any FFmpeg encoder, such as `h264_nvenc` on an NVIDIA GPU. OpenCV's FourCC codes (`mp4v`, `MJPG`, `avc1`) still work.

Frame sizes that aren't even are padded by one pixel, because H.264 and most other codecs need even dimensions; motionflow says when it does this.

motionflow writes to a hidden temporary file (`.<name>.part.<ext>`) and only moves it into place when it has finished. If it fails or you press Ctrl-C, nothing is left behind and an existing output stays as it was. It never overwrites an existing file unless you pass `--overwrite`, and refuses to write over the input.

!!! note "One frame fewer"
    The output currently has one frame fewer than the input, because flow is computed between consecutive frames: output frame *i* shows the motion from input frame *i* to *i + 1*. This is being changed so that frames line up one to one ([#12](https://github.com/roaldarbol/motionflow/issues/12)).
