# CLI Reference

## Synopsis

```
motionflow [OPTIONS] COMMAND [ARGS]...
motionflow VIDEO... [OPTIONS]          # short for: motionflow flow VIDEO...
```

| Command | What it does |
|---------|--------------|
| `flow` | Optical flow: how fast, and in which direction, each pixel moves. The default. |
| `chromatic` | Chromatic motion (BehaveAI): recent movement blue, older green and red. See [Methods](methods.md#chromatic-motion-motionflow-chromatic). |
| `methods` | List the optical flow methods, whether they can run here, and whether their weights are cached. |
| `download` | Download the deep methods' weights now (they otherwise download on first use). |

`-h` / `--help` works on every command, and `-V` / `--version` prints the version. `python -m motionflow` is the same as `motionflow`.

## Inputs

`VIDEO...` is one or more of:

- **video files**;
- **folders**: every video in the folder (not in subfolders), leaving out files that look like earlier outputs (`*_flow.*`), so running a command twice doesn't process its own results;
- **patterns** such as `'trials/*.mp4'`, quoted so that they work on Windows too.

With several inputs, `-o` is a folder, created if needed; any `-o` without an extension is treated as a folder. If one video fails, the others still run, and motionflow exits with status 1 at the end.

## Scripting

Output paths go to stdout, one per line; progress, messages and errors go to stderr. So scripts can capture the outputs:

```bash
out=$(motionflow recording.mp4 -q)
```

Progress bars only appear in an interactive terminal, so logs from batch jobs (SLURM, CI) stay clean.

| Exit status | Meaning |
|-------------|---------|
| 0 | Every video was processed |
| 1 | At least one video failed (see the messages) |
| 2 | The command line was wrong: unknown option, missing file, … |
| 130 | Interrupted with Ctrl-C; the unfinished output was not kept |

## `flow` options

### Flow

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --method METHOD` | `dis` | Flow method. See [Methods](methods.md), or run `motionflow methods`. |
| `--stride INT` | `1` | Estimate flow every N frames, between frames N apart, and show it until the next estimate. Magnitudes stay in pixels per frame. Faster; the first N frames show no motion. |
| `--smooth INT` | `0` | Average the flow over a window of N frames centred on each frame, so smoothing doesn't lag behind the motion. |
| `--device TEXT` | `auto` | Deep methods: `auto`, `cuda`, `mps` or `cpu`. |
| `--model-size SIZE` | `large` | Deep methods: `small` or `large`; SEA-RAFT also `medium`. |

### Drawing

| Option | Default | Description |
|--------|---------|-------------|
| `--viz STYLE` | `hsv` | `hsv` (hue is direction, brightness speed), `heatmap` (speed only) or `quiver` (arrows). See [Drawing the flow](#drawing-the-flow). |
| `--overlay FLOAT` | — | Draw the flow over the original frame at this opacity (0–1). |
| `--normalize MODE` | `auto` | How speeds map to colours: `auto`, `global` or `per-frame`. See [Which speed is full strength](#which-speed-is-full-strength). |
| `--percentile FLOAT` | `99` | For `auto` and `global`: the percentile of all motion drawn at full strength. |
| `--max-flow FLOAT` | — | Fixed speed (pixels per frame) drawn at full strength. Overrides `--normalize`. |
| `--min-flow FLOAT` | `0` | Draw motion below this many pixels per frame as none, to hide noise. |
| `--legend / --no-legend` | on | Draw a colour key in the bottom-right corner. |
| `--quiver-step INT` | `16` | `quiver`: grid spacing in pixels. |
| `--quiver-scale FLOAT` | auto | `quiver`: arrow length per pixel of motion. By default the full-strength speed is drawn as 90% of the grid spacing. |

Options that don't apply to the other choices you made (say, `--quiver-step` without `--viz quiver`) are reported with a warning.

### Input and output

Every command has these.

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output PATH` | next to each input | Output file, or folder. See [Inputs](#inputs) and [Output files](#output-files). |
| `--overwrite` | off | Replace outputs that already exist. |
| `--start TIME` | — | Start here: seconds (`12.5`), clock time (`1:30`, `01:02:03.5`) or a frame number (`120f`). |
| `--end TIME` | — | Stop here (exclusive), in the same formats as `--start`. |
| `--scale FLOAT` | `1.0` | Resize before processing, e.g. `0.5` for half size. Much faster on large videos. |
| `--fps FLOAT` | input's | Output frame rate. |
| `--codec TEXT` | `auto` | Output codec. See [Output files](#output-files). |
| `--crf INT` | — | Output quality, lower is better and bigger: the CRF for `h264`, `hevc`, `vp9` and `av1`; the quantiser (2–31) for `mjpeg` and `mpeg4`. |
| `-q, --quiet` | off | Print only output paths and errors. |
| `-v, --verbose` | off | Show full error details. |

## `chromatic` options

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | `exponential` | `exponential` (running averages) or `sequential` (the last three frames). |
| `--exp-a FLOAT` | `0.5` | `exponential`: fade rate of the medium-term average (green); higher fades slower. |
| `--exp-b FLOAT` | `0.8` | `exponential`: fade rate of the long-term average (red). |
| `--gain TEXT` | `4` | Amplification of the differences: one value, or three as `recent,medium,older`. |
| `--lum-weight FLOAT` | `0.7` | Brightness of the greyscale frame under the motion. |
| `--motion-only` | off | Motion on black (`--lum-weight 0`). |
| `--threshold FLOAT` | `0` | Subtracted from every channel to hide noise (0–255 scale). |
| `--tail-only` | off | Each colour shows only motion the more recent one doesn't. |
| `--frame-skip INT` | `0` | Use every (N+1)-th frame only: fewer output frames, larger motion between them. |

Plus the [input and output options](#input-and-output). The default output name is `<input>_chromatic.<ext>`.

## Examples

Basic usage with defaults (DIS, HSV visualisation):

```bash
motionflow recording.mp4
```

Chromatic motion, on black, from every 4th frame of a slow-motion video:

```bash
motionflow chromatic slowmo.mp4 --motion-only --frame-skip 3
```

Every video in a folder, results in another folder:

```bash
motionflow trials/ -o results/
```

TV-L1 method, heatmap drawn over the original at 70% opacity:

```bash
motionflow recording.mp4 --method tvl1 --viz heatmap --overlay 0.7
```

RAFT with GPU, every other frame, saved to a specific path:

```bash
motionflow recording.mp4 --method raft --device cuda --stride 2 -o flow.mp4
```

The same colour scale for several videos, so they can be compared:

```bash
motionflow trial1.mp4 --max-flow 5
motionflow trial2.mp4 --max-flow 5
```

Quiver arrows on a coarser grid:

```bash
motionflow recording.mp4 --viz quiver --quiver-step 32
```

## Drawing the flow

The flow is a speed and a direction for every pixel. The styles draw it as:

- **`hsv`**: a colour wheel. The hue shows the direction of motion and the brightness its speed, so black means still. The legend's wheel shows which colour is which direction.
- **`heatmap`**: speed only, from black (still) through purple and red to yellow (fast).
- **`quiver`**: arrows over the original frame, each showing the average motion in its grid cell.

Lucas-Kanade tracks individual points rather than every pixel, so its points are drawn as dots (`hsv`, `heatmap`) or arrows (`quiver`).

### Which speed is full strength

How speeds map to brightness decides what a viewer sees, so the default uses one scale for the whole video:

- **`auto`** (default): full strength is the 99th percentile of all motion in the video (`--percentile`). The same brightness means the same speed in every frame. Short videos are measured exactly; long ones are sampled at 48 points, which costs 48 flow estimates rather than a full extra pass.
- **`global`**: the same, measured on every frame. Exact, but it computes the flow twice.
- **`per-frame`**: each frame is scaled to its own largest motion. A still frame then shows its sensor noise at full brightness, so stillness looks like motion. This was the default in motionflow 0.1.
- **`--max-flow`**: a fixed speed in pixels per frame. Use it to compare videos on the same scale.

The scale never goes below 1 pixel per frame, so a still video stays dark. For smaller motions, set `--max-flow`. The legend shows the speed used, and `--min-flow` hides motion slower than a threshold.

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

The output has one frame per input frame, so the two line up: output frame *i* shows the motion from input frame *i − 1* to frame *i*. Frame 0 has no frame before it and shows no motion.
