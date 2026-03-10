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
| `-o, --output PATH` | `<input>_flow.<ext>` | Output file path. |
| `-m, --method METHOD` | `farneback` | Flow algorithm. See [Methods](methods.md). |
| `--viz STYLE` | `hsv` | Visualisation style. |
| `--blend FLOAT` | `0.0` | Opacity of original frame over flow (0 = pure flow, 1 = pure original). |
| `--stride INT` | `1` | Compute flow every N frames. |
| `--normalize MODE` | `per-frame` | Magnitude normalisation: `per-frame` or `global`. |
| `--max-flow FLOAT` | — | Manual magnitude cap. Overrides `--normalize`. |
| `--smooth INT` | `0` | Temporal smoothing window in frames. |
| `--codec TEXT` | `mp4v` | FourCC codec string for output. |
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
