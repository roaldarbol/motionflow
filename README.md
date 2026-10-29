# motionflow

CLI and Python API for optical flow video processing.

```bash
motionflow input.mp4
```

## Install

```bash
pip install motionflow            # classical methods
pip install "motionflow[torch]"   # + deep methods (RAFT)
```

A conda-forge package is planned. See the [installation guide](https://roaldarbol.github.io/motionflow/installation/) for PyTorch builds and GPU support.

## Usage

### CLI

```bash
# HSV colour-wheel flow (default: Farneback)
motionflow recording.mp4

# RAFT on GPU, heatmap style, every other frame
motionflow recording.mp4 --method raft --device cuda --viz heatmap --stride 2

# Show the original frame at 40% opacity over the flow, with temporal smoothing
motionflow recording.mp4 --blend 0.4 --smooth 3

# Quiver arrow overlay
motionflow recording.mp4 --viz quiver --quiver-step 24
```

### Python API

```python
from motionflow import process_video

out = process_video(
    "recording.mp4",
    method="raft",
    viz="hsv",
    stride=2,
    device="auto",
)
```

### Using estimators directly

```python
import cv2
from motionflow.methods import FarnebackFlow
from motionflow.viz import flow_to_hsv

flow = FarnebackFlow()(prev_frame, curr_frame)  # (H, W, 2)
vis  = flow_to_hsv(flow)                         # BGR image
```

## Methods

| Method | Type | GPU | Notes |
|--------|------|-----|-------|
| `farneback` | Dense | No | Default, fast |
| `tvl1` | Dense | No | Sharp edges, slow |
| `lucas-kanade` | Sparse | No | Experimental: output is currently near-empty ([#15](https://github.com/roaldarbol/motionflow/issues/15)) |
| `raft` | Dense | Yes | High quality; weights auto-downloaded |
| `sea-raft` | Dense | Yes | ECCV 2024; not working in this release ([#16](https://github.com/roaldarbol/motionflow/issues/16)) |
| `neuflow` | Dense | Yes | Edge-optimised; not working in this release ([#16](https://github.com/roaldarbol/motionflow/issues/16)) |

## Documentation

[roaldarbol.github.io/motionflow](https://roaldarbol.github.io/motionflow)

## License

MIT
