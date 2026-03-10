# motionflow

CLI and Python API for optical flow video processing.

```bash
motionflow input.mp4
```

## Install

```bash
pip install motionflow          # classical methods only
pip install motionflow[torch]   # + RAFT and deep methods
```

```bash
conda install -c conda-forge motionflow    # includes PyTorch
pixi global install motionflow             # auto-selects GPU/CPU torch
```

## Usage

### CLI

```bash
# HSV colour-wheel flow (default: Farneback)
motionflow recording.mp4

# RAFT on GPU, heatmap style, every other frame
motionflow recording.mp4 --method raft --device cuda --viz heatmap --stride 2

# Blend flow 40% over original, with temporal smoothing
motionflow recording.mp4 --viz blend --blend 0.4 --smooth 3

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
| `tvl1` | Dense | No | Sharp edges, slow; requires opencv-contrib |
| `lucas-kanade` | Sparse | No | Good with `--viz quiver` |
| `raft` | Dense | Yes | High quality; weights auto-downloaded |
| `sea-raft` | Dense | Yes | ECCV 2024; coming soon |
| `neuflow` | Dense | Yes | Edge-optimised; coming soon |

## Documentation

[roaldarbol.github.io/motionflow](https://roaldarbol.github.io/motionflow)

## License

MIT
