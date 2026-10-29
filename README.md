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

# The flow over the original frame at 60% opacity, with temporal smoothing
motionflow recording.mp4 --overlay 0.6 --smooth 3

# Quiver arrow overlay
motionflow recording.mp4 --viz quiver --quiver-step 24

# One minute from 1:30, at half size
motionflow recording.mp4 --start 1:30 --end 2:30 --scale 0.5
```

### Python API

```python
import motionflow

result = motionflow.flow("recording.mp4", method="raft", start="1:30", end="2:30")
print(result.output)  # recording_flow.mp4, one frame per input frame
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
| `lucas-kanade` | Sparse | No | Tracks points; drawn as dots or arrows |
| `raft` | Dense | Yes | High quality; weights auto-downloaded |
| `sea-raft` | Dense | Yes | ECCV 2024; not working in this release ([#16](https://github.com/roaldarbol/motionflow/issues/16)) |
| `neuflow` | Dense | Yes | Edge-optimised; not working in this release ([#16](https://github.com/roaldarbol/motionflow/issues/16)) |

## Documentation

[roaldarbol.github.io/motionflow](https://roaldarbol.github.io/motionflow)

## License

MIT
