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
# HSV colour-wheel flow (default method: DIS)
motionflow recording.mp4

# RAFT on GPU, heatmap style, every other frame
motionflow recording.mp4 --method raft --device cuda --viz heatmap --stride 2

# The flow over the original frame at 60% opacity, with temporal smoothing
motionflow recording.mp4 --overlay 0.6 --smooth 3

# Quiver arrow overlay
motionflow recording.mp4 --viz quiver --quiver-step 24

# One minute from 1:30, at half size
motionflow recording.mp4 --start 1:30 --end 2:30 --scale 0.5

# Every video in a folder; results in another folder
motionflow trials/ -o results/

# BehaveAI's chromatic motion: recent movement blue, older green and red
motionflow chromatic recording.mp4

# Which methods can run here
motionflow methods
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
from motionflow.methods import DISFlow
from motionflow.viz import flow_to_hsv

flow = DISFlow()(prev_frame, curr_frame)  # (H, W, 2)
vis  = flow_to_hsv(flow)                         # BGR image
```

## Methods

| Method | Type | GPU | Notes |
|--------|------|-----|-------|
| `dis` | Dense | No | Default; fast and accurate |
| `farneback` | Dense | No | The default before 0.2 |
| `tvl1` | Dense | No | Sharp edges, slow |
| `lucas-kanade` | Sparse | No | Tracks points; drawn as dots or arrows |
| `raft` | Dense | Yes | High quality; weights auto-downloaded |
| `sea-raft` | Dense | Yes | ECCV 2024; the most accurate |
| `neuflow` | Dense | Yes | The fastest deep method |

## Documentation

[roaldarbol.github.io/motionflow](https://roaldarbol.github.io/motionflow)

## License

MIT
