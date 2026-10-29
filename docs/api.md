# Python API

All CLI options are available as keyword arguments to `process_video`. This makes it straightforward to use motionflow in scripts, Jupyter notebooks, or larger pipelines.

## Basic usage

```python
from motionflow import process_video

out = process_video("recording.mp4")
print(out)  # recording_flow.mp4
```

## In a Jupyter notebook

```python
from pathlib import Path
from IPython.display import Video
from motionflow import process_video

out = process_video(
    "recording.mp4",
    method="raft",
    viz="hsv",
    stride=2,
    device="auto",
)

Video(str(out))
```

## Progress tracking

The `on_progress` callback receives the current and total frame count, which you can use to drive a progress bar:

```python
from tqdm.notebook import tqdm as tqdm_nb
from motionflow import process_video

total_frames = None
pbar = None

def on_progress(current: int, total: int) -> None:
    global total_frames, pbar
    if pbar is None:
        pbar = tqdm_nb(total=total, unit="frame")
        total_frames = total
    pbar.update(1)
    if current >= total:
        pbar.close()

process_video("recording.mp4", on_progress=on_progress)
```

## Using estimators directly

The flow estimators are also importable for custom pipelines:

```python
import cv2
from motionflow.methods import FarnebackFlow, RAFTFlow
from motionflow.viz import flow_to_hsv

estimator = FarnebackFlow()

prev = cv2.imread("frame_001.png")
curr = cv2.imread("frame_002.png")

flow = estimator(prev, curr)       # numpy array (H, W, 2)
vis  = flow_to_hsv(flow)           # BGR image (H, W, 3)

cv2.imwrite("flow.png", vis)
```

## API reference

See the full [API Reference](reference.md) for all parameters and return types.
