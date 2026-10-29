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

The `on_progress` callback receives the current and total frame count, which you can use to drive a progress bar. The total is `None` when the video doesn't record its length (raw `.h264`, for example):

```python
from tqdm.notebook import tqdm as tqdm_nb
from motionflow import process_video

total_frames = None
pbar = None

def on_progress(current: int, total: int | None) -> None:
    global total_frames, pbar
    if pbar is None:
        pbar = tqdm_nb(total=total, unit="frame")
        total_frames = total
    pbar.update(1)
    if total is not None and current >= total:
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

## Reading and writing video yourself

`VideoReader` and `VideoWriter` handle the video side, so a custom pipeline only has to deal with arrays. Frames are BGR `uint8` arrays of shape `(H, W, 3)`, as in OpenCV:

```python
from motionflow.methods import FarnebackFlow
from motionflow.video import VideoReader, VideoWriter
from motionflow.viz import flow_to_hsv

estimator = FarnebackFlow()

with VideoReader("recording.mp4", start="1:30", end="2:30", scale=0.5) as reader:
    with VideoWriter("flow.mp4", reader.rate, reader.width, reader.height) as writer:
        prev = None
        for frame in reader.frames():          # frame.index, frame.time, frame.image
            if prev is not None:
                writer.write(flow_to_hsv(estimator(prev, frame.image)))
            prev = frame.image
```

`reader.frame_count` estimates how many frames the reader will yield, or is `None` when the video doesn't record its length.

## API reference

See the full [API Reference](reference.md) for all parameters and return types.
