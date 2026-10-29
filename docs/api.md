# Python API

`motionflow.flow` takes the same options as the command line, as keyword arguments, so motionflow works the same in scripts, Jupyter notebooks and larger pipelines.

## Basic usage

```python
import motionflow

result = motionflow.flow("recording.mp4")
print(result.output)          # recording_flow.mp4
print(result.frames_written)  # same as the number of input frames
```

`flow` returns a [`Result`](reference.md#motionflow.pipeline.Result) with the output path, frame counts, frame rate, encoder and processing time. `result.info["max_flow"]` is the flow magnitude shown at full brightness (`None` when each frame is scaled on its own).

## Frames line up with the input

The output has one frame per input frame. Output frame *i* shows the motion from input frame *i − 1* to frame *i*, and frame 0, which has no frame before it, shows no motion. So output frame *i* can be matched to anything else recorded at input frame *i*, such as pose tracking or neural recordings.

## In a Jupyter notebook

```python
from IPython.display import Video
import motionflow

result = motionflow.flow(
    "recording.mp4",
    method="raft",
    start="1:30",
    end="2:00",
    scale=0.5,
)

Video(str(result.output))
```

The progress bar adapts to notebooks. Pass `progress=False` to turn it off.

## Progress tracking

The `on_progress` callback receives the number of frames written and the total, which is `None` when the video doesn't record its length (raw `.h264`, for example):

```python
def on_progress(done: int, total: int | None) -> None:
    if total:
        print(f"{done / total:.0%}", end="\r")

motionflow.flow("recording.mp4", progress=False, on_progress=on_progress)
```

## Building your own pipeline

### Transforms and `run`

Every method is a *transform*: it takes a stream of frames and yields images. [`run`](reference.md#motionflow.pipeline.run) does the rest the same way for every transform: reading and writing, the frame range and scaling, output naming, never overwriting the input, and leaving nothing behind if something fails.

`flow` is a thin wrapper around a `FlowTransform`:

```python
from motionflow import FlowRenderer, FlowTransform, IOOptions, run

transform = FlowTransform(
    "farneback",
    FlowRenderer("heatmap", max_flow=10),  # 10 px/frame at full brightness
    smooth=5,
)
result = run(transform, "recording.mp4", io=IOOptions(scale=0.5))
```

To compute something of your own, subclass `FrameTransform` and implement `__call__`. It gets a [`Frame`](reference.md#motionflow.video.Frame) (with `index`, `time` and a BGR `image`) and returns an image:

```python
import cv2
from motionflow import FrameTransform, run

class FrameDifference(FrameTransform):
    name = "diff"  # output: recording_diff.mp4

    def reset(self):
        self.previous = None

    def __call__(self, frame):
        diff = frame.image if self.previous is None else cv2.absdiff(frame.image, self.previous)
        self.previous = frame.image
        return diff

run(FrameDifference(), "recording.mp4")
```

A transform that needs to look ahead, or works on chunks of frames, can subclass `Transform` and implement `process(frames)` as a generator instead.

### The flow fields themselves

`FlowTransform.flows` yields each frame with the flow into it, as an `(H, W, 2)` array of `(dx, dy)` in pixels per frame, without drawing anything:

```python
import numpy as np
from motionflow import FlowTransform, VideoReader

transform = FlowTransform("farneback")
with VideoReader("recording.mp4") as reader:
    for frame, flow in transform.flows(reader.frames()):
        speed = np.linalg.norm(flow, axis=2).mean()
        print(frame.index, frame.time, speed)
```

### Estimators directly

The flow estimators work on any two BGR frames:

```python
import cv2
from motionflow.methods import FarnebackFlow
from motionflow.viz import flow_to_hsv

estimator = FarnebackFlow()

prev = cv2.imread("frame_001.png")
curr = cv2.imread("frame_002.png")

flow = estimator(prev, curr)       # numpy array (H, W, 2)
vis  = flow_to_hsv(flow)           # BGR image (H, W, 3)

cv2.imwrite("flow.png", vis)
```

### Reading and writing video

`VideoReader` and `VideoWriter` handle the video side on their own. Frames are BGR `uint8` arrays of shape `(H, W, 3)`, as in OpenCV:

```python
from motionflow import VideoReader, VideoWriter

with VideoReader("recording.mp4", start="1:30", end="2:30", scale=0.5) as reader:
    with VideoWriter("copy.mp4", reader.rate, reader.width, reader.height) as writer:
        for frame in reader.frames():          # frame.index, frame.time, frame.image
            writer.write(frame.image)
```

`reader.frame_count` estimates how many frames the reader will yield, or is `None` when the video doesn't record its length.

## Migrating from `process_video`

`process_video` from motionflow 0.1 still works but is deprecated. It returns the output path, where `flow` returns a `Result`:

```python
out = process_video("recording.mp4", method="raft")         # before
out = motionflow.flow("recording.mp4", method="raft").output  # now
```

Its output now has as many frames as the input (it used to have one fewer), and `stride` and `smooth` behave as described in the [CLI reference](cli.md#flow).

## API reference

See the full [API Reference](reference.md) for all parameters and return types.
