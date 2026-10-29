"""motionflow: motion visualisation for video, from the command line or Python."""

from motionflow._version import __version__
from motionflow.api import process_video
from motionflow.chromatic import ChromaticMotion, chromatic
from motionflow.events import EventCamera, events
from motionflow.flow import FlowRenderer, FlowTransform, flow
from motionflow.methods import FlowEstimator, make_estimator
from motionflow.pipeline import (
    FrameTransform,
    IOOptions,
    Result,
    Transform,
    default_output_path,
    run,
)
from motionflow.types import Method, ModelSize, NormalizeMode, VizStyle
from motionflow.video import Frame, VideoReader, VideoWriter, parse_time
from motionflow.weights import cache_dir

__all__ = [
    "__version__",
    # Optical flow
    "flow",
    "FlowTransform",
    "FlowRenderer",
    "FlowEstimator",
    "make_estimator",
    "Method",
    "ModelSize",
    "NormalizeMode",
    "VizStyle",
    # Chromatic motion
    "chromatic",
    "ChromaticMotion",
    # Event camera
    "events",
    "EventCamera",
    # Pipeline
    "run",
    "Transform",
    "FrameTransform",
    "IOOptions",
    "Result",
    "default_output_path",
    # Video
    "VideoReader",
    "VideoWriter",
    "Frame",
    "parse_time",
    # Other
    "cache_dir",
    "process_video",
]
