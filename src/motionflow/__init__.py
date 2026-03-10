"""motionflow: CLI and Python API for optical flow video processing."""

from motionflow._version import __version__
from motionflow.api import process_video
from motionflow.types import Method, ModelSize, NormalizeMode, VizStyle
from motionflow.weights import cache_dir

__all__ = [
    "__version__",
    "process_video",
    "Method",
    "VizStyle",
    "NormalizeMode",
    "ModelSize",
    "cache_dir",
]
