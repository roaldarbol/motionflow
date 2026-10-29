"""The 0.1 entry point, kept for compatibility. Use :func:`motionflow.flow`."""

from __future__ import annotations

import warnings
from pathlib import Path

from motionflow.flow import flow
from motionflow.pipeline import ProgressCallback, default_output_path
from motionflow.types import Method, ModelSize, NormalizeMode, VizStyle
from motionflow.video import TimePoint

__all__ = ["default_output_path", "process_video"]


def process_video(
    input: Path | str,
    output: Path | str | None = None,
    method: str | Method = Method.FARNEBACK,
    viz: str | VizStyle = VizStyle.HSV,
    blend: float = 0.0,
    stride: int = 1,
    normalize: str | NormalizeMode = NormalizeMode.PER_FRAME,
    max_flow: float | None = None,
    smooth: int = 0,
    codec: str = "auto",
    crf: int | None = None,
    fps: float | None = None,
    start: str | float | TimePoint | None = None,
    end: str | float | TimePoint | None = None,
    scale: float = 1.0,
    overwrite: bool = False,
    device: str = "auto",
    model_size: str | ModelSize = ModelSize.LARGE,
    quiver_step: int = 16,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Compute optical flow for a video and write a visualisation of it.

    .. deprecated:: 0.2
        Use :func:`motionflow.flow`, which takes the same arguments (as
        keywords) and returns a :class:`~motionflow.pipeline.Result` with
        more detail than the output path.

    Returns:
        Path to the written output video.
    """
    warnings.warn(
        "process_video() is deprecated; use motionflow.flow(), which returns a Result "
        "(the output path is result.output).",
        DeprecationWarning,
        stacklevel=2,
    )
    return flow(
        input,
        output,
        method=method,
        viz=viz,
        blend=blend,
        stride=stride,
        normalize=normalize,
        max_flow=max_flow,
        smooth=smooth,
        codec=codec,
        crf=crf,
        fps=fps,
        start=start,
        end=end,
        scale=scale,
        overwrite=overwrite,
        device=device,
        model_size=model_size,
        quiver_step=quiver_step,
        on_progress=on_progress,
    ).output
