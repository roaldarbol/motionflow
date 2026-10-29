"""Optical flow as a transform: estimate flow between frames and render it.

Output frame *i* shows the motion from input frame *i - 1* to frame *i*, so
the output has as many frames as the input and lines up with it frame for
frame. Frame 0 has no previous frame and shows no motion.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from motionflow.methods import FlowEstimator, make_estimator
from motionflow.pipeline import (
    IOOptions,
    ProgressCallback,
    Result,
    Transform,
    check_paths,
    default_output_path,
    prefetch,
    run,
)
from motionflow.types import Method, ModelSize, NormalizeMode, VizStyle
from motionflow.video import Frame, TimePoint, VideoReader
from motionflow.viz import visualize_flow


class FlowRenderer:
    """Turns a flow field into an image.

    Args:
        style: Visualisation style; see :class:`~motionflow.types.VizStyle`.
        blend: Opacity of the original frame over the flow image, 0-1.
        max_flow: Flow magnitude (pixels per frame) shown at full brightness.
            ``None`` scales each frame to its own largest magnitude.
        quiver_step: Grid spacing in pixels for quiver arrows.
    """

    def __init__(
        self,
        style: str | VizStyle = VizStyle.HSV,
        blend: float = 0.0,
        max_flow: float | None = None,
        quiver_step: int = 16,
    ) -> None:
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"blend must be between 0 and 1, got {blend}")
        if max_flow is not None and max_flow <= 0:
            raise ValueError(f"max_flow must be positive, got {max_flow}")
        if quiver_step < 1:
            raise ValueError(f"quiver_step must be at least 1, got {quiver_step}")
        self.style = VizStyle(style)
        self.blend = blend
        self.max_flow = max_flow
        self.quiver_step = quiver_step

    def __call__(self, flow: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Render ``flow`` (H, W, 2), with ``image`` (the frame it ends on) for overlays."""
        return visualize_flow(
            flow,
            style=self.style,
            frame=image,
            blend=self.blend,
            max_flow=self.max_flow,
            quiver_step=self.quiver_step,
        )


class FlowTransform(Transform):
    """Optical flow between consecutive frames, rendered as images.

    Args:
        estimator: A :class:`~motionflow.methods.FlowEstimator`, or a method
            name (see :class:`~motionflow.types.Method`).
        renderer: How to draw the flow. Defaults to HSV, scaled per frame.
        stride: Estimate flow every ``stride`` frames, between frames
            ``stride`` apart, and show it until the next estimate. The flow is
            divided by ``stride`` so magnitudes stay in pixels per frame. The
            first ``stride`` frames show no motion.
        smooth: Average the flow over a window of this many frames, centred
            on each frame (an even window reaches one frame further back).
            ``0`` or ``1`` disables smoothing.
        device: Compute device for deep methods, when ``estimator`` is a name.
        model_size: Model size for deep methods, when ``estimator`` is a name.
    """

    name = "flow"

    def __init__(
        self,
        estimator: FlowEstimator | str | Method = Method.FARNEBACK,
        renderer: FlowRenderer | None = None,
        *,
        stride: int = 1,
        smooth: int = 0,
        device: str = "auto",
        model_size: str | ModelSize = ModelSize.LARGE,
    ) -> None:
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")
        if smooth < 0:
            raise ValueError(f"smooth must be 0 or more, got {smooth}")
        if not isinstance(estimator, FlowEstimator):
            estimator = make_estimator(estimator, device=device, model_size=model_size)
        self.estimator = estimator
        self.renderer = renderer if renderer is not None else FlowRenderer()
        self.stride = stride
        self.smooth = smooth

    def process(self, frames: Iterable[Frame]) -> Iterator[np.ndarray]:
        for frame, flow in self.flows(frames):
            yield self.renderer(flow, frame.image)

    def flows(self, frames: Iterable[Frame]) -> Iterator[tuple[Frame, np.ndarray]]:
        """Yield each frame with the flow into it, ``(H, W, 2)`` in pixels per frame."""
        raw = self._raw_flows(frames)
        if self.smooth <= 1:
            yield from raw
        else:
            yield from _centred_mean(raw, self.smooth)

    def _raw_flows(self, frames: Iterable[Frame]) -> Iterator[tuple[Frame, np.ndarray]]:
        anchor: np.ndarray | None = None  # image the next estimate starts from
        since = 0  # frames since the anchor
        held: np.ndarray | None = None
        for frame in frames:
            if anchor is None:
                anchor = frame.image
                held = np.zeros((*frame.image.shape[:2], 2), dtype=np.float32)
                yield frame, held
                continue
            since += 1
            if since == self.stride:
                flow = self.estimator(anchor, frame.image)
                held = flow / self.stride if self.stride > 1 else flow
                anchor, since = frame.image, 0
            assert held is not None
            yield frame, held


def _centred_mean(
    items: Iterator[tuple[Frame, np.ndarray]], window: int
) -> Iterator[tuple[Frame, np.ndarray]]:
    """Average flows over a centred window, shrinking it at the ends."""
    ahead = (window - 1) // 2  # frames needed after the centre
    behind = window - 1 - ahead
    buffer: deque[tuple[Frame, np.ndarray]] = deque()
    centre = 0  # index in buffer of the next frame to emit

    def emit() -> tuple[Frame, np.ndarray]:
        lo = max(0, centre - behind)
        flows = [flow for _, flow in list(buffer)[lo : centre + ahead + 1]]
        return buffer[centre][0], np.mean(flows, axis=0).astype(np.float32)

    for item in items:
        buffer.append(item)
        if len(buffer) - 1 - centre >= ahead:
            yield emit()
            centre += 1
            if centre > behind:
                buffer.popleft()
                centre -= 1
    while centre < len(buffer):
        yield emit()
        centre += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def flow(
    input: Path | str,
    output: Path | str | None = None,
    *,
    method: str | Method = Method.FARNEBACK,
    viz: str | VizStyle = VizStyle.HSV,
    blend: float = 0.0,
    stride: int = 1,
    normalize: str | NormalizeMode = NormalizeMode.PER_FRAME,
    max_flow: float | None = None,
    smooth: int = 0,
    quiver_step: int = 16,
    device: str = "auto",
    model_size: str | ModelSize = ModelSize.LARGE,
    start: str | float | TimePoint | None = None,
    end: str | float | TimePoint | None = None,
    scale: float = 1.0,
    fps: float | None = None,
    codec: str = "auto",
    crf: int | None = None,
    overwrite: bool = False,
    progress: bool = True,
    on_progress: ProgressCallback | None = None,
) -> Result:
    """Compute optical flow for a video and write a visualisation of it.

    The output has one frame per input frame: frame *i* shows the motion from
    input frame *i - 1* to frame *i*, and frame 0 shows none.

    Args:
        input: Path to the input video.
        output: Output path. Defaults to ``<input>_flow.<ext>`` next to the
            input (``.mp4`` for inputs in a container motionflow can't write).
        method: Optical flow algorithm; see :class:`~motionflow.types.Method`.
        viz: Visualisation style; see :class:`~motionflow.types.VizStyle`.
        blend: Opacity of the original frame over the flow image, 0-1.
        stride: Estimate flow every N frames; see :class:`FlowTransform`.
        normalize: ``"per-frame"`` scales each frame to its own largest
            motion; ``"global"`` to the largest in the whole video, which
            needs a first pass over it.
        max_flow: Fixed magnitude (pixels per frame) for full brightness.
            Overrides ``normalize``.
        smooth: Centred averaging window in frames; ``0`` disables it.
        quiver_step: Grid spacing in pixels for quiver arrows.
        device: Compute device for deep methods (``"auto"``, ``"cuda"``,
            ``"mps"``, ``"cpu"``).
        model_size: Model size for RAFT (``"small"`` or ``"large"``).
        start: Where to start: seconds (``12.5``), clock time (``"1:30"``)
            or a frame number (``"120f"``).
        end: Where to stop (exclusive), in the same formats as ``start``.
        scale: Resize factor applied before computing flow, e.g. ``0.5``.
        fps: Output frame rate. Defaults to the input's.
        codec: Output codec; see :class:`~motionflow.video.VideoWriter`.
        crf: Output quality (lower is better and bigger).
        overwrite: Replace ``output`` if it already exists.
        progress: Show a progress bar.
        on_progress: Called after each output frame as
            ``on_progress(frames_done, total)``; ``total`` is ``None`` when
            the input doesn't record its length.

    Returns:
        A :class:`~motionflow.pipeline.Result`; ``result.info["max_flow"]``
        is the magnitude used for full brightness (``None`` for per-frame).
    """
    io = IOOptions(
        start=start, end=end, scale=scale, fps=fps, codec=codec, crf=crf, overwrite=overwrite
    )
    norm_mode = NormalizeMode(normalize)
    renderer = FlowRenderer(viz, blend=blend, max_flow=max_flow, quiver_step=quiver_step)
    input = Path(input)
    output = default_output_path(input, "_flow") if output is None else Path(output)
    check_paths(input, output, overwrite)
    transform = FlowTransform(
        method, renderer, stride=stride, smooth=smooth, device=device, model_size=model_size
    )

    two_pass = norm_mode == NormalizeMode.GLOBAL and max_flow is None
    if two_pass:
        renderer.max_flow = _largest_flow(transform, input, io, progress)

    return run(
        transform,
        input,
        output,
        io=io,
        progress=("Pass 2/2" if two_pass else "Processing") if progress else False,
        on_progress=on_progress,
        info={"method": Method(method).value, "max_flow": renderer.max_flow},
    )


def _largest_flow(transform: FlowTransform, input: Path, io: IOOptions, progress: bool) -> float:
    """First pass for global normalisation: the largest flow magnitude."""
    from tqdm.auto import tqdm

    largest = 1.0  # floor, so a still video doesn't divide by zero
    with VideoReader(input, start=io.start, end=io.end, scale=io.scale) as reader:
        total = reader.frame_count
        source = prefetch(reader.frames())
        try:
            for _, flow in tqdm(
                transform.flows(source),
                total=total,
                desc="Pass 1/2",
                unit="frame",
                disable=not progress,
            ):
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                largest = max(largest, float(magnitude.max()))
        finally:
            source.close()
    return largest
