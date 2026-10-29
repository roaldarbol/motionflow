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

from motionflow import viz as draw
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


class FlowRenderer:
    """Turns a flow field into an image.

    Args:
        style: ``hsv`` (colour wheel: hue is direction, brightness speed),
            ``heatmap`` (speed only) or ``quiver`` (arrows over the frame).
        overlay: Draw the flow over the original frame at this opacity (0-1).
            ``None`` draws the flow alone (for quiver: arrows on the frame).
        max_flow: Speed in pixels per frame drawn at full strength. ``None``
            scales each frame to its own largest motion.
        min_flow: Motion below this many pixels per frame is drawn as none,
            hiding sensor noise.
        quiver_step: Grid spacing in pixels for quiver arrows.
        quiver_scale: Arrow length per pixel of motion. ``None`` draws
            ``max_flow`` as 90% of the grid spacing.
        legend: Draw a key (colour wheel, colour bar or reference arrow) in
            the bottom-right corner.
        sparse: Draw only the points where the flow is non-zero, for point
            trackers such as Lucas-Kanade. :class:`FlowTransform` sets this
            from its estimator.
    """

    def __init__(
        self,
        style: str | VizStyle = VizStyle.HSV,
        *,
        overlay: float | None = None,
        max_flow: float | None = None,
        min_flow: float = 0.0,
        quiver_step: int = 16,
        quiver_scale: float | None = None,
        legend: bool = True,
        sparse: bool = False,
    ) -> None:
        if overlay is not None and not 0.0 <= overlay <= 1.0:
            raise ValueError(f"overlay must be between 0 and 1, got {overlay}")
        if max_flow is not None and max_flow <= 0:
            raise ValueError(f"max_flow must be positive, got {max_flow}")
        if min_flow < 0:
            raise ValueError(f"min_flow must be 0 or more, got {min_flow}")
        if quiver_step < 1:
            raise ValueError(f"quiver_step must be at least 1, got {quiver_step}")
        if quiver_scale is not None and quiver_scale <= 0:
            raise ValueError(f"quiver_scale must be positive, got {quiver_scale}")
        self.style = VizStyle(style)
        self.overlay = overlay
        self.max_flow = max_flow
        self.min_flow = min_flow
        self.quiver_step = quiver_step
        self.quiver_scale = quiver_scale
        self.legend = legend
        self.sparse = sparse

    def __call__(self, flow: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Render ``flow`` (H, W, 2), with ``image`` (the frame it ends on) for overlays."""
        style = self.style
        scale = self.quiver_scale
        if style == VizStyle.QUIVER and scale is None:
            scale = draw.quiver_scale(self.quiver_step, self.max_flow, flow)
        if self.sparse:
            drawn = draw.draw_sparse(
                flow, style, image, self.max_flow, self.min_flow, self.quiver_step, scale
            )
        elif style == VizStyle.QUIVER:
            drawn = draw.flow_to_quiver(
                flow, image, self.quiver_step, scale, min_flow=self.min_flow
            )
        elif style == VizStyle.HEATMAP:
            drawn = draw.flow_to_heatmap(flow, self.max_flow, self.min_flow)
        else:
            drawn = draw.flow_to_hsv(flow, self.max_flow, self.min_flow)

        if self.overlay is not None:
            drawn = draw.overlay(image, drawn, self.overlay)
        if self.legend:
            draw.add_legend(drawn, style, self.max_flow, self.quiver_step, scale)
        return drawn


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
        estimator: FlowEstimator | str | Method = Method.DIS,
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
        if estimator.sparse:
            self.renderer.sparse = True
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
# Normalisation
# ---------------------------------------------------------------------------

# Below this, "auto" measures every frame; above, it samples.
SAMPLES = 48
# Never scale below this speed (pixels per frame), so that the sensor noise
# of a still video stays dark instead of being stretched to full brightness.
MIN_SCALE = 1.0

# Magnitude histogram: log-spaced bins from 0.001 to 10,000 px/frame, ~0.5%
# wide, so percentiles need no memory proportional to the video's length.
_BINS = np.concatenate([[0.0], np.geomspace(1e-3, 1e4, 3000)])


class _MagnitudeHistogram:
    def __init__(self, sparse: bool = False) -> None:
        self.counts = np.zeros(len(_BINS) - 1, dtype=np.int64)
        self.sparse = sparse

    def add(self, flow: np.ndarray) -> None:
        if self.sparse:
            # Only the tracked points: everywhere else is zero by construction.
            magnitude = draw.flow_magnitude(flow)
            magnitude = magnitude[magnitude > 0]
        else:
            magnitude = draw.flow_magnitude(flow[::2, ::2])  # every 4th pixel is plenty
        self.counts += np.histogram(np.minimum(magnitude, _BINS[-1]), _BINS)[0]

    def percentile(self, q: float) -> float:
        total = self.counts.sum()
        if total == 0:
            return 0.0
        i = int(np.searchsorted(np.cumsum(self.counts), q / 100 * total))
        return float(_BINS[min(i + 1, len(_BINS) - 1)])


def measure_motion(
    transform: FlowTransform,
    input: Path | str,
    io: IOOptions | None = None,
    *,
    percentile: float = 99.0,
    sample: bool = True,
    progress: bool = True,
) -> float:
    """The speed (pixels per frame) at a percentile of all motion in a video.

    Used to scale colours to the whole video. With ``sample`` (and a video
    longer than a few hundred frames) it measures :data:`SAMPLES` frame
    pairs spread evenly through the range; otherwise every frame.
    """
    from tqdm.auto import tqdm

    io = io or IOOptions()
    histogram = _MagnitudeHistogram(sparse=transform.estimator.sparse)
    with VideoReader(input, start=io.start, end=io.end, scale=io.scale) as reader:
        total, rate = reader.frame_count, reader.rate
        if sample and total is not None and total > 4 * SAMPLES:
            first = next(reader.frames()).index
            positions = [first + int((k + 0.5) * total / SAMPLES) for k in range(SAMPLES)]
        else:
            positions = None
            source = prefetch(reader.frames())
            try:
                flows = tqdm(
                    transform.flows(source),
                    total=total,
                    desc="Measuring motion",
                    unit="frame",
                    disable=not progress,
                )
                for _, flow in flows:
                    histogram.add(flow)
            finally:
                source.close()

    if positions is not None:
        gap = transform.stride
        for index in tqdm(positions, desc="Measuring motion", unit="sample", disable=not progress):
            pair = _read_pair(input, index, gap, rate, io.scale)
            if pair is not None:
                histogram.add(transform.estimator(*pair) / gap)
    return histogram.percentile(percentile)


def _read_pair(
    input: Path | str, index: int, gap: int, rate, scale: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Frames ``index`` and ``index + gap``, reached by seeking."""
    try:
        with VideoReader(
            input, start=float(index / rate), end=f"{index + gap + 1}f", scale=scale
        ) as r:
            images = [frame.image for frame in r.frames()]
    except OSError:
        return None
    return (images[0], images[gap]) if len(images) > gap else None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def flow(
    input: Path | str,
    output: Path | str | None = None,
    *,
    method: str | Method = Method.DIS,
    viz: str | VizStyle = VizStyle.HSV,
    overlay: float | None = None,
    normalize: str | NormalizeMode = NormalizeMode.AUTO,
    percentile: float = 99.0,
    max_flow: float | None = None,
    min_flow: float = 0.0,
    legend: bool = True,
    stride: int = 1,
    smooth: int = 0,
    quiver_step: int = 16,
    quiver_scale: float | None = None,
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
        viz: ``hsv`` (hue is direction, brightness speed), ``heatmap`` (speed
            only) or ``quiver`` (arrows over the frame).
        overlay: Draw the flow over the original frame at this opacity (0-1).
        normalize: How speeds map to colours; see
            :class:`~motionflow.types.NormalizeMode`. The default, ``auto``,
            uses one scale for the whole video so that brightness means the
            same speed in every frame.
        percentile: For ``auto`` and ``global``: the percentile of all
            motion drawn at full strength. Faster motion is clipped.
        max_flow: Fixed speed (pixels per frame) drawn at full strength.
            Overrides ``normalize``.
        min_flow: Motion below this many pixels per frame is drawn as none.
        legend: Draw a key in the bottom-right corner.
        stride: Estimate flow every N frames; see :class:`FlowTransform`.
        smooth: Centred averaging window in frames; ``0`` disables it.
        quiver_step: Grid spacing in pixels for quiver arrows.
        quiver_scale: Arrow length per pixel of motion; automatic if ``None``.
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
        progress: Show progress bars.
        on_progress: Called after each output frame as
            ``on_progress(frames_done, total)``; ``total`` is ``None`` when
            the input doesn't record its length.

    Returns:
        A :class:`~motionflow.pipeline.Result`. ``result.info["max_flow"]``
        is the speed drawn at full strength (``None`` for per-frame).
    """
    io = IOOptions(
        start=start, end=end, scale=scale, fps=fps, codec=codec, crf=crf, overwrite=overwrite
    )
    norm_mode = NormalizeMode(normalize)
    if not 0 < percentile <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    renderer = FlowRenderer(
        viz,
        overlay=overlay,
        max_flow=max_flow,
        min_flow=min_flow,
        quiver_step=quiver_step,
        quiver_scale=quiver_scale,
        legend=legend,
    )
    input = Path(input)
    output = default_output_path(input, "_flow") if output is None else Path(output)
    check_paths(input, output, overwrite)
    transform = FlowTransform(
        method, renderer, stride=stride, smooth=smooth, device=device, model_size=model_size
    )

    measured = max_flow is None and norm_mode != NormalizeMode.PER_FRAME
    if measured:
        speed = measure_motion(
            transform,
            input,
            io,
            percentile=percentile,
            sample=norm_mode == NormalizeMode.AUTO,
            progress=progress,
        )
        renderer.max_flow = max(speed, MIN_SCALE)

    return run(
        transform,
        input,
        output,
        io=io,
        progress=("Rendering" if measured else "Processing") if progress else False,
        on_progress=on_progress,
        info={
            "method": Method(method).value,
            "normalize": "fixed" if max_flow is not None else norm_mode.value,
            "max_flow": renderer.max_flow,
        },
    )
