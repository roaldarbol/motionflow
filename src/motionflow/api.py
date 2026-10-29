from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from motionflow.methods import make_estimator
from motionflow.types import Method, ModelSize, NormalizeMode, VizStyle
from motionflow.video import SUPPORTED_EXTENSIONS, TimePoint, VideoReader, VideoWriter
from motionflow.viz import visualize_flow


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
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Process a video and write an optical flow visualisation.

    This is the primary entry point for the Python API. All CLI options map
    directly to arguments here.

    Args:
        input: Path to the input video.
        output: Output path. Defaults to ``<input_stem>_flow.<ext>`` next to
            the input; inputs in a container motionflow can't write (e.g.
            ``.ts``) get ``.mp4``.
        method: Optical flow algorithm. See :class:`~motionflow.types.Method`.
        viz: Visualisation style. See :class:`~motionflow.types.VizStyle`.
        blend: Opacity of original frame composited over flow visualisation.
            ``0.0`` is pure flow, ``1.0`` is pure original. Only applies when
            ``viz`` is not ``"quiver"`` or ``"blend"``.
        stride: Compute flow every N frames. Frames in between repeat the last
            computed flow.
        normalize: Flow magnitude normalisation strategy. ``"global"`` requires
            a full first pass through the video.
        max_flow: Manual magnitude cap. Overrides ``normalize``.
        smooth: Temporal smoothing window in frames. ``0`` disables smoothing.
        codec: Output codec. ``"auto"`` picks one for the output's extension
            (H.264 for ``.mp4``/``.mov``/``.mkv``, VP9 for ``.webm``, MJPEG for
            ``.avi``). See :class:`~motionflow.video.VideoWriter` for the
            others.
        crf: Output quality for H.264, HEVC, VP9 and AV1 (lower is better and
            bigger). ``None`` uses a near-lossless default.
        fps: Output frames per second. Preserves the source FPS if ``None``.
        start: Where to start: seconds (``12.5``), clock time (``"1:30"``) or
            a frame number (``"120f"``). See :func:`~motionflow.video.parse_time`.
        end: Where to stop (exclusive), in the same formats as ``start``.
        scale: Resize factor applied before computing flow, e.g. ``0.5``.
        overwrite: Replace ``output`` if it already exists. Otherwise an
            existing output raises :class:`FileExistsError`.
        device: Compute device for deep methods (``"auto"``, ``"cuda"``,
            ``"mps"``, ``"cpu"``).
        model_size: Model size for RAFT/SEA-RAFT (``"small"`` or ``"large"``).
        quiver_step: Grid spacing in pixels for quiver arrows.
        on_progress: Optional callback ``(current_frame, total_frames)`` called
            after each output frame is written. ``total_frames`` is ``None``
            when the input doesn't record its length.

    Returns:
        Path to the written output video.

    Raises:
        FileExistsError: If ``output`` exists and ``overwrite`` is false.
        ValueError: For invalid arguments, including an output path that is
            the input itself.
        OSError: If the input can't be read or the output can't be written.
    """
    input = Path(input)
    norm_mode = NormalizeMode(normalize)
    viz_style = VizStyle(viz)
    _check_arguments(
        stride=stride,
        smooth=smooth,
        blend=blend,
        fps=fps,
        max_flow=max_flow,
        quiver_step=quiver_step,
        scale=scale,
    )

    output = default_output_path(input) if output is None else Path(output)
    if output.resolve() == input.resolve():
        raise ValueError(f"The output would overwrite the input: {input}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    if not input.is_file():
        raise OSError(f"Video not found: {input}")

    estimator = make_estimator(method, device=device, model_size=model_size)
    read = {"start": start, "end": end, "scale": scale}

    # First pass: determine global max flow if needed.
    global_max: float | None = max_flow
    if norm_mode == NormalizeMode.GLOBAL and max_flow is None:
        global_max = _compute_global_max_flow(input, estimator, stride, read)

    # Second (or only) pass: compute flow and write visualisation.
    _render_video(
        input=input,
        output=output,
        estimator=estimator,
        viz_style=viz_style,
        blend=blend,
        stride=stride,
        max_flow=global_max,
        smooth=smooth,
        writer_options={"codec": codec, "crf": crf, "overwrite": overwrite},
        read=read,
        fps=fps,
        quiver_step=quiver_step,
        on_progress=on_progress,
    )

    return output


def default_output_path(input: Path | str, suffix: str = "_flow") -> Path:
    """Default output path: ``<stem><suffix><ext>`` next to ``input``.

    Keeps the input's extension when motionflow can write that container,
    and uses ``.mp4`` otherwise (e.g. for ``.ts`` or ``.h264`` inputs).
    """
    input = Path(input)
    ext = input.suffix if input.suffix.lower() in SUPPORTED_EXTENSIONS else ".mp4"
    return input.with_name(f"{input.stem}{suffix}{ext}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_arguments(
    *,
    stride: int,
    smooth: int,
    blend: float,
    fps: float | None,
    max_flow: float | None,
    quiver_step: int,
    scale: float,
) -> None:
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    if smooth < 0:
        raise ValueError(f"smooth must be 0 or more, got {smooth}")
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be between 0 and 1, got {blend}")
    if fps is not None and fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if max_flow is not None and max_flow <= 0:
        raise ValueError(f"max_flow must be positive, got {max_flow}")
    if quiver_step < 1:
        raise ValueError(f"quiver_step must be at least 1, got {quiver_step}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")


def _compute_global_max_flow(
    input: Path,
    estimator,
    stride: int,
    read: dict,
) -> float:
    """First pass: return the max flow magnitude across all sampled frames."""
    global_max = 1.0  # floor to avoid division by zero on static videos

    with VideoReader(input, **read) as reader:
        total = reader.frame_count
        frames = iter(reader)
        try:
            prev = next(frames)
        except StopIteration:
            return global_max

        for i, curr in enumerate(
            tqdm(
                frames,
                total=None if total is None else total - 1,
                desc="Pass 1/2 (global normalisation)",
                unit="frame",
            )
        ):
            if i % stride != 0:
                continue
            flow = estimator(prev, curr)
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            global_max = max(global_max, float(magnitude.max()))
            prev = curr

    return global_max


def _render_video(
    input: Path,
    output: Path,
    estimator,
    viz_style: VizStyle,
    blend: float,
    stride: int,
    max_flow: float | None,
    smooth: int,
    writer_options: dict,
    read: dict,
    fps: float | None,
    quiver_step: int,
    on_progress: Callable[[int, int | None], None] | None,
) -> None:
    needs_orig = viz_style in {VizStyle.QUIVER, VizStyle.BLEND} or blend > 0

    smooth_buf: deque[np.ndarray] = deque(maxlen=max(smooth, 1))
    last_flow: np.ndarray | None = None
    frame_idx = 0

    with VideoReader(input, **read) as reader:
        out_fps = fps if fps is not None else reader.rate
        total = reader.frame_count
        out_total = None if total is None else max(total - 1, 0)
        desc = "Pass 2/2" if max_flow is not None else "Processing"

        with VideoWriter(output, out_fps, reader.width, reader.height, **writer_options) as writer:
            frames = iter(reader)
            prev = next(frames)  # the reader guarantees at least one frame

            with tqdm(total=out_total, desc=desc, unit="frame") as pbar:
                for curr in frames:
                    if frame_idx % stride == 0:
                        raw_flow = estimator(prev, curr)
                        smooth_buf.append(raw_flow)
                        last_flow = np.mean(smooth_buf, axis=0)
                        prev = curr

                    if last_flow is not None:
                        orig = curr if needs_orig else None
                        vis = visualize_flow(
                            last_flow,
                            style=viz_style,
                            frame=orig,
                            blend=blend,
                            max_flow=max_flow,
                            quiver_step=quiver_step,
                        )
                        writer.write(vis)

                        if on_progress is not None:
                            on_progress(frame_idx + 1, out_total)

                    frame_idx += 1
                    pbar.update(1)

            if writer.frames_written == 0:
                raise ValueError(
                    f"{input.name} has only one frame in the selected range; "
                    "optical flow needs at least two."
                )
