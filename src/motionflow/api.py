from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from motionflow.methods import make_estimator
from motionflow.types import Method, ModelSize, NormalizeMode, VizStyle
from motionflow.video import VideoReader, VideoWriter
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
    fps: float | None = None,
    device: str = "auto",
    model_size: str | ModelSize = ModelSize.LARGE,
    quiver_step: int = 16,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Process a video and write an optical flow visualisation.

    This is the primary entry point for the Python API. All CLI options map
    directly to arguments here.

    Args:
        input: Path to the input video.
        output: Output path. Defaults to ``<input_stem>_flow.<ext>``.
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
        codec: FourCC codec string for the output file.
        fps: Output frames per second. Preserves the source FPS if ``None``.
        device: Compute device for deep methods (``"auto"``, ``"cuda"``,
            ``"mps"``, ``"cpu"``).
        model_size: Model size for RAFT/SEA-RAFT (``"small"`` or ``"large"``).
        quiver_step: Grid spacing in pixels for quiver arrows.
        on_progress: Optional callback ``(current_frame, total_frames)`` called
            after each output frame is written.

    Returns:
        Path to the written output video.
    """
    input = Path(input)
    norm_mode = NormalizeMode(normalize)
    viz_style = VizStyle(viz)

    if output is None:
        output = input.with_name(f"{input.stem}_flow{input.suffix}")
    output = Path(output)

    estimator = make_estimator(method, device=device, model_size=model_size)

    # First pass: determine global max flow if needed.
    global_max: float | None = max_flow
    if norm_mode == NormalizeMode.GLOBAL and max_flow is None:
        global_max = _compute_global_max_flow(input, estimator, stride)

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
        codec=codec,
        fps=fps,
        quiver_step=quiver_step,
        on_progress=on_progress,
    )

    return output


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_global_max_flow(
    input: Path,
    estimator,
    stride: int,
) -> float:
    """First pass: return the max flow magnitude across all sampled frames."""
    global_max = 1.0  # floor to avoid division by zero on static videos

    with VideoReader(input) as reader:
        frames = iter(reader)
        try:
            prev = next(frames)
        except StopIteration:
            return global_max

        for i, curr in enumerate(
            tqdm(frames, desc="Pass 1/2 (global normalisation)", unit="frame")
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
    codec: str,
    fps: float | None,
    quiver_step: int,
    on_progress: Callable[[int, int], None] | None,
) -> None:
    needs_orig = viz_style in {VizStyle.QUIVER, VizStyle.BLEND} or blend > 0

    smooth_buf: deque[np.ndarray] = deque(maxlen=max(smooth, 1))
    last_flow: np.ndarray | None = None
    frame_idx = 0

    with VideoReader(input) as reader:
        out_fps = fps if fps is not None else reader.fps
        total = reader.frame_count
        desc = "Pass 2/2" if max_flow is not None else "Processing"

        with VideoWriter(output, out_fps, reader.width, reader.height, codec) as writer:
            frames = iter(reader)
            try:
                prev = next(frames)
            except StopIteration:
                return

            with tqdm(total=total - 1, desc=desc, unit="frame") as pbar:
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
                            on_progress(frame_idx + 1, total - 1)

                    frame_idx += 1
                    pbar.update(1)
