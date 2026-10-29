"""Chromatic motion: recent movement in blue, older in green and red.

The motion encoding of BehaveAI (Troscianko et al., *PLOS Biology* 2026),
written for motionflow from the method's description. Each frame is
compared with three references of increasing age; the differences become
the blue (recent), green (medium-term) and red (older) channels, drawn over
a dimmed greyscale copy of the frame. A moving animal leaves a trail that
fades from blue through green to red, and still parts of the scene stay
grey.

Unlike optical flow, this needs no motion estimation: it's frame
differencing, so it's fast and makes no assumptions about texture, but it
shows *where* things changed rather than *which way* they moved.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from motionflow.pipeline import (
    IOOptions,
    ProgressCallback,
    Result,
    Transform,
    check_paths,
    default_output_path,
    run,
)
from motionflow.video import Frame, TimePoint

Strategy = Literal["exponential", "sequential"]


class ChromaticMotion(Transform):
    """Chromatic motion encoding (BehaveAI).

    Args:
        strategy: How the three references are kept.
            ``"exponential"``: the previous frame, and two running averages
            of past frames that fade at rates ``exp_a`` and ``exp_b``, so
            green and red show motion over longer, smoothly weighted spans.
            ``"sequential"``: the previous three frames.
        exp_a: Fade rate of the medium-term average (0-1; higher is slower).
        exp_b: Fade rate of the long-term average (0-1; higher is slower).
        lum_weight: Weight of the greyscale frame under the motion (0 = motion
            on black).
        gain: Amplification of the differences: one value for all channels,
            or three for (recent/blue, medium/green, older/red).
        threshold: Subtracted from every channel, hiding small differences
            such as sensor noise (0-255 scale).
        tail_only: Show in each channel only the motion that isn't in the
            more recent one, so the colours separate into distinct bands.
        frame_skip: Use every ``frame_skip + 1``-th frame only, so the output
            has fewer frames (and plays faster at the same frame rate). In
            slow-motion or high-frame-rate video this makes motion between
            the frames used larger.

    Reference: Troscianko, O'Shea-Wheller, Galloway & Gaston, "BehaveAI
    enables rapid detection and classification of objects and behavior from
    motion", PLOS Biology 24(2): e3003632 (2026).
    https://doi.org/10.1371/journal.pbio.3003632
    """

    name = "chromatic"

    def __init__(
        self,
        strategy: Strategy = "exponential",
        *,
        exp_a: float = 0.5,
        exp_b: float = 0.8,
        lum_weight: float = 0.7,
        gain: float | tuple[float, float, float] = 4.0,
        threshold: float = 0.0,
        tail_only: bool = False,
        frame_skip: int = 0,
    ) -> None:
        if strategy not in ("exponential", "sequential"):
            raise ValueError(f"strategy must be 'exponential' or 'sequential', got {strategy!r}")
        for name, value in (("exp_a", exp_a), ("exp_b", exp_b)):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")
        if lum_weight < 0:
            raise ValueError(f"lum_weight must be 0 or more, got {lum_weight}")
        gains = (gain, gain, gain) if isinstance(gain, (int, float)) else tuple(gain)
        if len(gains) != 3 or any(g < 0 for g in gains):
            raise ValueError(f"gain must be one value or three, all 0 or more; got {gain!r}")
        if threshold < 0:
            raise ValueError(f"threshold must be 0 or more, got {threshold}")
        if frame_skip < 0:
            raise ValueError(f"frame_skip must be 0 or more, got {frame_skip}")
        self.strategy = strategy
        self.exp_a = exp_a
        self.exp_b = exp_b
        self.lum_weight = lum_weight
        self.gains = gains
        self.threshold = threshold
        self.tail_only = tail_only
        self.frame_skip = frame_skip

    def process(self, frames: Iterable[Frame]) -> Iterator[np.ndarray]:
        refs: list[np.ndarray] | None = None
        step = self.frame_skip + 1
        for n, frame in enumerate(frames):
            if n % step:
                continue
            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if refs is None:
                # Nothing to compare the first frame with: it shows no motion.
                refs = [gray, gray.copy(), gray.copy()]
            diffs = [cv2.absdiff(gray, ref) for ref in refs]
            refs = self._update(refs, gray)
            yield self._compose(gray, diffs)

    def _update(self, refs: list[np.ndarray], gray: np.ndarray) -> list[np.ndarray]:
        if self.strategy == "sequential":
            return [gray, refs[0], refs[1]]
        # Running averages: ref <- fade * ref + (1 - fade) * gray
        cv2.accumulateWeighted(gray, refs[1], 1.0 - self.exp_a)
        cv2.accumulateWeighted(gray, refs[2], 1.0 - self.exp_b)
        return [gray, refs[1], refs[2]]

    def _compose(self, gray: np.ndarray, diffs: list[np.ndarray]) -> np.ndarray:
        recent, medium, older = diffs
        if self.tail_only:
            # Each channel keeps only what the more recent one doesn't show.
            recent, medium, older = (
                np.maximum(recent - medium, 0),
                np.maximum(medium - recent, 0),
                np.maximum(older - medium, 0),
            )
        base = gray * self.lum_weight - self.threshold
        channels = [  # BGR: blue = recent, green = medium, red = older
            base + gain * diff
            for gain, diff in zip(self.gains, (recent, medium, older), strict=True)
        ]
        image = cv2.merge(channels)
        return np.clip(image, 0, 255).astype(np.uint8)


def chromatic(
    input: Path | str,
    output: Path | str | None = None,
    *,
    strategy: Strategy = "exponential",
    exp_a: float = 0.5,
    exp_b: float = 0.8,
    lum_weight: float = 0.7,
    gain: float | tuple[float, float, float] = 4.0,
    threshold: float = 0.0,
    tail_only: bool = False,
    motion_only: bool = False,
    frame_skip: int = 0,
    start: str | float | TimePoint | None = None,
    end: str | float | TimePoint | None = None,
    scale: float = 1.0,
    fps: float | None = None,
    codec: str = "auto",
    crf: int | None = None,
    overwrite: bool = False,
    progress: bool | str = True,
    on_progress: ProgressCallback | None = None,
) -> Result:
    """Write a chromatic motion video: recent movement blue, older green and red.

    The BehaveAI motion encoding; see :class:`ChromaticMotion` for what the
    options do. ``motion_only`` draws the motion on black (``lum_weight=0``).
    Without ``frame_skip`` the output has one frame per input frame; frame 0
    shows no motion.

    Returns:
        A :class:`~motionflow.pipeline.Result`.
    """
    io = IOOptions(
        start=start, end=end, scale=scale, fps=fps, codec=codec, crf=crf, overwrite=overwrite
    )
    transform = ChromaticMotion(
        strategy,
        exp_a=exp_a,
        exp_b=exp_b,
        lum_weight=0.0 if motion_only else lum_weight,
        gain=gain,
        threshold=threshold,
        tail_only=tail_only,
        frame_skip=frame_skip,
    )
    input = Path(input)
    output = default_output_path(input, "_chromatic") if output is None else Path(output)
    check_paths(input, output, overwrite)
    return run(
        transform,
        input,
        output,
        io=io,
        progress=progress,
        on_progress=on_progress,
        info={"strategy": strategy, "frame_skip": frame_skip},
    )
