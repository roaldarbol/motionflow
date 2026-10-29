"""The processing pipeline: read a video, transform its frames, write the result.

Every method in motionflow is a :class:`Transform`: it takes a stream of
frames and yields images. :func:`run` does everything else the same way for
all of them: reading and writing video, the frame range and scaling, output
naming and safety, progress, and decoding in a background thread so that
decoding overlaps with computing.

A transform that turns each frame into one image independently of how far
it has got, apart from state it keeps itself, can subclass
:class:`FrameTransform` and implement ``__call__``. One that needs to look
ahead (a centred smoothing window, say) or works in chunks implements
:meth:`Transform.process` as a generator instead.
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from tqdm.auto import tqdm

from motionflow.video import (
    SUPPORTED_EXTENSIONS,
    Frame,
    TimePoint,
    VideoReader,
    VideoWriter,
    parse_time,
)

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class Transform(ABC):
    """A stream processor: video frames in, BGR images out.

    Transforms normally yield one image per input frame, so that output frame
    *i* belongs to input frame *i*; they may keep state across frames and
    look ahead.

    Attributes:
        name: Short name, used for the default output file name
            (``<input>_<name>.<ext>``).
    """

    name: str = "out"

    @abstractmethod
    def process(self, frames: Iterable[Frame]) -> Iterator[np.ndarray]:
        """Yield output images for ``frames``.

        Called once per video. Implementations should start from a clean
        state, so the same transform can process several videos in turn.
        """


class FrameTransform(Transform):
    """A transform that maps each frame to one image as it arrives.

    Subclasses implement :meth:`__call__`, and :meth:`reset` if they keep
    state between frames.
    """

    def reset(self) -> None:  # noqa: B027 - intentionally empty default
        """Forget all state from previous frames."""

    @abstractmethod
    def __call__(self, frame: Frame) -> np.ndarray:
        """Return the output image for ``frame``."""

    def process(self, frames: Iterable[Frame]) -> Iterator[np.ndarray]:
        self.reset()
        for frame in frames:
            yield self(frame)


# ---------------------------------------------------------------------------
# Options and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IOOptions:
    """Reading and writing options shared by every method.

    Attributes:
        start: Where to start: seconds (``12.5``), clock time (``"1:30"``)
            or a frame number (``"120f"``).
        end: Where to stop (exclusive), in the same formats as ``start``.
        scale: Resize factor applied while reading, e.g. ``0.5``.
        fps: Output frame rate. Defaults to the input's.
        codec: Output codec; see :class:`~motionflow.video.VideoWriter`.
        crf: Output quality (lower is better and bigger).
        overwrite: Replace the output if it already exists.
    """

    start: str | float | TimePoint | None = None
    end: str | float | TimePoint | None = None
    scale: float = 1.0
    fps: float | Fraction | None = None
    codec: str = "auto"
    crf: int | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        # Parse now, so a typo fails before any work is done.
        if self.start is not None:
            object.__setattr__(self, "start", parse_time(self.start))
        if self.end is not None:
            object.__setattr__(self, "end", parse_time(self.end))
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if self.fps is not None and self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.crf is not None and self.crf < 0:
            raise ValueError(f"crf must be 0 or more, got {self.crf}")


@dataclass(frozen=True)
class Result:
    """What :func:`run` produced.

    Attributes:
        output: Path of the written video.
        frames_read: Number of input frames processed.
        frames_written: Number of frames in the output.
        fps: Frame rate of the output.
        encoder: FFmpeg encoder used, e.g. ``"libx264"``.
        seconds: Wall-clock processing time.
        info: Method-specific details, e.g. the flow magnitude used for
            normalisation.
    """

    output: Path
    frames_read: int
    frames_written: int
    fps: Fraction
    encoder: str
    seconds: float
    info: dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[int, int | None], None]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def default_output_path(input: Path | str, suffix: str = "_flow") -> Path:
    """Default output path: ``<stem><suffix><ext>`` next to ``input``.

    Keeps the input's extension when motionflow can write that container,
    and uses ``.mp4`` otherwise (e.g. for ``.ts`` or ``.h264`` inputs).
    """
    input = Path(input)
    ext = input.suffix if input.suffix.lower() in SUPPORTED_EXTENSIONS else ".mp4"
    return input.with_name(f"{input.stem}{suffix}{ext}")


def check_paths(input: Path, output: Path, overwrite: bool) -> None:
    """Fail early on an output path that would destroy the input or a file."""
    if output.resolve() == input.resolve():
        raise ValueError(f"The output would overwrite the input: {input}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    if not input.is_file():
        raise OSError(f"Video not found: {input}")


def run(
    transform: Transform,
    input: Path | str,
    output: Path | str | None = None,
    *,
    io: IOOptions | None = None,
    progress: bool | str = True,
    on_progress: ProgressCallback | None = None,
    info: dict[str, Any] | None = None,
) -> Result:
    """Run ``transform`` over a video and write the images it yields.

    Args:
        transform: What to compute.
        input: Path to the input video.
        output: Output path. Defaults to ``<input>_<transform.name>.<ext>``
            next to the input.
        io: Reading and writing options; see :class:`IOOptions`.
        progress: Show a progress bar. A string is used as its label.
        on_progress: Called after each output frame as
            ``on_progress(frames_done, total)``; ``total`` is ``None`` when
            the input doesn't record its length.
        info: Extra details to include in the :class:`Result`.

    Returns:
        A :class:`Result` describing the output.

    Raises:
        FileExistsError: If the output exists and ``io.overwrite`` is false.
        ValueError: If the output is the input, or for invalid options.
        OSError: If the input can't be read or the output can't be written.
    """
    io = io or IOOptions()
    input = Path(input)
    output = default_output_path(input, f"_{transform.name}") if output is None else Path(output)
    check_paths(input, output, io.overwrite)

    started = time.perf_counter()
    read = 0
    with VideoReader(input, start=io.start, end=io.end, scale=io.scale) as reader:
        fps = Fraction(io.fps).limit_denominator(1001) if io.fps is not None else reader.rate
        total = reader.frame_count
        label = progress if isinstance(progress, str) else "Processing"

        def counted(frames: Iterator[Frame]) -> Iterator[Frame]:
            nonlocal read
            for frame in frames:
                read += 1
                yield frame

        source = prefetch(reader.frames())
        images = transform.process(counted(source))
        writer: VideoWriter | None = None
        try:
            with tqdm(total=total, desc=label, unit="frame", disable=not progress) as bar:
                for image in images:
                    if writer is None:
                        # Opened on the first image: a transform may change the size.
                        height, width = image.shape[:2]
                        writer = VideoWriter(
                            output, fps, width, height, io.codec, crf=io.crf, overwrite=io.overwrite
                        ).__enter__()
                    writer.write(image)
                    bar.update(1)
                    if on_progress is not None:
                        on_progress(writer.frames_written, total)
        except BaseException as exc:
            if writer is not None:
                writer.__exit__(type(exc), exc, exc.__traceback__)
            raise
        finally:
            # Stop the transform, then the background reader, before the
            # reader's container is closed: it's in use on that thread.
            close = getattr(images, "close", None)
            if close is not None:
                close()
            source.close()

        if writer is None:
            raise ValueError(f"{transform.name} produced no frames for {input.name}.")
        writer.__exit__(None, None, None)

    return Result(
        output=output,
        frames_read=read,
        frames_written=writer.frames_written,
        fps=fps,
        encoder=writer.encoder,
        seconds=time.perf_counter() - started,
        info=dict(info or {}),
    )


# ---------------------------------------------------------------------------
# Background reading
# ---------------------------------------------------------------------------

T = TypeVar("T")
_END = object()


class _Failed:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def prefetch(items: Iterable[T], size: int = 8) -> Iterator[T]:
    """Iterate ``items`` in a background thread, ``size`` items ahead.

    Lets decoding (which releases the GIL inside FFmpeg) overlap with
    processing. Exceptions in the background thread are raised here, and
    closing the returned generator stops the thread.
    """
    buffer: queue.Queue = queue.Queue(maxsize=size)
    stop = threading.Event()

    def put(item: object) -> bool:
        while not stop.is_set():
            try:
                buffer.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def work() -> None:
        try:
            for item in items:
                if not put(item):
                    return
            put(_END)
        except BaseException as exc:
            put(_Failed(exc))

    thread = threading.Thread(target=work, name="motionflow-reader", daemon=True)
    thread.start()
    try:
        while True:
            item = buffer.get()
            if item is _END:
                return
            if isinstance(item, _Failed):
                raise item.exc
            yield item
    finally:
        stop.set()
        thread.join(timeout=5)
