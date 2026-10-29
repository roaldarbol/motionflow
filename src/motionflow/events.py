"""Event-camera (DVS) simulation from ordinary video.

An event camera doesn't record frames. Each pixel independently reports an
*event* when its brightness changes by more than a contrast threshold: an ON
event when it gets brighter, an OFF event when it gets darker. This module
simulates that from video: every pixel keeps a reference log brightness and
emits one event per threshold crossed between frames.

This is the core model (Lichtsteiner et al. 2008; as in ESIM and v2e). It
assumes an ideal sensor: no noise, no threshold mismatch between pixels, no
bandwidth limit or refractory period. Events between two frames get
timestamps spread linearly across the interval.

The output is a video of the events, and optionally the event stream itself
as ``(t, x, y, p)`` records.
"""

from __future__ import annotations

import os
from collections import deque
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

Render = Literal["polarity", "count"]

EVENT_FORMATS = (".h5", ".hdf5", ".npz", ".txt")

# BGR colours for the polarity rendering.
ON_COLOUR = np.array([255, 160, 40], np.float32)  # blue: brighter
OFF_COLOUR = np.array([40, 60, 255], np.float32)  # red: darker


# ---------------------------------------------------------------------------
# Event stream files
# ---------------------------------------------------------------------------


class EventWriter:
    """Writes events to ``.h5``, ``.npz`` or ``.txt``, as they're produced.

    Like :class:`~motionflow.video.VideoWriter`, it writes to a hidden
    temporary file that becomes ``path`` only when :meth:`close` is called;
    :meth:`discard` removes it.

    Formats, each with ``t`` (microseconds; seconds in ``.txt``), ``x``,
    ``y`` and ``p`` (1 = ON, brighter; 0 = OFF, darker):

    - ``.h5`` / ``.hdf5``: datasets ``events/t``, ``events/x``, ``events/y``,
      ``events/p``, with ``width`` and ``height`` attributes. Needs h5py
      (``pip install "motionflow[events]"``).
    - ``.npz``: arrays ``t``, ``x``, ``y``, ``p``, ``width``, ``height``. Held
      in memory until the end, so best for short clips.
    - ``.txt``: one event per line, ``t x y p``, with ``t`` in seconds: the
      text format of the event-camera datasets of Mueggler et al. (2017).
    """

    def __init__(self, path: Path | str, width: int, height: int, overwrite: bool = False) -> None:
        self.path = Path(path)
        self.format = self.path.suffix.lower()
        if self.format not in EVENT_FORMATS:
            raise ValueError(
                f"Can't write events to {self.path.name!r}: use one of {', '.join(EVENT_FORMATS)}."
            )
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {self.path}")
        if self.format in (".h5", ".hdf5"):
            try:
                import h5py  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    'Writing events to HDF5 needs h5py: pip install "motionflow[events]", '
                    "or use .npz or .txt."
                ) from None
        self.width, self.height = width, height
        self.count = 0
        self._tmp = self.path.with_name(f".{self.path.stem}.part{self.path.suffix}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._chunks: list[tuple[np.ndarray, ...]] = []
        self._file = None
        if self.format in (".h5", ".hdf5"):
            import h5py

            self._file = h5py.File(self._tmp, "w")
            group = self._file.create_group("events")
            for name, dtype in (("t", "i8"), ("x", "u2"), ("y", "u2"), ("p", "u1")):
                group.create_dataset(name, (0,), dtype=dtype, maxshape=(None,), chunks=(1 << 16,))
            self._file.attrs["width"] = width
            self._file.attrs["height"] = height
        elif self.format == ".txt":
            self._file = open(self._tmp, "w")  # noqa: SIM115 - closed in close()/discard()

    def write(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> None:
        """Append events: ``t`` in seconds, ``x``/``y`` pixel coordinates, ``p`` 1 or 0."""
        n = len(t)
        if n == 0:
            return
        t_us = np.round(t * 1e6).astype(np.int64)
        if self.format in (".h5", ".hdf5"):
            group = self._file["events"]
            for name, values in (("t", t_us), ("x", x), ("y", y), ("p", p)):
                dataset = group[name]
                dataset.resize((self.count + n,))
                dataset[self.count :] = values
        elif self.format == ".txt":
            lines = np.column_stack([t, x, y, p])
            np.savetxt(self._file, lines, fmt=["%.6f", "%d", "%d", "%d"])
        else:
            self._chunks.append(
                (t_us, x.astype(np.uint16), y.astype(np.uint16), p.astype(np.uint8))
            )
        self.count += n

    def close(self) -> None:
        """Finish the file and move it into place."""
        if self.format == ".npz":
            parts = list(zip(*self._chunks, strict=True)) if self._chunks else [[]] * 4
            dtypes = (np.int64, np.uint16, np.uint16, np.uint8)
            t, x, y, p = (
                np.concatenate(c) if len(c) else np.zeros(0, d)
                for c, d in zip(parts, dtypes, strict=True)
            )
            with open(self._tmp, "wb") as f:
                np.savez(f, t=t, x=x, y=y, p=p, width=self.width, height=self.height)
        elif self._file is not None:
            self._file.close()
            self._file = None
        os.replace(self._tmp, self.path)

    def discard(self) -> None:
        """Abandon the file: nothing is left behind."""
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
        self._tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------


class EventCamera(Transform):
    """Simulates an event camera (DVS) and draws its events.

    Args:
        threshold: Contrast threshold for ON and OFF events, as a change in
            log brightness (0.2 is about a 22% change). Lower gives more
            events, including from noise.
        on_threshold: Threshold for ON events, overriding ``threshold``.
        off_threshold: Threshold for OFF events, overriding ``threshold``.
        log_offset: Added to brightness (0-255) before taking the log, so
            that dark pixels don't fire on noise.
        render: ``"polarity"``: ON events blue, OFF red, on black.
            ``"count"``: the number of events per pixel as a heatmap.
        window: Draw the events of the last ``window`` frames together.
            Event cameras are often shown this way, as frames of a few
            milliseconds; at high frame rates a window of several frames
            makes motion easier to see.
        overlay: Draw the events over the greyscale original at this
            opacity, instead of on black.
        full_count: For ``render="count"``: the event count drawn at full
            strength.
        save_events: Also write the event stream to this file (``.h5``,
            ``.npz`` or ``.txt``; see :class:`EventWriter`).
        overwrite: Replace ``save_events`` if it exists.

    The first frame only sets the reference, so it has no events.

    References: Lichtsteiner, Posch & Delbruck, "A 128x128 120 dB 15 µs
    latency asynchronous temporal contrast vision sensor", IEEE JSSC 2008.
    Rebecq, Gehrig & Scaramuzza, "ESIM: an open event camera simulator",
    CoRL 2018. Hu, Liu & Delbruck, "v2e: From video frames to realistic DVS
    events", CVPR Workshops 2021.
    """

    name = "events"

    def __init__(
        self,
        threshold: float = 0.2,
        *,
        on_threshold: float | None = None,
        off_threshold: float | None = None,
        log_offset: float = 10.0,
        render: Render = "polarity",
        window: int = 1,
        overlay: float | None = None,
        full_count: int = 4,
        save_events: Path | str | None = None,
        overwrite: bool = False,
    ) -> None:
        self.on_threshold = threshold if on_threshold is None else on_threshold
        self.off_threshold = threshold if off_threshold is None else off_threshold
        for name, value in (
            ("on_threshold", self.on_threshold),
            ("off_threshold", self.off_threshold),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if log_offset <= 0:
            raise ValueError(f"log_offset must be positive, got {log_offset}")
        if render not in ("polarity", "count"):
            raise ValueError(f"render must be 'polarity' or 'count', got {render!r}")
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        if overlay is not None and not 0.0 <= overlay <= 1.0:
            raise ValueError(f"overlay must be between 0 and 1, got {overlay}")
        if full_count < 1:
            raise ValueError(f"full_count must be at least 1, got {full_count}")
        if save_events is not None and Path(save_events).suffix.lower() not in EVENT_FORMATS:
            raise ValueError(f"save_events must end in one of {', '.join(EVENT_FORMATS)}")
        self.log_offset = log_offset
        self.render = render
        self.window = window
        self.overlay = overlay
        self.full_count = full_count
        self.save_events = Path(save_events) if save_events is not None else None
        self.overwrite = overwrite
        self.total_events = 0

    def process(self, frames: Iterable[Frame]) -> Iterator[np.ndarray]:
        writer: EventWriter | None = None
        reference: np.ndarray | None = None
        previous_time = 0.0
        recent: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.window)
        self.total_events = 0
        try:
            for frame in frames:
                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                log = np.log(gray.astype(np.float32) + self.log_offset)
                if reference is None:
                    reference = log
                    previous_time = frame.time
                    if self.save_events is not None:
                        h, w = gray.shape
                        writer = EventWriter(self.save_events, w, h, self.overwrite)
                    on = off = np.zeros(gray.shape, np.int32)
                else:
                    on, off, reference = self._fire(log, reference)
                    if writer is not None:
                        writer.write(
                            *self._events(on, off, log, reference, previous_time, frame.time)
                        )
                    previous_time = frame.time
                self.total_events += int(on.sum() + off.sum())
                recent.append((on, off))
                yield self._draw(recent, gray)
        except BaseException:
            if writer is not None:
                writer.discard()
            raise
        if writer is not None:
            writer.close()

    def _fire(
        self, log: np.ndarray, reference: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Events per pixel since the last frame, and the updated reference."""
        delta = log - reference
        on = np.floor(np.maximum(delta, 0) / self.on_threshold).astype(np.int32)
        off = np.floor(np.maximum(-delta, 0) / self.off_threshold).astype(np.int32)
        reference = reference + on * self.on_threshold - off * self.off_threshold
        return on, off, reference

    def _events(
        self,
        on: np.ndarray,
        off: np.ndarray,
        log: np.ndarray,
        reference: np.ndarray,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The individual events, timestamps spread linearly over [t0, t1]."""
        ts, xs, ys, ps = [], [], [], []
        for counts, polarity, step in ((on, 1, self.on_threshold), (off, 0, self.off_threshold)):
            y, x = np.nonzero(counts)
            if len(x) == 0:
                continue
            n = counts[y, x]
            # Where the threshold was crossed, as a fraction of the change.
            change = np.abs(log[y, x] - reference[y, x]) + n * step
            # Crossing number 1..n for each pixel's events, without a Python loop.
            total = int(n.sum())
            k = np.arange(total) - np.repeat(np.cumsum(n) - n, n) + 1
            fraction = k * step / np.repeat(change, n)
            ts.append(t0 + (t1 - t0) * fraction)
            xs.append(np.repeat(x, n))
            ys.append(np.repeat(y, n))
            ps.append(np.full(total, polarity, np.uint8))
        if not ts:
            empty = np.zeros(0)
            return empty, empty, empty, empty
        t, x, y, p = (np.concatenate(a) for a in (ts, xs, ys, ps))
        order = np.argsort(t, kind="stable")
        return t[order], x[order], y[order], p[order]

    def _draw(self, recent: deque[tuple[np.ndarray, np.ndarray]], gray: np.ndarray) -> np.ndarray:
        on = np.sum([o for o, _ in recent], axis=0) if len(recent) > 1 else recent[0][0]
        off = np.sum([f for _, f in recent], axis=0) if len(recent) > 1 else recent[0][1]
        if self.render == "count":
            strength = np.clip((on + off) * (255.0 / self.full_count), 0, 255).astype(np.uint8)
            image = cv2.applyColorMap(strength, cv2.COLORMAP_INFERNO)
            image[(on + off) == 0] = 0
        else:
            net = (on - off).astype(np.float32)
            strength = np.clip(np.abs(net) / self.full_count, 0, 1)[..., None]
            colour = np.where(net[..., None] > 0, ON_COLOUR, OFF_COLOUR)
            image = (strength * colour).astype(np.uint8)
        if self.overlay is not None:
            base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            fired = ((on + off) > 0)[..., None]
            blended = cv2.addWeighted(image, self.overlay, base, 1 - self.overlay, 0)
            image = np.where(fired, blended, base)
        return image


def events(
    input: Path | str,
    output: Path | str | None = None,
    *,
    threshold: float = 0.2,
    on_threshold: float | None = None,
    off_threshold: float | None = None,
    log_offset: float = 10.0,
    render: Render = "polarity",
    window: int = 1,
    overlay: float | None = None,
    full_count: int = 4,
    save_events: Path | str | None = None,
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
    """Simulate an event camera on a video: write a video of its events.

    See :class:`EventCamera` for the options. With ``save_events``, the
    event stream is written too; ``result.info["events"]`` is the number of
    events.

    Returns:
        A :class:`~motionflow.pipeline.Result`.
    """
    io = IOOptions(
        start=start, end=end, scale=scale, fps=fps, codec=codec, crf=crf, overwrite=overwrite
    )
    camera = EventCamera(
        threshold,
        on_threshold=on_threshold,
        off_threshold=off_threshold,
        log_offset=log_offset,
        render=render,
        window=window,
        overlay=overlay,
        full_count=full_count,
        save_events=save_events,
        overwrite=overwrite,
    )
    input = Path(input)
    output = default_output_path(input, "_events") if output is None else Path(output)
    check_paths(input, output, overwrite)
    if camera.save_events is not None:
        if camera.save_events.resolve() == input.resolve():
            raise ValueError(f"The events file would overwrite the input: {input}")
        if camera.save_events.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {camera.save_events}")
    result = run(camera, input, output, io=io, progress=progress, on_progress=on_progress)
    result.info.update({"events": camera.total_events, "events_file": camera.save_events})
    return result
