"""Event-camera (DVS) simulation from ordinary video.

An event camera doesn't record frames. Each pixel independently reports an
*event* when its brightness changes by more than a contrast threshold: an ON
event when it gets brighter, an OFF event when it gets darker. This module
simulates that from video: every pixel keeps a reference log brightness and
emits one event per threshold crossed between frames.

By default the sensor is ideal (Lichtsteiner et al. 2008; as in ESIM and
v2e): identical, noiseless pixels. The realistic effects of v2e can be
added: threshold mismatch between pixels, limited photoreceptor bandwidth,
leak events, shot noise and a refractory period; and in-between frames
along optical flow, so fast motion fires along its path at the right times.

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
# Sensor model
# ---------------------------------------------------------------------------

Sensor = Literal["ideal", "clean", "noisy"]
Interpolate = Literal["none", "flow"]

# The non-ideal effects of each sensor preset; v2e's "clean" and "noisy".
SENSORS: dict[str, dict[str, float]] = {
    "ideal": {
        "threshold_sigma": 0.0,
        "cutoff_hz": 0.0,
        "leak_rate_hz": 0.0,
        "shot_noise_rate_hz": 0.0,
        "refractory": 0.0,
    },
    "clean": {
        "threshold_sigma": 0.02,
        "cutoff_hz": 0.0,
        "leak_rate_hz": 0.0,
        "shot_noise_rate_hz": 0.0,
        "refractory": 0.0,
    },
    "noisy": {
        "threshold_sigma": 0.05,
        "cutoff_hz": 30.0,
        "leak_rate_hz": 0.1,
        "shot_noise_rate_hz": 5.0,
        "refractory": 0.0,
    },
}

# v2e's constants: spread of leak rates between pixels (decades), jitter of
# the leak, and how much shot noise falls off with intensity.
_NOISE_RATE_DECADES = 0.1
_LEAK_JITTER = 0.1
_SHOT_NOISE_INTENSITY_FACTOR = 0.25


def lin_log(gray: np.ndarray, linear_below: float = 20.0) -> np.ndarray:
    """Photoreceptor response: linear below ``linear_below`` (0-255), log above.

    As in v2e: the slope of the linear part meets the log at the transition,
    so dark pixels respond in proportion to brightness instead of having their
    noise amplified by the log.
    """
    x = gray.astype(np.float32)
    slope = np.log(linear_below) / linear_below
    return np.where(x <= linear_below, x * slope, np.log(np.maximum(x, linear_below)))


class _PixelState:
    """Everything the simulated sensor remembers between steps."""

    def __init__(self, shape: tuple[int, int], camera: EventCamera, first: np.ndarray) -> None:
        rng = camera._rng
        self.photoreceptor = first.copy()  # low-pass filtered log brightness
        self.reference = first.copy()  # the level the last event left
        self.last_event = np.full(shape, -np.inf, np.float64)
        if camera.threshold_sigma > 0:
            noise_on = rng.normal(0.0, camera.threshold_sigma, shape)
            noise_off = rng.normal(0.0, camera.threshold_sigma, shape)
            self.on_threshold = np.maximum(camera.on_threshold + noise_on, 0.01).astype(np.float32)
            self.off_threshold = np.maximum(camera.off_threshold + noise_off, 0.01).astype(
                np.float32
            )
        else:
            self.on_threshold = np.full(shape, camera.on_threshold, np.float32)
            self.off_threshold = np.full(shape, camera.off_threshold, np.float32)
        # Leak and shot noise vary between pixels (log-normally).
        self.noise_rate = np.exp(np.log(10) * _NOISE_RATE_DECADES * rng.standard_normal(shape))


# ---------------------------------------------------------------------------
# In-between frames along optical flow
# ---------------------------------------------------------------------------


class _FlowInterpolator:
    """Interpolates frames along optical flow, in both directions.

    v2e uses a neural network (SuperSloMo) for this; motionflow uses its own
    optical flow. A frame at fraction ``a`` of the interval is the blend of
    the first frame moved ``a`` of the way forward and the second moved
    ``1 - a`` of the way back. Occlusions aren't modelled.
    """

    def __init__(self, method: str, max_steps: int, steps: int | None) -> None:
        from motionflow.methods import make_estimator

        self.estimator = make_estimator(method)
        self.max_steps = max_steps
        self.steps = steps

    def __call__(
        self, prev: np.ndarray, curr: np.ndarray, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> list[np.ndarray]:
        """Grey frames at fractions 1/K, 2/K, ..., 1 of the interval (the last is ``curr``)."""
        forward = self.estimator(prev, curr)
        steps = self.steps
        if steps is None:
            # About a pixel of motion per step.
            speed = float(np.percentile(np.hypot(forward[..., 0], forward[..., 1]), 99.9))
            steps = int(np.clip(np.ceil(speed), 1, self.max_steps))
        if steps == 1:
            return [curr_gray.astype(np.float32)]
        backward = self.estimator(curr, prev)
        h, w = prev_gray.shape
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        a_img, b_img = prev_gray.astype(np.float32), curr_gray.astype(np.float32)
        frames = []
        for k in range(1, steps):
            a = k / steps
            from_a = cv2.remap(a_img, xs - a * forward[..., 0], ys - a * forward[..., 1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
            from_b = cv2.remap(b_img, xs - (1 - a) * backward[..., 0], ys - (1 - a) * backward[..., 1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
            frames.append((1 - a) * from_a + a * from_b)
        frames.append(b_img)
        return frames


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------


class EventCamera(Transform):
    """Simulates an event camera (DVS) and draws its events.

    Every pixel keeps a reference brightness; when its (photoreceptor)
    brightness moves more than a contrast threshold from it, the pixel emits
    one event per threshold crossed (ON brighter, OFF darker) and moves its
    reference by as many thresholds. Brightness is linear below
    ``linear_below`` and logarithmic above, as in a real photoreceptor.

    ``sensor`` picks how realistic the sensor is, after v2e: ``"ideal"``
    (identical, noiseless pixels), ``"clean"`` (thresholds vary slightly
    between pixels) or ``"noisy"`` (also a 30 Hz bandwidth limit, leak events
    and shot noise). Any of the effects can also be set on its own.

    Args:
        threshold: Contrast threshold for ON and OFF events, as a change in
            log brightness (0.2 is about a 22% change).
        on_threshold: Threshold for ON events, overriding ``threshold``.
        off_threshold: Threshold for OFF events, overriding ``threshold``.
        sensor: ``"ideal"``, ``"clean"`` or ``"noisy"``; sets the defaults
            of the five effects below.
        threshold_sigma: Pixel-to-pixel spread of the thresholds (standard
            deviation, log units).
        cutoff_hz: Photoreceptor bandwidth at full brightness; darker pixels
            respond more slowly, smearing motion as real sensors do. 0 means
            unlimited.
        leak_rate_hz: Rate at which each reference drifts down by one
            threshold, producing occasional ON events in still scenes.
        shot_noise_rate_hz: Rate of random ON and OFF events per pixel.
        refractory: Seconds after an event during which a pixel can't fire.
        seed: Random seed for the threshold spread and noise.
        linear_below: Brightness (0-255) below which the response is linear.
        interpolate: ``"flow"`` makes in-between frames along optical flow,
            so fast motion fires along its whole path at the right times
            rather than only where it starts and ends. ``"none"`` uses the
            frames as they are.
        upsample: In-between steps per frame interval with
            ``interpolate="flow"``; ``None`` chooses per interval, about one
            pixel of motion per step, up to ``max_upsample``.
        max_upsample: Most in-between steps when ``upsample`` is ``None``.
        flow_method: Optical flow method for the interpolation.
        render: ``"polarity"``: ON events blue, OFF red, on black.
            ``"count"``: the number of events per pixel as a heatmap.
        window: Draw the events of the last ``window`` frames together.
        overlay: Draw the events over the greyscale original at this opacity.
        full_count: Events per pixel drawn at full strength.
        save_events: Also write the event stream to this file (``.h5``,
            ``.npz`` or ``.txt``; see :class:`EventWriter`).
        overwrite: Replace ``save_events`` if it exists.

    The first frame only sets the references, so it has no events. After
    processing, ``frame_counts`` holds the (ON, OFF) totals of each frame and
    ``total_events`` their sum.

    References: Lichtsteiner, Posch & Delbruck, "A 128x128 120 dB 15 µs
    latency asynchronous temporal contrast vision sensor", IEEE JSSC 2008.
    Rebecq, Gehrig & Scaramuzza, "ESIM: an open event camera simulator",
    CoRL 2018. Hu, Liu & Delbruck, "v2e: From video frames to realistic DVS
    events", CVPR Workshops 2021, whose sensor model this follows
    (github.com/SensorsINI/v2e, MIT licence).
    """

    name = "events"

    def __init__(
        self,
        threshold: float = 0.2,
        *,
        on_threshold: float | None = None,
        off_threshold: float | None = None,
        sensor: Sensor = "ideal",
        threshold_sigma: float | None = None,
        cutoff_hz: float | None = None,
        leak_rate_hz: float | None = None,
        shot_noise_rate_hz: float | None = None,
        refractory: float | None = None,
        seed: int = 0,
        linear_below: float = 20.0,
        interpolate: Interpolate = "none",
        upsample: int | None = None,
        max_upsample: int = 16,
        flow_method: str = "dis",
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
        if sensor not in SENSORS:
            raise ValueError(f"sensor must be one of {', '.join(SENSORS)}, got {sensor!r}")
        preset = SENSORS[sensor]
        given = {
            "threshold_sigma": threshold_sigma,
            "cutoff_hz": cutoff_hz,
            "leak_rate_hz": leak_rate_hz,
            "shot_noise_rate_hz": shot_noise_rate_hz,
            "refractory": refractory,
        }
        for name, value in given.items():
            value = preset[name] if value is None else value
            if value < 0:
                raise ValueError(f"{name} must be 0 or more, got {value}")
            setattr(self, name, float(value))
        self.sensor = sensor
        if linear_below <= 1:
            raise ValueError(f"linear_below must be more than 1, got {linear_below}")
        if interpolate not in ("none", "flow"):
            raise ValueError(f"interpolate must be 'none' or 'flow', got {interpolate!r}")
        if upsample is not None and upsample < 1:
            raise ValueError(f"upsample must be at least 1, got {upsample}")
        if max_upsample < 1:
            raise ValueError(f"max_upsample must be at least 1, got {max_upsample}")
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
        self.seed = seed
        self.linear_below = linear_below
        self.interpolate = interpolate
        self.upsample = upsample
        self.max_upsample = max_upsample
        self.flow_method = flow_method
        self.render = render
        self.window = window
        self.overlay = overlay
        self.full_count = full_count
        self.save_events = Path(save_events) if save_events is not None else None
        self.overwrite = overwrite
        self.total_events = 0
        self.frame_counts: list[tuple[int, int]] = []  # (ON, OFF) per frame
        self.steps_used: list[int] = []

    # -- the loop -------------------------------------------------------------

    def process(self, frames: Iterable[Frame]) -> Iterator[np.ndarray]:
        self._rng = np.random.default_rng(self.seed)
        interpolator = (
            _FlowInterpolator(self.flow_method, self.max_upsample, self.upsample)
            if self.interpolate == "flow"
            else None
        )
        writer: EventWriter | None = None
        state: _PixelState | None = None
        previous: Frame | None = None
        previous_gray: np.ndarray | None = None
        recent: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.window)
        self.total_events = 0
        self.frame_counts = []
        self.steps_used = []
        try:
            for frame in frames:
                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                on = np.zeros(gray.shape, np.int32)
                off = np.zeros(gray.shape, np.int32)
                if state is None:
                    state = _PixelState(gray.shape, self, lin_log(gray, self.linear_below))
                    if self.save_events is not None:
                        writer = EventWriter(
                            self.save_events, gray.shape[1], gray.shape[0], self.overwrite
                        )
                else:
                    assert previous is not None and previous_gray is not None
                    if interpolator is not None:
                        steps = interpolator(previous.image, frame.image, previous_gray, gray)
                    else:
                        steps = [gray.astype(np.float32)]
                    self.steps_used.append(len(steps))
                    dt = (frame.time - previous.time) / len(steps)
                    for k, step_gray in enumerate(steps):
                        t0 = previous.time + k * dt
                        events = self._step(state, step_gray, t0, t0 + dt)
                        on += events.on_count
                        off += events.off_count
                        if writer is not None:
                            writer.write(*events.stream)
                self.frame_counts.append((int(on.sum()), int(off.sum())))
                self.total_events += self.frame_counts[-1][0] + self.frame_counts[-1][1]
                recent.append((on, off))
                previous, previous_gray = frame, gray
                yield self._draw(recent, gray)
        except BaseException:
            if writer is not None:
                writer.discard()
            raise
        if writer is not None:
            writer.close()

    def _step(self, state: _PixelState, gray: np.ndarray, t0: float, t1: float) -> _StepEvents:
        """Advance the sensor from time t0 to t1, ending at brightness ``gray``."""
        dt = t1 - t0
        log = lin_log(gray, self.linear_below)
        intensity = (gray.astype(np.float32) + 20) / 275  # v2e's 0-1 scale for rates

        # Photoreceptor: first-order low-pass, faster when brighter.
        if self.cutoff_hz > 0 and dt > 0:
            tau = 1 / (2 * np.pi * self.cutoff_hz)
            eps = np.minimum(intensity * (dt / tau), 1.0)
            state.photoreceptor += eps * (log - state.photoreceptor)
        else:
            state.photoreceptor = log

        # Leak: the reference drifts down, so pixels slowly fire ON events.
        if self.leak_rate_hz > 0 and dt > 0:
            jitter = 1 - _LEAK_JITTER * self._rng.standard_normal(log.shape)
            rate = self.leak_rate_hz * state.noise_rate * jitter
            state.reference -= (dt * rate * state.on_threshold).astype(np.float32)

        diff = state.photoreceptor - state.reference
        on = np.floor(np.maximum(diff, 0) / state.on_threshold).astype(np.int32)
        off = np.floor(np.maximum(-diff, 0) / state.off_threshold).astype(np.int32)
        stream = _crossings(on, off, diff, state, t0, t1, self.refractory)
        on_allowed, off_allowed = stream.on_count, stream.off_count
        state.reference += on_allowed * state.on_threshold - off_allowed * state.off_threshold

        # Shot noise: random events, more where thresholds are low.
        if self.shot_noise_rate_hz > 0 and dt > 0:
            factor = (
                (self.shot_noise_rate_hz / 2)
                * dt
                * ((_SHOT_NOISE_INTENSITY_FACTOR - 1) * intensity + 1)
            )
            draw = self._rng.random(log.shape)
            shot_on = draw > 1 - factor * (self.on_threshold / state.on_threshold)
            shot_off = draw < factor * (self.off_threshold / state.off_threshold)
            # A noise event resets the pixel, as in v2e.
            state.reference[shot_on] = state.photoreceptor[shot_on]
            state.reference[shot_off] = state.photoreceptor[shot_off]
            state.last_event[shot_on | shot_off] = t1
            stream = stream.with_noise(shot_on, shot_off, t1)
        return stream

    # -- drawing ----------------------------------------------------------------

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


class _StepEvents:
    """The events of one step: counts per pixel, and the stream (t, x, y, p)."""

    def __init__(self, on_count, off_count, t, x, y, p) -> None:
        self.on_count, self.off_count = on_count, off_count
        self.stream = (t, x, y, p)

    def with_noise(self, shot_on: np.ndarray, shot_off: np.ndarray, t: float) -> _StepEvents:
        parts = [self.stream]
        for mask, polarity in ((shot_on, 1), (shot_off, 0)):
            y, x = np.nonzero(mask)
            parts.append((np.full(len(x), t), x, y, np.full(len(x), polarity, np.uint8)))
        t_, x_, y_, p_ = (np.concatenate([part[i] for part in parts]) for i in range(4))
        order = np.argsort(t_, kind="stable")
        return _StepEvents(
            self.on_count + shot_on, self.off_count + shot_off,
            t_[order], x_[order], y_[order], p_[order],
        )  # fmt: skip


def _crossings(
    on: np.ndarray,
    off: np.ndarray,
    diff: np.ndarray,
    state: _PixelState,
    t0: float,
    t1: float,
    refractory: float,
) -> _StepEvents:
    """The individual events of one step, with timestamps and the refractory period.

    Brightness is taken to change linearly over [t0, t1], so the k-th of n
    crossings in a pixel happens at t0 + (t1 - t0) * k * threshold / |diff|.
    After an event a pixel can't fire for ``refractory`` seconds; crossings it
    misses stay pending in its reference and fire later.
    """
    allowed = {1: np.zeros_like(on), 0: np.zeros_like(off)}
    ts, xs, ys, ps = [], [], [], []
    for counts, polarity, thresholds in (
        (on, 1, state.on_threshold),
        (off, 0, state.off_threshold),
    ):
        y, x = np.nonzero(counts)
        if len(x) == 0:
            continue
        n = counts[y, x].astype(np.int64)
        spacing = (t1 - t0) * thresholds[y, x] / np.maximum(np.abs(diff[y, x]), 1e-12)
        if refractory > 0:
            # First crossing after the pixel's refractory period ends, and every
            # m-th one after that.
            ready = state.last_event[y, x] + refractory
            first = np.maximum(1, np.ceil((ready - t0) / spacing - 1e-9)).astype(np.int64)
            every = np.maximum(1, np.ceil(refractory / spacing - 1e-9)).astype(np.int64)
            count = np.where(first > n, 0, 1 + (n - first) // every)
        else:
            first = np.ones_like(n)
            every = np.ones_like(n)
            count = n
        keep = count > 0
        y, x, first, every, count, spacing = (a[keep] for a in (y, x, first, every, count, spacing))
        if len(x) == 0:
            continue
        allowed[polarity][y, x] = count
        total = int(count.sum())
        j = np.arange(total) - np.repeat(np.cumsum(count) - count, count)  # 0..count-1
        k = np.repeat(first, count) + j * np.repeat(every, count)
        t = t0 + k * np.repeat(spacing, count)
        ts.append(t)
        xs.append(np.repeat(x, count))
        ys.append(np.repeat(y, count))
        ps.append(np.full(total, polarity, np.uint8))
        last = first + (count - 1) * every
        state.last_event[y, x] = np.maximum(state.last_event[y, x], t0 + last * spacing)
    if not ts:
        empty = np.zeros(0)
        return _StepEvents(allowed[1], allowed[0], empty, empty.astype(np.int64),
                           empty.astype(np.int64), empty.astype(np.uint8))  # fmt: skip
    t, x, y, p = (np.concatenate(a) for a in (ts, xs, ys, ps))
    order = np.argsort(t, kind="stable")
    return _StepEvents(allowed[1], allowed[0], t[order], x[order], y[order], p[order])


def events(
    input: Path | str,
    output: Path | str | None = None,
    *,
    threshold: float = 0.2,
    on_threshold: float | None = None,
    off_threshold: float | None = None,
    sensor: Sensor = "ideal",
    threshold_sigma: float | None = None,
    cutoff_hz: float | None = None,
    leak_rate_hz: float | None = None,
    shot_noise_rate_hz: float | None = None,
    refractory: float | None = None,
    seed: int = 0,
    interpolate: Interpolate = "none",
    upsample: int | None = None,
    max_upsample: int = 16,
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
    event stream is written too. ``result.info`` has the number of events
    and, with ``interpolate="flow"``, the average in-between steps used.

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
        sensor=sensor,
        threshold_sigma=threshold_sigma,
        cutoff_hz=cutoff_hz,
        leak_rate_hz=leak_rate_hz,
        shot_noise_rate_hz=shot_noise_rate_hz,
        refractory=refractory,
        seed=seed,
        interpolate=interpolate,
        upsample=upsample,
        max_upsample=max_upsample,
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
    steps = camera.steps_used
    result.info.update(
        {
            "events": camera.total_events,
            "events_file": camera.save_events,
            "sensor": sensor,
            "mean_steps": float(np.mean(steps)) if steps else 1.0,
        }
    )
    return result
