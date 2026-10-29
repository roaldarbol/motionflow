"""Video reading and writing, built on PyAV (FFmpeg).

`VideoReader` decodes frames in order, without seeking back, so it also reads
streams that can't seek (MPEG-TS, raw H.264). It applies rotation metadata
and can resize and restrict to a time range while decoding.

`VideoWriter` picks an encoder from the file extension (H.264 for ``.mp4``,
``.mov`` and ``.mkv``; VP9 for ``.webm``; MJPEG for ``.avi``), writes to a
hidden temporary file, and only moves it into place once every frame has
been written, so a failed or interrupted run never leaves a file that looks
complete.
"""

from __future__ import annotations

import contextlib
import os
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Literal

import av
import numpy as np

# ---------------------------------------------------------------------------
# Time points (start/end of a range)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimePoint:
    """A position in a video: seconds, or a frame number.

    Create one with :func:`parse_time`.
    """

    value: float
    unit: Literal["s", "frame"]

    def __str__(self) -> str:
        return f"frame {int(self.value)}" if self.unit == "frame" else f"{self.value:g} s"


_HMS = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d*)?)$")


def parse_time(value: str | float | TimePoint) -> TimePoint:
    """Parse a time or frame position.

    Accepts seconds (``12.5``, ``"12.5"``, ``"12.5s"``), clock time
    (``"1:30"``, ``"01:02:03.5"``) or a frame number (``"120f"``).

    Raises:
        ValueError: If the value can't be parsed or is negative.
    """
    if isinstance(value, TimePoint):
        tp = value
    elif isinstance(value, (int, float)):
        tp = TimePoint(float(value), "s")
    else:
        text = value.strip().lower()
        if m := _HMS.match(text):
            hours, minutes, seconds = m.groups()
            tp = TimePoint(int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds), "s")
        elif text.endswith("f") and text[:-1].isdigit():
            tp = TimePoint(float(int(text[:-1])), "frame")
        else:
            try:
                tp = TimePoint(float(text.removesuffix("s")), "s")
            except ValueError:
                raise ValueError(
                    f"Can't read {value!r} as a time. Use seconds (12.5), "
                    "clock time (1:30 or 01:02:03.5) or a frame number (120f)."
                ) from None
    if tp.value < 0:
        raise ValueError(f"Time can't be negative: {value!r}")
    return tp


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """A decoded video frame.

    Attributes:
        index: Frame number in the source video, counting from 0. After
            seeking to a start time it is derived from the frame's timestamp.
        time: Presentation time in seconds from the start of the video.
        image: BGR image, ``uint8`` array of shape ``(H, W, 3)``.
    """

    index: int
    time: float
    image: np.ndarray


# Starting later than this (in seconds) seeks instead of decoding from the
# beginning. Seeking lands on a keyframe before the target, so the frames in
# between are still decoded and dropped.
_SEEK_THRESHOLD_S = 5.0


class VideoReader:
    """Streaming video reader. Use as a context manager.

    Iterating yields BGR frames as ``uint8`` arrays of shape ``(H, W, 3)``;
    :meth:`frames` yields :class:`Frame` objects with timestamps as well.

    Args:
        path: Path to the video file.
        start: First position to read (inclusive), as accepted by
            :func:`parse_time`. Reads from the beginning if ``None``.
        end: Position to stop at (exclusive). Reads to the end if ``None``.
        scale: Resize factor applied while decoding, e.g. ``0.5``.

    Raises:
        OSError: If the file can't be opened or contains no readable video.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        start: str | float | TimePoint | None = None,
        end: str | float | TimePoint | None = None,
        scale: float = 1.0,
    ) -> None:
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self._path = Path(path)
        self._start = parse_time(start) if start is not None else None
        self._end = parse_time(end) if end is not None else None
        self._scale = scale
        self._container: av.container.InputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._first: av.VideoFrame | None = None
        self._first_index = 0
        self._iterated = False

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> VideoReader:
        try:
            self._container = av.open(str(self._path))
        except FileNotFoundError:
            raise OSError(f"Video not found: {self._path}") from None
        except (av.error.FFmpegError, ValueError) as exc:
            raise OSError(f"Cannot open video: {self._path} ({exc})") from None
        if not self._container.streams.video:
            self._container.close()
            raise OSError(f"No video stream in: {self._path}")
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"  # FFmpeg's own decode threads

        self._rate = self._frame_rate()
        # Timestamps count from the stream's start, which isn't always 0
        # (MPEG-TS often starts at ~1.4 s).
        start_pts = self._stream.start_time
        tb = self._stream.time_base
        self._t0 = float(start_pts * tb) if start_pts is not None and tb is not None else 0.0
        self._decoded = self._decode()
        try:
            self._first, self._first_index = self._seek_to_start()
        except av.error.FFmpegError as exc:
            self.__exit__(None, None, None)
            raise OSError(f"Cannot read frames from video: {self._path} ({exc})") from None
        if self._first is None:
            self.__exit__(None, None, None)
            where = f" from {self._start}" if self._start else ""
            raise OSError(f"No frames to read{where} in: {self._path}")

        # Output geometry: rotation first (as players show it), then scaling.
        self._rot90 = round(self._first.rotation / 90) % 4
        raw_w, raw_h = self._first.width, self._first.height
        self._raw_size = (max(1, round(raw_w * self._scale)), max(1, round(raw_h * self._scale)))
        sw, sh = self._raw_size
        self._size = (sh, sw) if self._rot90 % 2 else (sw, sh)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None

    # -- properties -----------------------------------------------------------

    @property
    def fps(self) -> float:
        """Average frame rate in frames per second."""
        self._require_open()
        return float(self._rate)

    @property
    def rate(self) -> Fraction:
        """Average frame rate as an exact fraction (e.g. ``30000/1001``)."""
        self._require_open()
        return self._rate

    @property
    def width(self) -> int:
        """Width of the frames this reader yields (after rotation and scaling)."""
        self._require_open()
        return self._size[0]

    @property
    def height(self) -> int:
        """Height of the frames this reader yields (after rotation and scaling)."""
        self._require_open()
        return self._size[1]

    @property
    def rotation(self) -> int:
        """Rotation applied to the decoded frames, in degrees counter-clockwise."""
        self._require_open()
        return self._rot90 * 90

    @property
    def codec(self) -> str:
        """Name of the video codec, e.g. ``"h264"``."""
        self._require_open()
        return self._stream.codec_context.name  # type: ignore[union-attr]

    @property
    def frame_count(self) -> int | None:
        """Estimated number of frames this reader will yield, or ``None`` if unknown.

        Taken from the container's metadata, or from its duration when the
        metadata has no frame count. Not every format records either (raw
        H.264 doesn't), and the estimate can be off by a few frames.
        """
        self._require_open()
        stream = self._stream
        assert stream is not None and self._container is not None
        total: int | None = stream.frames or None
        if total is None and self._container.duration:
            total = round(self._container.duration / av.time_base * self._rate)
        first = self._first_index
        last = total
        if self._end is not None:
            end_frame = self._to_frame(self._end)
            last = end_frame if last is None else min(last, end_frame)
        return None if last is None else max(0, last - first)

    # -- iteration ------------------------------------------------------------

    def __iter__(self) -> Iterator[np.ndarray]:
        for frame in self.frames():
            yield frame.image

    def frames(self) -> Iterator[Frame]:
        """Yield the frames in the selected range as :class:`Frame` objects.

        Can only be iterated once per ``with`` block.
        """
        self._require_open()
        if self._iterated:
            raise RuntimeError("A VideoReader can only be iterated once; open it again to re-read.")
        self._iterated = True

        frame, index = self._first, self._first_index
        try:
            while frame is not None:
                time = self._time_of(frame, index)
                if self._past_end(index, time):
                    return
                yield Frame(index=index, time=time, image=self._to_image(frame))
                frame = next(self._decoded, None)
                index += 1
        except av.error.FFmpegError as exc:
            raise OSError(f"Error decoding {self._path} at frame {index}: {exc}") from None

    # -- internals ------------------------------------------------------------

    def _require_open(self) -> None:
        if self._container is None:
            raise RuntimeError("VideoReader must be used as a context manager.")

    def _frame_rate(self) -> Fraction:
        stream = self._stream
        assert stream is not None
        for rate in (stream.average_rate, stream.guessed_rate, stream.base_rate):
            if rate:
                return Fraction(rate)
        warnings.warn(
            f"{self._path.name} doesn't record a frame rate; assuming 25 fps.",
            stacklevel=3,
        )
        return Fraction(25)

    def _decode(self) -> Iterator[av.VideoFrame]:
        assert self._container is not None and self._stream is not None
        return self._container.decode(self._stream)

    def _to_frame(self, tp: TimePoint) -> int:
        return int(tp.value) if tp.unit == "frame" else round(tp.value * self._rate)

    def _time_of(self, frame: av.VideoFrame, index: int) -> float:
        # Raw streams carry no timestamps; fall back to the frame rate.
        if frame.time is None:
            return float(index / self._rate)
        return float(frame.time) - self._t0

    def _past_end(self, index: int, time: float) -> bool:
        if self._end is None:
            return False
        if self._end.unit == "frame":
            return index >= self._end.value
        return time >= self._end.value

    def _seek_to_start(self) -> tuple[av.VideoFrame | None, int]:
        """Return the first frame at or after ``start``, and its index."""
        start = self._start
        if start is not None and start.unit == "s" and start.value > _SEEK_THRESHOLD_S:
            found = self._seek_by_time(start.value)
            if found is not None:
                return found

        # From the beginning: exact frame numbers.
        index = 0
        for frame in self._decoded:
            if start is None:
                return frame, 0
            if start.unit == "frame":
                if index >= start.value:
                    return frame, index
            elif self._time_of(frame, index) >= start.value - 1e-6:
                return frame, index
            index += 1
        return None, index

    def _seek_by_time(self, seconds: float) -> tuple[av.VideoFrame | None, int] | None:
        """Seek near ``seconds``; ``None`` if seeking isn't possible here."""
        container, stream = self._container, self._stream
        assert container is not None and stream is not None
        # Raw streams (e.g. .h264) have no duration and no usable index to
        # seek in: decode from the beginning instead.
        if stream.time_base is None or not container.duration:
            return None
        try:
            target = int((seconds + self._t0) / stream.time_base)
            container.seek(target, stream=stream, backward=True)
        except av.error.FFmpegError:
            return None
        self._decoded = self._decode()
        first = True
        for frame in self._decoded:
            if frame.time is None:
                break  # no timestamps: can't tell where we are
            time = float(frame.time) - self._t0
            if first and time > seconds + 1e-6:
                break  # landed after the target: seeking didn't work
            first = False
            if time >= seconds - 1e-6:
                return frame, round(time * self._rate)
        # Seeking didn't work: reopen and decode from the beginning.
        container.close()
        self._container = av.open(str(self._path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._decoded = self._decode()
        return None

    def _to_image(self, frame: av.VideoFrame) -> np.ndarray:
        if self._scale == 1.0:
            image = frame.to_ndarray(format="bgr24")
        else:
            image = frame.to_ndarray(
                format="bgr24",
                width=self._raw_size[0],
                height=self._raw_size[1],
                interpolation="AREA" if self._scale < 1 else "BICUBIC",
            )
        if self._rot90:
            image = np.ascontiguousarray(np.rot90(image, self._rot90))
        return image


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

# Containers and the codecs "auto" tries for each, best first. (Some FFmpeg
# builds have no VP9 encoder, e.g. conda-forge's on Windows; WebM also takes
# AV1.)
_AUTO_CODEC = {
    ".mp4": ("h264",),
    ".m4v": ("h264",),
    ".mov": ("h264",),
    ".mkv": ("h264",),
    ".webm": ("vp9", "av1"),
    ".avi": ("mjpeg",),
}

# Encoders to try for each codec, best first. The first one this FFmpeg build
# has is used.
_ENCODERS = {
    "h264": ["libx264", "libopenh264", "h264_videotoolbox", "h264_mf"],
    "hevc": ["libx265", "hevc_videotoolbox", "hevc_mf"],
    "vp9": ["libvpx-vp9"],
    "av1": ["libsvtav1", "libaom-av1"],
    "ffv1": ["ffv1"],
    "mjpeg": ["mjpeg"],
    "mpeg4": ["mpeg4"],
}

# Other names people use, including OpenCV FourCC codes.
_ALIASES = {
    "avc1": "h264",
    "avc": "h264",
    "x264": "h264",
    "h265": "hevc",
    "hvc1": "hevc",
    "hev1": "hevc",
    "x265": "hevc",
    "vp09": "vp9",
    "av01": "av1",
    "mjpg": "mjpeg",
    "mp4v": "mpeg4",
    "xvid": "mpeg4",
    "divx": "mpeg4",
    "fmp4": "mpeg4",
}

# Default quality (CRF; lower is better) per encoder that supports it.
_DEFAULT_CRF = {"libx264": 18, "libx265": 20, "libvpx-vp9": 31, "libsvtav1": 30, "libaom-av1": 30}

# Encoders that take a fixed quantiser (2-31, lower is better) rather than a CRF.
_QSCALE_ENCODERS = {"mjpeg": 3, "mpeg4": 3}

SUPPORTED_EXTENSIONS = tuple(_AUTO_CODEC)
SUPPORTED_CODECS = tuple(_ENCODERS)


def _has_encoder(name: str) -> bool:
    try:
        av.codec.Codec(name, "w")
    except (av.error.FFmpegError, ValueError):
        return False
    return True


def _resolve_encoder(codec: str, path: Path) -> str:
    """Pick the FFmpeg encoder for ``codec`` (a name, alias or encoder) and ``path``."""
    ext = path.suffix.lower()
    if codec == "auto":
        if ext not in _AUTO_CODEC:
            raise ValueError(
                f"Can't write video to {path.name!r}: unknown extension {ext or '(none)'!r}. "
                f"Use one of: {', '.join(SUPPORTED_EXTENSIONS)}."
            )
        names = _AUTO_CODEC[ext]
    else:
        names = (_ALIASES.get(codec.lower(), codec.lower()),)

    if names[0] in _ENCODERS:
        tried = [encoder for name in names for encoder in _ENCODERS[name]]
        for encoder in tried:
            if _has_encoder(encoder):
                return encoder
        raise RuntimeError(
            f"This FFmpeg build has no {' or '.join(names)} encoder (tried {', '.join(tried)})."
        )
    if _has_encoder(codec):  # an FFmpeg encoder name, e.g. "h264_nvenc"
        return codec
    raise ValueError(
        f"Unknown codec {codec!r}. Use 'auto', one of {', '.join(SUPPORTED_CODECS)}, "
        "or the name of an FFmpeg encoder."
    )


def _pixel_format(encoder: str) -> str:
    if encoder == "ffv1":
        return "bgr0"  # lossless, no colour conversion
    if encoder == "mjpeg":
        return "yuvj420p"
    return "yuv420p"  # the format players and browsers expect


def _to_rate(fps: float | Fraction) -> Fraction:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if isinstance(fps, Fraction):
        return fps
    # Recover NTSC-style rates (29.97 -> 30000/1001) from their float value.
    return Fraction(fps).limit_denominator(1001)


class VideoWriter:
    """Streaming video writer. Use as a context manager.

    Frames go to a hidden temporary file next to ``path``
    (``.<name>.part<ext>``), which is moved to ``path`` when the ``with``
    block ends without an error. On an error, or Ctrl-C, the temporary file
    is deleted.

    Args:
        path: Output file path. The extension picks the container.
        fps: Frames per second, as a number or an exact fraction.
        width: Width of the frames passed to :meth:`write`.
        height: Height of the frames passed to :meth:`write`.
        codec: ``"auto"`` picks one for the extension (H.264 for ``.mp4``,
            ``.mov`` and ``.mkv``; VP9 for ``.webm``; MJPEG for ``.avi``).
            Otherwise one of ``h264``, ``hevc``, ``vp9``, ``av1``, ``ffv1``
            (lossless), ``mjpeg``, ``mpeg4``; an OpenCV FourCC such as
            ``mp4v``; or an FFmpeg encoder name such as ``h264_nvenc``.
        crf: Quality; lower is better and bigger. For H.264, HEVC, VP9 and
            AV1 this is the CRF; for MJPEG and MPEG-4, the quantiser (2-31).
            Defaults to a near-lossless setting per encoder.
        overwrite: Replace ``path`` if it exists. Otherwise an existing file
            raises :class:`FileExistsError`.

    Raises:
        FileExistsError: If ``path`` exists and ``overwrite`` is false.
        ValueError: For an unknown extension or codec.
    """

    def __init__(
        self,
        path: Path | str,
        fps: float | Fraction,
        width: int,
        height: int,
        codec: str = "auto",
        *,
        crf: int | None = None,
        overwrite: bool = False,
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError(f"Frame size must be positive, got {width}x{height}")
        self._path = Path(path)
        self._rate = _to_rate(fps)
        self._width, self._height = width, height
        self._encoder = _resolve_encoder(codec, self._path)
        self._crf = crf
        self._overwrite = overwrite
        self._pix_fmt = _pixel_format(self._encoder)
        # 4:2:0 chroma needs even dimensions: pad by a pixel rather than crop.
        even = self._pix_fmt in {"yuv420p", "yuvj420p"}
        self._out_size = (width + width % 2, height + height % 2) if even else (width, height)
        self._tmp_path = self._path.with_name(f".{self._path.stem}.part{self._path.suffix}")
        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._frames = 0

    @property
    def path(self) -> Path:
        """Final output path."""
        return self._path

    @property
    def encoder(self) -> str:
        """FFmpeg encoder in use, e.g. ``"libx264"``."""
        return self._encoder

    @property
    def frames_written(self) -> int:
        """Number of frames written so far."""
        return self._frames

    def __enter__(self) -> VideoWriter:
        if self._path.exists() and not self._overwrite:
            raise FileExistsError(f"Output already exists: {self._path}")
        if self._path.is_dir():
            raise IsADirectoryError(f"Output is a directory: {self._path}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._out_size != (self._width, self._height):
            warnings.warn(
                f"{self._encoder} needs even frame dimensions; padding "
                f"{self._width}x{self._height} to {self._out_size[0]}x{self._out_size[1]}.",
                stacklevel=2,
            )
        try:
            self._container = av.open(str(self._tmp_path), "w")
            self._stream = self._container.add_stream(self._encoder, rate=self._rate)
            self._configure(self._stream)
        except (av.error.FFmpegError, ValueError) as exc:
            self._discard()
            raise OSError(
                f"Cannot write {self._encoder} video to {self._path.name!r}: {exc}"
            ) from None
        return self

    def write(self, frame: np.ndarray) -> None:
        """Encode one BGR ``uint8`` frame of shape ``(height, width, 3)``."""
        if self._container is None or self._stream is None:
            raise RuntimeError("VideoWriter must be used as a context manager.")
        if frame.shape != (self._height, self._width, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"Expected a uint8 frame of shape {(self._height, self._width, 3)}, "
                f"got {frame.dtype} {frame.shape}"
            )
        pad_w, pad_h = self._out_size[0] - self._width, self._out_size[1] - self._height
        if pad_w or pad_h:
            frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
        try:
            for packet in self._stream.encode(video_frame):
                self._container.mux(packet)
        except av.error.FFmpegError as exc:
            raise OSError(f"Error encoding frame {self._frames} of {self._path}: {exc}") from None
        self._frames += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self._discard()
            return
        try:
            assert self._container is not None and self._stream is not None
            for packet in self._stream.encode():  # flush
                self._container.mux(packet)
            self._container.close()
            self._container = None
        except BaseException:
            self._discard()
            raise
        if self._frames == 0:
            self._discard()
            raise RuntimeError(f"No frames were written to {self._path}.")
        os.replace(self._tmp_path, self._path)

    def _configure(self, stream: av.video.stream.VideoStream) -> None:
        stream.width, stream.height = self._out_size
        stream.pix_fmt = self._pix_fmt
        options: dict[str, str] = {}
        if self._encoder in _DEFAULT_CRF:
            options["crf"] = str(
                self._crf if self._crf is not None else _DEFAULT_CRF[self._encoder]
            )
        if self._encoder == "libx264":
            options["preset"] = "medium"
        elif self._encoder == "libx265":
            options["x265-params"] = "log-level=error"
            stream.codec_tag = "hvc1"  # so Apple players accept it in .mp4/.mov
        elif self._encoder == "libvpx-vp9":
            options.update({"b": "0", "row-mt": "1"})  # constant quality
        elif self._encoder == "libsvtav1":
            options["preset"] = "8"
        elif self._encoder in _QSCALE_ENCODERS:
            q = str(self._crf if self._crf is not None else _QSCALE_ENCODERS[self._encoder])
            options.update({"qmin": q, "qmax": q})
        elif self._encoder != "ffv1":
            # Encoders without a quality mode (openh264, hardware): a generous bitrate.
            stream.bit_rate = int(self._out_size[0] * self._out_size[1] * float(self._rate) * 0.3)
        stream.options = options

    def _discard(self) -> None:
        if self._container is not None:
            with contextlib.suppress(Exception):
                self._container.close()
            self._container = None
        self._tmp_path.unlink(missing_ok=True)
