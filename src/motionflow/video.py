from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    """Streaming video reader. Use as a context manager.

    Yields BGR frames as numpy arrays on iteration.

    Args:
        path: Path to the video file.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> VideoReader:
        # Prefer the FFMPEG backend — it handles far more codecs than the
        # default backend in headless OpenCV builds.
        self._cap = cv2.VideoCapture(str(self._path), cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            # Fall back to default backend selection.
            self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise OSError(f"Cannot open video: {self._path}")
        # Read one frame to get true decoded dimensions — CAP_PROP_FRAME_WIDTH/HEIGHT
        # can be wrong for H.264 files with rotation metadata.
        ok, frame = self._cap.read()
        if not ok:
            raise OSError(f"Cannot read frames from video: {self._path}")
        self._frame_height, self._frame_width = frame.shape[:2]
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._cap is not None:
            self._cap.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("VideoReader must be used as a context manager.")
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    @property
    def fps(self) -> float:
        self._require_open()
        return float(self._cap.get(cv2.CAP_PROP_FPS))  # type: ignore[union-attr]

    @property
    def width(self) -> int:
        self._require_open()
        return self._frame_width

    @property
    def height(self) -> int:
        self._require_open()
        return self._frame_height

    @property
    def frame_count(self) -> int:
        self._require_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore[union-attr]

    def _require_open(self) -> None:
        if self._cap is None:
            raise RuntimeError("VideoReader must be used as a context manager.")


def _resolve_codec(codec: str, path: Path) -> str:
    """Resolve ``"auto"`` to a codec that can write the given container on this platform.

    On macOS with ``opencv-python-headless``, neither ``mp4v`` nor ``avc1`` can write
    ``.mp4``. In that case ``"auto"`` redirects to ``MJPG`` and renames the output
    path to ``.avi``. Use ``.avi`` output explicitly to avoid the redirect.
    """
    import platform

    if codec != "auto":
        return codec.ljust(4)[:4]

    ext = path.suffix.lower()
    if ext in {".mp4", ".m4v"} and platform.system() == "Darwin":
        # headless OpenCV on macOS cannot write MP4 — caller must handle the rename
        return "MJPG"
    if ext in {".mp4", ".m4v"}:
        return "mp4v"
    if ext == ".avi":
        return "MJPG"
    return "mp4v"


class VideoWriter:
    """Streaming video writer. Use as a context manager.

    Args:
        path: Output file path.
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        codec: FourCC codec string, e.g. ``"mp4v"`` or ``"avc1"``. ``"auto"`` selects a sensible default for the output container and platform.
    """

    def __init__(
        self,
        path: Path,
        fps: float,
        width: int,
        height: int,
        codec: str = "auto",
    ) -> None:
        self._path = Path(path)
        self._fps = fps
        self._width = width
        self._height = height
        self._codec = codec
        self._fourcc = cv2.VideoWriter_fourcc(*_resolve_codec(codec, self._path))
        self._writer: cv2.VideoWriter | None = None

    def __enter__(self) -> VideoWriter:
        import platform
        import sys

        self._path.parent.mkdir(parents=True, exist_ok=True)

        # On macOS with headless OpenCV, MP4 output is not supported. Redirect to
        # .avi transparently and warn the user.
        if (
            platform.system() == "Darwin"
            and self._path.suffix.lower() in {".mp4", ".m4v"}
            and self._codec == "auto"
        ):
            self._path = self._path.with_suffix(".avi")
            self._fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            print(
                f"Warning: MP4 output is not supported by opencv-headless on macOS. "
                f"Writing to {self._path} instead.",
                file=sys.stderr,
            )

        # Suppress the spurious cap_images probe warning OpenCV emits when
        # trying backends before settling on the right one.
        # setLogLevel is absent in some headless builds, so guard both calls.
        _has_log_control = hasattr(cv2, "setLogLevel")
        if _has_log_control:
            cv2.setLogLevel(0)
        try:
            self._writer = cv2.VideoWriter(
                str(self._path),
                self._fourcc,
                self._fps,
                (self._width, self._height),
            )
        finally:
            if _has_log_control:
                cv2.setLogLevel(3)  # LOG_LEVEL_WARNING

        if not self._writer.isOpened():
            raise OSError(f"Cannot open video for writing: {self._path}")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._writer is not None:
            self._writer.release()

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("VideoWriter must be used as a context manager.")
        self._writer.write(frame)