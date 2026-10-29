from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test videos with identifiable frames
# ---------------------------------------------------------------------------


def id_frame(index: int, width: int = 64, height: int = 48) -> np.ndarray:
    """A BGR frame whose number (0-255) is drawn as 8 black/white stripes.

    The stripes survive lossy compression, so :func:`frame_id` can read the
    number back from a decoded frame.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for bit in range(8):
        if index >> bit & 1:
            frame[:, bit * width // 8 : (bit + 1) * width // 8] = 255
    return frame


def frame_id(frame: np.ndarray) -> int:
    """Read back the number drawn by :func:`id_frame`."""
    width = frame.shape[1]
    margin = width // 32  # stay clear of compression blur at stripe edges
    index = 0
    for bit in range(8):
        stripe = frame[:, bit * width // 8 + margin : (bit + 1) * width // 8 - margin]
        if stripe.mean() > 127:
            index |= 1 << bit
    return index


def has_encoder(name: str) -> bool:
    """Whether this FFmpeg build has the encoder (conda-forge's Windows build lacks VP9)."""
    try:
        av.codec.Codec(name, "w")
    except (av.error.FFmpegError, ValueError):
        return False
    return True


# PyAV < 17 (the newest for Python 3.10) can't write rotation metadata.
CAN_WRITE_ROTATION = hasattr(av.video.stream.VideoStream, "set_display_rotation")


def write_video(
    path: Path,
    n: int = 10,
    width: int = 64,
    height: int = 48,
    fps: float | Fraction = 25,
    codec: str = "libx264",
    fmt: str | None = None,
    rotation: int = 0,
) -> Path:
    """Write a test video with PyAV directly (independent of VideoWriter).

    Frame ``i`` shows :func:`id_frame` ``(i)``.
    """
    pix_fmt = {"mjpeg": "yuvj420p", "ffv1": "bgr0"}.get(codec, "yuv420p")
    with av.open(str(path), "w", format=fmt) as container:
        stream = container.add_stream(codec, rate=Fraction(fps).limit_denominator(1001))
        stream.width, stream.height, stream.pix_fmt = width, height, pix_fmt
        if codec == "libx264":
            stream.options = {"qp": "10"}
        if rotation:
            stream.set_display_rotation(rotation)
        for i in range(n):
            image = id_frame(i, width, height)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def blank_frame() -> np.ndarray:
    """A solid grey BGR frame."""
    return np.full((120, 160, 3), 128, dtype=np.uint8)


@pytest.fixture()
def random_frame(rng: np.random.Generator) -> np.ndarray:
    """A random noise BGR frame."""
    return rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)


@pytest.fixture()
def checkerboard_frame() -> np.ndarray:
    """A checkerboard pattern useful for testing flow on textured content."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    for y in range(0, 120, 20):
        for x in range(0, 160, 20):
            if (y // 20 + x // 20) % 2 == 0:
                frame[y : y + 20, x : x + 20] = 200
    return frame


@pytest.fixture()
def shifted_frame(checkerboard_frame: np.ndarray) -> np.ndarray:
    """The checkerboard shifted 5 pixels right — produces known horizontal flow."""
    h, w = checkerboard_frame.shape[:2]
    M = np.float32([[1, 0, 5], [0, 1, 0]])
    return cv2.warpAffine(checkerboard_frame, M, (w, h))


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    """A short synthetic video with horizontal motion (20 frames, 160x120, H.264)."""
    path = tmp_path / "test_input.mp4"

    base = np.zeros((120, 160, 3), dtype=np.uint8)
    for y in range(0, 120, 20):
        for x in range(0, 160, 20):
            if (y // 20 + x // 20) % 2 == 0:
                base[y : y + 20, x : x + 20] = 180

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        stream.options = {"qp": "10"}
        for i in range(20):
            M = np.float32([[1, 0, i * 2], [0, 1, 0]])
            frame = cv2.warpAffine(base, M, (160, 120))
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
