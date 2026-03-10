from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


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
    """A short synthetic video with horizontal motion (20 frames, 160x120).

    Uses MJPG/avi rather than mp4v/mp4 because OpenCV cannot reliably
    read back mp4v-encoded files on macOS.
    """
    path = tmp_path / "test_input.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (160, 120))

    base = np.zeros((120, 160, 3), dtype=np.uint8)
    for y in range(0, 120, 20):
        for x in range(0, 160, 20):
            if (y // 20 + x // 20) % 2 == 0:
                base[y : y + 20, x : x + 20] = 180

    for i in range(20):
        M = np.float32([[1, 0, i * 2], [0, 1, 0]])
        frame = cv2.warpAffine(base, M, (160, 120))
        writer.write(frame)

    writer.release()
    return path