# Tests for the deep-learning methods (motionflow.methods.deep). Skipped
# without torch. Tests that need weights run when they're cached, or when
# MOTIONFLOW_TEST_WEIGHTS=1 allows downloading them (CI does).
# - Device and model-size validation
# - Every method and size: shape, dtype, and accuracy on a known shift, at
#   a frame size that needs padding
# - NeuFlow handles frame-size changes; flow() runs end to end
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import write_video  # noqa: E402

from motionflow import flow, weights  # noqa: E402
from motionflow.methods import make_estimator  # noqa: E402
from motionflow.methods.deep import RAFTFlow, SEARaftFlow, resolve_device  # noqa: E402

ALLOW_DOWNLOAD = os.environ.get("MOTIONFLOW_TEST_WEIGHTS") == "1"


def raft_cached(size: str) -> bool:
    hub = torch.hub.get_dir()
    return any(name.startswith(f"raft_{size}_") for name in _listdir(f"{hub}/checkpoints"))


def _listdir(path: str) -> list[str]:
    try:
        return os.listdir(path)
    except FileNotFoundError:
        return []


def available(method: str, size: str) -> bool:
    if ALLOW_DOWNLOAD:
        return True
    if method == "raft":
        return raft_cached(size)
    name = {
        "sea-raft": f"sea-raft-{'medium' if size == 'large' else size}",
        "neuflow": "neuflow-v2",
    }
    return weights.is_cached(name[method])


def needs_weights(method: str, size: str = "large"):
    return pytest.mark.skipif(
        not available(method, size),
        reason=f"{method} {size} weights not cached (set MOTIONFLOW_TEST_WEIGHTS=1)",
    )


def shifted_pair(dx: int = 3, dy: int = -2, w: int = 330, h: int = 250):
    """A smooth texture and the same shifted by (dx, dy), at a size needing padding."""
    import cv2

    rng = np.random.default_rng(0)
    base = cv2.resize(
        rng.integers(0, 256, (h // 4, w // 4), dtype=np.uint8), (w, h), cv2.INTER_CUBIC
    )
    base = cv2.cvtColor(cv2.GaussianBlur(base, (5, 5), 1.5), cv2.COLOR_GRAY2BGR)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return base, cv2.warpAffine(base, M, (w, h), borderMode=cv2.BORDER_REFLECT)


class TestValidation:
    def test_auto_device(self):
        assert resolve_device("auto").type in {"cuda", "mps", "cpu"}

    def test_unknown_device(self):
        with pytest.raises(ValueError, match="auto, cuda, mps or cpu"):
            resolve_device("gpu")

    @pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available here")
    def test_unavailable_cuda(self):
        with pytest.raises(RuntimeError, match="No CUDA GPU"):
            resolve_device("cuda")

    def test_raft_has_no_medium(self):
        with pytest.raises(ValueError, match="RAFT sizes are small, large"):
            RAFTFlow(size="medium", device="cpu")

    def test_sea_raft_sizes(self):
        with pytest.raises(ValueError, match="SEA-RAFT sizes are small, medium, large"):
            SEARaftFlow(size="huge", device="cpu")


CASES = [
    # method, size, largest acceptable mean endpoint error (px)
    ("raft", "small", 0.3),
    ("raft", "large", 0.15),
    ("sea-raft", "small", 0.1),
    ("sea-raft", "medium", 0.1),
    ("sea-raft", "large", 0.1),
    ("neuflow", "large", 0.15),
]


@pytest.mark.parametrize(("method", "size", "tolerance"), CASES)
def test_accuracy_on_known_shift(method, size, tolerance):
    if not available(method, size):
        pytest.skip(f"{method} {size} weights not cached (set MOTIONFLOW_TEST_WEIGHTS=1)")
    a, b = shifted_pair(3, -2)
    flow_ = make_estimator(method, device="cpu", model_size=size)(a, b)
    assert flow_.shape == (250, 330, 2)
    assert flow_.dtype == np.float32
    inner = flow_[16:-16, 16:-16]
    error = np.linalg.norm(inner - np.array([3, -2], np.float32), axis=2).mean()
    assert error < tolerance, f"{method} {size}: mean endpoint error {error:.3f} px"


@needs_weights("neuflow")
def test_neuflow_handles_size_changes():
    estimator = make_estimator("neuflow", device="cpu")
    for w, h in [(330, 250), (200, 120), (330, 250)]:
        a, b = shifted_pair(3, -2, w, h)
        flow_ = estimator(a, b)
        assert flow_.shape == (h, w, 2)
        assert abs(float(flow_[16:-16, 16:-16, 0].mean()) - 3) < 0.3


@pytest.mark.parametrize(
    ("method", "size"), [("raft", "small"), ("sea-raft", "small"), ("neuflow", "large")]
)
def test_tiny_frames(method, size):
    # Smaller than the models' correlation pyramids allow: padded internally.
    if not available(method, size):
        pytest.skip(f"{method} {size} weights not cached")
    a, b = shifted_pair(2, 1, w=64, h=48)
    assert make_estimator(method, device="cpu", model_size=size)(a, b).shape == (48, 64, 2)


@needs_weights("sea-raft", "small")
def test_flow_end_to_end(tmp_path):
    src = write_video(tmp_path / "clip.mp4", n=4)
    result = flow(src, method="sea-raft", model_size="small", device="cpu", progress=False)
    assert result.frames_written == 4
