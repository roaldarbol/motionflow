# Tests for classical optical flow estimators:
# - DISFlow: shape, dtype, direction, accuracy on a known shift, presets
# - FarnebackFlow: output shape, dtype, non-zero flow on motion
# - LucasKanadeFlow: output shape, dtype, sparse flow characteristics
# - TVL1Flow: output shape, dtype, non-zero flow on motion
# - make_estimator: correct class returned per Method
import cv2
import numpy as np
import pytest

import motionflow.methods
from motionflow.methods import make_estimator
from motionflow.methods.classical import DISFlow, FarnebackFlow, LucasKanadeFlow, TVL1Flow
from motionflow.types import Method, ModelSize


def smooth_texture_pair(dx: int = 3, dy: int = -2) -> tuple[np.ndarray, np.ndarray]:
    """A smooth random texture and the same shifted by (dx, dy) pixels."""
    rng = np.random.default_rng(0)
    base = cv2.resize(rng.integers(0, 256, (60, 80), dtype=np.uint8), (320, 240), cv2.INTER_CUBIC)
    base = cv2.cvtColor(cv2.GaussianBlur(base, (5, 5), 1.5), cv2.COLOR_GRAY2BGR)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return base, cv2.warpAffine(base, M, (320, 240), borderMode=cv2.BORDER_REFLECT)


class TestDISFlow:
    def test_output_shape_and_dtype(self, checkerboard_frame, shifted_frame):
        flow = DISFlow()(checkerboard_frame, shifted_frame)
        assert flow.shape == (120, 160, 2)
        assert flow.dtype == np.float32

    @pytest.mark.parametrize("preset", DISFlow.PRESETS)
    def test_accuracy_on_known_shift(self, preset):
        a, b = smooth_texture_pair(3, -2)
        flow = DISFlow(preset)(a, b)[16:-16, 16:-16]
        error = np.linalg.norm(flow - np.array([3, -2], np.float32), axis=2).mean()
        assert error < 0.2, f"{preset}: mean endpoint error {error:.3f} px"

    def test_static_frames_produce_low_flow(self, checkerboard_frame):
        flow = DISFlow()(checkerboard_frame, checkerboard_frame)
        assert float(np.hypot(flow[..., 0], flow[..., 1]).max()) < 0.5

    def test_overrides(self):
        dis = DISFlow("fast", patch_size=12, patch_stride=6, refinement_iterations=0)
        assert dis._dis.getPatchSize() == 12
        assert dis._dis.getPatchStride() == 6
        assert dis._dis.getVariationalRefinementIterations() == 0

    def test_invalid_preset(self):
        with pytest.raises(ValueError, match="preset"):
            DISFlow("slow")


class TestFarnebackFlow:
    def test_output_shape(self, checkerboard_frame, shifted_frame):
        estimator = FarnebackFlow()
        flow = estimator(checkerboard_frame, shifted_frame)
        assert flow.shape == (120, 160, 2)

    def test_output_dtype(self, checkerboard_frame, shifted_frame):
        flow = FarnebackFlow()(checkerboard_frame, shifted_frame)
        assert flow.dtype == np.float32

    def test_detects_horizontal_motion(self, checkerboard_frame, shifted_frame):
        flow = FarnebackFlow()(checkerboard_frame, shifted_frame)
        # Mean dx should be positive (rightward shift), dy near zero
        mean_dx = float(np.mean(flow[..., 0]))
        mean_dy = float(np.mean(flow[..., 1]))
        assert mean_dx > 0.5, f"Expected positive dx, got {mean_dx}"
        assert abs(mean_dy) < 1.0, f"Expected near-zero dy, got {mean_dy}"

    def test_static_frames_produce_low_flow(self, checkerboard_frame):
        flow = FarnebackFlow()(checkerboard_frame, checkerboard_frame)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        assert float(magnitude.max()) < 0.5


class TestLucasKanadeFlow:
    def test_output_shape(self, checkerboard_frame, shifted_frame):
        flow = LucasKanadeFlow()(checkerboard_frame, shifted_frame)
        assert flow.shape == (120, 160, 2)

    def test_output_dtype(self, checkerboard_frame, shifted_frame):
        flow = LucasKanadeFlow()(checkerboard_frame, shifted_frame)
        assert flow.dtype == np.float32

    def test_sparse_output(self, checkerboard_frame, shifted_frame):
        flow = LucasKanadeFlow()(checkerboard_frame, shifted_frame)
        # LK is sparse: most pixels should be zero
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        nonzero_ratio = float(np.count_nonzero(magnitude)) / magnitude.size
        assert nonzero_ratio < 0.05, f"Expected sparse flow, got {nonzero_ratio:.1%} nonzero"

    def test_blank_frame_returns_zero_flow(self, blank_frame):
        flow = LucasKanadeFlow()(blank_frame, blank_frame)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        assert float(magnitude.max()) == 0.0


class TestTVL1Flow:
    def test_output_shape_and_dtype(self, checkerboard_frame, shifted_frame):
        flow = TVL1Flow()(checkerboard_frame, shifted_frame)
        assert flow.shape == (120, 160, 2)
        assert flow.dtype == np.float32

    def test_detects_horizontal_motion(self, checkerboard_frame, shifted_frame):
        flow = TVL1Flow()(checkerboard_frame, shifted_frame)
        mean_dx = float(np.mean(flow[..., 0]))
        mean_dy = float(np.mean(flow[..., 1]))
        assert mean_dx > 0.5, f"Expected positive dx, got {mean_dx}"
        assert abs(mean_dy) < 1.0, f"Expected near-zero dy, got {mean_dy}"


class TestMakeEstimator:
    @pytest.mark.parametrize(
        "method,expected_class",
        [
            (Method.DIS, "DISFlow"),
            (Method.FARNEBACK, "FarnebackFlow"),
            (Method.LUCAS_KANADE, "LucasKanadeFlow"),
        ],
    )
    def test_returns_correct_class(self, method, expected_class):
        estimator = make_estimator(method)
        assert type(estimator).__name__ == expected_class

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            make_estimator("not-a-method")

    def test_string_input_accepted(self):
        estimator = make_estimator("farneback")
        assert isinstance(estimator, FarnebackFlow)

    @pytest.mark.parametrize("size", [ModelSize.SMALL, "small", ModelSize.LARGE, "large"])
    def test_model_size_reaches_estimator(self, monkeypatch, size):
        # Regression: the enum used to arrive as the string "ModelSize.SMALL",
        # so --model-size small silently loaded the large model.
        received = {}

        class FakeRAFT:
            def __init__(self, size, device):
                received["size"] = size

        monkeypatch.setattr(motionflow.methods, "RAFTFlow", FakeRAFT)
        make_estimator(Method.RAFT, model_size=size)
        assert received["size"] == ModelSize(size).value


class TestLucasKanadeGrid:
    def test_tracks_grid_points(self, checkerboard_frame, shifted_frame):
        flow = LucasKanadeFlow(points="grid", grid_step=20)(checkerboard_frame, shifted_frame)
        tracked = np.count_nonzero(np.hypot(flow[..., 0], flow[..., 1]))
        assert 0 < tracked <= (120 // 20) * (160 // 20)

    def test_invalid_points(self):
        with pytest.raises(ValueError, match="points"):
            LucasKanadeFlow(points="random")
