# Tests for visualisation functions:
# - flow_to_hsv: output shape, dtype, zero flow produces black image
# - flow_to_heatmap: output shape, dtype, colormap applied
# - flow_to_quiver: overlay has same shape as input frame
# - flow_to_blend: alpha=0 matches hsv, alpha=1 matches original frame
# - visualize_flow: correct dispatch, error on missing frame for quiver/blend
import numpy as np
import pytest

from motionflow.types import VizStyle
from motionflow.viz import (
    flow_to_blend,
    flow_to_heatmap,
    flow_to_hsv,
    flow_to_quiver,
    visualize_flow,
)


@pytest.fixture()
def zero_flow():
    return np.zeros((120, 160, 2), dtype=np.float32)


@pytest.fixture()
def horizontal_flow():
    flow = np.zeros((120, 160, 2), dtype=np.float32)
    flow[..., 0] = 5.0  # dx = 5, dy = 0
    return flow


@pytest.fixture()
def frame():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[..., 2] = 200  # red channel
    return img


class TestFlowToHsv:
    def test_output_shape(self, horizontal_flow):
        out = flow_to_hsv(horizontal_flow)
        assert out.shape == (120, 160, 3)

    def test_output_dtype(self, horizontal_flow):
        assert flow_to_hsv(horizontal_flow).dtype == np.uint8

    def test_zero_flow_is_black(self, zero_flow):
        out = flow_to_hsv(zero_flow)
        assert int(out.max()) == 0

    def test_max_flow_clips_brightness(self, horizontal_flow):
        out_full = flow_to_hsv(horizontal_flow, max_flow=5.0)
        out_clipped = flow_to_hsv(horizontal_flow, max_flow=10.0)
        # Full normalisation should be brighter than halved cap
        assert int(out_full.max()) > int(out_clipped.max())


class TestFlowToHeatmap:
    def test_output_shape(self, horizontal_flow):
        assert flow_to_heatmap(horizontal_flow).shape == (120, 160, 3)

    def test_output_dtype(self, horizontal_flow):
        assert flow_to_heatmap(horizontal_flow).dtype == np.uint8

    def test_zero_flow_dark(self, zero_flow):
        out = flow_to_heatmap(zero_flow)
        # INFERNO: zero maps to near-black
        assert int(out.mean()) < 30


class TestFlowToQuiver:
    def test_output_shape(self, horizontal_flow, frame):
        out = flow_to_quiver(horizontal_flow, frame)
        assert out.shape == frame.shape

    def test_does_not_modify_input_frame(self, horizontal_flow, frame):
        original = frame.copy()
        flow_to_quiver(horizontal_flow, frame)
        np.testing.assert_array_equal(frame, original)


class TestFlowToBlend:
    def test_output_shape(self, horizontal_flow, frame):
        assert flow_to_blend(horizontal_flow, frame, alpha=0.5).shape == frame.shape

    def test_alpha_zero_matches_hsv(self, horizontal_flow, frame):
        blended = flow_to_blend(horizontal_flow, frame, alpha=0.0)
        hsv = flow_to_hsv(horizontal_flow)
        np.testing.assert_array_equal(blended, hsv)

    def test_alpha_one_matches_original(self, horizontal_flow, frame):
        blended = flow_to_blend(horizontal_flow, frame, alpha=1.0)
        np.testing.assert_array_equal(blended, frame)


class TestVisualizeFlow:
    def test_hsv_dispatch(self, horizontal_flow):
        out = visualize_flow(horizontal_flow, VizStyle.HSV)
        assert out.shape == (120, 160, 3)

    def test_heatmap_dispatch(self, horizontal_flow):
        out = visualize_flow(horizontal_flow, VizStyle.HEATMAP)
        assert out.shape == (120, 160, 3)

    def test_quiver_requires_frame(self, horizontal_flow):
        with pytest.raises(ValueError, match="requires a source frame"):
            visualize_flow(horizontal_flow, VizStyle.QUIVER)

    def test_blend_requires_frame(self, horizontal_flow):
        with pytest.raises(ValueError, match="requires a source frame"):
            visualize_flow(horizontal_flow, VizStyle.BLEND)

    def test_quiver_with_frame(self, horizontal_flow, frame):
        out = visualize_flow(horizontal_flow, VizStyle.QUIVER, frame=frame)
        assert out.shape == frame.shape

    def test_blend_overlay_applied_when_blend_gt_zero(self, horizontal_flow, frame):
        pure = visualize_flow(horizontal_flow, VizStyle.HSV, blend=0.0)
        composited = visualize_flow(horizontal_flow, VizStyle.HSV, frame=frame, blend=0.5)
        # They should differ since the original frame is mixed in
        assert not np.array_equal(pure, composited)
