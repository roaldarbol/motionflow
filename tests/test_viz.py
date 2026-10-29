# Tests for visualisation functions:
# - flow_to_hsv: output shape, dtype, zero flow produces black image
# - flow_to_heatmap: output shape, dtype, colormap applied
# - flow_to_quiver: overlay has same shape as input frame
# - flow_to_blend: alpha=0 matches hsv, alpha=1 matches original frame
# - visualize_flow: correct dispatch, error on missing frame for quiver/blend
# - min_flow hides slow motion; quiver arrows are visible, averaged per
#   cell, and absent where nothing moves; sparse points are drawn
# - overlay; the HSV key matches the flow colours; legends stay in their
#   corner and are skipped on tiny frames
import numpy as np
import pytest

from motionflow.types import VizStyle
from motionflow.viz import (
    add_legend,
    draw_sparse,
    flow_to_blend,
    flow_to_heatmap,
    flow_to_hsv,
    flow_to_quiver,
    hsv_wheel,
    overlay,
    quiver_scale,
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
            visualize_flow(horizontal_flow, "blend")

    def test_quiver_with_frame(self, horizontal_flow, frame):
        out = visualize_flow(horizontal_flow, VizStyle.QUIVER, frame=frame)
        assert out.shape == frame.shape

    def test_blend_overlay_applied_when_blend_gt_zero(self, horizontal_flow, frame):
        pure = visualize_flow(horizontal_flow, VizStyle.HSV, blend=0.0)
        composited = visualize_flow(horizontal_flow, VizStyle.HSV, frame=frame, blend=0.5)
        # They should differ since the original frame is mixed in
        assert not np.array_equal(pure, composited)


class TestMinFlow:
    @pytest.mark.parametrize("render", [flow_to_hsv, flow_to_heatmap])
    def test_slow_motion_is_hidden(self, horizontal_flow, render):
        slow = horizontal_flow * 0.1  # 0.5 px/frame
        assert render(slow, max_flow=5.0).max() > 0
        assert int(render(slow, max_flow=5.0, min_flow=1.0).mean()) < 30


class TestQuiverRendering:
    def test_arrows_are_visible(self, frame):
        flow = np.zeros((120, 160, 2), dtype=np.float32)
        flow[..., 0] = 1.0  # 1 px/frame: invisible at true length
        out = flow_to_quiver(flow, frame, step=16, max_flow=1.0)
        changed = np.any(out != frame, axis=2)
        assert changed.sum() > 300

    def test_no_arrows_without_motion(self, zero_flow, frame):
        np.testing.assert_array_equal(flow_to_quiver(zero_flow, frame), frame)

    def test_arrows_average_their_cell(self, frame):
        # Motion only in the top-left 16x16 cell: one arrow, starting there.
        flow = np.zeros((120, 160, 2), dtype=np.float32)
        flow[:16, :16, 0] = 4.0
        out = flow_to_quiver(flow, frame, step=16, max_flow=4.0)
        changed = np.any(out != frame, axis=2)
        assert changed[:16, :32].any()
        assert not changed[32:].any()

    def test_scale(self):
        assert quiver_scale(16, max_flow=2.0) == pytest.approx(7.2)
        assert quiver_scale(16, max_flow=None, flow=np.zeros((32, 32, 2), np.float32)) == 0.0


class TestSparse:
    @pytest.fixture()
    def sparse_flow(self):
        flow = np.zeros((120, 160, 2), dtype=np.float32)
        flow[30, 40] = (3.0, 0.0)
        flow[90, 120] = (0.0, -2.0)
        return flow

    @pytest.mark.parametrize("style", ["hsv", "heatmap"])
    def test_points_become_dots(self, sparse_flow, style):
        out = draw_sparse(sparse_flow, style, max_flow=3.0)
        lit = np.any(out > 0, axis=2)
        assert lit[30, 40] and lit[90, 120]
        assert 2 < lit.sum() < 200

    def test_quiver_draws_arrows_from_points(self, sparse_flow, frame):
        out = draw_sparse(sparse_flow, "quiver", frame, max_flow=3.0)
        changed = np.any(out != frame, axis=2)
        assert changed[28:33, 38:60].any()
        assert changed[70:93, 118:123].any()


class TestOverlay:
    def test_opacity(self, frame):
        layer = np.full_like(frame, 100)
        np.testing.assert_array_equal(overlay(frame, layer, 1.0), layer)
        np.testing.assert_array_equal(overlay(frame, layer, 0.0), frame)
        assert overlay(frame, layer, 0.5)[0, 0, 2] == 150


class TestLegend:
    def test_wheel_matches_flow_colours(self):
        wheel, mask = hsv_wheel(101)
        right = np.zeros((1, 1, 2), np.float32)
        right[..., 0] = 1.0
        down = np.zeros((1, 1, 2), np.float32)
        down[..., 1] = 1.0
        np.testing.assert_allclose(wheel[50, 100], flow_to_hsv(right, 1.0)[0, 0], atol=8)
        np.testing.assert_allclose(wheel[100, 50], flow_to_hsv(down, 1.0)[0, 0], atol=8)
        assert mask[50, 50] and not mask[0, 0]

    @pytest.mark.parametrize("style", ["hsv", "heatmap", "quiver"])
    def test_only_the_corner_changes(self, style):
        image = np.zeros((240, 320, 3), np.uint8)
        add_legend(image, style, max_flow=5.0, quiver_scale=2.0)
        changed = np.any(image > 0, axis=2)
        assert changed[160:, 220:].any()
        assert not changed[:150].any() and not changed[:, :200].any()

    def test_label(self):
        image = np.zeros((240, 320, 3), np.uint8)
        add_legend(image, "heatmap", max_flow=None)  # "per frame"
        assert image[220:, 200:].any()

    def test_skipped_on_tiny_frames(self):
        image = np.zeros((40, 40, 3), np.uint8)
        add_legend(image, "hsv", max_flow=5.0)
        assert not image.any()

    @pytest.mark.parametrize("style", ["hsv", "heatmap", "quiver"])
    def test_no_points(self, zero_flow, frame, style):
        out = draw_sparse(zero_flow, style, frame, max_flow=3.0)
        expected = frame if style == "quiver" else np.zeros_like(frame)
        np.testing.assert_array_equal(out, expected)
