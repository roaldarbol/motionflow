# Tests for optical flow as a transform (motionflow.flow):
# - Frame alignment: output frame i shows motion from frame i-1 to i;
#   frame 0 shows none; N frames in, N out
# - stride: estimates between frames N apart, in pixels per frame, held
# - smooth: centred window (no lag), shrinking at the ends
# - flow(): every option end to end, global normalisation, the Result
# - FlowRenderer and FlowTransform validation
from pathlib import Path

import numpy as np
import pytest
from conftest import frame_id, write_video

from motionflow import FlowRenderer, FlowTransform, flow
from motionflow.methods import FlowEstimator
from motionflow.video import Frame, VideoReader


class FrameNumberFlow(FlowEstimator):
    """A fake estimator: dx is the difference between the frames' numbers.

    Records which pairs it was asked about.
    """

    def __init__(self, impulse_at: int | None = None) -> None:
        self.pairs: list[tuple[int, int]] = []
        self.impulse_at = impulse_at

    def estimate(self, prev, curr):
        a, b = frame_id(prev), frame_id(curr)
        self.pairs.append((a, b))
        dx = float(b == self.impulse_at) if self.impulse_at is not None else float(b - a)
        flow = np.zeros((*curr.shape[:2], 2), dtype=np.float32)
        flow[..., 0] = dx
        return flow


def frames(n: int) -> list[Frame]:
    from conftest import id_frame

    return [Frame(index=i, time=i / 25, image=id_frame(i)) for i in range(n)]


def dx_per_frame(transform: FlowTransform, n: int) -> list[float]:
    return [round(float(f[0, 0, 0]), 4) for _, f in transform.flows(frames(n))]


class TestAlignment:
    def test_flow_into_each_frame(self):
        estimator = FrameNumberFlow()
        out = list(FlowTransform(estimator).flows(frames(5)))
        assert [frame.index for frame, _ in out] == [0, 1, 2, 3, 4]
        assert estimator.pairs == [(0, 1), (1, 2), (2, 3), (3, 4)]
        assert dx_per_frame(FlowTransform(FrameNumberFlow()), 5) == [0, 1, 1, 1, 1]

    def test_first_frame_shows_no_motion(self):
        renderer = FlowRenderer("hsv")
        images = list(FlowTransform(FrameNumberFlow(), renderer).process(frames(3)))
        assert len(images) == 3
        assert images[0].max() == 0  # no motion is black in HSV
        assert images[1].max() > 0

    def test_single_frame(self):
        assert dx_per_frame(FlowTransform(FrameNumberFlow()), 1) == [0]


class TestStride:
    def test_estimates_between_frames_stride_apart(self):
        estimator = FrameNumberFlow()
        list(FlowTransform(estimator, stride=3).flows(frames(10)))
        assert estimator.pairs == [(0, 3), (3, 6), (6, 9)]

    def test_magnitudes_stay_per_frame_and_are_held(self):
        # Frames 0-2 come before the first estimate.
        assert dx_per_frame(FlowTransform(FrameNumberFlow(), stride=3), 8) == [
            0, 0, 0, 1, 1, 1, 1, 1
        ]  # fmt: skip


class TestSmoothing:
    def test_centred_window_does_not_lag(self):
        # Motion only into frame 5; a centred 3-frame window spreads it over 4-6.
        transform = FlowTransform(FrameNumberFlow(impulse_at=5), smooth=3)
        assert dx_per_frame(transform, 9) == [0, 0, 0, 0, 0.3333, 0.3333, 0.3333, 0, 0]

    def test_even_window_reaches_one_further_back(self):
        transform = FlowTransform(FrameNumberFlow(impulse_at=5), smooth=4)
        assert dx_per_frame(transform, 9) == [0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0]

    def test_window_shrinks_at_the_ends(self):
        transform = FlowTransform(FrameNumberFlow(impulse_at=1), smooth=3)
        # Frame 0's window is frames 0-1 (2 frames).
        assert dx_per_frame(transform, 4) == [0.5, 0.3333, 0.3333, 0]

    @pytest.mark.parametrize("smooth", [0, 1])
    def test_off(self, smooth):
        transform = FlowTransform(FrameNumberFlow(impulse_at=2), smooth=smooth)
        assert dx_per_frame(transform, 4) == [0, 0, 1, 0]

    @pytest.mark.parametrize("smooth", [2, 3, 5, 8])
    def test_same_number_of_frames(self, smooth):
        transform = FlowTransform(FrameNumberFlow(), smooth=smooth)
        out = list(transform.flows(frames(6)))
        assert [frame.index for frame, _ in out] == list(range(6))


@pytest.fixture()
def moving(tmp_path) -> Path:
    return write_video(tmp_path / "moving.mp4", n=12)


class TestFlowFunction:
    def test_n_frames_in_n_out(self, tiny_video):
        result = flow(tiny_video, progress=False)
        assert result.output == tiny_video.with_name("test_input_flow.mp4")
        assert result.frames_written == 20
        with VideoReader(result.output) as reader:
            assert sum(1 for _ in reader) == 20
        assert result.info == {"method": "farneback", "max_flow": None}

    def test_detects_motion(self, tiny_video, tmp_path):
        # tiny_video shifts right by 2 px per frame.
        result = flow(tiny_video, tmp_path / "out.mkv", codec="ffv1", progress=False)
        with VideoReader(result.output) as reader:
            images = list(reader)
        assert images[0].max() == 0
        assert images[5].mean() > 20

    @pytest.mark.parametrize("viz", ["hsv", "heatmap", "quiver", "blend"])
    def test_viz_styles(self, tiny_video, tmp_path, viz):
        result = flow(tiny_video, tmp_path / f"{viz}.mp4", viz=viz, blend=0.4, progress=False)
        assert result.frames_written == 20

    def test_global_normalisation(self, tiny_video, tmp_path):
        result = flow(tiny_video, tmp_path / "out.mp4", normalize="global", progress=False)
        assert result.info["max_flow"] > 1.0

    def test_max_flow(self, tiny_video, tmp_path):
        result = flow(tiny_video, tmp_path / "out.mp4", max_flow=20.0, progress=False)
        assert result.info["max_flow"] == 20.0

    @pytest.mark.parametrize("kwargs", [{"stride": 2}, {"smooth": 3}, {"scale": 0.5}])
    def test_options(self, tiny_video, tmp_path, kwargs):
        result = flow(tiny_video, tmp_path / "out.mp4", progress=False, **kwargs)
        assert result.frames_written == 20

    def test_range(self, moving, tmp_path):
        result = flow(moving, tmp_path / "out.mp4", start="2f", end="8f", progress=False)
        assert result.frames_written == 6

    @pytest.mark.parametrize(("codec", "name"), [("ffv1", "out.mkv"), ("mp4v", "out.mp4")])
    def test_codec(self, moving, tmp_path, codec, name):
        result = flow(moving, tmp_path / name, codec=codec, progress=False)
        assert result.frames_written == 12

    def test_single_frame_input(self, tmp_path):
        src = write_video(tmp_path / "one.mp4", n=1)
        result = flow(src, progress=False)
        assert result.frames_written == 1

    def test_on_progress(self, moving, tmp_path):
        calls = []
        flow(
            moving,
            tmp_path / "out.mp4",
            progress=False,
            on_progress=lambda done, total: calls.append((done, total)),
        )
        assert calls[-1] == (12, 12)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"stride": 0},
            {"smooth": -1},
            {"blend": 2.0},
            {"fps": 0},
            {"max_flow": 0},
            {"max_flow": -3},
            {"quiver_step": 0},
            {"scale": 0},
            {"start": "abc"},
            {"method": "invalid"},
            {"viz": "invalid"},
            {"normalize": "invalid"},
        ],
    )
    def test_invalid_arguments(self, moving, tmp_path, kwargs):
        with pytest.raises(ValueError):
            flow(moving, tmp_path / "out.mp4", progress=False, **kwargs)
        assert not (tmp_path / "out.mp4").exists()

    def test_refuses_to_overwrite_input(self, moving):
        with pytest.raises(ValueError, match="overwrite the input"):
            flow(moving, moving, progress=False)


class TestRenderer:
    @pytest.mark.parametrize(
        "kwargs", [{"blend": -0.1}, {"blend": 1.5}, {"max_flow": 0}, {"quiver_step": 0}]
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            FlowRenderer(**kwargs)

    def test_invalid_style(self):
        with pytest.raises(ValueError):
            FlowRenderer("sparkles")
