# Tests for event-camera simulation (motionflow.events):
# - The ideal model: no events without change; a brightness step gives
#   exactly floor(delta lin-log / C) events of the right polarity; leftover
#   change carries over; a moving edge fires only at the edge
# - Sensor realism (v2e): threshold spread (seeded), bandwidth limit (dark
#   pixels lag), leak and shot noise rates, refractory period, presets
# - In-between frames along optical flow: fast motion fires along its path
# - Event streams: sorted, inside the frame interval; .h5/.npz/.txt round
#   trips; nothing left behind on failure
# - Rendering, validation; events() and the CLI end to end
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from motionflow import VideoReader
from motionflow.cli import app
from motionflow.events import EventCamera, EventWriter, events, lin_log
from motionflow.video import Frame

H, W = 40, 60


def flat(value: float, h: int = H, w: int = W) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


def frames_of(images: list[np.ndarray], fps: float = 100.0) -> list[Frame]:
    return [Frame(i, i / fps, image) for i, image in enumerate(images)]


def run(camera: EventCamera, images: list[np.ndarray], fps: float = 100.0) -> list[tuple[int, int]]:
    """Run the camera over the images; (ON, OFF) totals per frame."""
    list(camera.process(frames_of(images, fps)))
    return camera.frame_counts


def read_stream(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path) as data:
            return {k: data[k] for k in ("t", "x", "y", "p")}
    if path.suffix == ".h5":
        import h5py

        with h5py.File(path) as f:
            return {k: f["events"][k][:] for k in ("t", "x", "y", "p")}
    rows = np.loadtxt(path, ndmin=2)
    return {"t": rows[:, 0], "x": rows[:, 1], "y": rows[:, 2], "p": rows[:, 3]}


def per_pixel(a: float, b: float, threshold: float = 0.2) -> int:
    """Events per pixel for a step from brightness a to b (ideal sensor)."""
    change = float(lin_log(np.array([b]))[0] - lin_log(np.array([a]))[0])
    return math.floor(abs(change) / threshold)


class TestLinLog:
    def test_log_above_linear_below(self):
        assert lin_log(np.array([100.0]))[0] == pytest.approx(math.log(100), rel=1e-6)
        assert lin_log(np.array([10.0]))[0] == pytest.approx(10 * math.log(20) / 20, rel=1e-6)

    def test_continuous_at_transition(self):
        assert lin_log(np.array([20.0]))[0] == pytest.approx(math.log(20), rel=1e-6)


class TestIdealModel:
    def test_no_change_no_events(self):
        assert run(EventCamera(), [flat(100)] * 5) == [(0, 0)] * 5

    def test_brightness_step_gives_exact_count(self):
        n = per_pixel(100, 150)  # log(1.5) = 0.405 -> 2
        assert run(EventCamera(0.2), [flat(100), flat(150)])[1] == (n * H * W, 0)

    def test_darkening_gives_off_events(self):
        n = per_pixel(100, 40)
        assert run(EventCamera(0.2), [flat(100), flat(40)])[1] == (0, n * H * W)

    def test_leftover_change_carries_over(self):
        # Two steps of 0.15 in log: none fires alone, but the second does (0.30).
        b1, b2 = 100 * math.exp(0.15), 100 * math.exp(0.30)
        totals = run(EventCamera(0.2), [flat(100), flat(round(b1)), flat(round(b2))])
        assert totals[1] == (0, 0)
        assert totals[2][0] == H * W

    def test_separate_thresholds(self):
        camera = EventCamera(on_threshold=0.1, off_threshold=1.0)
        assert run(camera, [flat(100), flat(150)])[1][0] == per_pixel(100, 150, 0.1) * H * W
        assert run(camera, [flat(100), flat(60)])[1] == (0, 0)

    def test_moving_edge_fires_only_at_the_edge(self, tmp_path):
        def edge(x):
            image = flat(50)
            image[:, x:] = 200
            return image

        out = tmp_path / "ev.npz"
        list(EventCamera(save_events=out).process(frames_of([edge(30), edge(25)])))
        stream = read_stream(out)
        assert set(np.unique(stream["x"])) == set(range(25, 30))
        assert np.all(stream["p"] == 1)  # dark -> bright where the edge moved


class TestSensorRealism:
    def test_presets(self):
        assert EventCamera(sensor="ideal").threshold_sigma == 0
        clean = EventCamera(sensor="clean")
        assert clean.threshold_sigma == 0.02 and clean.shot_noise_rate_hz == 0
        noisy = EventCamera(sensor="noisy")
        assert (noisy.threshold_sigma, noisy.cutoff_hz, noisy.leak_rate_hz) == (0.05, 30, 0.1)
        assert noisy.shot_noise_rate_hz == 5
        assert EventCamera(sensor="noisy", shot_noise_rate_hz=0).shot_noise_rate_hz == 0

    def test_threshold_spread_varies_between_pixels(self, tmp_path):
        out = tmp_path / "ev.npz"
        # A step right at the nominal threshold: with spread, some pixels fire, some don't.
        b = round(100 * math.exp(0.2))
        list(
            EventCamera(threshold_sigma=0.03, save_events=out).process(
                frames_of([flat(100), flat(b)])
            )
        )
        fired = len(read_stream(out)["t"])
        assert 0.1 * H * W < fired < 0.9 * H * W

    def test_seed_reproducible(self):
        b = round(100 * math.exp(0.2))
        images = [flat(100), flat(b)]
        a = run(EventCamera(threshold_sigma=0.03, seed=1), images)
        again = run(EventCamera(threshold_sigma=0.03, seed=1), images)
        other = run(EventCamera(threshold_sigma=0.03, seed=2), images)
        assert a == again
        assert a != other

    def test_bandwidth_limit_delays_and_dark_pixels_lag(self):
        # A step up in a bright and a dark half: with a low cutoff the events
        # spread over several frames, and the dark half responds more slowly.
        def scene(bright, dark):
            image = flat(0)
            image[:, : W // 2] = bright
            image[:, W // 2 :] = dark
            return image

        camera = EventCamera(cutoff_hz=20)
        images = [scene(100, 30)] + [scene(250, 75)] * 10
        list(camera.process(frames_of(images, fps=200)))
        totals = [on for on, _ in camera.frame_counts]
        assert totals[1] < sum(totals)  # not all at once
        # The ideal sensor fires everything at the step.
        ideal = run(EventCamera(), images, fps=200)
        assert sum(on for on, _ in ideal[2:]) == 0

    def test_leak_fires_on_events_in_a_still_scene(self):
        camera = EventCamera(leak_rate_hz=10.0)
        totals = run(camera, [flat(150)] * 101, fps=100)  # 1 s
        on = sum(on for on, _ in totals)
        off = sum(off for _, off in totals)
        assert off == 0
        assert 5 * H * W < on < 15 * H * W  # ~10 per pixel

    def test_shot_noise_rate(self):
        camera = EventCamera(shot_noise_rate_hz=20.0)
        totals = run(camera, [flat(128)] * 101, fps=100)  # 1 s
        events_per_pixel = sum(on + off for on, off in totals) / (H * W)
        intensity = (128 + 20) / 275
        expected = 20 * ((0.25 - 1) * intensity + 1)  # v2e's intensity factor
        assert events_per_pixel == pytest.approx(expected, rel=0.15)

    def test_refractory_limits_rate(self, tmp_path):
        out = tmp_path / "ev.npz"
        # A huge step (many crossings) but 50 ms refractory over a 10 ms frame.
        camera = EventCamera(0.05, refractory=0.05, save_events=out)
        list(camera.process(frames_of([flat(20), flat(250)], fps=100)))
        stream = read_stream(out)
        _, per_pixel_counts = np.unique(stream["y"] * W + stream["x"], return_counts=True)
        assert per_pixel_counts.max() == 1

    def test_refractory_keeps_change_pending(self):
        camera = EventCamera(0.05, refractory=0.025)
        totals = run(camera, [flat(20)] + [flat(250)] * 6, fps=100)
        # Fires in later frames too, as the refractory period runs out.
        assert sum(1 for on, _ in totals[2:] if on > 0) >= 2


class TestInterpolation:
    @staticmethod
    def square(x: int, w: int = 160, h: int = 60) -> np.ndarray:
        rng = np.random.default_rng(0)
        image = np.full((h, w, 3), 60, np.uint8)
        patch = rng.integers(150, 255, (20, 20, 1), dtype=np.uint8).repeat(3, axis=2)
        image[20:40, x : x + 20] = patch
        return image

    def test_fast_motion_fires_along_the_path(self, tmp_path):
        # A textured square jumps 40 px per frame: without interpolation the
        # events are only at the start and end; with it, also in between.
        images = [self.square(10), self.square(50), self.square(90)]
        plain, smooth = tmp_path / "plain.npz", tmp_path / "smooth.npz"
        list(EventCamera(save_events=plain).process(frames_of(images)))
        camera = EventCamera(interpolate="flow", save_events=smooth)
        list(camera.process(frames_of(images)))
        assert min(camera.steps_used) > 4  # about a pixel per step
        t_plain = np.unique(read_stream(plain)["t"])
        t_smooth = np.unique(read_stream(smooth)["t"])
        assert len(t_smooth) > 2 * len(t_plain)  # many more distinct times

    def test_fixed_upsample(self):
        camera = EventCamera(interpolate="flow", upsample=3)
        list(camera.process(frames_of([self.square(10), self.square(30)])))
        assert camera.steps_used == [3]

    def test_no_motion_needs_no_steps(self):
        camera = EventCamera(interpolate="flow")
        list(camera.process(frames_of([self.square(10)] * 3)))
        assert camera.steps_used == [1, 1]


class TestStream:
    @pytest.mark.parametrize("ext", [".npz", ".h5", ".txt"])
    def test_round_trip(self, tmp_path, ext):
        images = [flat(100), flat(150), flat(150), flat(60)]
        path = tmp_path / f"ev{ext}"
        camera = EventCamera(save_events=path)
        list(camera.process(frames_of(images, fps=100)))
        stream = read_stream(path)
        # The reference moves by whole thresholds, so the leftover of the ON
        # step counts towards the OFF events.
        on_each = per_pixel(100, 150)
        reference = float(lin_log(np.array([100.0]))[0]) + on_each * 0.2
        off_each = math.floor((reference - float(lin_log(np.array([60.0]))[0])) / 0.2)
        n_on, n_off = on_each * H * W, off_each * H * W
        assert len(stream["t"]) == n_on + n_off == camera.total_events
        assert int((stream["p"] == 1).sum()) == n_on
        assert set(np.unique(stream["x"])) == set(range(W))
        assert set(np.unique(stream["y"])) == set(range(H))
        t = stream["t"] / (1e6 if ext != ".txt" else 1)  # seconds
        assert np.all(np.diff(t) >= 0)  # sorted
        on_t, off_t = t[stream["p"] == 1], t[stream["p"] == 0]
        assert on_t.min() > 0.0 and on_t.max() <= 0.01 + 1e-6  # between frames 0 and 1
        assert off_t.min() > 0.02 and off_t.max() <= 0.03 + 1e-6  # between frames 2 and 3

    def test_multiple_crossings_are_spread_in_time(self, tmp_path):
        path = tmp_path / "ev.npz"
        list(EventCamera(0.1, save_events=path).process(frames_of([flat(100), flat(200)], fps=10)))
        t = read_stream(path)["t"]
        assert len(np.unique(t)) > 1
        assert t.max() <= 100_000  # microseconds, within the 0.1 s interval

    def test_failure_leaves_nothing_behind(self, tmp_path):
        path = tmp_path / "ev.h5"
        camera = EventCamera(save_events=path)

        def broken():
            yield from frames_of([flat(100), flat(150)])
            raise RuntimeError("decode failed")

        with pytest.raises(RuntimeError):
            list(camera.process(broken()))
        assert list(tmp_path.iterdir()) == []

    def test_unknown_format(self, tmp_path):
        with pytest.raises(ValueError, match="h5"):
            EventWriter(tmp_path / "ev.csv", W, H)

    def test_refuses_to_overwrite(self, tmp_path):
        path = tmp_path / "ev.npz"
        path.write_bytes(b"old")
        with pytest.raises(FileExistsError):
            EventWriter(path, W, H)


class TestRendering:
    def test_polarity_colours(self):
        images = list(EventCamera().process(frames_of([flat(100), flat(150), flat(60)])))
        assert images[0].max() == 0
        b, _, r = images[1][0, 0].astype(int)
        assert b > r  # ON is blue
        b, _, r = images[2][0, 0].astype(int)
        assert r > b  # OFF is red

    def test_count(self):
        images = list(EventCamera(render="count").process(frames_of([flat(100), flat(150)])))
        assert images[1].max() > 0

    def test_window_keeps_earlier_events(self):
        seq = [flat(100), flat(150), flat(150), flat(150)]
        single = list(EventCamera(window=1).process(frames_of(seq)))
        windowed = list(EventCamera(window=3).process(frames_of(seq)))
        assert single[3].max() == 0
        assert windowed[3].max() > 0

    def test_overlay_shows_the_scene(self):
        images = list(EventCamera(overlay=0.8).process(frames_of([flat(100), flat(100)])))
        assert images[1][0, 0, 0] == 100

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"threshold": 0},
            {"on_threshold": -1},
            {"sensor": "perfect"},
            {"threshold_sigma": -0.1},
            {"cutoff_hz": -1},
            {"refractory": -1},
            {"linear_below": 1},
            {"interpolate": "magic"},
            {"upsample": 0},
            {"max_upsample": 0},
            {"render": "rainbow"},
            {"window": 0},
            {"overlay": 2},
            {"full_count": 0},
            {"save_events": "ev.csv"},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            EventCamera(**kwargs)


class TestEndToEnd:
    def test_events(self, tiny_video, tmp_path):
        stream = tmp_path / "ev.npz"
        result = events(tiny_video, tmp_path / "out.mp4", save_events=stream, progress=False)
        assert result.frames_written == 20
        assert result.info["events"] > 0
        assert len(read_stream(stream)["t"]) == result.info["events"]
        with VideoReader(result.output) as reader:
            assert reader.width == 160

    def test_noisy_with_interpolation(self, tiny_video, tmp_path):
        result = events(
            tiny_video, tmp_path / "out.mp4", sensor="noisy", interpolate="flow", progress=False
        )
        assert result.info["sensor"] == "noisy"
        assert result.info["mean_steps"] > 1  # tiny_video moves 2 px/frame

    def test_events_refuse_to_overwrite_stream(self, tiny_video, tmp_path):
        stream = tmp_path / "ev.npz"
        stream.write_bytes(b"old")
        with pytest.raises(FileExistsError):
            events(tiny_video, tmp_path / "out.mp4", save_events=stream, progress=False)

    def test_cli(self, tiny_video, tmp_path):
        out = tmp_path / "out.mp4"
        args = [
            "events", str(tiny_video), "-o", str(out), "--save-events", "h5", "--window", "3",
            "--sensor", "noisy", "--interpolate", "flow", "--max-upsample", "4",
        ]  # fmt: skip
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert result.stdout.splitlines() == [str(out.with_suffix(".h5")), str(out)]
        assert read_stream(out.with_suffix(".h5"))["t"].size > 0

    @pytest.mark.parametrize(
        "args", [["--save-events", "csv"], ["--sensor", "perfect"], ["--interpolate", "magic"]]
    )
    def test_cli_bad_options(self, tiny_video, args):
        result = CliRunner().invoke(app, ["events", str(tiny_video), *args])
        assert result.exit_code == 2
