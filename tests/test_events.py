# Tests for event-camera simulation (motionflow.events):
# - The model: no events without change; a brightness step gives exactly
#   floor(delta log / C) events of the right polarity; leftover change
#   carries over; a moving edge fires only at the edge
# - Event streams: timestamps sorted and inside the frame interval; .h5,
#   .npz and .txt round trips; nothing left behind on failure
# - Rendering (polarity, count, window, overlay); validation; events() and
#   the CLI end to end
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from motionflow import VideoReader
from motionflow.cli import app
from motionflow.events import EventCamera, EventWriter, events
from motionflow.video import Frame

H, W = 40, 60


def flat(value: float) -> np.ndarray:
    return np.full((H, W, 3), value, np.uint8)


def frames_of(images: list[np.ndarray], fps: float = 100.0) -> list[Frame]:
    return [Frame(i, i / fps, image) for i, image in enumerate(images)]


def counts(camera: EventCamera, images: list[np.ndarray]) -> list[tuple[int, int]]:
    """(ON, OFF) totals per frame, read from the camera's internal model."""
    totals = []
    reference = None
    for image in images:
        gray = image[..., 0].astype(np.float32)
        log = np.log(gray + camera.log_offset)
        if reference is None:
            reference = log
            totals.append((0, 0))
            continue
        on, off, reference = camera._fire(log, reference)
        totals.append((int(on.sum()), int(off.sum())))
    return totals


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


class TestModel:
    def test_no_change_no_events(self):
        assert counts(EventCamera(), [flat(100)] * 5) == [(0, 0)] * 5

    def test_brightness_step_gives_exact_count(self):
        # log((150 + 10) / (100 + 10)) = 0.375 -> one ON event per pixel at C = 0.2
        camera = EventCamera(0.2)
        expected = math.floor(math.log(160 / 110) / 0.2)
        assert counts(camera, [flat(100), flat(150)])[1] == (expected * H * W, 0)

    def test_darkening_gives_off_events(self):
        camera = EventCamera(0.2)
        expected = math.floor(math.log(110 / 50) / 0.2)  # 100 -> 40
        assert counts(camera, [flat(100), flat(40)])[1] == (0, expected * H * W)

    def test_leftover_change_carries_over(self):
        # Two steps of 0.15 each: none fires alone, but the second does (0.30).
        camera = EventCamera(0.2, log_offset=10)
        b0 = 100.0
        b1 = (b0 + 10) * math.exp(0.15) - 10
        b2 = (b0 + 10) * math.exp(0.30) - 10
        totals = counts(camera, [flat(b0), flat(round(b1)), flat(round(b2))])
        assert totals[1] == (0, 0)
        assert totals[2][0] == H * W

    def test_separate_thresholds(self):
        camera = EventCamera(on_threshold=0.1, off_threshold=1.0)
        assert counts(camera, [flat(100), flat(150)])[1][0] == 3 * H * W
        assert counts(camera, [flat(100), flat(60)])[1] == (0, 0)

    def test_moving_edge_fires_only_at_the_edge(self, tmp_path):
        def edge(x):
            image = flat(50)
            image[:, x:] = 200
            return image

        out = tmp_path / "ev.npz"
        camera = EventCamera(save_events=out)
        list(camera.process(frames_of([edge(30), edge(25)])))
        stream = read_stream(out)
        assert set(np.unique(stream["x"])) == set(range(25, 30))
        assert np.all(stream["p"] == 1)  # dark -> bright where the edge moved


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
        on_each = math.floor(math.log(160 / 110) / 0.2)
        reference = math.log(110) + on_each * 0.2
        off_each = math.floor((reference - math.log(70)) / 0.2)
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
        camera = EventCamera(0.1, save_events=path)
        list(camera.process(frames_of([flat(100), flat(200)], fps=10)))
        t = read_stream(path)["t"]
        assert len(np.unique(t)) > 1  # several crossings, several times
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
        assert images[0].max() == 0  # no events on the first frame
        b, g, r = images[1][0, 0].astype(int)
        assert b > r  # ON is blue
        b, g, r = images[2][0, 0].astype(int)
        assert r > b  # OFF is red

    def test_count(self):
        images = list(EventCamera(render="count").process(frames_of([flat(100), flat(150)])))
        assert images[1].max() > 0

    def test_window_keeps_earlier_events(self):
        seq = [flat(100), flat(150), flat(150), flat(150)]
        single = list(EventCamera(window=1).process(frames_of(seq)))
        windowed = list(EventCamera(window=3).process(frames_of(seq)))
        assert single[3].max() == 0  # nothing new in frame 3
        assert windowed[3].max() > 0  # still shows frame 1's events

    def test_overlay_shows_the_scene(self):
        images = list(EventCamera(overlay=0.8).process(frames_of([flat(100), flat(100)])))
        assert images[1][0, 0, 0] == 100  # no events: the greyscale frame

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"threshold": 0},
            {"on_threshold": -1},
            {"log_offset": 0},
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

    def test_events_refuse_to_overwrite_stream(self, tiny_video, tmp_path):
        stream = tmp_path / "ev.npz"
        stream.write_bytes(b"old")
        with pytest.raises(FileExistsError):
            events(tiny_video, tmp_path / "out.mp4", save_events=stream, progress=False)

    def test_cli(self, tiny_video, tmp_path):
        out = tmp_path / "out.mp4"
        args = ["events", str(tiny_video), "-o", str(out), "--save-events", "h5", "--window", "3"]
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert result.stdout.splitlines() == [str(out.with_suffix(".h5")), str(out)]
        assert read_stream(out.with_suffix(".h5"))["t"].size > 0

    def test_cli_bad_format(self, tiny_video):
        result = CliRunner().invoke(app, ["events", str(tiny_video), "--save-events", "csv"])
        assert result.exit_code == 2
