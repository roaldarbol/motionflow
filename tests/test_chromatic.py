# Tests for chromatic motion (motionflow.chromatic, BehaveAI's encoding):
# - A still scene stays grey: no colour, only the dimmed greyscale frame
# - A move shows in blue first; afterwards the trail ages to green and red
# - Both strategies; tail_only separates the bands; motion_only is black
#   where nothing moves; threshold, gains, frame_skip; validation
# - chromatic() and the CLI command end to end
from pathlib import Path

import numpy as np
import pytest
from conftest import write_video
from typer.testing import CliRunner

from motionflow import VideoReader
from motionflow.chromatic import ChromaticMotion, chromatic
from motionflow.cli import app
from motionflow.video import Frame

H, W = 60, 80


def square_frame(x: int | None) -> np.ndarray:
    """Grey background (100) with a bright 20x20 square at column x (none if None)."""
    image = np.full((H, W, 3), 100, np.uint8)
    if x is not None:
        image[20:40, x : x + 20] = 220
    return image


def run_frames(transform: ChromaticMotion, positions: list[int | None]) -> list[np.ndarray]:
    frames = [Frame(i, i / 25, square_frame(x)) for i, x in enumerate(positions)]
    return [image.astype(int) for image in transform.process(frames)]


B, G, R = 0, 1, 2


class TestEncoding:
    def test_still_scene_is_grey(self):
        out = run_frames(ChromaticMotion(), [10] * 6)
        assert len(out) == 6
        for image in out:
            # No differences: every channel is the dimmed grey value.
            assert np.all(image[..., B] == image[..., G])
            assert np.all(image[..., G] == image[..., R])
        assert out[0][0, 0, B] == round(100 * 0.7)

    def test_first_frame_shows_no_motion(self):
        out = run_frames(ChromaticMotion(), [10, 40])
        assert np.all(out[0][..., B] == out[0][..., R])

    def test_motion_is_blue_first(self):
        # The square jumps from x=10 to x=40 at frame 3 (sequential: refs are
        # the last three frames, so all three differ at that instant).
        out = run_frames(ChromaticMotion("sequential", gain=1.0, lum_weight=0), [10, 10, 10, 40])
        where = out[3][30, 50]  # newly covered by the square
        assert where[B] > 0 and where[G] > 0 and where[R] > 0

    def test_trail_ages_from_blue_to_red(self):
        # Moves once at frame 2, then stays still: blue fades first, red last.
        positions = [10, 10, 40, 40, 40, 40, 40, 40]
        out = run_frames(ChromaticMotion(gain=1.0, lum_weight=0), positions)
        pixel = [image[30, 50] for image in out]  # inside the square's new position
        assert pixel[2][B] > 0  # the move itself
        assert pixel[3][B] == 0  # one frame later, nothing new
        assert pixel[3][G] > 0 and pixel[3][R] > 0  # but the averages still remember
        assert pixel[7][R] > pixel[7][G]  # red (slowest to fade) lasts longest

    def test_tail_only_separates_bands(self):
        positions = [10, 10, 40, 40, 40]
        plain = run_frames(ChromaticMotion(gain=1.0, lum_weight=0), positions)
        tail = run_frames(ChromaticMotion(gain=1.0, lum_weight=0, tail_only=True), positions)
        # At the move, blue (recent) minus green (medium) is smaller than blue.
        assert tail[2][30, 50][B] <= plain[2][30, 50][B]
        assert np.all(tail[2] >= 0)

    def test_motion_only_is_black_when_still(self):
        out = run_frames(ChromaticMotion(lum_weight=0), [10] * 4)
        assert all(image.max() == 0 for image in out)

    def test_threshold_hides_small_differences(self):
        positions = [10, 10, 40]
        out = run_frames(ChromaticMotion(gain=1.0, lum_weight=0, threshold=500), positions)
        assert out[2].max() == 0

    def test_per_channel_gains(self):
        positions = [10, 10, 10, 40]
        out = run_frames(
            ChromaticMotion("sequential", gain=(1.0, 0.0, 0.0), lum_weight=0), positions
        )
        assert out[3][..., B].max() > 0
        assert out[3][..., G].max() == 0 and out[3][..., R].max() == 0

    def test_frame_skip(self):
        out = run_frames(ChromaticMotion(frame_skip=2), [10] * 9)
        assert len(out) == 3  # frames 0, 3, 6

    def test_can_run_twice(self):
        transform = ChromaticMotion()
        first = run_frames(transform, [10, 40, 40])
        second = run_frames(transform, [10, 40, 40])
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"strategy": "random"},
            {"exp_a": 1.0},
            {"exp_b": -0.1},
            {"lum_weight": -1},
            {"gain": (1, 2)},
            {"gain": -1},
            {"threshold": -1},
            {"frame_skip": -1},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            ChromaticMotion(**kwargs)


class TestEndToEnd:
    def test_chromatic(self, tiny_video, tmp_path):
        result = chromatic(tiny_video, progress=False)
        assert result.output == tiny_video.with_name("test_input_chromatic.mp4")
        assert result.frames_written == 20
        with VideoReader(result.output) as reader:
            images = list(reader)
        # tiny_video moves: colour (channel differences) appears after frame 0.
        spread = [int(np.ptp(image.astype(int), axis=2).max()) for image in images]
        assert spread[0] < 10 and max(spread[1:]) > 50

    def test_motion_only_and_frame_skip(self, tiny_video, tmp_path):
        result = chromatic(
            tiny_video, tmp_path / "out.mp4", motion_only=True, frame_skip=1, progress=False
        )
        assert result.frames_written == 10

    def test_cli(self, tiny_video, tmp_path):
        out = tmp_path / "out.mp4"
        args = ["chromatic", str(tiny_video), "-o", str(out), "--gain", "2,4,6", "--tail-only"]
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert result.stdout.strip() == str(out)

    def test_cli_bad_gain(self, tiny_video):
        result = CliRunner().invoke(app, ["chromatic", str(tiny_video), "--gain", "1,2"])
        assert result.exit_code == 2

    def test_cli_folder_skips_own_outputs(self, tmp_path):
        folder = tmp_path / "videos"
        folder.mkdir()
        write_video(folder / "a.mp4", n=4)
        runner = CliRunner()
        runner.invoke(app, ["chromatic", str(folder)])
        result = runner.invoke(app, ["chromatic", str(folder), "--overwrite"])
        assert result.stdout.splitlines() == [str(Path(folder) / "a_chromatic.mp4")]
