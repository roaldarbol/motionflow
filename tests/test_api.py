# Tests for the process_video Python API:
# - Output creation, naming and content (frame count, size)
# - Options: stride, normalisation, smoothing, viz styles, progress callback,
#   start/end, scale, codec, overwrite
# - Output safety: never overwrites the input or an existing file; leaves
#   nothing behind on errors or interrupts
# - Argument validation
from pathlib import Path

import pytest
from conftest import write_video

from motionflow import process_video
from motionflow.api import default_output_path
from motionflow.video import VideoReader


def count_frames(path: Path) -> int:
    with VideoReader(path) as reader:
        return sum(1 for _ in reader)


@pytest.fixture()
def clip(tmp_path) -> Path:
    """A 20-frame, 64x48 H.264 clip at 25 fps."""
    return write_video(tmp_path / "clip.mp4", n=20)


class TestProcessVideoBasic:
    def test_output_file_created(self, tiny_video, tmp_path):
        out = process_video(tiny_video, output=tmp_path / "out.mp4")
        assert out.exists()
        assert count_frames(out) == 19  # one flow per consecutive pair

    def test_default_output_path(self, tiny_video):
        out = process_video(tiny_video)
        assert out.name == f"{tiny_video.stem}_flow{tiny_video.suffix}"
        assert count_frames(out) == 19

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("rec.mp4", "rec_flow.mp4"),
            ("rec.MOV", "rec_flow.MOV"),
            ("rec.avi", "rec_flow.avi"),
            ("rec.ts", "rec_flow.mp4"),
            ("rec.h264", "rec_flow.mp4"),
            ("rec.wmv", "rec_flow.mp4"),
        ],
    )
    def test_default_output_name(self, name, expected):
        assert default_output_path(Path("/data") / name) == Path("/data") / expected

    def test_custom_output_path_respected(self, tiny_video, tmp_path):
        custom = tmp_path / "custom_name.mp4"
        out = process_video(tiny_video, output=custom)
        assert out == custom

    def test_returns_path_object(self, tiny_video, tmp_path):
        out = process_video(tiny_video, output=tmp_path / "out.mp4")
        assert isinstance(out, Path)

    @pytest.mark.parametrize(("fmt", "name"), [("mpegts", "clip.ts"), ("h264", "clip.h264")])
    def test_streams_that_cannot_seek(self, tmp_path, fmt, name):
        src = write_video(tmp_path / name, n=10, fmt=fmt)
        out = process_video(src)
        assert out.name == "clip_flow.mp4"
        assert count_frames(out) == 9


class TestProcessVideoOptions:
    def test_stride_2(self, tiny_video, tmp_path):
        out = process_video(tiny_video, output=tmp_path / "out.mp4", stride=2)
        assert out.exists()

    def test_global_normalize(self, tiny_video, tmp_path):
        out = process_video(tiny_video, output=tmp_path / "out.mp4", normalize="global")
        assert out.exists()

    def test_manual_max_flow(self, tiny_video, tmp_path):
        out = process_video(tiny_video, output=tmp_path / "out.mp4", max_flow=20.0)
        assert out.exists()

    def test_smooth(self, tiny_video, tmp_path):
        out = process_video(tiny_video, output=tmp_path / "out.mp4", smooth=3)
        assert out.exists()

    @pytest.mark.parametrize("viz", ["hsv", "heatmap", "quiver", "blend"])
    def test_viz_styles(self, tiny_video, tmp_path, viz):
        out = process_video(tiny_video, output=tmp_path / f"out_{viz}.mp4", viz=viz, blend=0.4)
        assert out.exists()

    def test_on_progress_callback(self, tiny_video, tmp_path):
        calls: list[tuple[int, int | None]] = []
        process_video(
            tiny_video,
            output=tmp_path / "out.mp4",
            on_progress=lambda current, total: calls.append((current, total)),
        )
        assert calls[-1] == (19, 19)

    def test_on_progress_unknown_total(self, tmp_path):
        src = write_video(tmp_path / "clip.h264", n=5, fmt="h264")
        calls: list[tuple[int, int | None]] = []
        process_video(src, on_progress=lambda current, total: calls.append((current, total)))
        assert calls == [(1, None), (2, None), (3, None), (4, None)]

    def test_start_and_end(self, clip, tmp_path):
        out = process_video(clip, output=tmp_path / "out.mp4", start="5f", end="15f")
        assert count_frames(out) == 9

    def test_scale(self, clip, tmp_path):
        out = process_video(clip, output=tmp_path / "out.mp4", scale=0.5)
        with VideoReader(out) as reader:
            assert (reader.width, reader.height) == (32, 24)

    @pytest.mark.parametrize(("codec", "name"), [("ffv1", "out.mkv"), ("mp4v", "out.mp4")])
    def test_codec(self, clip, tmp_path, codec, name):
        out = process_video(clip, output=tmp_path / name, codec=codec)
        assert count_frames(out) == 19


class TestOutputSafety:
    def test_refuses_to_overwrite_input(self, clip):
        before = clip.read_bytes()
        with pytest.raises(ValueError, match="overwrite the input"):
            process_video(clip, output=clip)
        assert clip.read_bytes() == before

    def test_refuses_to_overwrite_input_via_other_path(self, clip, tmp_path):
        (tmp_path / "sub").mkdir()
        before = clip.read_bytes()
        with pytest.raises(ValueError, match="overwrite the input"):
            process_video(clip, output=tmp_path / "sub" / ".." / clip.name, overwrite=True)
        assert clip.read_bytes() == before

    def test_refuses_to_overwrite_existing_output(self, clip, tmp_path):
        out = tmp_path / "out.mp4"
        out.write_bytes(b"precious")
        with pytest.raises(FileExistsError):
            process_video(clip, output=out)
        assert out.read_bytes() == b"precious"

    def test_overwrite(self, clip, tmp_path):
        out = tmp_path / "out.mp4"
        out.write_bytes(b"old")
        process_video(clip, output=out, overwrite=True)
        assert count_frames(out) == 19

    def test_single_frame_input_fails_cleanly(self, tmp_path):
        src = write_video(tmp_path / "one.mp4", n=1)
        with pytest.raises(ValueError, match="at least two"):
            process_video(src)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["one.mp4"]

    def test_interrupt_leaves_nothing_behind(self, clip, tmp_path):
        def interrupt(current, total):
            if current == 5:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            process_video(clip, output=tmp_path / "out.mp4", on_progress=interrupt)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4"]

    def test_interrupt_keeps_existing_output(self, clip, tmp_path):
        out = tmp_path / "out.mp4"
        out.write_bytes(b"old")

        def interrupt(current, total):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            process_video(clip, output=out, overwrite=True, on_progress=interrupt)
        assert out.read_bytes() == b"old"


class TestProcessVideoErrors:
    def test_nonexistent_input_raises(self, tmp_path):
        with pytest.raises(OSError):
            process_video(tmp_path / "does_not_exist.mp4")

    def test_invalid_method_raises(self, tiny_video, tmp_path):
        with pytest.raises(ValueError):
            process_video(tiny_video, output=tmp_path / "out.mp4", method="invalid")

    def test_invalid_viz_raises(self, tiny_video, tmp_path):
        with pytest.raises(ValueError):
            process_video(tiny_video, output=tmp_path / "out.mp4", viz="invalid")

    def test_unknown_output_extension(self, clip, tmp_path):
        with pytest.raises(ValueError, match="Use one of"):
            process_video(clip, output=tmp_path / "out.gif")

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
        ],
    )
    def test_invalid_arguments(self, clip, tmp_path, kwargs):
        with pytest.raises(ValueError):
            process_video(clip, output=tmp_path / "out.mp4", **kwargs)
        assert not (tmp_path / "out.mp4").exists()
