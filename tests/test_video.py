# Tests for video I/O (motionflow.video):
# - parse_time: seconds, clock time, frame numbers, errors
# - VideoReader: every container, including streams that can't seek (MPEG-TS,
#   raw H.264); frame order and content; fps, size and frame count; rotation;
#   start/end ranges; scaling; unreadable files
# - VideoWriter: codec per extension, aliases, round trips; odd sizes;
#   overwrite protection; nothing left behind on errors or zero frames
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from conftest import CAN_WRITE_ROTATION, frame_id, has_encoder, id_frame, write_video

from motionflow.video import TimePoint, VideoReader, VideoWriter, parse_time

# (file name, encoder, FFmpeg format)
CONTAINERS = [
    ("clip.mp4", "libx264", None),
    ("clip.mov", "libx264", None),
    ("clip.mkv", "ffv1", None),
    ("clip.avi", "mjpeg", None),
    pytest.param(
        "clip.webm",
        "libvpx-vp9",
        None,
        marks=pytest.mark.skipif(not has_encoder("libvpx-vp9"), reason="no VP9 encoder"),
    ),
    ("clip.ts", "libx264", "mpegts"),
    ("clip.h264", "libx264", "h264"),
]


def read_ids(path: Path, **kwargs) -> list[int]:
    with VideoReader(path, **kwargs) as reader:
        return [frame_id(image) for image in reader]


class TestParseTime:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("12.5", TimePoint(12.5, "s")),
            ("12.5s", TimePoint(12.5, "s")),
            (3, TimePoint(3.0, "s")),
            ("1:30", TimePoint(90.0, "s")),
            ("01:02:03.5", TimePoint(3723.5, "s")),
            ("120f", TimePoint(120.0, "frame")),
            ("0f", TimePoint(0.0, "frame")),
        ],
    )
    def test_formats(self, text, expected):
        assert parse_time(text) == expected

    @pytest.mark.parametrize("text", ["abc", "1:2:3:4", "-5", "f", "12.5f"])
    def test_invalid(self, text):
        with pytest.raises(ValueError):
            parse_time(text)


class TestVideoReader:
    @pytest.mark.parametrize(("name", "codec", "fmt"), CONTAINERS)
    def test_reads_every_frame_in_order(self, tmp_path, name, codec, fmt):
        path = write_video(tmp_path / name, n=12, codec=codec, fmt=fmt)
        with VideoReader(path) as reader:
            assert (reader.width, reader.height) == (64, 48)
            assert reader.fps == pytest.approx(25)
            frames = list(reader.frames())
        assert [frame_id(f.image) for f in frames] == list(range(12))
        assert [f.index for f in frames] == list(range(12))
        assert [f.time for f in frames] == pytest.approx([i / 25 for i in range(12)], abs=1e-3)

    @pytest.mark.parametrize(("name", "codec", "fmt"), CONTAINERS)
    def test_frame_count(self, tmp_path, name, codec, fmt):
        path = write_video(tmp_path / name, n=12, codec=codec, fmt=fmt)
        with VideoReader(path) as reader:
            count = reader.frame_count
        if name.endswith(".h264"):
            assert count is None  # raw H.264 records neither count nor duration
        else:
            assert count == 12

    def test_frame_image_format(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4")
        with VideoReader(path) as reader:
            image = next(iter(reader))
        assert image.dtype == np.uint8
        assert image.shape == (48, 64, 3)

    def test_ntsc_rate_is_exact(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", fps=Fraction(30000, 1001))
        with VideoReader(path) as reader:
            assert reader.rate == Fraction(30000, 1001)

    def test_odd_size(self, tmp_path):
        path = write_video(tmp_path / "clip.mkv", width=65, height=49, codec="ffv1")
        with VideoReader(path) as reader:
            image = next(iter(reader))
        assert image.shape == (49, 65, 3)

    def test_single_frame(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", n=1)
        assert read_ids(path) == [0]

    @pytest.mark.skipif(not CAN_WRITE_ROTATION, reason="this PyAV can't write rotation")
    @pytest.mark.parametrize("rotation", [90, -90, 180])
    def test_rotation_is_applied(self, tmp_path, rotation):
        path = write_video(tmp_path / "clip.mp4", rotation=rotation)
        with VideoReader(path) as reader:
            size = (reader.width, reader.height)
            image = next(iter(reader))
        expected = np.rot90(id_frame(0), round(rotation / 90) % 4)
        assert size == (expected.shape[1], expected.shape[0])
        assert image.shape == expected.shape

    def test_start_and_end_frames(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", n=20)
        assert read_ids(path, start="5f", end="9f") == [5, 6, 7, 8]

    def test_start_and_end_seconds(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", n=20)  # 25 fps
        assert read_ids(path, start=0.2, end=0.4) == [5, 6, 7, 8, 9]

    @pytest.mark.parametrize(("name", "codec", "fmt"), CONTAINERS)
    def test_start_after_seek_threshold(self, tmp_path, name, codec, fmt):
        # Starting later than 5 s seeks; frames and indices must still be exact.
        path = write_video(tmp_path / name, n=200, codec=codec, fmt=fmt)  # 8 s
        with VideoReader(path, start=6.0, end="153f") as reader:
            frames = list(reader.frames())
            count = reader.frame_count
        assert [frame_id(f.image) for f in frames] == [150, 151, 152]
        assert [f.index for f in frames] == [150, 151, 152]
        if count is not None:
            assert count == 3

    def test_start_past_end(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", n=5)
        with pytest.raises(OSError, match="No frames"):
            VideoReader(path, start="10f").__enter__()

    def test_scale(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", width=64, height=48)
        with VideoReader(path, scale=0.5) as reader:
            assert (reader.width, reader.height) == (32, 24)
            image = next(iter(reader))
        assert image.shape == (24, 32, 3)
        assert frame_id(image) == 0

    def test_can_only_iterate_once(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4")
        with VideoReader(path) as reader:
            list(reader)
            with pytest.raises(RuntimeError, match="once"):
                list(reader)

    def test_missing_file(self, tmp_path):
        with pytest.raises(OSError, match="not found"):
            VideoReader(tmp_path / "missing.mp4").__enter__()

    @pytest.mark.parametrize("content", [b"", b"not a video at all"])
    def test_unreadable_file(self, tmp_path, content):
        path = tmp_path / "broken.mp4"
        path.write_bytes(content)
        with pytest.raises(OSError):
            VideoReader(path).__enter__()


class TestVideoWriter:
    @pytest.mark.parametrize(
        ("name", "encoder"),
        [
            ("out.mp4", "libx264"),
            ("out.mov", "libx264"),
            ("out.mkv", "libx264"),
            # VP9, or AV1 where the FFmpeg build has no VP9 encoder
            ("out.webm", {"libvpx-vp9", "libsvtav1", "libaom-av1"}),
            ("out.avi", "mjpeg"),
        ],
    )
    def test_round_trip(self, tmp_path, name, encoder):
        path = tmp_path / name
        with VideoWriter(path, 25, 64, 48) as writer:
            assert writer.encoder in ({encoder} if isinstance(encoder, str) else encoder)
            for i in range(10):
                writer.write(id_frame(i))
        assert read_ids(path) == list(range(10))
        with VideoReader(path) as reader:
            assert reader.fps == pytest.approx(25)

    @pytest.mark.parametrize(
        ("codec", "encoder"),
        [("mp4v", "mpeg4"), ("MJPG", "mjpeg"), ("avc1", "libx264"), ("ffv1", "ffv1")],
    )
    def test_codec_aliases(self, tmp_path, codec, encoder):
        path = tmp_path / ("out.avi" if encoder in ("mjpeg", "ffv1") else "out.mp4")
        with VideoWriter(path, 25, 64, 48, codec=codec) as writer:
            assert writer.encoder == encoder
            writer.write(id_frame(3))
        assert read_ids(path) == [3]

    def test_ffv1_is_lossless(self, tmp_path):
        path = tmp_path / "out.mkv"
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        with VideoWriter(path, 25, 64, 48, codec="ffv1") as writer:
            writer.write(frame)
        with VideoReader(path) as reader:
            np.testing.assert_array_equal(next(iter(reader)), frame)

    def test_ntsc_rate(self, tmp_path):
        path = tmp_path / "out.mp4"
        with VideoWriter(path, 29.97002997, 64, 48) as writer:
            writer.write(id_frame(0))
        with VideoReader(path) as reader:
            assert reader.rate == Fraction(30000, 1001)

    def test_odd_size_is_padded(self, tmp_path):
        path = tmp_path / "out.mp4"
        with (
            pytest.warns(UserWarning, match="padding 65x49 to 66x50"),
            VideoWriter(path, 25, 65, 49) as writer,
        ):
            writer.write(id_frame(5, 65, 49))
        with VideoReader(path) as reader:
            assert (reader.width, reader.height) == (66, 50)
            assert frame_id(next(iter(reader))[:, :65]) == 5

    @pytest.mark.parametrize(
        ("ext", "codec", "low", "high"), [(".mp4", "h264", 10, 40), (".avi", "mjpeg", 2, 31)]
    )
    def test_lower_crf_is_bigger(self, tmp_path, ext, codec, low, high):
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (48, 64, 3), dtype=np.uint8) for _ in range(5)]
        sizes = []
        for crf in (low, high):
            path = tmp_path / f"crf{crf}{ext}"
            with VideoWriter(path, 25, 64, 48, codec=codec, crf=crf) as writer:
                for frame in frames:
                    writer.write(frame)
            sizes.append(path.stat().st_size)
        assert sizes[0] > sizes[1]

    @pytest.mark.parametrize("name", ["out.gif", "out.txt", "out"])
    def test_unknown_extension(self, tmp_path, name):
        with pytest.raises(ValueError, match="Use one of"):
            VideoWriter(tmp_path / name, 25, 64, 48)

    def test_unknown_codec(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown codec"):
            VideoWriter(tmp_path / "out.mp4", 25, 64, 48, codec="nonsense")

    @pytest.mark.parametrize("fps", [0, -5])
    def test_invalid_fps(self, tmp_path, fps):
        with pytest.raises(ValueError, match="fps"):
            VideoWriter(tmp_path / "out.mp4", fps, 64, 48)

    def test_wrong_frame_shape(self, tmp_path):
        with VideoWriter(tmp_path / "out.mp4", 25, 64, 48) as writer:
            writer.write(id_frame(0))
            with pytest.raises(ValueError, match="shape"):
                writer.write(np.zeros((10, 10, 3), np.uint8))

    def test_refuses_to_overwrite(self, tmp_path):
        path = tmp_path / "out.mp4"
        path.write_bytes(b"precious")
        with pytest.raises(FileExistsError):
            VideoWriter(path, 25, 64, 48).__enter__()
        assert path.read_bytes() == b"precious"

    def test_overwrite(self, tmp_path):
        path = tmp_path / "out.mp4"
        path.write_bytes(b"old")
        with VideoWriter(path, 25, 64, 48, overwrite=True) as writer:
            writer.write(id_frame(1))
        assert read_ids(path) == [1]

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "a" / "b" / "out.mp4"
        with VideoWriter(path, 25, 64, 48) as writer:
            writer.write(id_frame(0))
        assert path.exists()

    def test_error_leaves_nothing_behind(self, tmp_path):
        path = tmp_path / "out.mp4"
        with pytest.raises(KeyboardInterrupt), VideoWriter(path, 25, 64, 48) as writer:
            writer.write(id_frame(0))
            raise KeyboardInterrupt
        assert list(tmp_path.iterdir()) == []

    def test_error_keeps_existing_file(self, tmp_path):
        path = tmp_path / "out.mp4"
        path.write_bytes(b"old")
        with (
            pytest.raises(RuntimeError, match="boom"),
            VideoWriter(path, 25, 64, 48, overwrite=True) as writer,
        ):
            writer.write(id_frame(0))
            raise RuntimeError("boom")
        assert path.read_bytes() == b"old"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mp4"]

    def test_zero_frames_is_an_error(self, tmp_path):
        with (
            pytest.raises(RuntimeError, match="No frames"),
            VideoWriter(tmp_path / "out.mp4", 25, 64, 48),
        ):
            pass
        assert list(tmp_path.iterdir()) == []
