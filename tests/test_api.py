# Tests for the process_video Python API:
# - Output file is created with correct extension
# - Output path defaulting (stem + _flow)
# - Custom output path is respected
# - stride argument reduces processing (smoke test)
# - normalize="global" completes without error
# - on_progress callback is called
# - Invalid method raises ValueError
# - Non-existent input raises OSError
from pathlib import Path

import pytest

from motionflow import process_video

# MJPG/avi is used throughout because mp4v cannot write .mp4 on macOS.
_CODEC = "MJPG"


class TestProcessVideoBasic:
    def test_output_file_created(self, tiny_video, tmp_path):
        out = process_video(tiny_video, output=tmp_path / "out.avi", codec=_CODEC)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_default_output_path(self, tiny_video):
        out = process_video(tiny_video, codec=_CODEC)
        assert out.name == f"{tiny_video.stem}_flow{tiny_video.suffix}"
        out.unlink(missing_ok=True)

    def test_custom_output_path_respected(self, tiny_video, tmp_path):
        custom = tmp_path / "custom_name.avi"
        out = process_video(tiny_video, output=custom, codec=_CODEC)
        assert out == custom

    def test_returns_path_object(self, tiny_video, tmp_path):
        out = process_video(tiny_video, output=tmp_path / "out.avi", codec=_CODEC)
        assert isinstance(out, Path)


class TestProcessVideoOptions:
    def test_stride_2(self, tiny_video, tmp_path):
        out = process_video(tiny_video, output=tmp_path / "out.avi", stride=2, codec=_CODEC)
        assert out.exists()

    def test_global_normalize(self, tiny_video, tmp_path):
        out = process_video(
            tiny_video,
            output=tmp_path / "out.avi",
            normalize="global",
            codec=_CODEC,
        )
        assert out.exists()

    def test_manual_max_flow(self, tiny_video, tmp_path):
        out = process_video(
            tiny_video,
            output=tmp_path / "out.avi",
            max_flow=20.0,
            codec=_CODEC,
        )
        assert out.exists()

    def test_smooth(self, tiny_video, tmp_path):
        out = process_video(tiny_video, output=tmp_path / "out.avi", smooth=3, codec=_CODEC)
        assert out.exists()

    @pytest.mark.parametrize("viz", ["hsv", "heatmap", "quiver", "blend"])
    def test_viz_styles(self, tiny_video, tmp_path, viz):
        out = process_video(
            tiny_video,
            output=tmp_path / f"out_{viz}.avi",
            viz=viz,
            blend=0.4,
            codec=_CODEC,
        )
        assert out.exists()

    def test_on_progress_callback(self, tiny_video, tmp_path):
        calls: list[tuple[int, int]] = []

        def callback(current: int, total: int) -> None:
            calls.append((current, total))

        process_video(tiny_video, output=tmp_path / "out.avi", codec=_CODEC, on_progress=callback)
        assert len(calls) > 0
        assert all(c <= t for c, t in calls)


class TestProcessVideoErrors:
    def test_nonexistent_input_raises(self, tmp_path):
        with pytest.raises(OSError):
            process_video(tmp_path / "does_not_exist.avi")

    def test_invalid_method_raises(self, tiny_video, tmp_path):
        with pytest.raises(ValueError):
            process_video(tiny_video, output=tmp_path / "out.avi", method="invalid")

    def test_invalid_viz_raises(self, tiny_video, tmp_path):
        with pytest.raises(ValueError):
            process_video(tiny_video, output=tmp_path / "out.avi", viz="invalid")
