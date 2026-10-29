# Tests for the deprecated process_video API (motionflow.api):
# - It still works, returns a Path, and warns that it is deprecated
# - It passes its arguments through to flow()
from pathlib import Path

import pytest

from motionflow import process_video
from motionflow.video import VideoReader


def test_works_and_warns(tiny_video, tmp_path):
    with pytest.warns(DeprecationWarning, match="motionflow.flow"):
        out = process_video(tiny_video, output=tmp_path / "out.mp4")
    assert isinstance(out, Path)
    with VideoReader(out) as reader:
        assert sum(1 for _ in reader) == 20


def test_default_output(tiny_video):
    with pytest.warns(DeprecationWarning):
        out = process_video(tiny_video)
    assert out.name == "test_input_flow.mp4"


def test_arguments_pass_through(tiny_video, tmp_path):
    with pytest.warns(DeprecationWarning):
        out = process_video(
            tiny_video, output=tmp_path / "out.mkv", codec="ffv1", start="5f", scale=0.5
        )
    with VideoReader(out) as reader:
        assert (reader.width, reader.height) == (80, 60)
        assert sum(1 for _ in reader) == 15


def test_errors_pass_through(tiny_video, tmp_path):
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
        process_video(tiny_video, output=tmp_path / "out.mp4", stride=0)


@pytest.mark.parametrize(("viz", "blend"), [("blend", 0.3), ("hsv", 0.4), ("blend", 0.0)])
def test_old_blend_options(tiny_video, tmp_path, viz, blend):
    with pytest.warns(DeprecationWarning):
        out = process_video(tiny_video, output=tmp_path / "out.mp4", viz=viz, blend=blend)
    assert out.exists()
