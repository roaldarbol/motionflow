# Tests for the command-line interface:
# - Help, version, a successful run, the new I/O options
# - Error reporting: existing output, bad arguments, Ctrl-C; error text with
#   square brackets survives Rich markup
# - Warnings are shown as short messages
from pathlib import Path

import pytest
from conftest import write_video
from typer.testing import CliRunner

import motionflow.cli
from motionflow.cli import app
from motionflow.video import VideoReader

runner = CliRunner()


@pytest.fixture()
def clip(tmp_path) -> Path:
    return write_video(tmp_path / "clip.mp4", n=10)


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for option in ("--start", "--end", "--scale", "--crf", "--overwrite", "--codec"):
        assert option in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("motionflow ")


def test_run(clip):
    result = runner.invoke(app, [str(clip)])
    assert result.exit_code == 0, result.output
    assert "Done." in result.output
    assert (clip.parent / "clip_flow.mp4").exists()


def test_range_and_scale(clip, tmp_path):
    out = tmp_path / "out.mp4"
    result = runner.invoke(
        app, [str(clip), "-o", str(out), "--start", "2f", "--end", "8f", "--scale", "0.5"]
    )
    assert result.exit_code == 0, result.output
    with VideoReader(out) as reader:
        assert (reader.width, reader.height) == (32, 24)
        assert sum(1 for _ in reader) == 5


def test_existing_output(clip, tmp_path):
    out = tmp_path / "out.mp4"
    out.write_bytes(b"precious")
    result = runner.invoke(app, [str(clip), "-o", str(out)])
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert "--overwrite" in result.output
    assert out.read_bytes() == b"precious"

    result = runner.invoke(app, [str(clip), "-o", str(out), "--overwrite"])
    assert result.exit_code == 0, result.output
    assert out.read_bytes() != b"precious"


def test_output_is_input(clip):
    result = runner.invoke(app, [str(clip), "-o", str(clip)])
    assert result.exit_code == 1
    assert "overwrite the input" in result.output


def test_bad_time(clip, tmp_path):
    result = runner.invoke(app, [str(clip), "-o", str(tmp_path / "o.mp4"), "--start", "soon"])
    assert result.exit_code == 1
    assert "Can't read 'soon' as a time" in result.output


def test_square_brackets_survive(clip, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError('Install with: pip install "motionflow[torch]"')

    monkeypatch.setattr(motionflow.cli, "process_video", fail)
    result = runner.invoke(app, [str(clip)])
    assert result.exit_code == 1
    assert 'pip install "motionflow[torch]"' in result.output


def test_interrupt(clip, monkeypatch):
    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(motionflow.cli, "process_video", interrupt)
    result = runner.invoke(app, [str(clip)])
    assert result.exit_code == 130
    assert "Interrupted" in result.output


def test_warnings_are_short(clip, tmp_path):
    # 64x48 at 0.53 is 34x25: odd height, so the writer pads and warns.
    result = runner.invoke(app, [str(clip), "-o", str(tmp_path / "o.mp4"), "--scale", "0.53"])
    assert result.exit_code == 0, result.output
    assert "Warning: libx264 needs even frame dimensions" in result.output
    assert "UserWarning" not in result.output
