# Tests for the command-line interface (motionflow.cli):
# - Structure: subcommands, `motionflow VIDEO` as short for `motionflow flow
#   VIDEO`, -h, -V, python -m motionflow, the methods command
# - Output: paths on stdout (one per line), messages on stderr
# - Batch input: folders (skipping earlier outputs), patterns, -o as folder;
#   per-video errors and the exit code
# - Errors: existing output, output is input, bad times, Ctrl-C; error text
#   with square brackets survives Rich markup; warnings are one line
# - Options that have no effect warn
import re
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import write_video
from typer.testing import CliRunner

import motionflow.cli
from motionflow.cli import app
from motionflow.video import VideoReader

runner = CliRunner()


def strip(text: str) -> str:
    """Without colour codes (CI terminals get them)."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def stdout(result) -> str:
    return strip(result.stdout)


def stderr(result) -> str:
    return strip(result.stderr)


@pytest.fixture()
def clip(tmp_path) -> Path:
    return write_video(tmp_path / "clip.mp4", n=10)


@pytest.fixture()
def folder(tmp_path) -> Path:
    folder = tmp_path / "videos"
    folder.mkdir()
    write_video(folder / "a.mp4", n=6)
    write_video(folder / "b.mov", n=6)
    (folder / "notes.txt").write_text("not a video")
    return folder


class TestStructure:
    def test_help(self):
        result = runner.invoke(app, ["-h"])
        assert result.exit_code == 0
        assert "flow" in stdout(result) and "methods" in stdout(result)

    def test_flow_help(self):
        result = runner.invoke(app, ["flow", "--help"])
        assert result.exit_code == 0
        for option in ("--start", "--end", "--scale", "--crf", "--overwrite", "--overlay"):
            assert option in stdout(result)
        assert "[default:" in stdout(result)  # not swallowed as Rich markup

    def test_no_arguments_shows_help(self):
        result = runner.invoke(app, [])
        assert "Usage" in strip(result.output)

    @pytest.mark.parametrize("flag", ["--version", "-V"])
    def test_version(self, flag):
        result = runner.invoke(app, [flag])
        assert result.exit_code == 0
        assert stdout(result).startswith("motionflow ")

    def test_python_dash_m(self):
        result = subprocess.run(
            [sys.executable, "-m", "motionflow", "--version"], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert result.stdout.startswith("motionflow ")

    def test_methods(self):
        result = runner.invoke(app, ["methods"])
        assert result.exit_code == 0
        for method in ("farneback", "tvl1", "lucas-kanade", "raft"):
            assert method in stdout(result)


class TestRun:
    def test_short_form(self, clip):
        result = runner.invoke(app, [str(clip)])
        assert result.exit_code == 0, strip(result.output)
        assert stdout(result) == f"{clip.with_name('clip_flow.mp4')}\n"
        assert "Done:" in stderr(result)

    def test_explicit_command(self, clip, tmp_path):
        out = tmp_path / "out.mp4"
        result = runner.invoke(app, ["flow", str(clip), "-o", str(out)])
        assert result.exit_code == 0, strip(result.output)
        assert stdout(result) == f"{out}\n"

    def test_quiet(self, clip):
        result = runner.invoke(app, [str(clip), "-q"])
        assert result.exit_code == 0
        assert stderr(result) == ""
        assert stdout(result).strip().endswith("clip_flow.mp4")

    def test_range_and_scale(self, clip, tmp_path):
        out = tmp_path / "out.mp4"
        args = [str(clip), "-o", str(out), "--start", "2f", "--end", "8f", "--scale", "0.5"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, strip(result.output)
        with VideoReader(out) as reader:
            assert (reader.width, reader.height) == (32, 24)
            assert sum(1 for _ in reader) == 6

    def test_rendering_options(self, clip, tmp_path):
        args = [
            str(clip), "-o", str(tmp_path / "out.mp4"), "--viz", "quiver", "--overlay", "0.5",
            "--no-legend", "--normalize", "global", "--percentile", "95", "--min-flow", "0.2",
            "--quiver-scale", "4", "--quiver-step", "8",
        ]  # fmt: skip
        result = runner.invoke(app, args)
        assert result.exit_code == 0, strip(result.output)
        assert "Warning" not in stderr(result)

    def test_blend_option_is_gone(self, clip):
        result = runner.invoke(app, [str(clip), "--blend", "0.4"])
        assert result.exit_code == 2


class TestBatch:
    def test_folder(self, folder):
        result = runner.invoke(app, [str(folder)])
        assert result.exit_code == 0, strip(result.output)
        assert stdout(result).splitlines() == [
            str(folder / "a_flow.mp4"),
            str(folder / "b_flow.mov"),
        ]

    def test_folder_skips_earlier_outputs(self, folder):
        runner.invoke(app, [str(folder)])
        result = runner.invoke(app, [str(folder), "--overwrite"])
        assert result.exit_code == 0
        assert len(stdout(result).splitlines()) == 2  # not a_flow_flow.mp4

    def test_output_folder(self, folder, tmp_path):
        out = tmp_path / "results"
        result = runner.invoke(app, [str(folder), "-o", str(out)])
        assert result.exit_code == 0, strip(result.output)
        assert sorted(p.name for p in out.iterdir()) == ["a_flow.mp4", "b_flow.mov"]

    def test_pattern(self, folder, tmp_path):
        result = runner.invoke(app, [str(folder / "*.mov"), "-o", str(tmp_path / "out")])
        assert result.exit_code == 0, strip(result.output)
        assert stdout(result).splitlines() == [str(tmp_path / "out" / "b_flow.mov")]

    def test_output_without_extension_is_a_folder(self, clip, tmp_path):
        result = runner.invoke(app, [str(clip), "-o", str(tmp_path / "results")])
        assert result.exit_code == 0, strip(result.output)
        assert (tmp_path / "results" / "clip_flow.mp4").exists()

    def test_output_file_with_several_inputs(self, folder, tmp_path):
        result = runner.invoke(app, [str(folder), "-o", str(tmp_path / "one.mp4")])
        assert result.exit_code == 2
        assert "must be a folder" in strip(result.output)

    def test_one_failure_does_not_stop_the_rest(self, folder):
        (folder / "a_flow.mp4").write_bytes(b"precious")
        result = runner.invoke(app, [str(folder)])
        assert result.exit_code == 1
        assert stdout(result).splitlines() == [str(folder / "b_flow.mov")]
        assert "1 of 2 videos failed" in stderr(result)
        assert (folder / "a_flow.mp4").read_bytes() == b"precious"

    @pytest.mark.parametrize("args", [["nope.mp4"], ["*.nothing"]])
    def test_nothing_to_read(self, tmp_path, monkeypatch, args):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, args)
        assert result.exit_code == 2

    def test_empty_folder(self, tmp_path):
        result = runner.invoke(app, [str(tmp_path)])
        assert result.exit_code == 2
        assert "No videos" in strip(result.output)


class TestErrors:
    def test_existing_output(self, clip, tmp_path):
        out = tmp_path / "out.mp4"
        out.write_bytes(b"precious")
        result = runner.invoke(app, [str(clip), "-o", str(out)])
        assert result.exit_code == 1
        assert "already exists" in stderr(result)
        assert "--overwrite" in stderr(result)
        assert stdout(result) == ""
        assert out.read_bytes() == b"precious"

        result = runner.invoke(app, [str(clip), "-o", str(out), "--overwrite"])
        assert result.exit_code == 0, strip(result.output)
        assert out.read_bytes() != b"precious"

    def test_output_is_input(self, clip):
        result = runner.invoke(app, [str(clip), "-o", str(clip)])
        assert result.exit_code == 1
        assert "overwrite the input" in stderr(result)

    def test_bad_time(self, clip, tmp_path):
        result = runner.invoke(app, [str(clip), "-o", str(tmp_path / "o.mp4"), "--start", "soon"])
        assert result.exit_code == 1
        assert "Can't read 'soon' as a time" in stderr(result)

    def test_verbose_shows_traceback(self, clip, tmp_path):
        args = [str(clip), "-o", str(tmp_path / "o.mp4"), "--start", "soon", "-v"]
        result = runner.invoke(app, args)
        assert "Traceback" in stderr(result)

    def test_square_brackets_survive(self, clip, monkeypatch):
        def fail(*args, **kwargs):
            raise RuntimeError('Install with: pip install "motionflow[torch]"')

        monkeypatch.setattr(motionflow.cli, "flow", fail)
        result = runner.invoke(app, [str(clip)])
        assert result.exit_code == 1
        assert 'pip install "motionflow[torch]"' in stderr(result)

    def test_interrupt(self, clip, monkeypatch):
        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(motionflow.cli, "flow", interrupt)
        result = runner.invoke(app, [str(clip)])
        assert result.exit_code == 130
        assert "Interrupted" in stderr(result)

    def test_warnings_are_short(self, clip, tmp_path):
        # 64x48 at 0.53 is 34x25: odd height, so the writer pads and warns.
        args = [str(clip), "-o", str(tmp_path / "o.mp4"), "--scale", "0.53"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, strip(result.output)
        assert "Warning: libx264 needs even frame dimensions" in stderr(result)
        assert "UserWarning" not in stderr(result)


class TestUnusedOptions:
    @pytest.mark.parametrize(
        ("args", "flag"),
        [
            (["--quiver-step", "8"], "--quiver-step"),
            (["--quiver-scale", "2"], "--quiver-scale"),
            (["--device", "cpu"], "--device"),
            (["--model-size", "small"], "--model-size"),
            (["--max-flow", "5", "--percentile", "90"], "--percentile"),
            (["--normalize", "per-frame", "--percentile", "90"], "--percentile"),
        ],
    )
    def test_warns(self, clip, tmp_path, args, flag):
        result = runner.invoke(app, [str(clip), "-o", str(tmp_path / "o.mp4"), *args])
        assert result.exit_code == 0, strip(result.output)
        assert f"{flag} has no effect" in stderr(result)
