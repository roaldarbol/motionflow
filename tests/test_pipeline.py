# Tests for the pipeline (motionflow.pipeline):
# - run(): one output frame per input frame, in order, for any transform;
#   ranges and scaling; the Result; transforms that change the frame size
# - Output safety: never overwrites the input or an existing file; leaves
#   nothing behind on errors, interrupts, or when nothing is produced
# - IOOptions validation, default output names, progress reporting
# - prefetch(): errors from the background thread surface, and stopping
#   early stops the thread
import threading
from pathlib import Path

import numpy as np
import pytest
from conftest import frame_id, write_video

from motionflow.pipeline import (
    FrameTransform,
    IOOptions,
    Transform,
    default_output_path,
    prefetch,
    run,
)
from motionflow.video import VideoReader


class Identity(FrameTransform):
    name = "same"

    def __call__(self, frame):
        return frame.image


class Half(FrameTransform):
    """Halves the frame size, so the output differs from the input."""

    def __call__(self, frame):
        return np.ascontiguousarray(frame.image[::2, ::2])


class Nothing(Transform):
    def process(self, frames):
        for _ in frames:
            pass
        return iter(())


class Fails(FrameTransform):
    def __call__(self, frame):
        if frame.index == 3:
            raise RuntimeError("boom")
        return frame.image


def ids(path: Path) -> list[int]:
    with VideoReader(path) as reader:
        return [frame_id(image) for image in reader]


@pytest.fixture()
def clip(tmp_path) -> Path:
    return write_video(tmp_path / "clip.mp4", n=12)


class TestRun:
    def test_one_output_frame_per_input_frame(self, clip):
        result = run(Identity(), clip, progress=False)
        assert result.output == clip.with_name("clip_same.mp4")
        assert ids(result.output) == list(range(12))
        assert (result.frames_read, result.frames_written) == (12, 12)
        assert result.fps == 25
        assert result.encoder == "libx264"

    def test_range_and_scale(self, clip, tmp_path):
        io = IOOptions(start="3f", end="9f", scale=0.5)
        result = run(Identity(), clip, tmp_path / "out.mp4", io=io, progress=False)
        assert ids(result.output) == [3, 4, 5, 6, 7, 8]
        with VideoReader(result.output) as reader:
            assert (reader.width, reader.height) == (32, 24)

    def test_output_size_follows_the_transform(self, clip, tmp_path):
        result = run(Half(), clip, tmp_path / "out.mp4", progress=False)
        with VideoReader(result.output) as reader:
            assert (reader.width, reader.height) == (32, 24)

    def test_fps_override(self, clip, tmp_path):
        result = run(Identity(), clip, tmp_path / "out.mp4", io=IOOptions(fps=10), progress=False)
        with VideoReader(result.output) as reader:
            assert reader.fps == pytest.approx(10)

    def test_info_is_passed_through(self, clip, tmp_path):
        result = run(Identity(), clip, tmp_path / "o.mp4", progress=False, info={"a": 1})
        assert result.info == {"a": 1}

    def test_on_progress(self, clip, tmp_path):
        calls = []
        run(
            Identity(),
            clip,
            tmp_path / "out.mp4",
            progress=False,
            on_progress=lambda done, total: calls.append((done, total)),
        )
        assert calls == [(i, 12) for i in range(1, 13)]

    def test_on_progress_unknown_total(self, tmp_path):
        src = write_video(tmp_path / "clip.h264", n=4, fmt="h264")
        calls = []
        run(Identity(), src, progress=False, on_progress=lambda d, t: calls.append((d, t)))
        assert calls == [(1, None), (2, None), (3, None), (4, None)]

    @pytest.mark.parametrize(("fmt", "name"), [("mpegts", "clip.ts"), ("h264", "clip.h264")])
    def test_streams_that_cannot_seek(self, tmp_path, fmt, name):
        src = write_video(tmp_path / name, n=10, fmt=fmt)
        result = run(Identity(), src, progress=False)
        assert result.output.name == "clip_same.mp4"
        assert ids(result.output) == list(range(10))

    def test_transform_can_run_twice(self, clip, tmp_path):
        transform = Identity()
        run(transform, clip, tmp_path / "a.mp4", progress=False)
        run(transform, clip, tmp_path / "b.mp4", progress=False)
        assert ids(tmp_path / "b.mp4") == list(range(12))


class TestOutputSafety:
    def test_refuses_to_overwrite_input(self, clip):
        before = clip.read_bytes()
        with pytest.raises(ValueError, match="overwrite the input"):
            run(Identity(), clip, clip, progress=False)
        assert clip.read_bytes() == before

    def test_refuses_to_overwrite_input_via_other_path(self, clip, tmp_path):
        (tmp_path / "sub").mkdir()
        before = clip.read_bytes()
        with pytest.raises(ValueError, match="overwrite the input"):
            run(
                Identity(),
                clip,
                tmp_path / "sub" / ".." / clip.name,
                io=IOOptions(overwrite=True),
                progress=False,
            )
        assert clip.read_bytes() == before

    def test_refuses_to_overwrite_existing_output(self, clip, tmp_path):
        out = tmp_path / "out.mp4"
        out.write_bytes(b"precious")
        with pytest.raises(FileExistsError):
            run(Identity(), clip, out, progress=False)
        assert out.read_bytes() == b"precious"

    def test_overwrite(self, clip, tmp_path):
        out = tmp_path / "out.mp4"
        out.write_bytes(b"old")
        run(Identity(), clip, out, io=IOOptions(overwrite=True), progress=False)
        assert ids(out) == list(range(12))

    def test_error_in_transform_leaves_nothing_behind(self, clip, tmp_path):
        with pytest.raises(RuntimeError, match="boom"):
            run(Fails(), clip, tmp_path / "out.mp4", progress=False)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4"]

    def test_interrupt_leaves_nothing_behind(self, clip, tmp_path):
        def interrupt(done, total):
            if done == 5:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run(Identity(), clip, tmp_path / "out.mp4", progress=False, on_progress=interrupt)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4"]

    def test_interrupt_keeps_existing_output(self, clip, tmp_path):
        out = tmp_path / "out.mp4"
        out.write_bytes(b"old")

        def interrupt(done, total):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run(
                Identity(),
                clip,
                out,
                io=IOOptions(overwrite=True),
                progress=False,
                on_progress=interrupt,
            )
        assert out.read_bytes() == b"old"

    def test_no_output_frames(self, clip, tmp_path):
        with pytest.raises(ValueError, match="no frames"):
            run(Nothing(), clip, tmp_path / "out.mp4", progress=False)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4"]

    def test_missing_input(self, tmp_path):
        with pytest.raises(OSError, match="not found"):
            run(Identity(), tmp_path / "missing.mp4", progress=False)

    def test_unknown_output_extension(self, clip, tmp_path):
        with pytest.raises(ValueError, match="Use one of"):
            run(Identity(), clip, tmp_path / "out.gif", progress=False)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4"]


class TestIOOptions:
    @pytest.mark.parametrize(
        "kwargs", [{"scale": 0}, {"scale": -1}, {"fps": 0}, {"crf": -1}, {"start": "soon"}]
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            IOOptions(**kwargs)

    def test_times_are_parsed(self):
        io = IOOptions(start="1:30", end="120f")
        assert (io.start.value, io.start.unit) == (90.0, "s")
        assert (io.end.value, io.end.unit) == (120.0, "frame")


@pytest.mark.parametrize(
    ("name", "suffix", "expected"),
    [
        ("rec.mp4", "_flow", "rec_flow.mp4"),
        ("rec.MOV", "_flow", "rec_flow.MOV"),
        ("rec.avi", "_x", "rec_x.avi"),
        ("rec.ts", "_flow", "rec_flow.mp4"),
        ("rec.h264", "_flow", "rec_flow.mp4"),
        ("rec.wmv", "_flow", "rec_flow.mp4"),
    ],
)
def test_default_output_path(name, suffix, expected):
    assert default_output_path(Path("/data") / name, suffix) == Path("/data") / expected


class TestPrefetch:
    def test_yields_everything_in_order(self):
        assert list(prefetch(range(100), size=3)) == list(range(100))

    def test_error_in_background_is_raised(self):
        def items():
            yield 1
            raise OSError("decode failed")

        with pytest.raises(OSError, match="decode failed"):
            list(prefetch(items()))

    def test_stopping_early_stops_the_thread(self):
        def endless():
            i = 0
            while True:
                yield i
                i += 1

        source = prefetch(endless(), size=2)
        assert next(source) == 0
        source.close()
        assert not any(t.name == "motionflow-reader" for t in threading.enumerate())
