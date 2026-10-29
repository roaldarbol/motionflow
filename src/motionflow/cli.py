"""The ``motionflow`` command line.

``motionflow VIDEO`` is short for ``motionflow flow VIDEO``. Output paths go to
stdout, one per line, and everything else (progress, messages, errors) to
stderr, so ``out=$(motionflow video.mp4)`` works.
"""

from __future__ import annotations

import glob
import importlib.util
import sys
import traceback
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from typer.core import TyperGroup

from motionflow._version import __version__
from motionflow.chromatic import chromatic
from motionflow.events import events
from motionflow.flow import flow
from motionflow.pipeline import Result, default_output_path
from motionflow.types import Method, ModelSize, NormalizeMode, VizStyle

console = Console(stderr=True)

# Video files a folder input expands to.
VIDEO_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".mpg", ".mpeg",
    ".ts", ".mts", ".m2ts", ".h264", ".264", ".flv", ".3gp", ".ogv",
}  # fmt: skip

IO = "Input and output"
DRAW = "Drawing"
FLOW = "Flow"
CHROMA = "Chromatic motion"
EVENTS = "Event camera"


class DefaultCommandGroup(TyperGroup):
    """Runs ``flow`` when the first argument isn't a command (``motionflow video.mp4``)."""

    default_command = "flow"

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    name="motionflow",
    cls=DefaultCommandGroup,
    help=(
        "Motion visualisation for video.\n\n"
        "[bold]motionflow VIDEO[/bold] is short for [bold]motionflow flow VIDEO[/bold]."
    ),
    epilog=(
        "Examples:\n\n"
        "  motionflow recording.mp4\n\n"
        "  motionflow flow recording.mp4 -m raft --viz heatmap\n\n"
        "  motionflow videos/ -o results/ --start 1:00 --end 2:00"
    ),
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"motionflow {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = None,
) -> None:
    """Motion visualisation for video."""


# ---------------------------------------------------------------------------
# Options shared by every command
# ---------------------------------------------------------------------------

Inputs = Annotated[
    list[str],
    typer.Argument(
        help="Video files, folders (every video in them) or patterns such as 'trials/*.mp4'.",
        metavar="VIDEO...",
        show_default=False,
    ),
]
Output = Annotated[
    Path | None,
    typer.Option(
        "--output",
        "-o",
        help=(
            "Output file, or folder (anything without an extension). "
            "\\[default: <input>_<command>.<ext> next to each input]"
        ),
        show_default=False,
        rich_help_panel=IO,
    ),
]
Overwrite = Annotated[
    bool,
    typer.Option("--overwrite", help="Replace outputs that already exist.", rich_help_panel=IO),
]
Start = Annotated[
    str | None,
    typer.Option(
        "--start",
        help="Start here: seconds (12.5), clock time (1:30) or a frame number (120f).",
        show_default=False,
        rich_help_panel=IO,
    ),
]
End = Annotated[
    str | None,
    typer.Option(
        "--end",
        help="Stop here (exclusive), in the same formats as --start.",
        show_default=False,
        rich_help_panel=IO,
    ),
]
Scale = Annotated[
    float,
    typer.Option(
        "--scale", help="Resize before processing, e.g. 0.5 for half size.", rich_help_panel=IO
    ),
]
Fps = Annotated[
    float | None,
    typer.Option(
        "--fps",
        help="Output frame rate. \\[default: the input's]",
        show_default=False,
        rich_help_panel=IO,
    ),
]
Codec = Annotated[
    str,
    typer.Option(
        "--codec",
        help=(
            "auto (from the extension), h264, hevc, vp9, av1, ffv1 (lossless), "
            "mjpeg, mpeg4, or an FFmpeg encoder."
        ),
        rich_help_panel=IO,
    ),
]
Crf = Annotated[
    int | None,
    typer.Option(
        "--crf",
        min=0,
        help=(
            "Output quality, lower is better: CRF for h264/hevc/vp9/av1, "
            "quantiser (2-31) for mjpeg/mpeg4."
        ),
        show_default=False,
        rich_help_panel=IO,
    ),
]
Quiet = Annotated[
    bool,
    typer.Option("--quiet", "-q", help="Print only output paths and errors.", rich_help_panel=IO),
]
Verbose = Annotated[
    bool,
    typer.Option("--verbose", "-v", help="Show full error details.", rich_help_panel=IO),
]


# ---------------------------------------------------------------------------
# flow
# ---------------------------------------------------------------------------


@app.command("flow")
def flow_command(
    inputs: Inputs,
    method: Annotated[
        Method,
        typer.Option(
            "--method",
            "-m",
            help="Optical flow method. See: motionflow methods.",
            rich_help_panel=FLOW,
        ),
    ] = Method.DIS,
    stride: Annotated[
        int,
        typer.Option(
            "--stride",
            min=1,
            help="Estimate flow every N frames (faster); speeds stay in pixels per frame.",
            rich_help_panel=FLOW,
        ),
    ] = 1,
    smooth: Annotated[
        int,
        typer.Option(
            "--smooth",
            min=0,
            help="Average over a centred window of N frames.",
            rich_help_panel=FLOW,
        ),
    ] = 0,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            help="Deep methods: auto, cuda, mps or cpu. \\[default: auto]",
            show_default=False,
            rich_help_panel=FLOW,
        ),
    ] = None,
    model_size: Annotated[
        ModelSize | None,
        typer.Option(
            "--model-size",
            help="Deep methods: model size. \\[default: large]",
            show_default=False,
            rich_help_panel=FLOW,
        ),
    ] = None,
    viz: Annotated[
        VizStyle,
        typer.Option(
            "--viz",
            help="hsv: hue is direction, brightness speed. heatmap: speed only. quiver: arrows.",
            rich_help_panel=DRAW,
        ),
    ] = VizStyle.HSV,
    overlay: Annotated[
        float | None,
        typer.Option(
            "--overlay",
            min=0.0,
            max=1.0,
            help="Draw over the original frame at this opacity (0-1).",
            show_default=False,
            rich_help_panel=DRAW,
        ),
    ] = None,
    normalize: Annotated[
        NormalizeMode,
        typer.Option(
            "--normalize",
            help="How speeds map to colours: auto (one scale per video), global (exact), per-frame.",
            rich_help_panel=DRAW,
        ),
    ] = NormalizeMode.AUTO,
    percentile: Annotated[
        float | None,
        typer.Option(
            "--percentile",
            min=0.0,
            max=100.0,
            help="auto/global: percentile of all motion drawn at full strength. \\[default: 99]",
            show_default=False,
            rich_help_panel=DRAW,
        ),
    ] = None,
    max_flow: Annotated[
        float | None,
        typer.Option(
            "--max-flow",
            help="Fixed speed (px/frame) at full strength, e.g. to compare videos.",
            show_default=False,
            rich_help_panel=DRAW,
        ),
    ] = None,
    min_flow: Annotated[
        float,
        typer.Option(
            "--min-flow",
            min=0.0,
            help="Draw motion below this (px/frame) as none.",
            rich_help_panel=DRAW,
        ),
    ] = 0.0,
    legend: Annotated[
        bool,
        typer.Option(
            "--legend/--no-legend", help="Draw a colour key in the corner.", rich_help_panel=DRAW
        ),
    ] = True,
    quiver_step: Annotated[
        int | None,
        typer.Option(
            "--quiver-step",
            min=1,
            help="quiver: grid spacing in pixels. \\[default: 16]",
            show_default=False,
            rich_help_panel=DRAW,
        ),
    ] = None,
    quiver_scale: Annotated[
        float | None,
        typer.Option(
            "--quiver-scale",
            help="quiver: arrow length per pixel of motion. \\[default: automatic]",
            show_default=False,
            rich_help_panel=DRAW,
        ),
    ] = None,
    output: Output = None,
    overwrite: Overwrite = False,
    start: Start = None,
    end: End = None,
    scale: Scale = 1.0,
    fps: Fps = None,
    codec: Codec = "auto",
    crf: Crf = None,
    quiet: Quiet = False,
    verbose: Verbose = False,
) -> None:
    """Optical flow: how fast, and in which direction, each pixel moves.

    The output has one frame per input frame: frame i shows the motion from
    frame i-1 to frame i.
    """
    if viz != VizStyle.QUIVER:
        _warn_unused(f"--viz {viz.value}", quiver_step=quiver_step, quiver_scale=quiver_scale)
    if not method.requires_torch:
        _warn_unused(f"--method {method.value}", device=device, model_size=model_size)
    if max_flow is not None:
        _warn_unused("--max-flow", percentile=percentile)
    elif normalize == NormalizeMode.PER_FRAME:
        _warn_unused("--normalize per-frame", percentile=percentile)

    def process(input: Path, out: Path, progress: bool | str) -> Result:
        return flow(
            input,
            out,
            method=method,
            viz=viz,
            overlay=overlay,
            normalize=normalize,
            percentile=99.0 if percentile is None else percentile,
            max_flow=max_flow,
            min_flow=min_flow,
            legend=legend,
            stride=stride,
            smooth=smooth,
            quiver_step=16 if quiver_step is None else quiver_step,
            quiver_scale=quiver_scale,
            device=device or "auto",
            model_size=model_size or ModelSize.LARGE,
            start=start,
            end=end,
            scale=scale,
            fps=fps,
            codec=codec,
            crf=crf,
            overwrite=overwrite,
            progress=progress,
        )

    _run_batch(inputs, output, "_flow", process, quiet=quiet, verbose=verbose)


# ---------------------------------------------------------------------------
# chromatic
# ---------------------------------------------------------------------------


def _parse_gain(value: str) -> float | tuple[float, float, float]:
    parts = [p for p in value.replace(",", " ").split() if p]
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        raise typer.BadParameter(f"not a number: {value!r}") from None
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 3:
        return (numbers[0], numbers[1], numbers[2])
    raise typer.BadParameter("give one value, or three separated by commas (recent,medium,older)")


@app.command("chromatic")
def chromatic_command(
    inputs: Inputs,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            help="exponential: running averages fading at --exp-a/--exp-b. sequential: the last three frames.",
            rich_help_panel=CHROMA,
        ),
    ] = "exponential",
    exp_a: Annotated[
        float | None,
        typer.Option(
            "--exp-a",
            min=0.0,
            max=0.99,
            help="exponential: fade rate of the medium-term average (green). \\[default: 0.5]",
            show_default=False,
            rich_help_panel=CHROMA,
        ),
    ] = None,
    exp_b: Annotated[
        float | None,
        typer.Option(
            "--exp-b",
            min=0.0,
            max=0.99,
            help="exponential: fade rate of the long-term average (red). \\[default: 0.8]",
            show_default=False,
            rich_help_panel=CHROMA,
        ),
    ] = None,
    gain: Annotated[
        str,
        typer.Option(
            "--gain",
            help="Amplification of the differences: one value, or recent,medium,older.",
            rich_help_panel=CHROMA,
        ),
    ] = "4",
    lum_weight: Annotated[
        float,
        typer.Option(
            "--lum-weight",
            min=0.0,
            help="Brightness of the greyscale frame under the motion.",
            rich_help_panel=CHROMA,
        ),
    ] = 0.7,
    motion_only: Annotated[
        bool,
        typer.Option(
            "--motion-only",
            help="Motion on black (same as --lum-weight 0).",
            rich_help_panel=CHROMA,
        ),
    ] = False,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            min=0.0,
            help="Subtract this from every channel to hide noise (0-255).",
            rich_help_panel=CHROMA,
        ),
    ] = 0.0,
    tail_only: Annotated[
        bool,
        typer.Option(
            "--tail-only",
            help="Each colour shows only motion the more recent one doesn't.",
            rich_help_panel=CHROMA,
        ),
    ] = False,
    frame_skip: Annotated[
        int,
        typer.Option(
            "--frame-skip",
            min=0,
            help="Use every (N+1)-th frame only: fewer output frames, larger motion between them.",
            rich_help_panel=CHROMA,
        ),
    ] = 0,
    output: Output = None,
    overwrite: Overwrite = False,
    start: Start = None,
    end: End = None,
    scale: Scale = 1.0,
    fps: Fps = None,
    codec: Codec = "auto",
    crf: Crf = None,
    quiet: Quiet = False,
    verbose: Verbose = False,
) -> None:
    """Chromatic motion (BehaveAI): recent movement blue, older green and red.

    Moving things leave a trail that fades from blue through green to red over
    a dimmed greyscale frame; still parts stay grey.
    """
    if strategy not in ("exponential", "sequential"):
        raise typer.BadParameter("exponential or sequential", param_hint="--strategy")
    if strategy == "sequential":
        _warn_unused("--strategy sequential", exp_a=exp_a, exp_b=exp_b)
    gains = _parse_gain(gain)

    def process(input: Path, out: Path, progress: bool | str) -> Result:
        return chromatic(
            input,
            out,
            strategy=strategy,  # type: ignore[arg-type]
            exp_a=0.5 if exp_a is None else exp_a,
            exp_b=0.8 if exp_b is None else exp_b,
            gain=gains,
            lum_weight=lum_weight,
            motion_only=motion_only,
            threshold=threshold,
            tail_only=tail_only,
            frame_skip=frame_skip,
            start=start,
            end=end,
            scale=scale,
            fps=fps,
            codec=codec,
            crf=crf,
            overwrite=overwrite,
            progress=progress,
        )

    _run_batch(inputs, output, "_chromatic", process, quiet=quiet, verbose=verbose)


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


@app.command("events")
def events_command(
    inputs: Inputs,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            min=0.0,
            help="Contrast threshold: change in log brightness per event (0.2 is ~22%).",
            rich_help_panel=EVENTS,
        ),
    ] = 0.2,
    on_threshold: Annotated[
        float | None,
        typer.Option(
            "--on-threshold",
            help="Threshold for ON (brighter) events. \\[default: --threshold]",
            show_default=False,
            rich_help_panel=EVENTS,
        ),
    ] = None,
    off_threshold: Annotated[
        float | None,
        typer.Option(
            "--off-threshold",
            help="Threshold for OFF (darker) events. \\[default: --threshold]",
            show_default=False,
            rich_help_panel=EVENTS,
        ),
    ] = None,
    log_offset: Annotated[
        float,
        typer.Option(
            "--log-offset",
            help="Added to brightness (0-255) before the log; higher keeps dark noise quiet.",
            rich_help_panel=EVENTS,
        ),
    ] = 10.0,
    render: Annotated[
        str,
        typer.Option(
            "--render",
            help="polarity: ON blue, OFF red. count: events per pixel as a heatmap.",
            rich_help_panel=EVENTS,
        ),
    ] = "polarity",
    window: Annotated[
        int,
        typer.Option(
            "--window",
            min=1,
            help="Draw the events of the last N frames together.",
            rich_help_panel=EVENTS,
        ),
    ] = 1,
    overlay: Annotated[
        float | None,
        typer.Option(
            "--overlay",
            min=0.0,
            max=1.0,
            help="Draw the events over the greyscale video at this opacity.",
            show_default=False,
            rich_help_panel=EVENTS,
        ),
    ] = None,
    full_count: Annotated[
        int,
        typer.Option(
            "--full-count",
            min=1,
            help="Events per pixel drawn at full strength.",
            rich_help_panel=EVENTS,
        ),
    ] = 4,
    save_events: Annotated[
        str | None,
        typer.Option(
            "--save-events",
            help="Also write the event stream next to each output: h5 (needs h5py), npz or txt.",
            show_default=False,
            rich_help_panel=EVENTS,
        ),
    ] = None,
    output: Output = None,
    overwrite: Overwrite = False,
    start: Start = None,
    end: End = None,
    scale: Scale = 1.0,
    fps: Fps = None,
    codec: Codec = "auto",
    crf: Crf = None,
    quiet: Quiet = False,
    verbose: Verbose = False,
) -> None:
    """Simulate an event camera: pixels report brightness changes as ON/OFF events.

    The video shows the events (ON blue, OFF red); --save-events also writes
    them as (t, x, y, polarity) records.
    """
    if render not in ("polarity", "count"):
        raise typer.BadParameter("polarity or count", param_hint="--render")
    if save_events is not None and save_events.lower().lstrip(".") not in (
        "h5",
        "hdf5",
        "npz",
        "txt",
    ):
        raise typer.BadParameter("h5, npz or txt", param_hint="--save-events")

    def process(input: Path, out: Path, progress: bool | str) -> Result:
        stream = out.with_suffix("." + save_events.lower().lstrip(".")) if save_events else None
        result = events(
            input,
            out,
            threshold=threshold,
            on_threshold=on_threshold,
            off_threshold=off_threshold,
            log_offset=log_offset,
            render=render,  # type: ignore[arg-type]
            window=window,
            overlay=overlay,
            full_count=full_count,
            save_events=stream,
            start=start,
            end=end,
            scale=scale,
            fps=fps,
            codec=codec,
            crf=crf,
            overwrite=overwrite,
            progress=progress,
        )
        if stream is not None:
            typer.echo(str(stream))  # stdout, like the video path
        return result

    _run_batch(inputs, output, "_events", process, quiet=quiet, verbose=verbose)


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------


# Weight files each deep method needs (RAFT's come from torchvision).
DEEP_WEIGHTS = {
    Method.SEA_RAFT: {
        "small": "sea-raft-small",
        "medium": "sea-raft-medium",
        "large": "sea-raft-medium",
    },
    Method.NEUFLOW: {"large": "neuflow-v2"},
}


def _torchvision_raft_cached(size: str) -> bool:
    import os

    hub = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub"
    return any((hub / "checkpoints").glob(f"raft_{size}_*.pth"))


@app.command("methods")
def methods_command() -> None:
    """List the optical flow methods and whether they can run here."""
    import cv2

    from motionflow import weights

    torch = importlib.util.find_spec("torch") is not None
    contrib = hasattr(cv2, "optflow")

    def deep(method: Method) -> tuple[bool, str]:
        if not torch:
            return False, "needs torch"
        if method == Method.RAFT:
            cached = all(_torchvision_raft_cached(s) for s in ("small", "large"))
        else:
            cached = all(weights.is_cached(n) for n in set(DEEP_WEIGHTS[method].values()))
        return True, "" if cached else "weights download on first use"

    rows = [
        (Method.DIS, "dense", "fast and accurate; the default", True, ""),
        (Method.FARNEBACK, "dense", "the default before 0.2", True, ""),
        (
            Method.TVL1,
            "dense",
            "sharp motion edges; slow",
            contrib,
            "" if contrib else "needs OpenCV contrib",
        ),
        (Method.LUCAS_KANADE, "points", "tracks points; drawn as dots or arrows", True, ""),
        (Method.RAFT, "dense, deep", "small, large", *deep(Method.RAFT)),
        (
            Method.SEA_RAFT,
            "dense, deep",
            "small, medium, large; very accurate",
            *deep(Method.SEA_RAFT),
        ),
        (Method.NEUFLOW, "dense, deep", "fastest deep method", *deep(Method.NEUFLOW)),
    ]
    table = Table(box=None, pad_edge=False)
    for column in ("Method", "Kind", "Notes", "Available"):
        table.add_column(column)
    for method, kind, notes, ok, extra in rows:
        if ok:
            status = "[green]yes[/green]" + (f" [dim]({escape(extra)})[/dim]" if extra else "")
        else:
            status = f"[red]no[/red] [dim]({escape(extra)})[/dim]"
        table.add_row(method.value, kind, escape(notes), status)
    Console().print(table)
    if not torch:
        Console().print(escape('\nDeep methods need PyTorch: pip install "motionflow[torch]"'))


@app.command("download")
def download_command(
    methods: Annotated[
        list[Method] | None,
        typer.Argument(
            help="Deep methods to fetch weights for: raft, sea-raft, neuflow. [default: all]",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Download model weights now, e.g. on a login node before a batch job.

    Weights otherwise download on first use.
    """
    from motionflow import weights

    chosen = methods or [Method.RAFT, Method.SEA_RAFT, Method.NEUFLOW]
    for method in chosen:
        if not method.requires_torch:
            console.print(f"{method.value} needs no weights.")
            continue
        if method == Method.RAFT:
            try:
                from torchvision.models import optical_flow as of
            except ImportError:
                console.print(
                    "[red]Error:[/red] "
                    + escape('RAFT needs torch: pip install "motionflow[torch]"')
                )
                raise typer.Exit(code=1) from None
            for size_weights in (of.Raft_Small_Weights.DEFAULT, of.Raft_Large_Weights.DEFAULT):
                size_weights.get_state_dict(progress=True)
            console.print("[green]Cached:[/green] raft (small, large)")
            continue
        for name in sorted(set(DEEP_WEIGHTS[method].values())):
            try:
                path = weights.fetch(name)
            except RuntimeError as exc:
                console.print(f"[red]Error:[/red] {escape(str(exc))}")
                raise typer.Exit(code=1) from None
            console.print(f"[green]Cached:[/green] {name} [dim]({escape(str(path))})[/dim]")


# ---------------------------------------------------------------------------
# Running several inputs
# ---------------------------------------------------------------------------


def _warn_unused(why: str, **options: Any) -> None:
    for name, value in options.items():
        if value is not None:
            flag = "--" + name.replace("_", "-")
            console.print(f"[yellow]Warning:[/yellow] {flag} has no effect with {escape(why)}.")


def expand_inputs(inputs: list[str], suffix: str) -> list[Path]:
    """Files, folders and glob patterns to a list of video files.

    Folders give every video in them (not recursively), leaving out files
    that look like this command's own outputs (``*<suffix>.*``).
    """
    found: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            videos = sorted(
                p
                for p in path.iterdir()
                if p.suffix.lower() in VIDEO_EXTENSIONS
                and not p.stem.endswith(suffix)
                and not p.name.startswith(".")
            )
            if not videos:
                raise typer.BadParameter(f"No videos in folder: {item}")
            found.extend(videos)
        elif path.exists():
            found.append(path)
        elif any(char in item for char in "*?["):
            matches = sorted(Path(p) for p in glob.glob(item) if Path(p).is_file())
            if not matches:
                raise typer.BadParameter(f"No files match: {item}")
            found.extend(matches)
        else:
            raise typer.BadParameter(f"File not found: {item}")
    return list(dict.fromkeys(found))  # drop duplicates, keep order


def plan_outputs(inputs: list[Path], output: Path | None, suffix: str) -> list[tuple[Path, Path]]:
    """Pair each input with its output path.

    ``output`` is a folder if it is one already or has no extension (a video
    file needs one); otherwise it's the output file for a single input.
    """
    if output is None:
        return [(p, default_output_path(p, suffix)) for p in inputs]
    if output.is_dir() or output.suffix == "":
        return [(p, output / default_output_path(p, suffix).name) for p in inputs]
    if len(inputs) > 1:
        raise typer.BadParameter(
            f"With several inputs, -o must be a folder, not a file: {output}",
            param_hint="-o",
        )
    return [(inputs[0], output)]


def _run_batch(
    inputs: list[str],
    output: Path | None,
    suffix: str,
    process: Callable[[Path, Path, bool | str], Result],
    *,
    quiet: bool,
    verbose: bool,
) -> None:
    jobs = plan_outputs(expand_inputs(inputs, suffix), output, suffix)
    show_progress = not quiet and sys.stderr.isatty()
    failed = 0
    for n, (input, out) in enumerate(jobs, 1):
        label = f"[{n}/{len(jobs)}] {input.name}" if len(jobs) > 1 else input.name
        try:
            with warnings.catch_warnings():
                warnings.showwarning = _show_warning
                result = process(input, out, label if show_progress else False)
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted.[/yellow] The unfinished output was not kept.")
            raise typer.Exit(code=130) from None
        except Exception as exc:
            failed += 1
            _report_error(exc, input, out, verbose)
            continue
        typer.echo(str(result.output))  # stdout, for scripts
        if not quiet:
            console.print(
                f"[green]Done:[/green] {escape(str(result.output))} "
                f"[dim]({result.frames_written} frames, {result.seconds:.1f} s)[/dim]"
            )
    if failed:
        if len(jobs) > 1:
            console.print(f"[red]{failed} of {len(jobs)} videos failed.[/red]")
        raise typer.Exit(code=1)


def _report_error(exc: Exception, input: Path, output: Path, verbose: bool) -> None:
    message = escape(str(exc))
    where = f"{escape(input.name)}: "
    if isinstance(exc, FileExistsError):
        message += ". Use --overwrite to replace it"
    elif isinstance(exc, PermissionError):
        message += ". Use -o to write somewhere else"
    elif isinstance(exc, NotImplementedError):
        console.print(f"[yellow]Not implemented:[/yellow] {where}{message}")
        return
    console.print(f"[red]Error:[/red] {where}{message}")
    if verbose:
        console.print(escape("".join(traceback.format_exception(exc))), style="dim")


def _show_warning(message, category, filename, lineno, file=None, line=None) -> None:
    console.print(f"[yellow]Warning:[/yellow] {escape(str(message))}")
