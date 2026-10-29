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
    ] = Method.FARNEBACK,
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
# methods
# ---------------------------------------------------------------------------


@app.command("methods")
def methods_command() -> None:
    """List the optical flow methods and whether they can run here."""
    import cv2

    torch = importlib.util.find_spec("torch") is not None
    contrib = hasattr(cv2, "optflow")
    rows = [
        (Method.FARNEBACK, "dense", "fast; the default", True, ""),
        (Method.TVL1, "dense", "sharp motion edges; slow", contrib, "needs OpenCV contrib"),
        (Method.LUCAS_KANADE, "points", "tracks points; drawn as dots or arrows", True, ""),
        (Method.RAFT, "dense, deep", "high quality; GPU recommended", torch, "needs torch"),
        (Method.SEA_RAFT, "dense, deep", "not working in this release (#16)", False, "#16"),
        (Method.NEUFLOW, "dense, deep", "not working in this release (#16)", False, "#16"),
    ]
    table = Table(box=None, pad_edge=False)
    for column in ("Method", "Kind", "Notes", "Available"):
        table.add_column(column)
    for method, kind, notes, ok, missing in rows:
        status = "[green]yes[/green]" if ok else f"[red]no[/red] [dim]({escape(missing)})[/dim]"
        table.add_row(method.value, kind, escape(notes), status)
    Console().print(table)
    if not torch:
        Console().print(escape('\nDeep methods need PyTorch: pip install "motionflow[torch]"'))


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
