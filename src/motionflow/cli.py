from __future__ import annotations

import warnings
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from motionflow._version import __version__
from motionflow.flow import flow
from motionflow.types import Method, ModelSize, NormalizeMode, VizStyle

app = typer.Typer(
    name="motionflow",
    help="Compute and visualise optical flow from a video file.",
    add_completion=False,
)
console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"motionflow {__version__}")
        raise typer.Exit()


@app.command()
def main(
    input: Path = typer.Argument(
        ...,
        help="Input video file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path. Defaults to <input_stem>_flow.<ext> next to the input.",
        show_default=False,
    ),
    method: Method = typer.Option(
        Method.FARNEBACK,
        "--method",
        "-m",
        help="Optical flow algorithm.",
        show_default=True,
    ),
    viz: VizStyle = typer.Option(
        VizStyle.HSV,
        "--viz",
        help="hsv: hue is direction, brightness speed. heatmap: speed only. quiver: arrows.",
        show_default=True,
    ),
    overlay: float | None = typer.Option(
        None,
        "--overlay",
        min=0.0,
        max=1.0,
        help="Draw the flow over the original frame at this opacity (0-1).",
        show_default=False,
    ),
    legend: bool = typer.Option(
        True,
        "--legend/--no-legend",
        help="Draw a colour key in the bottom-right corner.",
    ),
    stride: int = typer.Option(
        1,
        "--stride",
        min=1,
        help="Compute flow every N frames.",
        show_default=True,
    ),
    normalize: NormalizeMode = typer.Option(
        NormalizeMode.AUTO,
        "--normalize",
        help=(
            "How speeds map to colours. auto: one scale for the whole video (sampled "
            "on long videos); global: the same from every frame; per-frame: each frame "
            "on its own."
        ),
        show_default=True,
    ),
    percentile: float = typer.Option(
        99.0,
        "--percentile",
        min=0.0,
        max=100.0,
        help="For auto/global: the percentile of all motion drawn at full strength.",
        show_default=True,
    ),
    max_flow: float | None = typer.Option(
        None,
        "--max-flow",
        help="Fixed speed (pixels per frame) drawn at full strength. Overrides --normalize.",
        show_default=False,
    ),
    min_flow: float = typer.Option(
        0.0,
        "--min-flow",
        min=0.0,
        help="Draw motion below this many pixels per frame as none (hides noise).",
        show_default=True,
    ),
    smooth: int = typer.Option(
        0,
        "--smooth",
        min=0,
        help="Temporal smoothing window in frames. 0 = disabled.",
        show_default=True,
    ),
    codec: str = typer.Option(
        "auto",
        "--codec",
        help=(
            "Output codec: auto (from the extension), h264, hevc, vp9, av1, "
            "ffv1 (lossless), mjpeg, mpeg4, or an FFmpeg encoder name."
        ),
        show_default=True,
    ),
    crf: int | None = typer.Option(
        None,
        "--crf",
        min=0,
        help="Output quality, lower is better: CRF for h264/hevc/vp9/av1, quantiser (2-31) for mjpeg/mpeg4.",
        show_default=False,
    ),
    start: str | None = typer.Option(
        None,
        "--start",
        help="Start at this time: seconds (12.5), clock time (1:30) or frame (120f).",
        show_default=False,
    ),
    end: str | None = typer.Option(
        None,
        "--end",
        help="Stop at this time (exclusive), in the same formats as --start.",
        show_default=False,
    ),
    scale: float = typer.Option(
        1.0,
        "--scale",
        help="Resize before computing flow, e.g. 0.5 for half size.",
        show_default=True,
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace the output file if it already exists.",
    ),
    fps: float | None = typer.Option(
        None,
        "--fps",
        help="Output FPS. Preserves source FPS if not set.",
        show_default=False,
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Compute device for deep methods: auto, cuda, mps, or cpu.",
        show_default=True,
    ),
    model_size: ModelSize = typer.Option(
        ModelSize.LARGE,
        "--model-size",
        help="Model size for RAFT / SEA-RAFT.",
        show_default=True,
    ),
    quiver_step: int = typer.Option(
        16,
        "--quiver-step",
        min=1,
        help="Grid spacing in pixels for quiver arrows.",
        show_default=True,
    ),
    quiver_scale: float | None = typer.Option(
        None,
        "--quiver-scale",
        help="Arrow length per pixel of motion. Automatic if not set.",
        show_default=False,
    ),
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Compute and visualise optical flow from INPUT video."""
    try:
        with warnings.catch_warnings():
            warnings.showwarning = _show_warning
            result = flow(
                input,
                output,
                method=method,
                viz=viz,
                overlay=overlay,
                legend=legend,
                percentile=percentile,
                min_flow=min_flow,
                quiver_scale=quiver_scale,
                stride=stride,
                normalize=normalize,
                max_flow=max_flow,
                smooth=smooth,
                codec=codec,
                crf=crf,
                fps=fps,
                start=start,
                end=end,
                scale=scale,
                overwrite=overwrite,
                device=device,
                model_size=model_size,
                quiver_step=quiver_step,
            )
        console.print(f"[green]Done.[/green] Output: {escape(str(result.output))}")
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow] No output was written.")
        raise typer.Exit(code=130) from None
    except FileExistsError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}. Use --overwrite to replace it.")
        raise typer.Exit(code=1) from exc
    except NotImplementedError as exc:
        console.print(f"[yellow]Not implemented:[/yellow] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc


def _show_warning(message, category, filename, lineno, file=None, line=None) -> None:
    console.print(f"[yellow]Warning:[/yellow] {escape(str(message))}")
