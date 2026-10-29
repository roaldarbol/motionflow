"""Drawing flow fields as images.

All functions take a flow field of shape ``(H, W, 2)`` holding ``(dx, dy)``
in pixels per frame, and return BGR ``uint8`` images. ``max_flow`` is the
magnitude drawn at full strength; ``None`` scales each frame to its own
largest motion. Motion below ``min_flow`` is drawn as none.

:class:`~motionflow.flow.FlowRenderer` combines these into the styles the
command line offers.
"""

from __future__ import annotations

import cv2
import numpy as np

from motionflow.types import VizStyle

# ---------------------------------------------------------------------------
# Dense styles
# ---------------------------------------------------------------------------


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """Per-pixel flow magnitude, ``(H, W)`` float32, in pixels per frame."""
    return cv2.magnitude(flow[..., 0], flow[..., 1])


def _strength(magnitude: np.ndarray, max_flow: float | None, min_flow: float) -> np.ndarray:
    """Magnitude scaled to 0-255 (uint8), with motion below min_flow at 0."""
    norm = max_flow if max_flow is not None else float(magnitude.max())
    if norm <= 0:
        return np.zeros(magnitude.shape, dtype=np.uint8)
    strength = np.clip(magnitude * (255.0 / norm), 0, 255)
    if min_flow > 0:
        strength[magnitude < min_flow] = 0
    return strength.astype(np.uint8)


def flow_to_hsv(
    flow: np.ndarray, max_flow: float | None = None, min_flow: float = 0.0
) -> np.ndarray:
    """Colour wheel: hue shows the direction of motion, brightness its speed.

    Args:
        flow: Flow field ``(H, W, 2)``.
        max_flow: Magnitude shown at full brightness; per frame if ``None``.
        min_flow: Motion below this many pixels per frame is black.

    Returns:
        BGR image of shape ``(H, W, 3)``.
    """
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.empty((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle * (90.0 / np.pi)).astype(np.uint8)  # 0-180, OpenCV's hue range
    hsv[..., 1] = 255
    hsv[..., 2] = _strength(magnitude, max_flow, min_flow)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flow_to_heatmap(
    flow: np.ndarray, max_flow: float | None = None, min_flow: float = 0.0
) -> np.ndarray:
    """Speed only, on the inferno colour map (black is still).

    Args:
        flow: Flow field ``(H, W, 2)``.
        max_flow: Magnitude shown at full strength; per frame if ``None``.
        min_flow: Motion below this many pixels per frame is black.

    Returns:
        BGR image of shape ``(H, W, 3)``.
    """
    strength = _strength(flow_magnitude(flow), max_flow, min_flow)
    return cv2.applyColorMap(strength, cv2.COLORMAP_INFERNO)


# ---------------------------------------------------------------------------
# Arrows and points
# ---------------------------------------------------------------------------


def _line_width(shape: tuple[int, ...]) -> int:
    return max(1, round(min(shape[:2]) / 360))


def quiver_scale(step: int, max_flow: float | None, flow: np.ndarray | None = None) -> float:
    """Arrow length multiplier that draws ``max_flow`` as 90% of the grid spacing.

    With ``max_flow=None`` the frame's own largest grid-averaged motion is
    used. Returns 0 when there is no motion to scale to.
    """
    reference = max_flow
    if reference is None and flow is not None:
        reference = float(flow_magnitude(_grid_average(flow, step)).max())
    return 0.9 * step / reference if reference else 0.0


def _grid_average(flow: np.ndarray, step: int) -> np.ndarray:
    h, w = flow.shape[:2]
    rows, cols = max(1, h // step), max(1, w // step)
    return cv2.resize(flow, (cols, rows), interpolation=cv2.INTER_AREA)


def _arrows(
    image: np.ndarray,
    starts: np.ndarray,
    vectors: np.ndarray,
    scale: float,
    min_flow: float,
    color: tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    """Draw arrows from ``starts`` (N, 2) along ``vectors`` (N, 2) times ``scale``."""
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    keep = (lengths >= max(min_flow, 1e-6)) & (lengths * scale >= 1.5)
    for (x, y), (dx, dy) in zip(starts[keep], vectors[keep] * scale, strict=True):
        start = (round(float(x)), round(float(y)))
        end = (round(float(x + dx)), round(float(y + dy)))
        cv2.arrowedLine(image, start, end, color, thickness, cv2.LINE_AA, tipLength=0.3)
    return image


def flow_to_quiver(
    flow: np.ndarray,
    frame: np.ndarray,
    step: int = 16,
    scale: float | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int | None = None,
    max_flow: float | None = None,
    min_flow: float = 0.0,
) -> np.ndarray:
    """Arrows on a grid over the original frame.

    Each arrow shows the average motion in its grid cell. By default arrows
    are scaled so that ``max_flow`` (or the frame's largest motion) is 90%
    of the grid spacing, and cells without motion get no arrow.

    Args:
        flow: Flow field ``(H, W, 2)``.
        frame: BGR frame to draw on, ``(H, W, 3)``.
        step: Grid spacing in pixels.
        scale: Arrow length per pixel of motion. ``None`` scales automatically.
        color: BGR arrow colour.
        thickness: Line width; scales with the frame size if ``None``.
        max_flow: Motion drawn as a full-length arrow when ``scale`` is ``None``.
        min_flow: Cells with less motion than this get no arrow.

    Returns:
        BGR image with arrows drawn on a copy of ``frame``.
    """
    grid = _grid_average(flow, step)
    rows, cols = grid.shape[:2]
    if scale is None:
        scale = quiver_scale(step, max_flow, flow)
    ys, xs = np.mgrid[0:rows, 0:cols]
    starts = np.stack([(xs + 0.5) * step, (ys + 0.5) * step], axis=-1).reshape(-1, 2)
    width = thickness if thickness is not None else _line_width(frame.shape)
    return _arrows(frame.copy(), starts, grid.reshape(-1, 2), scale, min_flow, color, width)


def draw_sparse(
    flow: np.ndarray,
    style: VizStyle | str,
    frame: np.ndarray | None = None,
    max_flow: float | None = None,
    min_flow: float = 0.0,
    quiver_step: int = 16,
    scale: float | None = None,
) -> np.ndarray:
    """Draw a sparse flow field: only the tracked points (non-zero flow).

    Used for point trackers such as Lucas-Kanade. ``hsv`` and ``heatmap``
    draw a dot per point, coloured as the dense style would colour it, on
    black; ``quiver`` draws an arrow from each point over ``frame``.
    """
    style = VizStyle(style)
    magnitude = flow_magnitude(flow)
    ys, xs = np.nonzero(magnitude > 0)
    points = np.stack([xs, ys], axis=-1).astype(np.float32)
    vectors = flow[ys, xs]
    if style == VizStyle.QUIVER:
        if frame is None:
            raise ValueError("style='quiver' requires a source frame.")
        if scale is None:
            reference = max_flow if max_flow is not None else float(magnitude.max())
            scale = 0.9 * quiver_step / reference if reference else 0.0
        width = _line_width(frame.shape)
        return _arrows(frame.copy(), points, vectors, scale, min_flow, (0, 255, 0), width)

    canvas = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    if len(points) == 0:
        return canvas
    # Colour each point as the dense style would, then draw it as a dot.
    colours = (flow_to_hsv if style == VizStyle.HSV else flow_to_heatmap)(
        vectors.reshape(1, -1, 2).astype(np.float32), max_flow, min_flow
    ).reshape(-1, 3)
    radius = max(2, round(min(flow.shape[:2]) / 120))
    for (x, y), colour in zip(points, colours, strict=True):
        if colour.any():
            cv2.circle(canvas, (int(x), int(y)), radius, colour.tolist(), -1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Compositing and legends
# ---------------------------------------------------------------------------


def overlay(frame: np.ndarray, layer: np.ndarray, opacity: float) -> np.ndarray:
    """Draw ``layer`` over ``frame`` at ``opacity`` (0 = frame only, 1 = layer only)."""
    return cv2.addWeighted(layer, opacity, frame, 1.0 - opacity, 0)


def hsv_wheel(size: int) -> tuple[np.ndarray, np.ndarray]:
    """The HSV colour key: direction as hue, speed as brightness out to the rim.

    Returns:
        ``(image, mask)``: a BGR image ``(size, size, 3)`` and a boolean mask
        of the disc.
    """
    r = (size - 1) / 2
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    flow = np.stack([(xs - r) / r, (ys - r) / r], axis=-1)
    mask = flow_magnitude(flow) <= 1.0
    return flow_to_hsv(flow, max_flow=1.0), mask


def _label(image: np.ndarray, text: str, origin: tuple[int, int], height: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.3, height / 30)
    thickness = max(1, round(scale * 1.5))
    cv2.putText(image, text, origin, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _format_flow(max_flow: float | None) -> str:
    return "per frame" if max_flow is None else f"{max_flow:.3g} px/frame"


def add_legend(
    image: np.ndarray,
    style: VizStyle | str,
    max_flow: float | None,
    quiver_step: int = 16,
    quiver_scale: float | None = None,
) -> np.ndarray:
    """Draw a key in the bottom-right corner of ``image`` (in place) and return it.

    ``hsv`` gets a colour wheel, ``heatmap`` a colour bar, and ``quiver`` a
    reference arrow, each labelled with ``max_flow`` (the speed at full
    strength), or "per frame" when each frame is scaled on its own.
    """
    style = VizStyle(style)
    h, w = image.shape[:2]
    size = int(np.clip(min(h, w) // 6, 32, 160))
    margin = max(4, size // 8)
    text_height = max(8, size // 6)
    label = _format_flow(max_flow)
    x1, y1 = w - margin, h - margin - text_height - margin // 2  # key's bottom-right
    if x1 - size < 0 or y1 - size < 0:
        return image  # too small for a key

    if style == VizStyle.HSV:
        wheel, mask = hsv_wheel(size)
        region = image[y1 - size : y1, x1 - size : x1]
        region[mask] = wheel[mask]
        cv2.circle(image, (x1 - size // 2, y1 - size // 2), size // 2, (255, 255, 255), 1)
    elif style == VizStyle.HEATMAP:
        bar_w = max(6, size // 6)
        gradient = np.linspace(255, 0, size).astype(np.uint8)[:, None].repeat(bar_w, axis=1)
        image[y1 - size : y1, x1 - bar_w : x1] = cv2.applyColorMap(gradient, cv2.COLORMAP_INFERNO)
        cv2.rectangle(image, (x1 - bar_w, y1 - size), (x1, y1), (255, 255, 255), 1)
    else:
        if max_flow is not None and quiver_scale:
            length = max(2, round(max_flow * quiver_scale))
            y = y1 - text_height
            cv2.arrowedLine(
                image, (x1 - length, y), (x1, y), (0, 255, 0), _line_width(image.shape),
                cv2.LINE_AA, tipLength=min(0.3, 8 / length),
            )  # fmt: skip

    (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, max(0.3, text_height / 30), 1)
    _label(image, label, (max(0, w - margin - text_w), h - margin), text_height)
    return image


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def flow_to_blend(
    flow: np.ndarray,
    frame: np.ndarray,
    alpha: float = 0.5,
    max_flow: float | None = None,
) -> np.ndarray:
    """HSV flow blended with the original frame; ``alpha`` is the frame's weight.

    Kept from motionflow 0.1; :class:`~motionflow.flow.FlowRenderer` with
    ``overlay`` does the same for every style.
    """
    return overlay(frame, flow_to_hsv(flow, max_flow=max_flow), 1.0 - alpha)


def visualize_flow(
    flow: np.ndarray,
    style: VizStyle | str,
    frame: np.ndarray | None = None,
    blend: float = 0.0,
    max_flow: float | None = None,
    quiver_step: int = 16,
) -> np.ndarray:
    """Render a flow field in one of the 0.1 styles (including ``"blend"``).

    Kept from motionflow 0.1; use :class:`~motionflow.flow.FlowRenderer`.
    ``blend`` is the weight of the original frame.
    """
    if style == "blend":
        if frame is None:
            raise ValueError("style='blend' requires a source frame.")
        return flow_to_blend(flow, frame, alpha=blend, max_flow=max_flow)
    style = VizStyle(style)
    if style == VizStyle.QUIVER:
        if frame is None:
            raise ValueError("style='quiver' requires a source frame.")
        return flow_to_quiver(flow, frame, step=quiver_step, max_flow=max_flow)
    image = (flow_to_hsv if style == VizStyle.HSV else flow_to_heatmap)(flow, max_flow)
    if blend > 0 and frame is not None:
        image = overlay(frame, image, 1.0 - blend)
    return image
