from __future__ import annotations

import cv2
import numpy as np

from motionflow.types import VizStyle


def flow_to_hsv(flow: np.ndarray, max_flow: float | None = None) -> np.ndarray:
    """Render flow as an HSV colour wheel image.

    Hue encodes direction, brightness encodes magnitude.

    Args:
        flow: Flow array of shape ``(H, W, 2)`` with ``(dx, dy)`` components.
        max_flow: Magnitude used to normalise brightness. Computed per-frame
            if ``None``.

    Returns:
        BGR image of shape ``(H, W, 3)``.
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = (angle * 90.0 / np.pi).astype(np.uint8)

    norm = max_flow if max_flow is not None else float(magnitude.max())
    if norm > 0:
        hsv[..., 2] = np.clip(magnitude / norm * 255, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flow_to_heatmap(flow: np.ndarray, max_flow: float | None = None) -> np.ndarray:
    """Render flow magnitude as a heatmap (COLORMAP_INFERNO).

    Direction information is discarded.

    Args:
        flow: Flow array of shape ``(H, W, 2)``.
        max_flow: Magnitude cap for normalisation. Computed per-frame if ``None``.

    Returns:
        BGR image of shape ``(H, W, 3)``.
    """
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    norm = max_flow if max_flow is not None else float(magnitude.max())
    if norm > 0:
        scaled = np.clip(magnitude / norm * 255, 0, 255).astype(np.uint8)
    else:
        scaled = np.zeros(magnitude.shape, dtype=np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)


def flow_to_quiver(
    flow: np.ndarray,
    frame: np.ndarray,
    step: int = 16,
    scale: float = 1.0,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Render flow as arrow overlays on the original frame.

    Args:
        flow: Flow array of shape ``(H, W, 2)``.
        frame: BGR source frame of shape ``(H, W, 3)``.
        step: Grid spacing in pixels between arrow origins.
        scale: Arrow length multiplier.
        color: BGR arrow colour.
        thickness: Arrow line thickness.

    Returns:
        BGR image with arrows overlaid.
    """
    out = frame.copy()
    h, w = flow.shape[:2]
    y_coords, x_coords = np.mgrid[step // 2 : h : step, step // 2 : w : step]

    for y, x in zip(y_coords.ravel(), x_coords.ravel(), strict=True):
        dx, dy = flow[y, x]
        start = (int(x), int(y))
        end = (int(x + dx * scale), int(y + dy * scale))
        cv2.arrowedLine(out, start, end, color, thickness, tipLength=0.3)

    return out


def flow_to_blend(
    flow: np.ndarray,
    frame: np.ndarray,
    alpha: float = 0.5,
    max_flow: float | None = None,
) -> np.ndarray:
    """Blend HSV flow visualisation with the original frame.

    Args:
        flow: Flow array of shape ``(H, W, 2)``.
        frame: BGR source frame of shape ``(H, W, 3)``.
        alpha: Weight of the original frame (0 = pure flow, 1 = pure original).
        max_flow: Normalisation cap for flow visualisation.

    Returns:
        BGR blended image of shape ``(H, W, 3)``.
    """
    flow_img = flow_to_hsv(flow, max_flow=max_flow)
    return cv2.addWeighted(frame, alpha, flow_img, 1.0 - alpha, 0)


def visualize_flow(
    flow: np.ndarray,
    style: VizStyle,
    frame: np.ndarray | None = None,
    blend: float = 0.0,
    max_flow: float | None = None,
    quiver_step: int = 16,
) -> np.ndarray:
    """Dispatch flow visualisation to the appropriate renderer.

    Args:
        flow: Flow array of shape ``(H, W, 2)``.
        style: Visualisation style to use.
        frame: Original BGR frame. Required for ``quiver`` and ``blend`` styles,
            and when ``blend > 0`` for any other style.
        blend: When ``> 0`` and style is not ``quiver`` or ``blend``, composites
            the flow visualisation over ``frame`` at this opacity.
        max_flow: Normalisation cap.
        quiver_step: Grid spacing for quiver arrows.

    Returns:
        BGR visualised image.
    """
    if style == VizStyle.HSV:
        img = flow_to_hsv(flow, max_flow=max_flow)
    elif style == VizStyle.HEATMAP:
        img = flow_to_heatmap(flow, max_flow=max_flow)
    elif style == VizStyle.QUIVER:
        if frame is None:
            raise ValueError("style='quiver' requires a source frame.")
        return flow_to_quiver(flow, frame, step=quiver_step)
    elif style == VizStyle.BLEND:
        if frame is None:
            raise ValueError("style='blend' requires a source frame.")
        return flow_to_blend(flow, frame, alpha=blend, max_flow=max_flow)
    else:
        raise ValueError(f"Unknown viz style: {style}")

    if blend > 0 and frame is not None:
        img = cv2.addWeighted(frame, blend, img, 1.0 - blend, 0)

    return img
