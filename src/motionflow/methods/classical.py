from __future__ import annotations

import cv2
import numpy as np

from motionflow.methods.base import FlowEstimator


class FarnebackFlow(FlowEstimator):
    """Dense optical flow via Gunnar Farneback's polynomial expansion.

    The default before motionflow 0.2. :class:`DISFlow` is faster and more
    accurate on most footage.

    Args:
        pyr_scale: Scale between pyramid levels. ``0.5`` halves each level.
        levels: Number of pyramid levels.
        winsize: Averaging window size. Larger values increase robustness to
            noise and fast motion but blur flow boundaries.
        iterations: Iterations per pyramid level.
        poly_n: Pixel neighbourhood for polynomial expansion (typically 5 or 7).
        poly_sigma: Gaussian sigma for polynomial expansion smoothing.
        flags: Extra flags passed to ``cv2.calcOpticalFlowFarneback``.
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        flags: int = 0,
    ) -> None:
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags

    def estimate(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        return cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            self.pyr_scale,
            self.levels,
            self.winsize,
            self.iterations,
            self.poly_n,
            self.poly_sigma,
            self.flags,
        )


class DISFlow(FlowEstimator):
    """Dense optical flow via Dense Inverse Search (Kroeger et al., ECCV 2016).

    The default method: on 720p and 1080p test sequences with known motion
    it was 3-4x faster than Farneback and more accurate, including on large
    motions Farneback can't follow. Part of standard OpenCV.

    Args:
        preset: ``"ultrafast"``, ``"fast"`` or ``"medium"`` (slowest, most
            accurate). The other arguments override the preset's values.
        patch_size: Size of the patches matched between frames, in pixels.
        patch_stride: Spacing between patches; smaller is slower and denser.
        refinement_iterations: Iterations of the variational refinement that
            smooths the result; 0 turns it off.
    """

    PRESETS = ("ultrafast", "fast", "medium")

    def __init__(
        self,
        preset: str = "medium",
        patch_size: int | None = None,
        patch_stride: int | None = None,
        refinement_iterations: int | None = None,
    ) -> None:
        if preset not in self.PRESETS:
            raise ValueError(f"preset must be one of {', '.join(self.PRESETS)}, got {preset!r}")
        self.preset = preset
        self._dis = cv2.DISOpticalFlow_create(
            getattr(cv2, f"DISOPTICAL_FLOW_PRESET_{preset.upper()}")
        )
        if patch_size is not None:
            self._dis.setPatchSize(patch_size)
        if patch_stride is not None:
            self._dis.setPatchStride(patch_stride)
        if refinement_iterations is not None:
            self._dis.setVariationalRefinementIterations(refinement_iterations)

    def estimate(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        return self._dis.calc(prev_gray, curr_gray, None)


class TVL1Flow(FlowEstimator):
    """Dense optical flow via the TV-L1 variational method.

    Produces sharper motion boundaries than Farneback at the cost of speed.

    Uses OpenCV's contrib modules, which motionflow installs
    (``opencv-contrib-python-headless`` on PyPI; ``opencv`` on conda-forge).

    Args:
        tau: Time step of the numerical scheme.
        lambda_: Weight of the data term.
        theta: Weight of the regularisation term.
        scales_number: Number of scales in the pyramid.
        warps: Number of warpings per scale.
        epsilon: Stopping criterion threshold.
        inner_iterations: Iterations between outlier filtering steps.
        outer_iterations: Maximum number of outer iterations per warp; the
            scheme stops earlier once it converges to within ``epsilon``.
    """

    def __init__(
        self,
        tau: float = 0.25,
        lambda_: float = 0.15,
        theta: float = 0.3,
        scales_number: int = 5,
        warps: int = 5,
        epsilon: float = 0.01,
        inner_iterations: int = 30,
        outer_iterations: int = 10,
    ) -> None:
        self._estimator = self._create_estimator(
            tau,
            lambda_,
            theta,
            scales_number,
            warps,
            epsilon,
            inner_iterations,
            outer_iterations,
        )

    @staticmethod
    def _create_estimator(
        tau: float,
        lambda_: float,
        theta: float,
        scales_number: int,
        warps: int,
        epsilon: float,
        inner_iterations: int,
        outer_iterations: int,
    ) -> cv2.DualTVL1OpticalFlow:
        try:
            est = cv2.optflow.DualTVL1OpticalFlow_create()
        except AttributeError:
            try:
                est = cv2.DualTVL1OpticalFlow_create()  # type: ignore[attr-defined]
            except AttributeError:
                raise RuntimeError(
                    "TV-L1 optical flow is not available in your OpenCV build. "
                    "It needs OpenCV's contrib modules, which motionflow installs as "
                    "opencv-contrib-python-headless. If another OpenCV wheel "
                    "(opencv-python, opencv-python-headless, …) is installed too, "
                    "they overwrite each other's files: uninstall every OpenCV "
                    "wheel, then reinstall opencv-contrib-python-headless."
                ) from None
        est.setTau(tau)
        est.setLambda(lambda_)
        est.setTheta(theta)
        est.setScalesNumber(scales_number)
        est.setWarpingsNumber(warps)
        est.setEpsilon(epsilon)
        est.setInnerIterations(inner_iterations)
        est.setOuterIterations(outer_iterations)
        return est

    def estimate(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        return self._estimator.calc(prev_gray, curr_gray, None)


class LucasKanadeFlow(FlowEstimator):
    """Sparse optical flow via Lucas-Kanade with pyramidal refinement.

    Tracks points from one frame to the next: Shi-Tomasi corners by default,
    or a regular grid. The flow is zero except at the tracked points, which
    the renderers draw as dots or arrows.

    Args:
        points: ``"corners"`` tracks up to ``max_corners`` Shi-Tomasi
            corners; ``"grid"`` tracks points every ``grid_step`` pixels.
        grid_step: Spacing in pixels between points when ``points="grid"``.
        max_corners: Maximum number of corners to detect per frame.
        quality_level: Minimum corner quality (0–1).
        min_distance: Minimum distance between corners in pixels.
        win_size: Search window size for LK tracker (width, height).
        max_level: Maximum pyramid level.
    """

    sparse = True

    def __init__(
        self,
        points: str = "corners",
        grid_step: int = 16,
        max_corners: int = 500,
        quality_level: float = 0.01,
        min_distance: float = 10.0,
        win_size: tuple[int, int] = (21, 21),
        max_level: int = 3,
    ) -> None:
        if points not in ("corners", "grid"):
            raise ValueError(f"points must be 'corners' or 'grid', got {points!r}")
        if grid_step < 1:
            raise ValueError(f"grid_step must be at least 1, got {grid_step}")
        self.points = points
        self.grid_step = grid_step
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.win_size = win_size
        self.max_level = max_level
        self._lk_params = {
            "winSize": win_size,
            "maxLevel": max_level,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }

    def estimate(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        h, w = prev.shape[:2]
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        flow = np.zeros((h, w, 2), dtype=np.float32)

        if self.points == "grid":
            step = self.grid_step
            ys, xs = np.mgrid[step // 2 : h : step, step // 2 : w : step]
            pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float32)
        else:
            pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=7,
            )
        if pts is None or len(pts) == 0:
            return flow

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None, **self._lk_params
        )

        if next_pts is None or status is None:
            return flow

        good_prev = pts[status.ravel() == 1]
        good_next = next_pts[status.ravel() == 1]

        for (x0, y0), (x1, y1) in zip(
            good_prev.reshape(-1, 2), good_next.reshape(-1, 2), strict=True
        ):
            xi, yi = int(round(x0)), int(round(y0))
            if 0 <= yi < h and 0 <= xi < w:
                flow[yi, xi] = (x1 - x0, y1 - y0)

        return flow
