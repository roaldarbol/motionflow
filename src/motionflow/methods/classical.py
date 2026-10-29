from __future__ import annotations

import cv2
import numpy as np

from motionflow.methods.base import FlowEstimator


class FarnebackFlow(FlowEstimator):
    """Dense optical flow via Gunnar Farneback's polynomial expansion.

    This is the default method — no additional dependencies required.

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

    Tracks a regular grid of Shi-Tomasi corners. Untracked pixels are set to
    zero, producing a sparse flow field suited for quiver visualisation.

    Args:
        grid_step: Spacing in pixels between candidate tracking points.
        max_corners: Maximum number of corners to detect per frame.
        quality_level: Minimum corner quality (0–1).
        min_distance: Minimum distance between corners in pixels.
        win_size: Search window size for LK tracker (width, height).
        max_level: Maximum pyramid level.
    """

    def __init__(
        self,
        grid_step: int = 16,
        max_corners: int = 500,
        quality_level: float = 0.01,
        min_distance: float = 10.0,
        win_size: tuple[int, int] = (21, 21),
        max_level: int = 3,
    ) -> None:
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
