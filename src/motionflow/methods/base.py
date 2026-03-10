from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FlowEstimator(ABC):
    """Abstract base for all optical flow estimators.

    Subclasses must implement :meth:`estimate`. The estimator is also callable,
    delegating to :meth:`estimate`.
    """

    @abstractmethod
    def estimate(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Estimate optical flow between two consecutive BGR frames.

        Args:
            prev: Previous frame as a BGR uint8 array of shape ``(H, W, 3)``.
            curr: Current frame as a BGR uint8 array of shape ``(H, W, 3)``.

        Returns:
            Flow array of shape ``(H, W, 2)`` with ``(dx, dy)`` displacement
            components in pixels.
        """
        ...

    def __call__(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        return self.estimate(prev, curr)
