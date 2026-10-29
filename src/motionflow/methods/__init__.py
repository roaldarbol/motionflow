from __future__ import annotations

from motionflow.methods.base import FlowEstimator
from motionflow.methods.classical import FarnebackFlow, LucasKanadeFlow, TVL1Flow
from motionflow.methods.deep import NeuFlowV2Flow, RAFTFlow, SEARaftFlow
from motionflow.types import Method, ModelSize

__all__ = [
    "FlowEstimator",
    "FarnebackFlow",
    "TVL1Flow",
    "LucasKanadeFlow",
    "RAFTFlow",
    "SEARaftFlow",
    "NeuFlowV2Flow",
    "make_estimator",
]


def make_estimator(
    method: Method | str,
    device: str = "auto",
    model_size: ModelSize | str = "large",
) -> FlowEstimator:
    """Construct a flow estimator by name.

    Args:
        method: Flow method identifier (see :class:`~motionflow.types.Method`).
        device: Compute device for deep methods. ``"auto"`` picks the best
            available (CUDA > MPS > CPU).
        model_size: Model size for RAFT and SEA-RAFT (``"small"``/``"medium"``
            or ``"large"``).

    Returns:
        A configured :class:`FlowEstimator` instance ready to call.

    Raises:
        ValueError: For unknown method strings.
        RuntimeError: When required dependencies are missing.
    """
    m = Method(method)
    # .value, not str(): str() of a str-Enum member is 'ModelSize.SMALL'.
    s = model_size.value if isinstance(model_size, ModelSize) else str(model_size)

    if m == Method.FARNEBACK:
        return FarnebackFlow()
    if m == Method.TVL1:
        return TVL1Flow()
    if m == Method.LUCAS_KANADE:
        return LucasKanadeFlow()
    if m == Method.RAFT:
        return RAFTFlow(size=s, device=device)
    if m == Method.SEA_RAFT:
        return SEARaftFlow(size=s, device=device)
    if m == Method.NEUFLOW:
        return NeuFlowV2Flow(device=device)

    raise ValueError(f"Unknown method: {method!r}")
