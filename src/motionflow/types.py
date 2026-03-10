from enum import Enum


class Method(str, Enum):
    """Optical flow estimation method."""

    FARNEBACK = "farneback"
    TVL1 = "tvl1"
    LUCAS_KANADE = "lucas-kanade"
    RAFT = "raft"
    SEA_RAFT = "sea-raft"
    NEUFLOW = "neuflow"

    @property
    def requires_torch(self) -> bool:
        return self in {Method.RAFT, Method.SEA_RAFT, Method.NEUFLOW}

    @property
    def requires_contrib(self) -> bool:
        return self == Method.TVL1


class VizStyle(str, Enum):
    """Flow visualisation style."""

    HSV = "hsv"
    HEATMAP = "heatmap"
    QUIVER = "quiver"
    BLEND = "blend"


class NormalizeMode(str, Enum):
    """Flow magnitude normalisation strategy."""

    PER_FRAME = "per-frame"
    GLOBAL = "global"


class ModelSize(str, Enum):
    """Model size for deep learning methods."""

    SMALL = "small"
    LARGE = "large"
