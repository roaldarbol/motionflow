from enum import Enum


class Method(str, Enum):
    """Optical flow estimation method."""

    DIS = "dis"
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


class NormalizeMode(str, Enum):
    """How flow magnitudes are scaled to colours.

    - ``auto``: one scale for the whole video, from a percentile of its
      motion; exact for short videos, sampled for long ones.
    - ``global``: the same, from every frame (a full extra pass).
    - ``per-frame``: each frame scaled to its own largest motion. Still
      scenes then look busy, since their noise is stretched to full scale.
    """

    AUTO = "auto"
    GLOBAL = "global"
    PER_FRAME = "per-frame"


class ModelSize(str, Enum):
    """Model size for deep learning methods."""

    SMALL = "small"
    LARGE = "large"
