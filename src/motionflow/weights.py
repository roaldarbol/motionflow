"""Model weights: where they come from, and a checksummed download cache.

Weights download automatically on first use, with a progress bar, into
``~/.cache/motionflow/weights`` (or ``$MOTIONFLOW_CACHE_DIR/weights``). Each
file is checked against a SHA-256 checksum before it's used. To fetch them
ahead of time, for example on a cluster login node before a batch job, run
``motionflow download``.

RAFT's weights come from torchvision, which keeps its own cache.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WeightFile:
    """A downloadable weight file.

    Attributes:
        name: Key used in motionflow, e.g. ``"sea-raft-medium"``.
        url: Where to download it from.
        sha256: Expected SHA-256 checksum.
        size: Size in bytes, for the progress bar.
        source: Where the weights originate, for the docs.
    """

    name: str
    url: str
    sha256: str
    size: int
    source: str

    @property
    def filename(self) -> str:
        return Path(self.url).name


_NEUFLOW_SHA = "204b5e3744461d90303b9ff82caa7a1bb56a2ca2"

WEIGHTS: dict[str, WeightFile] = {
    w.name: w
    for w in [
        # SEA-RAFT's own release is on Hugging Face (MemorySlices/*); ptlflow
        # mirrors the same checkpoints on GitHub, which is reachable from more
        # networks. The file names carry the start of their SHA-256.
        WeightFile(
            "sea-raft-small",
            "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/"
            "sea_raft_s-spring-4d13c106.ckpt",
            "4d13c106d8b1158eb11e1a878a3e3d0d886b8eb8f8d05763c87b0f256995d992",
            35_660_885,
            "SEA-RAFT S, trained on Tartan, Chairs, Things, Sintel, KITTI, HD1K and Spring",
        ),
        WeightFile(
            "sea-raft-medium",
            "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/"
            "sea_raft_m-spring-de7c13e2.ckpt",
            "de7c13e2b6a402bdd45d44ba7028e19e052d6788f61da98f23cc2ac6d808e1af",
            78_892_661,
            "SEA-RAFT M, trained on Tartan, Chairs, Things, Sintel, KITTI, HD1K and Spring",
        ),
        WeightFile(
            "neuflow-v2",
            f"https://github.com/neufieldrobotics/NeuFlow_v2/raw/{_NEUFLOW_SHA}/neuflow_mixed.pth",
            "76152c8068f247a7d073aa13e61da8cb4c3c6a798076d4dc8e20f7995fcc019f",
            36_195_519,
            "NeuFlow v2, mixed training set (the authors' recommended checkpoint)",
        ),
    ]
}


def cache_dir() -> Path:
    """The weights cache: ``$MOTIONFLOW_CACHE_DIR/weights`` or ``~/.cache/motionflow/weights``.

    Created if it doesn't exist.
    """
    root = os.environ.get("MOTIONFLOW_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "motionflow"
    path = base / "weights"
    path.mkdir(parents=True, exist_ok=True)
    return path


def weights_path(name: str) -> Path:
    """Where the weight file ``name`` is (or would be) cached."""
    return cache_dir() / _lookup(name).filename


def is_cached(name: str) -> bool:
    """Whether the weight file ``name`` is in the cache."""
    return weights_path(name).exists()


def fetch(name: str, progress: bool = True) -> Path:
    """Return the cached path of weight file ``name``, downloading it if needed.

    Raises:
        RuntimeError: If the download fails or the checksum doesn't match.
    """
    weight = _lookup(name)
    path = weights_path(name)
    if path.exists():
        return path

    if progress:
        print(
            f"Downloading {weight.name} weights ({weight.size / 1e6:.0f} MB) to {path}",
            file=sys.stderr,
        )
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        digest = _download(weight, tmp_path, progress)
        if digest != weight.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {weight.name} weights from {weight.url}: "
                f"expected {weight.sha256}, got {digest}."
            )
        os.replace(tmp_path, path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            f"Couldn't download {weight.name} weights from {weight.url}: {exc}. "
            f"Download the file yourself and save it as {path}."
        ) from exc
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def _lookup(name: str) -> WeightFile:
    try:
        return WEIGHTS[name]
    except KeyError:
        raise ValueError(f"Unknown weights {name!r}; known: {', '.join(WEIGHTS)}") from None


def _download(weight: WeightFile, dest: Path, progress: bool) -> str:
    """Stream ``weight.url`` to ``dest`` and return the file's SHA-256."""
    from tqdm.auto import tqdm

    digest = hashlib.sha256()
    request = urllib.request.Request(weight.url, headers={"User-Agent": "motionflow"})
    with (
        urllib.request.urlopen(request, timeout=60) as response,
        open(dest, "wb") as out,
        tqdm(
            total=weight.size,
            unit="B",
            unit_scale=True,
            desc=weight.name,
            disable=not progress,
        ) as bar,
    ):
        while chunk := response.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


def clear(name: str) -> None:
    """Remove a cached weight file (e.g. to force a fresh download)."""
    weights_path(name).unlink(missing_ok=True)


__all__ = ["WEIGHTS", "WeightFile", "cache_dir", "clear", "fetch", "is_cached", "weights_path"]
