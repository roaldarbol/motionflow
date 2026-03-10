"""Vendor setup: downloads architecture source files from upstream research repos.

Run once after installing motionflow if you want to use SEA-RAFT or NeuFlow v2:

    python -m motionflow.methods._vendor_setup

Files are placed in motionflow/methods/_vendor/ alongside the installed package.
Re-running updates to the latest upstream commit.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Pinned commits — update these when you want to track a newer upstream version.
# Using full SHAs ensures reproducibility.
# ---------------------------------------------------------------------------

SEA_RAFT_COMMIT = "main"  # pin to a SHA for reproducibility, e.g. "a1b2c3d"
NEUFLOW_V2_COMMIT = "master"

_VENDOR_DIR = Path(__file__).parent / "_vendor"

# ---------------------------------------------------------------------------
# SEA-RAFT: core/ directory files needed for inference
# (architecture only — no training, datasets, or submission code)
# ---------------------------------------------------------------------------

_SEA_RAFT_BASE = (
    f"https://raw.githubusercontent.com/princeton-vl/SEA-RAFT/{SEA_RAFT_COMMIT}"
)
_SEA_RAFT_FILES: list[tuple[str, str]] = [
    # (remote path relative to repo root, local path relative to _vendor/sea_raft/)
    ("core/sea_raft.py", "sea_raft.py"),
    ("core/extractor.py", "extractor.py"),
    ("core/update.py", "update.py"),
    ("core/corr.py", "corr.py"),
    ("core/utils/utils.py", "utils/utils.py"),
    ("LICENSE", "LICENSE"),
]

# ---------------------------------------------------------------------------
# NeuFlow v2: inference-only files
# ---------------------------------------------------------------------------

_NEUFLOW_BASE = (
    f"https://raw.githubusercontent.com/neufieldrobotics/NeuFlow_v2/{NEUFLOW_V2_COMMIT}"
)
_NEUFLOW_FILES: list[tuple[str, str]] = [
    ("neuflow.py", "neuflow.py"),
    ("backbone.py", "backbone.py"),
    ("refine.py", "refine.py"),
    ("LICENSE", "LICENSE"),
]


def _fetch(remote_url: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Fetching {remote_url}", flush=True)
    try:
        urllib.request.urlretrieve(remote_url, local_path)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Failed to download {remote_url}: {exc}\n"
            "Check your internet connection and that the file path is correct.\n"
            "If the upstream repo has changed, update the paths in _vendor_setup.py."
        ) from exc


def setup_sea_raft(force: bool = False) -> None:
    """Download SEA-RAFT architecture files into the vendor directory."""
    dest_dir = _VENDOR_DIR / "sea_raft"
    sentinel = dest_dir / "sea_raft.py"

    if sentinel.exists() and not force:
        print("SEA-RAFT vendor files already present. Pass force=True to re-download.")
        return

    print("Fetching SEA-RAFT architecture files...")
    for remote_rel, local_rel in _SEA_RAFT_FILES:
        _fetch(f"{_SEA_RAFT_BASE}/{remote_rel}", dest_dir / local_rel)

    # Write an __init__ that exposes the model class cleanly
    (dest_dir / "__init__.py").write_text(
        "from .sea_raft import RAFT as SEARaft  # noqa: F401\n"
    )
    print("SEA-RAFT vendor setup complete.")


def setup_neuflow_v2(force: bool = False) -> None:
    """Download NeuFlow v2 architecture files into the vendor directory."""
    dest_dir = _VENDOR_DIR / "neuflow_v2"
    sentinel = dest_dir / "neuflow.py"

    if sentinel.exists() and not force:
        print("NeuFlow v2 vendor files already present. Pass force=True to re-download.")
        return

    print("Fetching NeuFlow v2 architecture files...")
    for remote_rel, local_rel in _NEUFLOW_FILES:
        _fetch(f"{_NEUFLOW_BASE}/{remote_rel}", dest_dir / local_rel)

    (dest_dir / "__init__.py").write_text(
        "from .neuflow import NeuFlow  # noqa: F401\n"
    )
    print("NeuFlow v2 vendor setup complete.")


def setup_all(force: bool = False) -> None:
    """Download all vendor architecture files."""
    setup_sea_raft(force=force)
    setup_neuflow_v2(force=force)


if __name__ == "__main__":
    force = "--force" in sys.argv
    setup_all(force=force)
    print("\nAll vendor files ready. SEA-RAFT and NeuFlow v2 are now available.")
