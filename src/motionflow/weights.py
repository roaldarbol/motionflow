from __future__ import annotations

import sys
from pathlib import Path

_CACHE_DIR = Path.home() / ".cache" / "motionflow" / "weights"


def cache_dir() -> Path:
    """Return (and create if needed) the motionflow weights cache directory."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def weights_path(filename: str) -> Path:
    """Full path for a cached weight file."""
    return cache_dir() / filename


def is_cached(filename: str) -> bool:
    """Return True if a weight file already exists in the cache."""
    return weights_path(filename).exists()


def prompt_download(method: str, source: str, filename: str) -> bool:
    """Ask the user whether to download model weights.

    Args:
        method: Human-readable method name shown in the prompt.
        source: URL or HuggingFace repo shown in the prompt.
        filename: The local filename that will be created on acceptance.

    Returns:
        ``True`` if the user accepted, ``False`` if they declined.
    """
    print(
        f"\n  Model weights for {method} are not cached locally.\n"
        f"  Source : {source}\n"
        f"  Destination : {weights_path(filename)}\n",
        file=sys.stderr,
    )
    while True:
        answer = input("  Download now? [y/N] ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        print("  Please enter y or n.", file=sys.stderr)


def require_weights(
    method: str,
    source: str,
    filename: str,
    downloader,
) -> Path:
    """Ensure a weight file is available, prompting the user if needed.

    Args:
        method: Human-readable method name.
        source: URL or HuggingFace repo string shown in the prompt.
        filename: Local cache filename.
        downloader: Zero-argument callable that performs the actual download
            and must place the file at ``weights_path(filename)``.

    Returns:
        Path to the cached weight file.

    Raises:
        RuntimeError: If the user declines the download.
    """
    path = weights_path(filename)
    if path.exists():
        return path

    if not prompt_download(method, source, filename):
        raise RuntimeError(
            f"Weights for {method} are required but were not downloaded. "
            "Re-run and accept the download prompt, or download manually "
            f"from {source} and place the file at {path}."
        )

    downloader()

    if not path.exists():
        raise RuntimeError(
            f"Download for {method} appeared to succeed but the file "
            f"was not found at {path}. Please report this issue."
        )

    return path
