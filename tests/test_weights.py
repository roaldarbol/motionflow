# Tests for the weights cache (motionflow.weights), without network access:
# downloads come from local file:// URLs.
# - fetch() downloads once, verifies the checksum, and reuses the cache
# - A checksum mismatch or a failed download leaves nothing behind
# - MOTIONFLOW_CACHE_DIR moves the cache; unknown names are rejected
import hashlib
from pathlib import Path

import pytest

from motionflow import weights
from motionflow.weights import WeightFile


@pytest.fixture()
def cache(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("MOTIONFLOW_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache" / "weights"


@pytest.fixture()
def served(tmp_path, monkeypatch) -> WeightFile:
    """A fake weight file served from a file:// URL, registered as 'test'."""
    source = tmp_path / "server" / "model-1234.pth"
    source.parent.mkdir()
    source.write_bytes(b"weights" * 1000)
    weight = WeightFile(
        "test",
        source.as_uri(),
        hashlib.sha256(source.read_bytes()).hexdigest(),
        source.stat().st_size,
        "a test",
    )
    monkeypatch.setitem(weights.WEIGHTS, "test", weight)
    return weight


def test_cache_dir_override(cache):
    assert weights.cache_dir() == cache
    assert cache.is_dir()


def test_fetch_downloads_and_verifies(cache, served):
    assert not weights.is_cached("test")
    path = weights.fetch("test", progress=False)
    assert path == cache / "model-1234.pth"
    assert path.read_bytes() == b"weights" * 1000
    assert weights.is_cached("test")
    assert [p.name for p in cache.iterdir()] == ["model-1234.pth"]


def test_fetch_uses_the_cache(cache, served):
    path = weights.fetch("test", progress=False)
    path.write_bytes(b"already here")  # a second fetch must not download again
    assert weights.fetch("test", progress=False).read_bytes() == b"already here"


def test_checksum_mismatch(cache, served, monkeypatch):
    monkeypatch.setitem(weights.WEIGHTS, "test", WeightFile(
        "test", served.url, "0" * 64, served.size, "a test"
    ))  # fmt: skip
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        weights.fetch("test", progress=False)
    assert list(cache.iterdir()) == []


def test_failed_download(cache, tmp_path, monkeypatch):
    missing = (tmp_path / "nowhere" / "model.pth").as_uri()
    monkeypatch.setitem(weights.WEIGHTS, "test", WeightFile("test", missing, "0" * 64, 1, "x"))
    with pytest.raises(RuntimeError, match="Couldn't download"):
        weights.fetch("test", progress=False)
    assert list(cache.iterdir()) == []


def test_clear(cache, served):
    weights.fetch("test", progress=False)
    weights.clear("test")
    assert not weights.is_cached("test")


def test_unknown_name():
    with pytest.raises(ValueError, match="Unknown weights"):
        weights.fetch("nope")


def test_registered_weights_look_right():
    for weight in weights.WEIGHTS.values():
        assert weight.url.startswith("https://github.com/")
        assert len(weight.sha256) == 64
        assert weight.size > 1_000_000
