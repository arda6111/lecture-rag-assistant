"""Tests for the corpus index cache directory and atomic builds."""

import os
import stat
from pathlib import Path

import pytest

from lecture_rag import corpus_index
from lecture_rag.corpus_index import (
    CORPUS_INDEX_NAME,
    build_index_atomically,
    corpus_cache_dir,
)
from lecture_rag.index import VectorIndex
from tests.conftest import HashingEmbeddingModel

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample_lectures"
MODEL_NAME = "all-MiniLM-L6-v2"


def test_corpus_cache_dir_creates_private_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The cache directory is created under XDG_CACHE_HOME with mode 0700."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    cache_dir = corpus_cache_dir()

    assert cache_dir == tmp_path / "lecture-rag"
    assert cache_dir.is_dir()
    assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700


def test_corpus_cache_dir_falls_back_for_shared_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A group or world accessible cache directory is not used."""
    shared_dir = tmp_path / "lecture-rag"
    shared_dir.mkdir()
    shared_dir.chmod(0o777)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    cache_dir = corpus_cache_dir()

    assert cache_dir != shared_dir
    assert shared_dir not in cache_dir.parents
    assert cache_dir.is_dir()


def test_corpus_cache_dir_falls_back_for_symlinked_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A symlinked cache directory is not used."""
    real_dir = tmp_path / "real-cache"
    real_dir.mkdir()
    link = tmp_path / "lecture-rag"
    os.symlink(real_dir, link)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    cache_dir = corpus_cache_dir()

    assert cache_dir != link
    assert cache_dir.is_dir()
    assert not cache_dir.is_symlink()


def test_corpus_cache_dir_falls_back_for_another_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A cache directory owned by another uid is not used."""
    if not hasattr(os, "getuid"):
        pytest.skip("os.getuid is not available on this platform")

    owned_dir = tmp_path / "lecture-rag"
    owned_dir.mkdir(mode=0o700)
    current_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: current_uid + 1)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    cache_dir = corpus_cache_dir()

    assert cache_dir != owned_dir
    assert cache_dir.is_dir()


def test_build_index_atomically_installs_loadable_index(tmp_path: Path) -> None:
    """An atomic build installs a loadable index and leaves no temporary dirs."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    index_dir = build_index_atomically(
        SAMPLE_DIR,
        cache_dir,
        HashingEmbeddingModel(),
        MODEL_NAME,
    )

    assert index_dir == cache_dir / CORPUS_INDEX_NAME
    index = VectorIndex.load(index_dir)
    assert index.documents
    assert index.model_name == MODEL_NAME
    assert list(cache_dir.glob(".corpus-index-*")) == []


def test_build_index_atomically_keeps_existing_index_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed build removes its temporary dir and keeps the existing index."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    index_dir = build_index_atomically(
        SAMPLE_DIR,
        cache_dir,
        HashingEmbeddingModel(),
        MODEL_NAME,
    )
    before = (index_dir / "documents.json").read_bytes()

    def failing_build(*args: object, **kwargs: object) -> int:
        raise RuntimeError("build failed")

    monkeypatch.setattr(corpus_index, "build_index", failing_build)

    with pytest.raises(RuntimeError, match="build failed"):
        build_index_atomically(
            SAMPLE_DIR,
            cache_dir,
            HashingEmbeddingModel(),
            MODEL_NAME,
        )

    assert (index_dir / "documents.json").read_bytes() == before
    assert list(cache_dir.glob(".corpus-index-*")) == []


def test_build_index_atomically_replaces_stale_index(tmp_path: Path) -> None:
    """An existing stale index directory is replaced by the new build."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stale_dir = cache_dir / CORPUS_INDEX_NAME
    stale_dir.mkdir()
    (stale_dir / "stale.txt").write_text("stale", encoding="utf-8")

    index_dir = build_index_atomically(
        SAMPLE_DIR,
        cache_dir,
        HashingEmbeddingModel(),
        MODEL_NAME,
    )

    assert not (index_dir / "stale.txt").exists()
    assert (index_dir / "index.faiss").exists()
    assert VectorIndex.load(index_dir).documents


def test_build_index_atomically_replaces_target_symlink(tmp_path: Path) -> None:
    """A symlink at the target is replaced without touching its destination."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    marker = destination / "marker.txt"
    marker.write_text("keep me", encoding="utf-8")
    target = cache_dir / CORPUS_INDEX_NAME
    os.symlink(destination, target)

    index_dir = build_index_atomically(
        SAMPLE_DIR,
        cache_dir,
        HashingEmbeddingModel(),
        MODEL_NAME,
    )

    assert index_dir == target
    assert not index_dir.is_symlink()
    assert index_dir.is_dir()
    assert marker.read_text(encoding="utf-8") == "keep me"
    assert VectorIndex.load(index_dir).documents
