"""Tests for the corpus index cache directory and safe installs."""

import os
import shutil
import stat
import tempfile
from pathlib import Path

import pytest

from lecture_rag import corpus_index
from lecture_rag.corpus_index import (
    CORPUS_INDEX_NAME,
    build_and_install_index,
    corpus_cache_dir,
    corpus_fingerprint,
    corpus_index_name,
    resolve_corpus_dir,
)
from lecture_rag.index import VectorIndex
from tests.conftest import HashingEmbeddingModel

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample_lectures"
MODEL_NAME = "all-MiniLM-L6-v2"


def use_fallback_tempdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Point the process fallback cache at a test-owned temporary directory."""
    tempdir = tmp_path / "tmp"
    tempdir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tempdir))
    monkeypatch.setattr(corpus_index, "_fallback_dir", None)
    return tempdir


def assert_private_fallback(cache_dir: Path, tempdir: Path) -> None:
    """Assert that a cache path is the expected private fallback directory."""
    assert cache_dir.is_relative_to(tempdir)
    assert cache_dir.is_dir()
    assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700


def test_resolve_corpus_dir_prefers_app_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The corpus next to the application directory is preferred over the cwd."""
    app_dir = tmp_path / "app"
    app_corpus = app_dir / "data" / "corpus"
    app_corpus.mkdir(parents=True)
    cwd_dir = tmp_path / "cwd"
    (cwd_dir / "data" / "corpus").mkdir(parents=True)
    monkeypatch.chdir(cwd_dir)

    assert resolve_corpus_dir(app_dir) == app_corpus


def test_resolve_corpus_dir_falls_back_to_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The cwd corpus is used when the application directory has none."""
    cwd_dir = tmp_path / "cwd"
    cwd_corpus = cwd_dir / "data" / "corpus"
    cwd_corpus.mkdir(parents=True)
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    assert resolve_corpus_dir(app_dir) == cwd_corpus


def test_resolve_corpus_dir_returns_app_candidate_when_none_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The expected app location is returned when no corpus directory exists."""
    app_dir = tmp_path / "app"
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    assert resolve_corpus_dir(app_dir) == app_dir / "data" / "corpus"


def test_resolve_corpus_dir_ignores_installed_package_location(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-editable install still resolves the corpus from the app directory."""
    fake_file = tmp_path / "venv/lib/python3.12/site-packages/lecture_rag/corpus_index.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(corpus_index, "__file__", str(fake_file))
    monkeypatch.chdir(tmp_path)

    assert resolve_corpus_dir(REPO_ROOT) == REPO_ROOT / "data" / "corpus"
    assert not hasattr(corpus_index, "CORPUS_DIR")


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
    tempdir = use_fallback_tempdir(monkeypatch, tmp_path)
    shared_dir = tmp_path / "lecture-rag"
    shared_dir.mkdir()
    shared_dir.chmod(0o777)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    cache_dir = corpus_cache_dir()

    assert cache_dir != shared_dir
    assert shared_dir not in cache_dir.parents
    assert_private_fallback(cache_dir, tempdir)
    assert corpus_cache_dir() == cache_dir


def test_corpus_cache_dir_falls_back_for_symlinked_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A symlinked cache directory is not used."""
    tempdir = use_fallback_tempdir(monkeypatch, tmp_path)
    real_dir = tmp_path / "real-cache"
    real_dir.mkdir()
    link = tmp_path / "lecture-rag"
    os.symlink(real_dir, link)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    cache_dir = corpus_cache_dir()

    assert cache_dir != link
    assert not cache_dir.is_symlink()
    assert_private_fallback(cache_dir, tempdir)
    assert corpus_cache_dir() == cache_dir


def test_corpus_cache_dir_falls_back_for_another_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A cache directory owned by another uid is not used."""
    if not hasattr(os, "getuid"):
        pytest.skip("os.getuid is not available on this platform")

    tempdir = use_fallback_tempdir(monkeypatch, tmp_path)
    owned_dir = tmp_path / "lecture-rag"
    owned_dir.mkdir(mode=0o700)
    current_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: current_uid + 1)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    cache_dir = corpus_cache_dir()

    assert cache_dir != owned_dir
    assert_private_fallback(cache_dir, tempdir)
    assert corpus_cache_dir() == cache_dir


def test_corpus_cache_dir_recreates_removed_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A new fallback directory is created after the previous one is removed."""
    tempdir = use_fallback_tempdir(monkeypatch, tmp_path)
    shared_dir = tmp_path / "lecture-rag"
    shared_dir.mkdir()
    shared_dir.chmod(0o777)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    first = corpus_cache_dir()
    shutil.rmtree(first)
    second = corpus_cache_dir()

    assert first != second
    assert_private_fallback(second, tempdir)
    assert corpus_cache_dir() == second


def test_build_and_install_index_installs_loadable_index(tmp_path: Path) -> None:
    """A build installs a loadable index and leaves no temporary dirs."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    index_dir = build_and_install_index(
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


def test_build_and_install_index_keeps_existing_index_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed build removes its temporary dir and keeps the existing index."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    index_dir = build_and_install_index(
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
        build_and_install_index(
            SAMPLE_DIR,
            cache_dir,
            HashingEmbeddingModel(),
            MODEL_NAME,
        )

    assert (index_dir / "documents.json").read_bytes() == before
    assert list(cache_dir.glob(".corpus-index-*")) == []


def test_build_and_install_index_replaces_stale_index(tmp_path: Path) -> None:
    """An existing stale index directory is replaced by the new build."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stale_dir = cache_dir / CORPUS_INDEX_NAME
    stale_dir.mkdir()
    (stale_dir / "stale.txt").write_text("stale", encoding="utf-8")

    index_dir = build_and_install_index(
        SAMPLE_DIR,
        cache_dir,
        HashingEmbeddingModel(),
        MODEL_NAME,
    )

    assert not (index_dir / "stale.txt").exists()
    assert (index_dir / "index.faiss").exists()
    assert VectorIndex.load(index_dir).documents


def test_build_and_install_index_replaces_target_symlink(tmp_path: Path) -> None:
    """A symlink at the target is replaced without touching its destination."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    marker = destination / "marker.txt"
    marker.write_text("keep me", encoding="utf-8")
    target = cache_dir / CORPUS_INDEX_NAME
    os.symlink(destination, target)

    index_dir = build_and_install_index(
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


def test_corpus_fingerprint_is_stable(tmp_path: Path) -> None:
    """The fingerprint is a stable SHA-256 digest of the PDF contents."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.pdf").write_bytes(b"first")
    (corpus_dir / "b.PDF").write_bytes(b"second")

    first = corpus_fingerprint(corpus_dir)

    assert first == corpus_fingerprint(corpus_dir)
    assert len(first) == 64


def test_corpus_fingerprint_changes_when_content_changes(tmp_path: Path) -> None:
    """Changing a PDF's content changes the fingerprint."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    pdf_path = corpus_dir / "a.pdf"
    pdf_path.write_bytes(b"first")
    before = corpus_fingerprint(corpus_dir)

    pdf_path.write_bytes(b"second")

    assert corpus_fingerprint(corpus_dir) != before


def test_corpus_fingerprint_changes_when_file_is_added(tmp_path: Path) -> None:
    """Adding a PDF changes the fingerprint."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.pdf").write_bytes(b"first")
    before = corpus_fingerprint(corpus_dir)

    (corpus_dir / "b.pdf").write_bytes(b"second")

    assert corpus_fingerprint(corpus_dir) != before


def test_corpus_fingerprint_changes_when_file_is_renamed(tmp_path: Path) -> None:
    """Renaming a PDF changes the fingerprint even when its content is unchanged."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.pdf").write_bytes(b"content")
    before = corpus_fingerprint(corpus_dir)

    (corpus_dir / "a.pdf").rename(corpus_dir / "b.pdf")

    assert corpus_fingerprint(corpus_dir) != before


def test_corpus_fingerprint_ignores_non_pdf_files(tmp_path: Path) -> None:
    """Files without a PDF suffix do not affect the fingerprint."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.pdf").write_bytes(b"content")
    before = corpus_fingerprint(corpus_dir)

    (corpus_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    (corpus_dir / "nested").mkdir()
    (corpus_dir / "nested" / "b.pdf").write_bytes(b"ignored")

    assert corpus_fingerprint(corpus_dir) == before


def test_corpus_index_name_uses_fingerprint_prefix() -> None:
    """The index name is derived from the first 16 fingerprint characters."""
    fingerprint = "0123456789abcdef" + "f" * 48

    assert corpus_index_name(fingerprint) == "corpus-index-0123456789abcdef"


def test_changed_corpus_builds_a_new_index_and_removes_the_old_one(
    tmp_path: Path,
) -> None:
    """A changed corpus uses its own index directory and removes stale siblings."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    shutil.copy(SAMPLE_DIR / "algorithms.pdf", corpus_dir / "algorithms.pdf")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    first_fingerprint = corpus_fingerprint(corpus_dir)
    first_name = corpus_index_name(first_fingerprint)
    first_index = build_and_install_index(
        corpus_dir,
        cache_dir,
        HashingEmbeddingModel(),
        MODEL_NAME,
        index_name=first_name,
    )

    assert first_index == cache_dir / first_name
    assert VectorIndex.load(first_index).documents

    shutil.copy(SAMPLE_DIR / "data_structures.pdf", corpus_dir / "data_structures.pdf")
    legacy_dir = cache_dir / CORPUS_INDEX_NAME
    legacy_dir.mkdir()
    (legacy_dir / "stale.txt").write_text("stale", encoding="utf-8")

    second_fingerprint = corpus_fingerprint(corpus_dir)
    second_name = corpus_index_name(second_fingerprint)
    assert second_fingerprint != first_fingerprint

    second_index = build_and_install_index(
        corpus_dir,
        cache_dir,
        HashingEmbeddingModel(),
        MODEL_NAME,
        index_name=second_name,
    )

    assert second_index == cache_dir / second_name
    assert VectorIndex.load(second_index).documents
    assert not first_index.exists()
    assert not legacy_dir.exists()
    assert sorted(path.name for path in cache_dir.glob("corpus-index-*")) == [second_name]
