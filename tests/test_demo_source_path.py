"""Tests that the demo imports the checkout's lecture_rag sources."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import demo

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_stale_package(root: Path) -> Path:
    """Create a stale lecture_rag package that shadows the checkout.

    Args:
        root: Directory in which to create the stale package.

    Returns:
        The directory to put first on sys.path.
    """
    package_dir = root / "stale" / "lecture_rag"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "corpus_index.py").write_text(
        'CORPUS_INDEX_NAME = "old"\n',
        encoding="utf-8",
    )
    return root / "stale"


def remove_lecture_rag_modules() -> None:
    """Remove lecture_rag and its submodules from sys.modules."""
    for name in list(sys.modules):
        if name == "lecture_rag" or name.startswith("lecture_rag."):
            del sys.modules[name]


def test_use_repo_sources_reloads_the_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stale package is replaced by the checkout's package."""
    stale_dir = write_stale_package(tmp_path)
    monkeypatch.setattr(sys, "path", [str(stale_dir), *sys.path])
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    remove_lecture_rag_modules()

    stale_index = importlib.import_module("lecture_rag.corpus_index")
    assert not hasattr(stale_index, "resolve_corpus_dir")

    assert demo.use_repo_sources(REPO_ROOT / "src") is True

    checkout_index = importlib.import_module("lecture_rag.corpus_index")
    checkout_src = (REPO_ROOT / "src").resolve()
    assert hasattr(checkout_index, "resolve_corpus_dir")
    assert Path(checkout_index.__file__).resolve().is_relative_to(checkout_src)
    assert sys.path[0] == str(REPO_ROOT / "src")
    assert sum(1 for entry in sys.path if Path(entry).resolve() == checkout_src) == 1


def test_use_repo_sources_ignores_a_directory_without_the_package(
    tmp_path: Path,
) -> None:
    """A directory without lecture_rag leaves sys.path unchanged."""
    original = list(sys.path)

    assert demo.use_repo_sources(tmp_path / "not_src") is False
    assert sys.path == original


def test_subprocess_import_uses_checkout_sources(tmp_path: Path) -> None:
    """The demo loads lecture_rag from the checkout even with a stale PYTHONPATH."""
    stale_dir = write_stale_package(tmp_path)
    code = (
        "import sys; "
        f"sys.path.insert(1, {str(REPO_ROOT)!r}); "
        "import demo, lecture_rag.corpus_index as c; "
        "print(c.__file__); "
        'print(hasattr(c, "resolve_corpus_dir"))'
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stale_dir)
    env["HF_HUB_OFFLINE"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )

    lines = result.stdout.strip().splitlines()
    assert len(lines) == 2
    assert Path(lines[0]).resolve().is_relative_to((REPO_ROOT / "src").resolve())
    assert lines[1] == "True"
