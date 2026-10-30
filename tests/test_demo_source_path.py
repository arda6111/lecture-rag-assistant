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


def write_minimal_package(root: Path) -> Path:
    """Create a minimal lecture_rag package with a single value.

    Args:
        root: Directory in which to create the package.

    Returns:
        The src directory that contains the package.
    """
    src_dir = root / "src"
    package_dir = src_dir / "lecture_rag"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "corpus_index.py").write_text("VALUE = 1\n", encoding="utf-8")
    return src_dir


def test_drop_stale_modules_reloads_changed_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Changing a source file drops the package so the new code is imported."""
    src_dir = write_minimal_package(tmp_path)
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    monkeypatch.setattr(sys, "path", [str(src_dir), *sys.path])
    remove_lecture_rag_modules()

    package = importlib.import_module("lecture_rag")
    index = importlib.import_module("lecture_rag.corpus_index")
    assert index.VALUE == 1
    setattr(package, demo.SOURCE_STAMP_ATTR, demo.source_stamp(src_dir))
    assert demo.drop_stale_modules(src_dir) is False
    assert importlib.import_module("lecture_rag.corpus_index") is index

    module_path = src_dir / "lecture_rag" / "corpus_index.py"
    module_path.write_text(
        "VALUE = 2\n\n\ndef new_feature() -> int:\n    return VALUE\n",
        encoding="utf-8",
    )
    stat = module_path.stat()
    new_mtime = stat.st_mtime_ns + 10_000_000_000
    os.utime(module_path, ns=(new_mtime, new_mtime))

    assert demo.drop_stale_modules(src_dir) is True
    reloaded = importlib.import_module("lecture_rag.corpus_index")
    assert reloaded.VALUE == 2
    assert reloaded.new_feature() == 2


def test_drop_stale_modules_treats_a_missing_stamp_as_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A loaded package without a recorded stamp is removed."""
    src_dir = write_minimal_package(tmp_path)
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    monkeypatch.setattr(sys, "path", [str(src_dir), *sys.path])
    remove_lecture_rag_modules()

    package = importlib.import_module("lecture_rag")
    importlib.import_module("lecture_rag.corpus_index")
    assert not hasattr(package, demo.SOURCE_STAMP_ATTR)

    assert demo.drop_stale_modules(src_dir) is True
    assert "lecture_rag" not in sys.modules


def test_drop_stale_modules_ignores_an_unloaded_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Nothing is dropped when lecture_rag was never imported."""
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    monkeypatch.setattr(sys, "path", list(sys.path))
    remove_lecture_rag_modules()

    assert demo.drop_stale_modules(tmp_path / "src") is False


def test_source_stamp_tracks_python_files_only(tmp_path: Path) -> None:
    """The stamp ignores caches and other files, and follows size and mtime."""
    src_dir = tmp_path / "src"
    package_dir = src_dir / "lecture_rag"
    package_dir.mkdir(parents=True)
    module_path = package_dir / "__init__.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    (package_dir / "notes.txt").write_text("not python\n", encoding="utf-8")
    cache_dir = package_dir / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "cached.py").write_text("VALUE = 99\n", encoding="utf-8")

    first = demo.source_stamp(src_dir)
    assert [entry[0] for entry in first] == ["lecture_rag/__init__.py"]

    module_path.write_text("VALUE = 100\n", encoding="utf-8")
    second = demo.source_stamp(src_dir)
    assert second != first
    assert second[0][2] != first[0][2]

    stat = module_path.stat()
    new_mtime = stat.st_mtime_ns + 10_000_000_000
    os.utime(module_path, ns=(new_mtime, new_mtime))
    third = demo.source_stamp(src_dir)
    assert third != second
    assert third[0][1] != second[0][1]
    assert third[0][2] == second[0][2]


def test_demo_records_the_checkout_source_stamp() -> None:
    """Importing the demo stamps the checkout's lecture_rag package."""
    assert getattr(sys.modules["lecture_rag"], demo.SOURCE_STAMP_ATTR) == demo.source_stamp(
        demo.SRC_DIR
    )


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


def test_drop_stale_modules_ignores_a_directory_without_the_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without src/lecture_rag the loaded package is never dropped."""
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    loaded = sys.modules["lecture_rag"]

    assert demo.drop_stale_modules(tmp_path) is False
    assert sys.modules["lecture_rag"] is loaded
