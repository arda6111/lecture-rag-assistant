"""Tests for the Streamlit demo configuration errors."""

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_demo() -> AppTest:
    """Run the demo app and return the testing harness."""
    return AppTest.from_file(str(REPO_ROOT / "demo.py"), default_timeout=60).run()


def test_invalid_retrieval_mode_shows_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An invalid retrieval mode is reported as a configuration error."""
    monkeypatch.setenv("LECTURE_RAG_RETRIEVAL_MODE", "bogus")
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(tmp_path / "missing"))

    app = run_demo()
    errors = [str(element.value) for element in app.error]

    assert any("LECTURE_RAG_RETRIEVAL_MODE" in error for error in errors)
    assert not any("Failed to load index" in error for error in errors)


def test_missing_index_shows_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A valid mode with a missing index reports that the index is missing."""
    monkeypatch.setenv("LECTURE_RAG_RETRIEVAL_MODE", "dense")
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(tmp_path / "missing"))

    app = run_demo()
    errors = [str(element.value) for element in app.error]

    assert any("not found" in error for error in errors)
