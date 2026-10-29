"""Tests for embedding model loading errors."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from lecture_rag.cli import main
from lecture_rag.embeddings import EmbeddingModel, EmbeddingModelError
from tests.conftest import HashingEmbeddingModel

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample_lectures"


def raise_oserror(*args: object, **kwargs: object) -> None:
    """Simulate a missing or unreadable model."""
    raise OSError("not found")


def assert_clean_failure(result: object) -> None:
    """The command failed with a usage or Click error, not a traceback."""
    assert getattr(result, "exit_code") != 0
    assert isinstance(getattr(result, "exception"), SystemExit)
    assert "Traceback" not in getattr(result, "output")


def test_embedding_model_wraps_load_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A load failure becomes an EmbeddingModelError with the model name."""
    monkeypatch.setattr("lecture_rag.embeddings.SentenceTransformer", raise_oserror)

    with pytest.raises(EmbeddingModelError, match="Could not load embedding model 'nope'"):
        EmbeddingModel("nope")


def test_ingest_reports_embedding_model_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ingest fails cleanly when the embedding model cannot be loaded."""
    monkeypatch.setattr("lecture_rag.embeddings.SentenceTransformer", raise_oserror)

    result = CliRunner().invoke(
        main,
        [
            "ingest",
            str(SAMPLE_DIR),
            "--model",
            "nope",
            "--index-dir",
            str(tmp_path / "index"),
        ],
    )

    assert_clean_failure(result)
    assert "Could not load embedding model 'nope'" in result.output


def test_ask_reports_embedding_model_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ask fails cleanly when the retriever cannot load the embedding model."""
    runner = CliRunner()
    index_dir = tmp_path / "index"
    monkeypatch.setattr("lecture_rag.cli.EmbeddingModel", HashingEmbeddingModel)

    ingest_result = runner.invoke(
        main,
        ["ingest", str(SAMPLE_DIR), "--index-dir", str(index_dir)],
    )
    assert ingest_result.exit_code == 0, ingest_result.output

    monkeypatch.setattr("lecture_rag.embeddings.SentenceTransformer", raise_oserror)

    result = runner.invoke(
        main,
        [
            "ask",
            "What is dynamic programming?",
            "--index-dir",
            str(index_dir),
            "--model",
            "nope",
        ],
    )

    assert_clean_failure(result)
    assert "Could not load embedding model 'nope'" in result.output
