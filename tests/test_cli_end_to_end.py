"""End-to-end CLI tests with a deterministic offline embedding model."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from lecture_rag.cli import main
from tests.conftest import HashingEmbeddingModel

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample_lectures"


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a CLI runner."""
    return CliRunner()


def test_ingest_and_ask_end_to_end(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ingest sample PDFs and answer a question with citations."""
    monkeypatch.setattr("lecture_rag.cli.EmbeddingModel", HashingEmbeddingModel)
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", HashingEmbeddingModel)

    index_dir = tmp_path / "index"

    ingest_result = cli_runner.invoke(
        main,
        ["ingest", str(SAMPLE_DIR), "--index-dir", str(index_dir)],
    )

    assert ingest_result.exit_code == 0, ingest_result.output
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "documents.json").exists()

    ask_result = cli_runner.invoke(
        main,
        [
            "ask",
            "What data structure does BFS use for traversal?",
            "--index-dir",
            str(index_dir),
            "--mode",
            "extractive",
            "--retrieval",
            "bm25",
            "--abstain-threshold",
            "0",
        ],
    )

    assert ask_result.exit_code == 0, ask_result.output
    assert "algorithms.pdf, page 2" in ask_result.output
