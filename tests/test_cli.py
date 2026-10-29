"""Tests for CLI functionality."""

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from lecture_rag.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample_lectures"


@pytest.fixture
def cli_runner():
    """Create a CLI runner."""
    return CliRunner()


def test_cli_help(cli_runner: CliRunner) -> None:
    """Test CLI help message."""
    result = cli_runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "Lecture RAG Assistant" in result.output


def test_ingest_help(cli_runner: CliRunner) -> None:
    """Test ingest command help."""
    result = cli_runner.invoke(main, ["ingest", "--help"])

    assert result.exit_code == 0
    assert "ingest" in result.output.lower()


def test_ask_help(cli_runner: CliRunner) -> None:
    """Test ask command help."""
    result = cli_runner.invoke(main, ["ask", "--help"])

    assert result.exit_code == 0
    assert "ask" in result.output.lower()


def test_eval_help(cli_runner: CliRunner) -> None:
    """Test eval command help."""
    result = cli_runner.invoke(main, ["eval", "--help"])

    assert result.exit_code == 0
    assert "eval" in result.output.lower()


def test_cli_loads_dotenv(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI loads environment values from a .env file."""
    monkeypatch.setenv("LECTURE_RAG_RETRIEVAL_MODE", "dense")
    monkeypatch.delenv("LECTURE_RAG_RETRIEVAL_MODE")

    with cli_runner.isolated_filesystem():
        Path(".env").write_text(
            "LECTURE_RAG_RETRIEVAL_MODE=bm25\n",
            encoding="utf-8",
        )
        result = cli_runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert os.environ.get("LECTURE_RAG_RETRIEVAL_MODE") == "bm25"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--max-file-size-mb", "0"),
        ("--max-file-size-mb", "-1"),
        ("--max-pages", "0"),
        ("--max-pages", "-1"),
    ],
)
def test_ingest_rejects_non_positive_limits(
    cli_runner: CliRunner,
    option: str,
    value: str,
) -> None:
    """Ingest rejects non-positive file size and page limits."""
    result = cli_runner.invoke(
        main,
        ["ingest", str(SAMPLE_DIR), option, value],
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output
