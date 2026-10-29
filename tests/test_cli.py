"""Tests for CLI functionality."""


import pytest
from click.testing import CliRunner

from src.lecture_rag.cli import main


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
