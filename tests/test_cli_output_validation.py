"""Validation and error handling tests for eval --output."""

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from lecture_rag.cli import main
from tests.conftest import HashingEmbeddingModel

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample_lectures"
EVAL_FILE = REPO_ROOT / "eval" / "eval_set.json"


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def index_dir(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Build an index over the sample lectures with the offline embedding model."""
    monkeypatch.setattr("lecture_rag.cli.EmbeddingModel", HashingEmbeddingModel)
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", HashingEmbeddingModel)
    path = tmp_path / "index"
    result = runner.invoke(main, ["ingest", str(SAMPLE_DIR), "--index-dir", str(path)])
    assert result.exit_code == 0, result.output
    return path


def assert_clean_failure(result: object) -> None:
    """The command failed with a usage or Click error, not a traceback."""
    assert getattr(result, "exit_code") != 0
    assert isinstance(getattr(result, "exception"), SystemExit)
    assert "Traceback" not in getattr(result, "output")


def eval_args(index_dir: Path, *extra: str) -> list[str]:
    """Build eval arguments for the bundled eval set."""
    return ["eval", "--eval-file", str(EVAL_FILE), "--index-dir", str(index_dir), *extra]


def test_output_rejects_empty_value(runner: CliRunner, index_dir: Path) -> None:
    """An empty --output is rejected instead of becoming the current directory."""
    result = runner.invoke(main, eval_args(index_dir, "--output", ""))

    assert_clean_failure(result)
    assert "must not be empty" in result.output


@pytest.mark.parametrize("flag", ["--compare", "--sweep"])
def test_output_cannot_be_combined_with_compare_or_sweep(
    runner: CliRunner,
    index_dir: Path,
    tmp_path: Path,
    flag: str,
) -> None:
    """--output already evaluates all modes, so the other modes are rejected."""
    result = runner.invoke(
        main,
        eval_args(index_dir, "--output", str(tmp_path / "out"), flag),
    )

    assert_clean_failure(result)
    assert "cannot be combined with --compare or --sweep" in result.output


def test_output_with_retrieval_warns_and_writes(
    runner: CliRunner,
    index_dir: Path,
    tmp_path: Path,
) -> None:
    """--retrieval is ignored with --output but the results are still written."""
    out_dir = tmp_path / "out"

    result = runner.invoke(
        main,
        eval_args(index_dir, "--output", str(out_dir), "--retrieval", "bm25"),
    )

    assert result.exit_code == 0, result.output
    assert "--retrieval is ignored with --output" in result.output
    assert (out_dir / "results.json").exists()
    data = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert set(data["modes"]) == {"dense", "bm25", "hybrid"}


def test_output_under_regular_file_fails(
    runner: CliRunner,
    index_dir: Path,
    tmp_path: Path,
) -> None:
    """A path below a regular file cannot become an output directory."""
    regular_file = tmp_path / "file"
    regular_file.write_text("not a directory", encoding="utf-8")

    result = runner.invoke(
        main,
        eval_args(index_dir, "--output", str(regular_file / "sub")),
    )

    assert_clean_failure(result)
    assert "Cannot create output directory" in result.output


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to read-only directories")
def test_output_read_only_directory_fails_before_evaluation(
    runner: CliRunner,
    index_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read-only output directory fails before the evaluation runs."""
    out_dir = tmp_path / "readonly"
    out_dir.mkdir()
    out_dir.chmod(0o555)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("evaluation must not run for a read-only output")

    monkeypatch.setattr("lecture_rag.cli.run_full_evaluation", fail_if_called)

    result = runner.invoke(main, eval_args(index_dir, "--output", str(out_dir)))

    assert_clean_failure(result)
    assert "Output directory is not writable" in result.output


def test_output_write_error_is_reported(
    runner: CliRunner,
    index_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write failure is reported as a clean Click error."""
    out_dir = tmp_path / "out"
    original_write_text = Path.write_text

    def fail_write(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if path.parent == out_dir:
            raise PermissionError("denied")
        return original_write_text(path, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", fail_write)

    result = runner.invoke(main, eval_args(index_dir, "--output", str(out_dir)))

    assert_clean_failure(result)
    assert "Could not write results" in result.output
    assert not (out_dir / "results.json").exists()
