"""CLI input validation and error handling tests."""

import json
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


def assert_clean_failure(result) -> None:  # type: ignore[no-untyped-def]
    """The command failed with a usage or Click error, not a traceback."""
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["--chunk-size", "0"],
        ["--chunk-size", "100", "--chunk-overlap", "100"],
        ["--chunk-size", "100", "--chunk-overlap", "150"],
        ["--chunk-overlap", "-10"],
    ],
)
def test_ingest_rejects_invalid_chunking(
    runner: CliRunner, tmp_path: Path, args: list[str]
) -> None:
    """Invalid chunk settings fail fast instead of looping forever."""
    result = runner.invoke(
        main, ["ingest", str(SAMPLE_DIR), "--index-dir", str(tmp_path / "idx"), *args]
    )

    assert_clean_failure(result)
    assert "chunk" in result.output.lower()
    assert not (tmp_path / "idx").exists()


def test_ingest_empty_directory_fails(runner: CliRunner, tmp_path: Path) -> None:
    """Ingest exits non-zero when there is nothing to index."""
    empty = tmp_path / "empty"
    empty.mkdir()

    result = runner.invoke(main, ["ingest", str(empty), "--index-dir", str(tmp_path / "idx")])

    assert_clean_failure(result)
    assert "No PDF documents could be loaded" in result.output


def test_ingest_only_fake_pdfs_fails(runner: CliRunner, tmp_path: Path) -> None:
    """A directory with only invalid PDFs exits non-zero."""
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    (fake_dir / "notes.pdf").write_text("plain text, not a PDF", encoding="utf-8")

    result = runner.invoke(main, ["ingest", str(fake_dir), "--index-dir", str(tmp_path / "idx")])

    assert_clean_failure(result)
    assert "No PDF documents could be loaded" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["--k", "0"],
        ["--k", "-1"],
        ["--abstain-threshold", "1.5"],
        ["--abstain-threshold", "-1"],
    ],
)
def test_ask_rejects_out_of_range_options(
    runner: CliRunner, index_dir: Path, args: list[str]
) -> None:
    """ask validates k and the abstain threshold."""
    result = runner.invoke(main, ["ask", "What is BFS?", "--index-dir", str(index_dir), *args])

    assert_clean_failure(result)
    assert "Invalid value" in result.output


@pytest.mark.parametrize(
    "args",
    [["--k", "0"], ["--abstain-threshold", "1.5"], ["--abstain-threshold", "-1"]],
)
def test_eval_rejects_out_of_range_options(
    runner: CliRunner, index_dir: Path, args: list[str]
) -> None:
    """eval validates k and the abstain threshold."""
    result = runner.invoke(
        main, ["eval", "--eval-file", str(EVAL_FILE), "--index-dir", str(index_dir), *args]
    )

    assert_clean_failure(result)
    assert "Invalid value" in result.output


def test_invalid_retrieval_mode_env_is_reported(
    runner: CliRunner, index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid LECTURE_RAG_RETRIEVAL_MODE gives a clear error."""
    monkeypatch.setenv("LECTURE_RAG_RETRIEVAL_MODE", "foo")

    for args in (
        ["ask", "What is BFS?", "--index-dir", str(index_dir)],
        ["eval", "--eval-file", str(EVAL_FILE), "--index-dir", str(index_dir)],
    ):
        result = runner.invoke(main, args)
        assert_clean_failure(result)
        assert "Invalid retrieval mode" in result.output


def test_eval_malformed_file_is_reported(
    runner: CliRunner, index_dir: Path, tmp_path: Path
) -> None:
    """A malformed eval file gives a clear error."""
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    result = runner.invoke(
        main, ["eval", "--eval-file", str(broken), "--index-dir", str(index_dir)]
    )

    assert_clean_failure(result)
    assert "Invalid eval file" in result.output


def test_eval_empty_set_is_reported(runner: CliRunner, index_dir: Path, tmp_path: Path) -> None:
    """An empty eval set is an error."""
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"questions": []}), encoding="utf-8")

    result = runner.invoke(main, ["eval", "--eval-file", str(empty), "--index-dir", str(index_dir)])

    assert_clean_failure(result)
    assert "Eval set is empty" in result.output


def test_ask_llm_without_key_fails_before_network(
    runner: CliRunner, index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """llm mode without a key fails with a message and makes no API call."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    def fail_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("network client must not be created")

    monkeypatch.setattr("lecture_rag.generation.OpenAI", fail_if_called)

    result = runner.invoke(
        main,
        ["ask", "What is BFS?", "--index-dir", str(index_dir), "--mode", "llm"],
    )

    assert_clean_failure(result)
    assert "OPENAI_API_KEY" in result.output


def test_ask_missing_index_is_reported(runner: CliRunner, tmp_path: Path) -> None:
    """An index directory without index files gives a clear error."""
    empty_index = tmp_path / "no-index"
    empty_index.mkdir()

    result = runner.invoke(main, ["ask", "What is BFS?", "--index-dir", str(empty_index)])

    assert_clean_failure(result)
    assert "Index files not found" in result.output

class SmallEmbeddingModel(HashingEmbeddingModel):
    """Offline embedding model that produces 128-dimensional vectors."""

    dimension = 128


def ask_args(index_dir: Path, model: str = "small") -> list[str]:
    """Build ask arguments against a test index."""
    return ["ask", "What is BFS?", "--index-dir", str(index_dir), "--model", model]


def eval_args(index_dir: Path, model: str = "small") -> list[str]:
    """Build eval arguments against a test index."""
    return [
        "eval",
        "--eval-file",
        str(EVAL_FILE),
        "--index-dir",
        str(index_dir),
        "--model",
        model,
    ]


@pytest.mark.parametrize("command", ["ask", "eval"])
def test_model_dimension_mismatch_fails_cleanly(
    runner: CliRunner,
    index_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """A model with a different dimension produces a clean CLI error."""
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", SmallEmbeddingModel)
    args = ask_args(index_dir) if command == "ask" else eval_args(index_dir)

    result = runner.invoke(main, args)

    assert_clean_failure(result)
    assert "128-dimensional" in result.output
    assert "Run `lecture-rag ingest` first" not in result.output


@pytest.mark.parametrize("command", ["ask", "eval"])
def test_legacy_index_dimension_mismatch_fails_cleanly(
    runner: CliRunner,
    index_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """A legacy index without metadata still produces a clean dimension error."""
    (index_dir / "index_meta.json").unlink()
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", SmallEmbeddingModel)
    args = ask_args(index_dir) if command == "ask" else eval_args(index_dir)

    result = runner.invoke(main, args)

    assert_clean_failure(result)
    assert "128-dimensional" in result.output
    assert "Run `lecture-rag ingest` first" not in result.output


@pytest.mark.parametrize("command", ["ask", "eval"])
def test_matching_model_still_works(
    runner: CliRunner,
    index_dir: Path,
    command: str,
) -> None:
    """The ingest model can still query the index for ask and eval."""
    args = ask_args(index_dir, model="all-MiniLM-L6-v2")
    if command == "eval":
        args = eval_args(index_dir, model="all-MiniLM-L6-v2")

    result = runner.invoke(main, args)

    assert result.exit_code == 0, result.output
