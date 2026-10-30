"""CLI input validation and error handling tests."""

import json
import logging
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from lecture_rag.cli import main
from lecture_rag.retrieval import Retriever
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


def test_ingest_empty_directory_fails(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ingest exits non-zero before loading the model when there is no PDF."""
    empty = tmp_path / "empty"
    empty.mkdir()

    def fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("embedding model must not be loaded")

    monkeypatch.setattr("lecture_rag.cli.EmbeddingModel", fail_if_called)

    result = runner.invoke(main, ["ingest", str(empty), "--index-dir", str(tmp_path / "idx")])

    assert_clean_failure(result)
    assert f"No PDF files found in {empty}" in result.output


@pytest.mark.parametrize("model", ["", "   "])
def test_ingest_rejects_empty_model(
    runner: CliRunner,
    tmp_path: Path,
    model: str,
) -> None:
    """Ingest rejects an empty or whitespace-only model name."""
    index_dir = tmp_path / "idx"

    result = runner.invoke(
        main,
        ["ingest", str(SAMPLE_DIR), "--model", model, "--index-dir", str(index_dir)],
    )

    assert_clean_failure(result)
    assert "must not be empty" in result.output
    assert not index_dir.exists()


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
    assert "missing index.faiss" in result.output
    assert "Re-run `lecture-rag ingest` to rebuild the index." in result.output


class SmallEmbeddingModel(HashingEmbeddingModel):
    """Offline embedding model that produces 128-dimensional vectors."""

    dimension = 128


def ask_args(index_dir: Path, model: str = "all-MiniLM-L6-v2") -> list[str]:
    """Build ask arguments against a test index."""
    return ["ask", "What is BFS?", "--index-dir", str(index_dir), "--model", model]


def eval_args(index_dir: Path, model: str = "all-MiniLM-L6-v2") -> list[str]:
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


class OtherEmbeddingModel(HashingEmbeddingModel):
    """Offline 384-dimensional model for name mismatch tests."""

    dimension = 384


@pytest.mark.parametrize("command", ["ask", "eval"])
def test_model_name_mismatch_fails_cleanly(
    runner: CliRunner,
    index_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """A different model name with the same dimension produces a clean error."""
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", OtherEmbeddingModel)
    args = (
        ask_args(index_dir, model="other-model")
        if command == "ask"
        else eval_args(index_dir, model="other-model")
    )

    result = runner.invoke(main, args)

    assert_clean_failure(result)
    assert "--allow-model-mismatch" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ["ask", "eval"])
def test_model_name_mismatch_allowed_with_warning(
    runner: CliRunner,
    index_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """Allowing a model name mismatch succeeds with a warning."""
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", OtherEmbeddingModel)
    args = (
        ask_args(index_dir, model="other-model")
        if command == "ask"
        else eval_args(index_dir, model="other-model")
    )
    args.append("--allow-model-mismatch")

    result = runner.invoke(main, args)

    assert result.exit_code == 0, result.output
    assert "Warning:" in result.output


@pytest.mark.parametrize("command", ["ask", "eval"])
def test_prefixed_matching_model_does_not_warn(
    runner: CliRunner,
    index_dir: Path,
    command: str,
) -> None:
    """A prefixed version of the ingest model matches without a warning."""
    model = "sentence-transformers/all-MiniLM-L6-v2"
    args = (
        ask_args(index_dir, model=model) if command == "ask" else eval_args(index_dir, model=model)
    )

    result = runner.invoke(main, args)

    assert result.exit_code == 0, result.output
    assert "Warning:" not in result.output


def test_eval_compare_model_name_mismatch_fails_cleanly(
    runner: CliRunner,
    index_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """eval --compare also rejects a different model name without the flag."""
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", OtherEmbeddingModel)
    args = eval_args(index_dir, model="other-model") + ["--compare"]

    result = runner.invoke(main, args)

    assert_clean_failure(result)
    assert "--allow-model-mismatch" in result.output


@pytest.mark.parametrize("command", ["ask", "eval"])
def test_corrupt_index_metadata_fails_cleanly(
    runner: CliRunner,
    index_dir: Path,
    command: str,
) -> None:
    """Corrupt metadata gives a clean failure without the generic suffix."""
    (index_dir / "index_meta.json").write_text("{not json", encoding="utf-8")
    args = ask_args(index_dir) if command == "ask" else eval_args(index_dir)

    result = runner.invoke(main, args)

    assert_clean_failure(result)
    assert "Re-run `lecture-rag ingest`" in result.output
    assert "Run `lecture-rag ingest` first" not in result.output


class FailingEmbeddingModel:
    """Embedding model that fails if it is constructed."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("embedding model must not be loaded")


@pytest.mark.parametrize("command", ["ask", "eval"])
def test_model_name_mismatch_does_not_load_the_model(
    runner: CliRunner,
    index_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """A model name mismatch is rejected before the embedding model is constructed."""
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", FailingEmbeddingModel)
    args = (
        ask_args(index_dir, model="other-model")
        if command == "ask"
        else eval_args(index_dir, model="other-model")
    )

    result = runner.invoke(main, args)

    assert_clean_failure(result)
    assert "does not match the model the index was built with" in result.output


def test_eval_output_model_name_mismatch_leaves_no_output_directory(
    runner: CliRunner,
    index_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A rejected model leaves no output directory behind."""
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", OtherEmbeddingModel)
    out_dir = tmp_path / "out"
    result = runner.invoke(
        main,
        eval_args(index_dir, model="other-model") + ["--output", str(out_dir)],
    )

    assert_clean_failure(result)
    assert not out_dir.exists()


def test_eval_sweep_model_name_mismatch_fails_cleanly(
    runner: CliRunner,
    index_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """eval --sweep rejects a different model name before sweeping."""
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", OtherEmbeddingModel)
    result = runner.invoke(main, eval_args(index_dir, model="other-model") + ["--sweep"])

    assert_clean_failure(result)
    assert "does not match the model the index was built with" in result.output


def test_dimension_mismatch_fails_cleanly_when_name_mismatch_is_allowed(
    runner: CliRunner,
    index_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allowing a name mismatch does not bypass the dimension check."""
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", SmallEmbeddingModel)
    result = runner.invoke(main, ask_args(index_dir) + ["--allow-model-mismatch"])

    assert_clean_failure(result)
    assert "128-dimensional" in result.output


@pytest.mark.parametrize("command", ["ask", "eval"])
def test_model_name_mismatch_warning_is_printed_once(
    runner: CliRunner,
    index_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    command: str,
) -> None:
    """The warning is printed once and index.py does not log a duplicate."""
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", OtherEmbeddingModel)
    args = (
        ask_args(index_dir, model="other-model")
        if command == "ask"
        else eval_args(index_dir, model="other-model")
    )
    args.append("--allow-model-mismatch")

    with caplog.at_level(logging.WARNING):
        result = runner.invoke(main, args)

    assert result.exit_code == 0, result.output
    assert result.output.count("Warning:") == 1
    assert "differs from the index model" not in caplog.text


def test_model_name_with_a_space_is_shown_as_a_quoted_hint(
    runner: CliRunner,
    index_dir: Path,
) -> None:
    """A model name with a space is quoted when suggested as a command."""
    metadata_path = index_dir / "index_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_name"] = "my model"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = runner.invoke(main, ask_args(index_dir, model="other-model"))

    assert_clean_failure(result)
    assert "--model 'my model'" in result.output


def test_long_index_model_name_is_not_shown_as_a_model_hint(
    runner: CliRunner,
    index_dir: Path,
) -> None:
    """A shortened index model name is not suggested as a runnable command."""
    long_name = "x" * 300
    metadata_path = index_dir / "index_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_name"] = long_name
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = runner.invoke(main, ask_args(index_dir, model="other-model"))

    assert_clean_failure(result)
    assert "--model " not in result.output
    assert "shortened" in result.output


def corrupt_index_case(
    index_dir: Path,
    case: str,
    temp_dir: Path,
) -> Path:
    """Apply one corrupt index case and return the path to query."""
    if case == "corrupt_faiss":
        (index_dir / "index.faiss").write_bytes(b"not a faiss index")
        return index_dir
    if case == "corrupt_documents":
        (index_dir / "documents.json").write_text("{not json", encoding="utf-8")
        return index_dir
    if case == "empty_dir":
        empty_dir = temp_dir / "empty_index"
        empty_dir.mkdir()
        return empty_dir
    if case == "symlinked_faiss":
        file_path = index_dir / "index.faiss"
        target = temp_dir / "valid_index.faiss"
        file_path.rename(target)
        os.symlink(target, file_path)
        return index_dir
    if case == "symlinked_documents":
        file_path = index_dir / "documents.json"
        target = temp_dir / "valid_documents.json"
        file_path.rename(target)
        os.symlink(target, file_path)
        return index_dir
    if case == "symlinked_dir":
        link_path = temp_dir / "linked_index"
        os.symlink(index_dir, link_path)
        return link_path
    raise AssertionError(f"unknown case {case}")


@pytest.mark.parametrize(
    "case",
    [
        "corrupt_faiss",
        "corrupt_documents",
        "empty_dir",
        "symlinked_faiss",
        "symlinked_documents",
    ],
)
def test_ask_rejects_invalid_index_cleanly(
    runner: CliRunner,
    index_dir: Path,
    tmp_path: Path,
    case: str,
) -> None:
    """Every invalid index case fails cleanly with the rebuild hint."""
    query_path = corrupt_index_case(index_dir, case, tmp_path)

    result = runner.invoke(main, ["ask", "What is BFS?", "--index-dir", str(query_path)])

    assert_clean_failure(result)
    assert "Re-run `lecture-rag ingest` to rebuild the index." in result.output


def test_ask_rejects_symlinked_index_directory_cleanly(
    runner: CliRunner,
    index_dir: Path,
    tmp_path: Path,
) -> None:
    """A symlinked index directory fails with the real path hint."""
    link_path = tmp_path / "linked_index"
    os.symlink(index_dir, link_path)

    result = runner.invoke(main, ["ask", "What is BFS?", "--index-dir", str(link_path)])

    assert_clean_failure(result)
    assert "Use the real (non-symlink) path" in result.output
    assert "Re-run" not in result.output


def test_ask_accepts_index_under_symlinked_parent(
    runner: CliRunner,
    index_dir: Path,
    tmp_path: Path,
) -> None:
    """An index reached through a symlinked parent directory can be queried."""
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    index_dir.rename(real_parent / "index")
    linked_parent = tmp_path / "linked_parent"
    os.symlink(real_parent, linked_parent)

    result = runner.invoke(main, ask_args(linked_parent / "index"))

    assert result.exit_code == 0, result.output


def test_ask_rejects_dangling_symlinked_index_cleanly(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """A dangling symlinked index directory fails with the real path hint."""
    link_path = tmp_path / "dangling_index"
    os.symlink(tmp_path / "missing_index", link_path)

    result = runner.invoke(main, ask_args(link_path))

    assert_clean_failure(result)
    assert "Use the real (non-symlink) path" in result.output
    assert "Re-run" not in result.output


@pytest.mark.parametrize("command", ["ask", "eval"])
def test_nonexistent_index_fails_cleanly(
    runner: CliRunner,
    tmp_path: Path,
    command: str,
) -> None:
    """A missing index directory fails with a clear message and no traceback."""
    missing = tmp_path / "missing_index"
    args = ask_args(missing) if command == "ask" else eval_args(missing)

    result = runner.invoke(main, args)

    assert_clean_failure(result)
    assert "Index directory not found" in result.output
    assert "Run `lecture-rag ingest` to build the index." in result.output


@pytest.mark.parametrize("command", ["ask", "eval"])
def test_value_error_fallback_has_no_ingest_first_hint(
    runner: CliRunner,
    index_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """An unexpected ValueError is reported without the generic ingest hint."""

    def fail_from_index_path(
        cls: type[Retriever],
        *args: object,
        **kwargs: object,
    ) -> Retriever:
        raise ValueError("boom")

    monkeypatch.setattr(
        Retriever,
        "from_index_path",
        classmethod(fail_from_index_path),
    )
    args = ask_args(index_dir) if command == "ask" else eval_args(index_dir)

    result = runner.invoke(main, args)

    assert_clean_failure(result)
    assert "Could not load the index" in result.output
    assert "boom" in result.output
    assert "Run `lecture-rag ingest` first" not in result.output
