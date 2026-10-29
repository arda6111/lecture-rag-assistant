"""Offline tests for optional cross-encoder reranking."""

import importlib.util
import json
from pathlib import Path

import pytest
import sentence_transformers
from click.testing import CliRunner

import lecture_rag.rerank
from lecture_rag.cli import main
from lecture_rag.eval import EvalQuestion, EvalSet, evaluate_retrieval, run_full_evaluation
from lecture_rag.index import VectorIndex
from lecture_rag.models import Document, RetrievalResult
from lecture_rag.rerank import (
    DEFAULT_RERANK_MODEL,
    CrossEncoderReranker,
    RerankerError,
    default_rerank_model,
    rerank,
)
from lecture_rag.retrieval import RetrievalMode, RetrievalOutcome, Retriever
from tests.conftest import HashingEmbeddingModel

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample_lectures"
EVAL_FILE = REPO_ROOT / "eval" / "eval_set.json"

SCRIPT_PATH = REPO_ROOT / "scripts" / "run_corpus_eval.py"
_SPEC = importlib.util.spec_from_file_location("run_corpus_eval_rerank", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
run_corpus_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_corpus_eval)


class MappingReranker:
    """Reranker that returns scores from a fixed mapping."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[list[str]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return the configured score for each text."""
        self.calls.append(list(texts))
        return [self.scores.get(text, 0.0) for text in texts]


class ReverseReranker:
    """Reranker that gives later candidates higher scores."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.model_name = "fake-reranker"

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score each text by its position, reversing the input order."""
        self.calls.append(list(texts))
        return [float(index) for index in range(len(texts))]


class OverlapReranker:
    """Reranker that scores by query word overlap."""

    def __init__(self, model_name: str = "fake-reranker") -> None:
        self.model_name = model_name

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score each text by the number of shared words."""
        words = set(query.lower().split())
        return [float(len(words & set(text.lower().split()))) for text in texts]


def make_document(chunk_id: str, text: str) -> Document:
    """Create a document for testing."""
    return Document(
        text=text,
        file_name="test.pdf",
        page_number=1,
        chunk_id=chunk_id,
    )


def make_result(chunk_id: str, text: str, score: float = 1.0) -> RetrievalResult:
    """Create a retrieval result for testing."""
    return RetrievalResult(
        document=make_document(chunk_id, text),
        score=score,
    )


def build_index(count: int = 25) -> tuple[HashingEmbeddingModel, VectorIndex]:
    """Build a deterministic index with the offline hashing model."""
    model = HashingEmbeddingModel()
    documents = [
        Document(
            text=f"alpha document {index}",
            file_name="test.pdf",
            page_number=index + 1,
            chunk_id=f"doc-{index}",
        )
        for index in range(count)
    ]
    index = VectorIndex(model.dimension)
    index.add_documents(documents, model.embed_texts([document.text for document in documents]))
    return model, index


def test_rerank_reorders_top_k_and_sets_score_type() -> None:
    """Reranking sorts by the new score and returns new rerank results."""
    results = [make_result("a", "a"), make_result("b", "b"), make_result("c", "c")]
    reranker = MappingReranker({"a": 0.1, "b": 0.9, "c": 0.5})

    reranked = rerank("query", results, reranker, 2)

    assert [result.document.chunk_id for result in reranked] == ["b", "c"]
    assert [result.score for result in reranked] == [0.9, 0.5]
    assert all(result.score_type == "rerank" for result in reranked)
    assert reranked[0] is not results[1]
    assert reranker.calls == [["a", "b", "c"]]


def test_rerank_breaks_ties_by_original_rank() -> None:
    """Equal rerank scores keep the original first-stage order."""
    results = [make_result("a", "a"), make_result("b", "b"), make_result("c", "c")]
    reranker = MappingReranker({"a": 0.5, "b": 0.5, "c": 0.9})

    reranked = rerank("query", results, reranker, 3)

    assert [result.document.chunk_id for result in reranked] == ["c", "a", "b"]


def test_rerank_empty_results_does_not_call_the_reranker() -> None:
    """Reranking an empty result list returns immediately."""
    reranker = MappingReranker({})

    assert rerank("query", [], reranker, 5) == []
    assert reranker.calls == []


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_search_rerank_reorders_and_preserves_dense_similarity(
    mode: RetrievalMode,
) -> None:
    """Reranking uses the default pool size in every mode."""
    model, index = build_index()
    reranker = ReverseReranker()
    retriever = Retriever(model, index, mode=mode, reranker=reranker)

    base = retriever.search("alpha", k=5, mode=mode)
    pool = retriever.search("alpha", k=20, mode=mode)
    reranked = retriever.search("alpha", k=5, mode=mode, rerank=True)

    assert len(reranker.calls[-1]) == min(20, len(index.documents))
    assert reranker.calls[-1] == [result.document.text for result in pool.results]
    assert [result.document.chunk_id for result in reranked.results] == [
        result.document.chunk_id for result in reversed(pool.results)
    ][:5]
    assert all(result.score_type == "rerank" for result in reranked.results)
    assert reranked.reranked is True
    assert base.reranked is False
    assert reranked.mode == base.mode
    assert reranked.max_dense_similarity == base.max_dense_similarity


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_search_rerank_uses_configured_candidate_pool(mode: RetrievalMode) -> None:
    """rerank_candidates controls how many texts the reranker sees."""
    model, index = build_index()
    reranker = ReverseReranker()
    retriever = Retriever(
        model,
        index,
        mode=mode,
        reranker=reranker,
        rerank_candidates=7,
    )

    retriever.search("alpha", k=5, mode=mode, rerank=True)

    assert len(reranker.calls[-1]) == 7


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_search_rerank_caps_the_pool_at_index_size(mode: RetrievalMode) -> None:
    """The candidate pool never exceeds the number of indexed documents."""
    model, index = build_index(count=3)
    reranker = ReverseReranker()
    retriever = Retriever(model, index, mode=mode, reranker=reranker)

    retriever.search("0 1 2", k=5, mode=mode, rerank=True)

    assert len(reranker.calls[-1]) == 3


def test_search_rerank_without_reranker_raises() -> None:
    """Requesting reranking without a configured reranker is an error."""
    model, index = build_index(count=3)
    retriever = Retriever(model, index, mode="dense")

    with pytest.raises(ValueError, match="reranker"):
        retriever.search("alpha", rerank=True)


def test_default_rerank_model_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rerank model comes from the environment and falls back to the default."""
    monkeypatch.delenv("LECTURE_RAG_RERANK_MODEL", raising=False)
    assert default_rerank_model() == DEFAULT_RERANK_MODEL

    monkeypatch.setenv("LECTURE_RAG_RERANK_MODEL", "   ")
    assert default_rerank_model() == DEFAULT_RERANK_MODEL

    monkeypatch.setenv("LECTURE_RAG_RERANK_MODEL", "custom-reranker")
    assert default_rerank_model() == "custom-reranker"


def test_cross_encoder_loading_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading failures become RerankerError messages that name the model."""

    def fail(*args: object, **kwargs: object) -> object:
        raise OSError("not available")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", fail)

    with pytest.raises(RerankerError, match="'broken-model'"):
        CrossEncoderReranker("broken-model")


def test_importing_rerank_does_not_construct_cross_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the rerank module has no model-loading side effect."""
    calls: list[str] = []

    def fake_cross_encoder(*args: object, **kwargs: object) -> object:
        calls.append("constructed")
        return object()

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", fake_cross_encoder)

    importlib.reload(lecture_rag.rerank)

    assert calls == []


@pytest.fixture
def cli_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Build an offline index over the sample lectures."""
    monkeypatch.setattr("lecture_rag.cli.EmbeddingModel", HashingEmbeddingModel)
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", HashingEmbeddingModel)
    index_dir = tmp_path / "index"
    result = CliRunner().invoke(
        main,
        ["ingest", str(SAMPLE_DIR), "--index-dir", str(index_dir)],
    )
    assert result.exit_code == 0, result.output
    return index_dir


def write_small_eval_file(path: Path) -> Path:
    """Write a small eval file for CLI tests."""
    eval_set = EvalSet(
        questions=[
            EvalQuestion(
                question="What is the Bellman equation?",
                expected_file="algorithms.pdf",
                expected_page=1,
            ),
            EvalQuestion(question="How does TCP congestion control work?"),
        ]
    )
    path.write_text(eval_set.model_dump_json(), encoding="utf-8")
    return path


def fake_reranker_class(instances: list[str]) -> type:
    """Create a fake CLI reranker that records model names."""

    class FakeReranker:
        def __init__(self, model_name: str) -> None:
            instances.append(model_name)

        def score(self, query: str, texts: list[str]) -> list[float]:
            return [float(index) for index in range(len(texts))]

    return FakeReranker


@pytest.mark.parametrize("command", ["ask", "eval"])
def test_cli_without_rerank_never_constructs_cross_encoder(
    command: str,
    cli_index: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ask and eval leave the cross-encoder untouched without --rerank."""

    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("CrossEncoder must not be constructed")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", fail)
    runner = CliRunner()
    if command == "ask":
        args = [
            "ask",
            "What is BFS?",
            "--index-dir",
            str(cli_index),
            "--abstain-threshold",
            "0",
        ]
    else:
        eval_file = write_small_eval_file(tmp_path / "eval.json")
        args = ["eval", "--eval-file", str(eval_file), "--index-dir", str(cli_index)]

    result = runner.invoke(main, args)

    assert result.exit_code == 0, result.output


def test_cli_ask_rerank_uses_fake_reranker(
    cli_index: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ask --rerank attaches a reranker and labels sources with rerank scores."""
    instances: list[str] = []
    monkeypatch.setattr(
        "lecture_rag.cli.CrossEncoderReranker",
        fake_reranker_class(instances),
    )

    result = CliRunner().invoke(
        main,
        [
            "ask",
            "What is the Bellman equation?",
            "--index-dir",
            str(cli_index),
            "--abstain-threshold",
            "0",
            "--rerank",
            "--rerank-candidates",
            "6",
            "--rerank-model",
            "fake-model",
        ],
    )

    assert result.exit_code == 0, result.output
    assert instances == ["fake-model"]
    assert "(rerank: " in result.output


@pytest.mark.parametrize("args", [[], ["--compare"]])
def test_cli_eval_rerank_modes(
    args: list[str],
    cli_index: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """eval and eval --compare accept --rerank through a fake reranker."""
    instances: list[str] = []
    monkeypatch.setattr(
        "lecture_rag.cli.CrossEncoderReranker",
        fake_reranker_class(instances),
    )
    eval_file = write_small_eval_file(tmp_path / "eval.json")

    result = CliRunner().invoke(
        main,
        [
            "eval",
            "--eval-file",
            str(eval_file),
            "--index-dir",
            str(cli_index),
            "--rerank",
            *args,
        ],
    )

    assert result.exit_code == 0, result.output
    assert instances == [DEFAULT_RERANK_MODEL]


def test_cli_eval_output_rerank_records_command(
    cli_index: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """eval --output records the rerank options in results.json."""
    instances: list[str] = []
    monkeypatch.setattr(
        "lecture_rag.cli.CrossEncoderReranker",
        fake_reranker_class(instances),
    )
    eval_file = write_small_eval_file(tmp_path / "eval.json")
    out_dir = tmp_path / "out"

    result = CliRunner().invoke(
        main,
        [
            "eval",
            "--eval-file",
            str(eval_file),
            "--index-dir",
            str(cli_index),
            "--rerank",
            "--rerank-candidates",
            "8",
            "--rerank-model",
            "fake-model",
            "--output",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert data["rerank"] is True
    assert " --rerank" in data["command"]
    assert "--rerank-candidates 8" in data["command"]
    assert "--rerank-model fake-model" in data["command"]


def test_cli_rerank_candidates_without_rerank_is_a_usage_error(cli_index: Path) -> None:
    """Rerank-only options require --rerank."""
    result = CliRunner().invoke(
        main,
        [
            "ask",
            "What is BFS?",
            "--index-dir",
            str(cli_index),
            "--rerank-candidates",
            "3",
        ],
    )

    assert result.exit_code == 2
    assert "--rerank-candidates requires --rerank" in result.output
    assert "Traceback" not in result.output


def test_cli_reranker_error_is_a_clean_failure(
    cli_index: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RerankerError becomes a Click error without a traceback."""

    class FailingReranker:
        def __init__(self, model_name: str) -> None:
            raise RerankerError(f"Could not load reranker model '{model_name}': boom")

    monkeypatch.setattr("lecture_rag.cli.CrossEncoderReranker", FailingReranker)

    result = CliRunner().invoke(
        main,
        [
            "ask",
            "What is BFS?",
            "--index-dir",
            str(cli_index),
            "--rerank",
        ],
    )

    assert result.exit_code == 1
    assert "Could not load reranker model 'cross-encoder/ms-marco-MiniLM-L-6-v2'" in result.output
    assert "Traceback" not in result.output


class RecordingRetriever:
    """Retriever stub that records whether reranking was requested."""

    def __init__(self) -> None:
        self.rerank_calls: list[bool] = []
        self.document = Document(
            text="answer text",
            file_name="a.pdf",
            page_number=1,
            chunk_id="a-1",
        )

    def search(
        self,
        query: str,
        k: int = 5,
        mode: str | None = None,
        rerank: bool = False,
    ) -> RetrievalOutcome:
        """Return one matching result and record the rerank flag."""
        self.rerank_calls.append(rerank)
        return RetrievalOutcome(
            results=[RetrievalResult(document=self.document, score=1.0)],
            max_dense_similarity=0.9,
            mode=mode or "dense",
            reranked=rerank,
        )


def test_evaluate_retrieval_passes_rerank_to_search() -> None:
    """evaluate_retrieval forwards rerank=True to the retriever."""
    retriever = RecordingRetriever()
    eval_set = EvalSet(
        questions=[
            EvalQuestion(question="q", expected_file="a.pdf", expected_page=1),
        ]
    )

    metrics = evaluate_retrieval(retriever, eval_set, k=5, rerank=True)  # type: ignore[arg-type]

    assert metrics["hit_at_5"] == 1.0
    assert retriever.rerank_calls == [True]


def test_run_full_evaluation_passes_rerank_and_records_it() -> None:
    """run_full_evaluation reranks every call and includes rerank in its result."""
    retriever = RecordingRetriever()
    eval_set = EvalSet(
        questions=[
            EvalQuestion(question="q", expected_file="a.pdf", expected_page=1),
        ]
    )

    results = run_full_evaluation(
        retriever,  # type: ignore[arg-type]
        eval_set,
        eval_file="eval/small.json",
        k=5,
        abstain_threshold=0.35,
        model_name="hashing",
        indexed_chunks=1,
        command="lecture-rag eval --output eval",
        rerank=True,
    )

    assert results["rerank"] is True
    assert set(results["modes"]) == {"dense", "bm25", "hybrid"}
    assert retriever.rerank_calls
    assert all(retriever.rerank_calls)


def test_compute_corpus_results_with_reranker() -> None:
    """The corpus script records reranked modes and latency when given a reranker."""
    reranker = OverlapReranker()

    results = run_corpus_eval.compute_corpus_results(
        HashingEmbeddingModel(),
        "hashing",
        reranker=reranker,
        rerank_model_name="fake-reranker",
        corpus_dir=SAMPLE_DIR,
        eval_file=EVAL_FILE,
        chunk_sizes=(512,),
    )

    assert set(results["reranker"]["modes"]) == {"dense", "bm25", "hybrid"}
    assert results["reranker"]["model"] == "fake-reranker"
    assert results["reranker"]["candidates"] == 20
    assert set(results["latency_ms"]) == {"dense", "bm25", "hybrid"}
    for mode in ("dense", "bm25", "hybrid"):
        assert set(results["latency_ms"][mode]) == {"base", "rerank"}
        assert isinstance(results["latency_ms"][mode]["base"], float)
        assert isinstance(results["latency_ms"][mode]["rerank"], float)


def synthetic_corpus_results(with_rerank: bool) -> dict[str, object]:
    """Build a minimal corpus result dictionary for rendering tests."""
    metrics = {
        "hit_at_k": 0.5,
        "mrr_at_k": 0.5,
        "hit_at_1": 0.25,
        "mrr_at_1": 0.25,
        "abstain_accuracy": 0.5,
        "abstain_precision": 0.5,
        "abstain_recall": 0.5,
        "hit_at_k_after_abstain": 0.5,
    }
    results: dict[str, object] = {
        "k": 5,
        "abstain_threshold": 0.35,
        "questions": {"total": 3, "answerable": 2, "unanswerable": 1},
        "corpus": {"directory": "data/corpus", "files": 1, "pages": 2},
        "indexed_chunks": 10,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "embedding_model": "hashing",
        "modes": {mode: dict(metrics) for mode in ("dense", "bm25", "hybrid")},
        "sweep": [{"threshold": 0.35, "abstain_accuracy": 0.5}],
        "chunk_sizes": [
            {
                "chunk_size": 512,
                "indexed_chunks": 10,
                "hit_at_k": 0.5,
                "mrr_at_k": 0.5,
                "abstain_accuracy": 0.5,
            }
        ],
    }
    results["rerank"] = False
    if with_rerank:
        reranked = dict(metrics)
        reranked["hit_at_k"] = 0.75
        reranked["mrr_at_k"] = 0.75
        results["reranker"] = {
            "model": "fake-reranker",
            "candidates": 20,
            "modes": {mode: dict(reranked) for mode in ("dense", "bm25", "hybrid")},
        }
        results["latency_ms"] = {
            mode: {"base": 1.0, "rerank": 2.0} for mode in ("dense", "bm25", "hybrid")
        }
    return results


def test_render_corpus_markdown_keeps_old_table_without_rerank() -> None:
    """Rendering without a rerank key keeps the original comparison table."""
    markdown = run_corpus_eval.render_corpus_markdown(synthetic_corpus_results(False))

    assert "| Mode | Hit@5 | MRR@5 | Hit@1 | MRR@1 |" in markdown
    assert "| Mode | Rerank |" not in markdown
    assert "ms/query with rerank" not in markdown


def test_render_corpus_markdown_adds_rerank_and_latency_tables() -> None:
    """Rendering with a rerank key adds reranked metrics and latency."""
    markdown = run_corpus_eval.render_corpus_markdown(synthetic_corpus_results(True))

    assert "| Mode | Rerank | Hit@5 | MRR@5 | Hit@1 | MRR@1 |" in markdown
    assert "| dense | no | 0.500 | 0.500 | 0.250 | 0.250 |" in markdown
    assert "| dense | yes | 0.750 | 0.750 | 0.250 | 0.250 |" in markdown
    assert "Abstention uses the first-stage dense similarity" in markdown
    assert "| Mode | ms/query | ms/query with rerank |" in markdown
    assert "Latency was measured on one CPU machine and varies between runs." in markdown
