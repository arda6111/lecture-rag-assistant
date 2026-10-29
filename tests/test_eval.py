"""Tests for the evaluation harness and abstention behaviour."""

from pathlib import Path

import numpy as np
import pytest

from lecture_rag.eval import EvalQuestion, EvalSet, evaluate_retrieval, load_eval_set
from lecture_rag.index import VectorIndex
from lecture_rag.models import Document, RetrievalResult
from lecture_rag.retrieval import DEFAULT_ABSTAIN_THRESHOLD, Retriever

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeEmbeddingModel:
    """Deterministic embedding model for tests."""

    dimension = 3

    def embed_query(self, query: str) -> np.ndarray:
        """Return a fixed query vector."""
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class StubRetriever:
    """Retriever stub with configurable results and dense similarities."""

    def __init__(
        self,
        results: dict[str, list[RetrievalResult]],
        similarities: dict[str, float],
    ) -> None:
        self.results = results
        self.similarities = similarities

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Return configured results for a query."""
        return self.results[query][:k]

    def should_abstain(self, query: str, threshold: float) -> bool:
        """Abstain when the configured similarity is below the threshold."""
        return self.similarities[query] < threshold


def make_document(file_name: str, page_number: int, chunk_id: str) -> Document:
    """Create a document for testing."""
    return Document(
        text=f"{file_name} page {page_number}",
        file_name=file_name,
        page_number=page_number,
        chunk_id=chunk_id,
    )


def result(document: Document, score: float = 1.0) -> RetrievalResult:
    """Create a retrieval result for testing."""
    return RetrievalResult(document=document, score=score)


def test_eval_question_answerable_property() -> None:
    """Test that answerable depends on both expected fields being set."""
    assert EvalQuestion(question="q", expected_file="a.pdf", expected_page=1).answerable
    assert not EvalQuestion(question="q").answerable
    assert not EvalQuestion(question="q", expected_file="a.pdf").answerable
    assert not EvalQuestion(question="q", expected_page=1).answerable


def test_evaluate_retrieval_uses_answerable_questions_for_hit_and_mrr() -> None:
    """Test that hit rate and MRR ignore unanswerable questions."""
    correct_one = make_document("a.pdf", 1, "a-1")
    correct_two = make_document("b.pdf", 2, "b-2")
    wrong = make_document("c.pdf", 1, "c-1")

    eval_set = EvalSet(
        questions=[
            EvalQuestion(question="first", expected_file="a.pdf", expected_page=1),
            EvalQuestion(question="second", expected_file="b.pdf", expected_page=2),
            EvalQuestion(question="unknown"),
        ]
    )
    retriever = StubRetriever(
        results={
            "first": [result(wrong, 0.9), result(correct_one, 0.8)],
            "second": [result(correct_two, 0.7)],
            "unknown": [result(wrong, 0.6)],
        },
        similarities={"first": 0.9, "second": 0.7, "unknown": 0.1},
    )

    metrics = evaluate_retrieval(retriever, eval_set, k=5)

    assert metrics["total_questions"] == 3
    assert metrics["answerable_questions"] == 2
    assert metrics["unanswerable_questions"] == 1
    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr_at_5"] == 0.75
    assert metrics["mean_score"] == pytest.approx((0.9 + 0.8 + 0.7 + 0.6) / 4)
    assert "abstain_accuracy" not in metrics


def test_evaluate_retrieval_abstain_metrics_math() -> None:
    """Test abstain accuracy, precision, recall, and post-abstain hits."""
    correct_one = make_document("a.pdf", 1, "a-1")
    correct_two = make_document("b.pdf", 2, "b-2")
    other = make_document("c.pdf", 1, "c-1")

    eval_set = EvalSet(
        questions=[
            EvalQuestion(question="answer-one", expected_file="a.pdf", expected_page=1),
            EvalQuestion(question="answer-two", expected_file="b.pdf", expected_page=2),
            EvalQuestion(question="unknown-one"),
            EvalQuestion(question="unknown-two"),
        ]
    )
    retriever = StubRetriever(
        results={
            "answer-one": [result(correct_one)],
            "answer-two": [result(correct_two)],
            "unknown-one": [result(other)],
            "unknown-two": [result(other)],
        },
        similarities={
            "answer-one": 0.80,
            "answer-two": 0.20,
            "unknown-one": 0.10,
            "unknown-two": 0.90,
        },
    )

    metrics = evaluate_retrieval(retriever, eval_set, k=5, abstain_threshold=0.35)

    assert metrics["abstain_accuracy"] == 0.5
    assert metrics["abstain_precision"] == 0.5
    assert metrics["abstain_recall"] == 0.5
    assert metrics["answerable_hit_at_5_after_abstain"] == 0.5
    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr_at_5"] == 1.0


def test_evaluate_retrieval_all_abstain_metrics_zero_without_positives() -> None:
    """Test abstain precision and recall fall back to zero."""
    eval_set = EvalSet(
        questions=[
            EvalQuestion(question="answer", expected_file="a.pdf", expected_page=1),
        ]
    )
    correct = make_document("a.pdf", 1, "a-1")
    retriever = StubRetriever(
        results={"answer": [result(correct)]},
        similarities={"answer": 0.9},
    )

    metrics = evaluate_retrieval(retriever, eval_set, k=1, abstain_threshold=0.35)

    assert metrics["abstain_accuracy"] == 1.0
    assert metrics["abstain_precision"] == 0.0
    assert metrics["abstain_recall"] == 0.0
    assert metrics["answerable_hit_at_1_after_abstain"] == 1.0


def test_max_dense_similarity_and_should_abstain() -> None:
    """Test dense similarity and abstention decisions."""
    documents = [
        make_document("a.pdf", 1, "a-1"),
        make_document("b.pdf", 2, "b-2"),
    ]
    index = VectorIndex(dimension=3)
    index.add_documents(
        documents,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    retriever = Retriever(FakeEmbeddingModel(), index, mode="bm25")

    assert retriever.max_dense_similarity("anything") == 1.0
    assert retriever.should_abstain("anything", 1.5)
    assert not retriever.should_abstain("anything", DEFAULT_ABSTAIN_THRESHOLD)


def test_max_dense_similarity_empty_index() -> None:
    """Test that an empty index has zero similarity and abstains."""
    retriever = Retriever(FakeEmbeddingModel(), VectorIndex(dimension=3), mode="dense")

    assert retriever.max_dense_similarity("anything") == 0.0
    assert retriever.should_abstain("anything", DEFAULT_ABSTAIN_THRESHOLD)


def test_real_eval_set_structure() -> None:
    """Test that the bundled evaluation set is valid and references real PDFs."""
    eval_set = load_eval_set(REPO_ROOT / "eval" / "eval_set.json")
    answerable = [question for question in eval_set.questions if question.answerable]
    unanswerable = [question for question in eval_set.questions if not question.answerable]

    assert len(eval_set.questions) >= 25
    assert len(unanswerable) >= 5
    assert len(answerable) >= 20

    pdf_dir = REPO_ROOT / "data" / "sample_lectures"
    covered_pages = set()
    for question in answerable:
        assert question.expected_file is not None
        assert question.expected_page is not None
        assert (pdf_dir / question.expected_file).exists()
        covered_pages.add((question.expected_file, question.expected_page))

    assert len(covered_pages) == 4
    assert all(
        question.expected_file is None and question.expected_page is None
        for question in unanswerable
    )


def test_load_eval_set_rejects_malformed_json(tmp_path: Path) -> None:
    """Malformed JSON raises a ValueError naming the file."""
    eval_file = tmp_path / "broken.json"
    eval_file.write_text('{"questions": [', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid eval file"):
        load_eval_set(eval_file)


def test_load_eval_set_rejects_schema_errors(tmp_path: Path) -> None:
    """JSON that does not match the schema raises a ValueError."""
    eval_file = tmp_path / "wrong.json"
    eval_file.write_text('{"questions": [{"expected_page": 1}]}', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid eval file"):
        load_eval_set(eval_file)


def test_evaluate_retrieval_rejects_empty_eval_set() -> None:
    """An empty eval set is an error instead of all-zero metrics."""
    with pytest.raises(ValueError, match="empty"):
        evaluate_retrieval(None, EvalSet(questions=[]))  # type: ignore[arg-type]
