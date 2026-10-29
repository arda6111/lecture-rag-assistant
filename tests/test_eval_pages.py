"""Tests for multi-page expected pages in evaluation questions."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from lecture_rag.eval import EvalQuestion, EvalSet, evaluate_retrieval, load_eval_set
from lecture_rag.models import Document, RetrievalResult
from lecture_rag.retrieval import RetrievalOutcome

REPO_ROOT = Path(__file__).resolve().parents[1]


class StubRetriever:
    """Retriever stub returning configured results for each query."""

    def __init__(self, results: dict[str, list[RetrievalResult]]) -> None:
        self.results = results

    def search(self, query: str, k: int = 5, mode: str | None = None) -> RetrievalOutcome:
        """Return configured results for a query."""
        return RetrievalOutcome(
            results=self.results[query][:k],
            max_dense_similarity=1.0,
            mode=mode or "dense",
        )


def make_document(file_name: str, page_number: int, chunk_id: str) -> Document:
    """Create a document for testing."""
    return Document(
        text=f"{file_name} page {page_number}",
        file_name=file_name,
        page_number=page_number,
        chunk_id=chunk_id,
    )


def result(document: Document) -> RetrievalResult:
    """Create a retrieval result for testing."""
    return RetrievalResult(document=document, score=1.0)


def test_multi_page_question_hits_when_any_page_is_retrieved() -> None:
    """A question with several expected pages is a hit on any of them."""
    wrong = make_document("a.pdf", 1, "a-1")
    expected = make_document("a.pdf", 3, "a-3")
    eval_set = EvalSet(
        questions=[
            EvalQuestion(question="q", expected_file="a.pdf", expected_pages=[2, 3]),
        ]
    )
    retriever = StubRetriever({"q": [result(wrong), result(expected)]})

    metrics = evaluate_retrieval(retriever, eval_set, k=5)

    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr_at_5"] == 0.5


def test_multi_page_mrr_uses_first_matching_rank() -> None:
    """MRR uses the first retrieved document that matches any expected page."""
    wrong = make_document("a.pdf", 1, "a-1")
    first_match = make_document("a.pdf", 2, "a-2")
    later_match = make_document("a.pdf", 3, "a-3")
    eval_set = EvalSet(
        questions=[
            EvalQuestion(question="q", expected_file="a.pdf", expected_pages=[2, 3]),
        ]
    )
    retriever = StubRetriever({"q": [result(wrong), result(first_match), result(later_match)]})

    metrics = evaluate_retrieval(retriever, eval_set, k=5)

    assert metrics["mrr_at_5"] == 0.5


@pytest.mark.parametrize(
    "fields",
    [
        {"expected_file": "a.pdf", "expected_page": 1, "expected_pages": [1]},
        {"expected_file": "a.pdf", "expected_pages": []},
        {"expected_file": "a.pdf", "expected_page": 0},
        {"expected_file": "a.pdf", "expected_pages": [1, 0]},
        {"expected_file": "a.pdf"},
        {"expected_page": 1},
        {"expected_pages": [1]},
    ],
)
def test_eval_question_rejects_invalid_expected_pages(fields: dict[str, object]) -> None:
    """Every invalid combination of expected file and page fields is rejected."""
    with pytest.raises(ValidationError):
        EvalQuestion(question="q", **fields)


def test_existing_eval_set_still_loads_with_same_counts() -> None:
    """The bundled sample eval set keeps working with expected_page."""
    eval_set = load_eval_set(REPO_ROOT / "eval" / "eval_set.json")
    answerable = [question for question in eval_set.questions if question.answerable]
    unanswerable = [question for question in eval_set.questions if not question.answerable]

    assert len(eval_set.questions) == 33
    assert len(answerable) == 25
    assert len(unanswerable) == 8
    assert all(question.expected_pages is None for question in eval_set.questions)
    assert all(len(question.pages) == 1 for question in answerable)
