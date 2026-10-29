"""Tests for BM25 retrieval."""

from lecture_rag.bm25 import BM25Index, tokenize
from lecture_rag.models import Document


def make_document(chunk_id: str, text: str) -> Document:
    """Create a document for testing."""
    return Document(
        text=text,
        file_name="test.pdf",
        page_number=1,
        chunk_id=chunk_id,
    )


def test_tokenize_lowercases_and_removes_stopwords() -> None:
    """Test tokenization normalization."""
    assert tokenize("The Binary Search Tree") == ["binary", "search", "tree"]


def test_bm25_search_ranks_matching_document_first() -> None:
    """Test that BM25 ranks the document containing query terms first."""
    documents = [
        make_document("doc-1", "binary search tree traversal algorithm"),
        make_document("doc-2", "hash table lookup with constant time"),
        make_document("doc-3", "dynamic programming optimization problem"),
    ]
    index = BM25Index(documents)

    results = index.search("binary search tree", k=3)

    assert results
    assert results[0][0].chunk_id == "doc-1"
    assert results[0][1] > 0
    assert all(score > 0 for _, score in results)


def test_bm25_search_limits_results() -> None:
    """Test that BM25 respects the result limit."""
    documents = [
        make_document("doc-1", "graph traversal"),
        make_document("doc-2", "graph search"),
        make_document("doc-3", "hash table"),
        make_document("doc-4", "binary tree"),
        make_document("doc-5", "dynamic programming"),
    ]
    index = BM25Index(documents)

    results = index.search("graph", k=1)

    assert len(results) == 1


def test_bm25_search_empty_cases() -> None:
    """Test searching with an empty index or empty query."""
    assert BM25Index([]).search("graph", k=5) == []

    documents = [make_document("doc-1", "graph traversal")]
    index = BM25Index(documents)

    assert index.search("", k=5) == []
    assert index.search("the and", k=5) == []
