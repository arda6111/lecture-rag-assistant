"""Tests for retrieval modes and reciprocal rank fusion."""

import numpy as np
import pytest
from click.testing import CliRunner

from lecture_rag.cli import main
from lecture_rag.index import VectorIndex
from lecture_rag.models import Document
from lecture_rag.retrieval import Retriever, reciprocal_rank_fusion


class FakeEmbeddingModel:
    """Deterministic embedding model for tests."""

    dimension = 3

    def embed_query(self, query: str) -> np.ndarray:
        """Return a fixed query vector."""
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def make_document(chunk_id: str, text: str) -> Document:
    """Create a document for testing."""
    return Document(
        text=text,
        file_name="test.pdf",
        page_number=1,
        chunk_id=chunk_id,
    )


@pytest.fixture
def documents() -> list[Document]:
    """Create documents with distinct dense and sparse signals."""
    return [
        make_document("doc-1", "binary search tree traversal"),
        make_document("doc-2", "hash table lookup"),
        make_document("doc-3", "dynamic programming optimization"),
    ]


@pytest.fixture
def vector_index(documents: list[Document]) -> VectorIndex:
    """Create a populated vector index."""
    index = VectorIndex(dimension=3)
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    index.add_documents(documents, embeddings)
    return index


def test_reciprocal_rank_fusion_math() -> None:
    """Test RRF scores and ordering on a small example."""
    doc_a = make_document("a", "document a")
    doc_b = make_document("b", "document b")
    doc_c = make_document("c", "document c")

    fused = reciprocal_rank_fusion([[doc_a, doc_b], [doc_b, doc_c]], k=60)

    scores = {document.chunk_id: score for document, score in fused}

    assert fused[0][0].chunk_id == "b"
    assert scores["a"] == pytest.approx(1.0 / 61)
    assert scores["b"] == pytest.approx(1.0 / 62 + 1.0 / 61)
    assert scores["c"] == pytest.approx(1.0 / 62)


def test_dense_mode(
    vector_index: VectorIndex,
    documents: list[Document],
) -> None:
    """Test dense retrieval mode."""
    retriever = Retriever(FakeEmbeddingModel(), vector_index, mode="dense")

    results = retriever.retrieve("binary search tree", k=2)

    assert results[0].document.chunk_id == "doc-1"
    assert len(results) == 2


def test_bm25_mode(
    vector_index: VectorIndex,
    documents: list[Document],
) -> None:
    """Test BM25 retrieval mode."""
    retriever = Retriever(FakeEmbeddingModel(), vector_index, mode="bm25")

    results = retriever.retrieve("binary search tree", k=2)

    assert results[0].document.chunk_id == "doc-1"
    assert results[0].score > 0


def test_hybrid_mode(
    vector_index: VectorIndex,
    documents: list[Document],
) -> None:
    """Test hybrid retrieval mode."""
    retriever = Retriever(FakeEmbeddingModel(), vector_index, mode="hybrid")

    results = retriever.retrieve("binary search tree", k=2)

    assert results[0].document.chunk_id == "doc-1"
    assert results[0].score > 0
    assert len(results) == 2


def test_mode_from_environment(
    vector_index: VectorIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the retrieval mode defaults from the environment."""
    monkeypatch.setenv("LECTURE_RAG_RETRIEVAL_MODE", "bm25")

    retriever = Retriever(FakeEmbeddingModel(), vector_index)

    assert retriever.mode == "bm25"


def test_set_mode(
    vector_index: VectorIndex,
    documents: list[Document],
) -> None:
    """Test changing retrieval mode after initialization."""
    retriever = Retriever(FakeEmbeddingModel(), vector_index, mode="dense")

    retriever.set_mode("hybrid")

    assert retriever.mode == "hybrid"
    assert retriever.retrieve("binary search tree", k=1)[0].document.chunk_id == "doc-1"


def test_invalid_mode_raises(
    vector_index: VectorIndex,
    documents: list[Document],
) -> None:
    """Test that an invalid retrieval mode raises an error."""
    with pytest.raises(ValueError, match="Invalid retrieval mode"):
        Retriever(FakeEmbeddingModel(), vector_index, mode="invalid")  # type: ignore[arg-type]

    retriever = Retriever(FakeEmbeddingModel(), vector_index, mode="dense")
    with pytest.raises(ValueError, match="Invalid retrieval mode"):
        retriever.set_mode("invalid")  # type: ignore[arg-type]


def test_eval_help_shows_compare() -> None:
    """Test that eval help includes the comparison and retrieval options."""
    result = CliRunner().invoke(main, ["eval", "--help"])

    assert result.exit_code == 0
    assert "--compare" in result.output
    assert "--retrieval" in result.output
