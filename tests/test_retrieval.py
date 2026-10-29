"""Tests for retrieval modes and reciprocal rank fusion."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from lecture_rag.chunking import chunk_documents
from lecture_rag.cli import main
from lecture_rag.index import VectorIndex
from lecture_rag.models import Document
from lecture_rag.pdf_loader import load_directory
from lecture_rag.retrieval import (
    DEFAULT_ABSTAIN_THRESHOLD,
    Retriever,
    parse_abstain_threshold,
    reciprocal_rank_fusion,
)
from tests.conftest import HashingEmbeddingModel

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_retrieve_mode_override_does_not_mutate_instance(
    vector_index: VectorIndex,
) -> None:
    """A per-call mode overrides retrieval without changing the instance mode."""
    query = "binary search tree"
    retriever = Retriever(FakeEmbeddingModel(), vector_index, mode="dense")
    bm25_retriever = Retriever(FakeEmbeddingModel(), vector_index, mode="bm25")

    expected = [result.model_dump() for result in bm25_retriever.retrieve(query, k=3)]
    actual = [result.model_dump() for result in retriever.retrieve(query, k=3, mode="bm25")]

    assert actual == expected
    assert retriever.mode == "dense"


def test_retrieve_invalid_mode_override_raises(
    vector_index: VectorIndex,
) -> None:
    """An invalid per-call mode raises without changing the instance mode."""
    retriever = Retriever(FakeEmbeddingModel(), vector_index, mode="dense")

    with pytest.raises(ValueError, match="Invalid retrieval mode"):
        retriever.retrieve("binary search tree", mode="invalid")  # type: ignore[arg-type]

    assert retriever.mode == "dense"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.37", 0.35),
        ("1.7", 1.0),
        ("-2", 0.0),
        ("abc", DEFAULT_ABSTAIN_THRESHOLD),
        (None, DEFAULT_ABSTAIN_THRESHOLD),
    ],
)
def test_parse_abstain_threshold(raw: str | None, expected: float) -> None:
    """Abstain thresholds are clamped, rounded, and validated."""
    assert parse_abstain_threshold(raw) == expected


def build_sample_index() -> tuple[HashingEmbeddingModel, VectorIndex]:
    """Build a deterministic index from the bundled sample lecture PDFs."""
    documents = load_directory(REPO_ROOT / "data" / "sample_lectures")
    chunks = chunk_documents(documents)
    embedding_model = HashingEmbeddingModel()
    index = VectorIndex(embedding_model.dimension)
    index.add_documents(
        chunks,
        embedding_model.embed_texts([chunk.text for chunk in chunks]),
    )
    return embedding_model, index


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize(
    ("query", "expected_file", "expected_page"),
    [
        (
            "Which collision resolution strategies do hash tables use?",
            "data_structures.pdf",
            2,
        ),
        (
            "What is the Bellman equation in dynamic programming?",
            "algorithms.pdf",
            1,
        ),
    ],
)
def test_sample_lectures_top_result(
    mode: str,
    query: str,
    expected_file: str,
    expected_page: int,
) -> None:
    """The expected page is ranked first across every retrieval mode."""
    embedding_model, index = build_sample_index()
    retriever = Retriever(embedding_model, index, mode="dense")

    results = retriever.retrieve(query, k=5, mode=mode)  # type: ignore[arg-type]

    assert results
    top = results[0].document
    assert (top.file_name, top.page_number) == (expected_file, expected_page)


def test_default_retrieval_mode_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default mode comes from the environment and falls back to dense."""
    from lecture_rag.retrieval import default_retrieval_mode

    monkeypatch.delenv("LECTURE_RAG_RETRIEVAL_MODE", raising=False)
    assert default_retrieval_mode() == "dense"

    monkeypatch.setenv("LECTURE_RAG_RETRIEVAL_MODE", "Hybrid")
    assert default_retrieval_mode() == "hybrid"

    monkeypatch.setenv("LECTURE_RAG_RETRIEVAL_MODE", "foo")
    with pytest.raises(ValueError, match="Invalid retrieval mode"):
        default_retrieval_mode()
