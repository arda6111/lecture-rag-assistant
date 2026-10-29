"""Tests for vector index functionality."""


import numpy as np
import pytest

from src.lecture_rag.index import VectorIndex
from src.lecture_rag.models import Document


def test_create_index() -> None:
    """Test creating an empty index."""
    index = VectorIndex(dimension=384)

    assert index.dimension == 384
    assert len(index.documents) == 0


def test_add_documents(sample_documents: list[Document], sample_embeddings: np.ndarray) -> None:
    """Test adding documents to index."""
    index = VectorIndex(dimension=384)
    index.add_documents(sample_documents, sample_embeddings)

    assert len(index.documents) == 3
    assert index.documents[0].text == sample_documents[0].text


def test_add_documents_dimension_mismatch(sample_documents: list[Document]) -> None:
    """Test adding documents with wrong embedding dimension."""
    index = VectorIndex(dimension=384)
    wrong_embeddings = np.random.rand(3, 512).astype(np.float32)

    with pytest.raises(ValueError, match="dimension"):
        index.add_documents(sample_documents, wrong_embeddings)


def test_add_documents_count_mismatch(sample_documents: list[Document]) -> None:
    """Test adding mismatched number of documents and embeddings."""
    index = VectorIndex(dimension=384)
    wrong_embeddings = np.random.rand(2, 384).astype(np.float32)

    with pytest.raises(ValueError, match="must match"):
        index.add_documents(sample_documents, wrong_embeddings)


def test_search(sample_documents: list[Document], sample_embeddings: np.ndarray) -> None:
    """Test searching the index."""
    index = VectorIndex(dimension=384)
    index.add_documents(sample_documents, sample_embeddings)

    query = sample_embeddings[0]
    results = index.search(query, k=2)

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc, _ in results)
    assert all(isinstance(score, float) for _, score in results)


def test_search_empty_index() -> None:
    """Test searching an empty index."""
    index = VectorIndex(dimension=384)
    query = np.random.rand(384).astype(np.float32)

    results = index.search(query, k=5)
    assert len(results) == 0


def test_save_and_load(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir,
) -> None:
    """Test saving and loading index."""
    index = VectorIndex(dimension=384)
    index.add_documents(sample_documents, sample_embeddings)

    index_path = temp_dir / "test_index"
    index.save(index_path)

    loaded_index = VectorIndex.load(index_path)

    assert loaded_index.dimension == 384
    assert len(loaded_index.documents) == 3
    assert loaded_index.documents[0].text == sample_documents[0].text
