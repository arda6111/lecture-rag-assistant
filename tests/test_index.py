"""Tests for vector index functionality."""

import json
import logging

import numpy as np
import pytest

from lecture_rag.index import IndexModelMismatchError, VectorIndex
from lecture_rag.models import Document


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


def test_load_rejects_document_count_mismatch(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir,
) -> None:
    """Loading fails when documents.json does not match the FAISS index size."""
    index = VectorIndex(dimension=384)
    index.add_documents(sample_documents, sample_embeddings)

    index_path = temp_dir / "mismatched_index"
    index.save(index_path)

    docs_path = index_path / "documents.json"
    documents = json.loads(docs_path.read_text(encoding="utf-8"))
    docs_path.write_text(json.dumps(documents[:-1]), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        VectorIndex.load(index_path)

def test_save_writes_index_metadata(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir,
) -> None:
    """Saving writes the model name and dimension alongside the index."""
    index = VectorIndex(dimension=384, model_name="model-a")
    index.add_documents(sample_documents, sample_embeddings)

    index_path = temp_dir / "metadata_index"
    index.save(index_path)

    metadata = json.loads((index_path / "index_meta.json").read_text(encoding="utf-8"))
    assert metadata == {"model_name": "model-a", "dimension": 384}


def test_load_restores_model_name(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir,
) -> None:
    """Loading restores the model name stored in the metadata."""
    index = VectorIndex(dimension=384, model_name="model-a")
    index.add_documents(sample_documents, sample_embeddings)
    index_path = temp_dir / "metadata_index"
    index.save(index_path)

    loaded_index = VectorIndex.load(index_path)

    assert loaded_index.model_name == "model-a"
    assert loaded_index.dimension == 384


def test_load_legacy_index_without_metadata(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir,
) -> None:
    """Indexes without metadata load with an unknown model name."""
    index = VectorIndex(dimension=384, model_name="model-a")
    index.add_documents(sample_documents, sample_embeddings)
    index_path = temp_dir / "legacy_index"
    index.save(index_path)
    (index_path / "index_meta.json").unlink()

    loaded_index = VectorIndex.load(index_path)

    assert loaded_index.model_name is None
    assert loaded_index.dimension == 384


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        {},
        {"dimension": 384},
        {"dimension": "384", "model_name": "model-a"},
        {"dimension": 384, "model_name": 1},
        {"dimension": True, "model_name": "model-a"},
    ],
)
def test_load_rejects_invalid_metadata(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir,
    metadata: object,
) -> None:
    """Metadata must be an object with an integer dimension and string name."""
    index = VectorIndex(dimension=384)
    index.add_documents(sample_documents, sample_embeddings)
    index_path = temp_dir / "invalid_metadata_index"
    index.save(index_path)
    (index_path / "index_meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid index metadata"):
        VectorIndex.load(index_path)


def test_load_rejects_malformed_metadata(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir,
) -> None:
    """Metadata that is not valid JSON is rejected."""
    index = VectorIndex(dimension=384)
    index.add_documents(sample_documents, sample_embeddings)
    index_path = temp_dir / "malformed_metadata_index"
    index.save(index_path)
    (index_path / "index_meta.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid index metadata"):
        VectorIndex.load(index_path)


def test_load_rejects_metadata_dimension_mismatch(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir,
) -> None:
    """Metadata that disagrees with the FAISS index is rejected."""
    index = VectorIndex(dimension=384)
    index.add_documents(sample_documents, sample_embeddings)
    index_path = temp_dir / "mismatched_metadata_index"
    index.save(index_path)
    (index_path / "index_meta.json").write_text(
        json.dumps({"model_name": "model-a", "dimension": 512}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match the FAISS index"):
        VectorIndex.load(index_path)


def test_check_model_reports_known_index_model() -> None:
    """A dimension mismatch names both the requested and indexed models."""
    index = VectorIndex(dimension=384, model_name="model-a")

    with pytest.raises(IndexModelMismatchError) as exc_info:
        index.check_model("model-b", 128)

    message = str(exc_info.value)
    assert "Embedding model 'model-b' produces 128-dimensional vectors" in message
    assert "built with 'model-a' (384 dimensions)" in message
    assert "Use --model model-a" in message


def test_check_model_reports_legacy_index() -> None:
    """A legacy index without a model name still reports the mismatch."""
    index = VectorIndex(dimension=384)

    with pytest.raises(IndexModelMismatchError) as exc_info:
        index.check_model("model-b", 128)

    message = str(exc_info.value)
    assert "Embedding model 'model-b' produces 128-dimensional vectors" in message
    assert "built with a 384-dimensional model" in message
    assert "Use the model the index was built with" in message


def test_check_model_warns_when_names_differ(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A different model name with the same dimension only logs a warning."""
    index = VectorIndex(dimension=384, model_name="model-a")

    with caplog.at_level(logging.WARNING):
        index.check_model("model-b", 384)

    assert "differs from the model used to build the index" in caplog.text
