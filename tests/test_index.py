"""Tests for vector index functionality."""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from lecture_rag.index import (
    IndexMetadataError,
    IndexModelMismatchError,
    InvalidIndexError,
    VectorIndex,
    normalize_model_name,
    shorten_model_name,
)
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

    with pytest.raises(InvalidIndexError, match="does not match") as exc_info:
        VectorIndex.load(index_path)

    message = str(exc_info.value)
    assert str(docs_path) in message
    assert message.endswith("Re-run `lecture-rag ingest` to rebuild the index.")

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

    with pytest.raises(IndexMetadataError, match="Invalid index metadata") as exc_info:
        VectorIndex.load(index_path)

    assert "Re-run `lecture-rag ingest`" in str(exc_info.value)


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

    with pytest.raises(IndexMetadataError, match="Invalid index metadata") as exc_info:
        VectorIndex.load(index_path)

    assert "Re-run `lecture-rag ingest`" in str(exc_info.value)


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

    with pytest.raises(
        IndexMetadataError,
        match="does not match the FAISS index",
    ) as exc_info:
        VectorIndex.load(index_path)

    assert "Re-run `lecture-rag ingest`" in str(exc_info.value)


def test_load_rejects_symlinked_metadata(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir,
) -> None:
    """A symlinked metadata file is rejected even when its target is valid."""
    index = VectorIndex(dimension=384)
    index.add_documents(sample_documents, sample_embeddings)
    index_path = temp_dir / "symlinked_metadata_index"
    index.save(index_path)

    meta_path = index_path / "index_meta.json"
    target = temp_dir / "valid_index_meta.json"
    meta_path.rename(target)
    os.symlink(target, meta_path)

    with pytest.raises(IndexMetadataError, match="symlink") as exc_info:
        VectorIndex.load(index_path)

    assert "Re-run `lecture-rag ingest`" in str(exc_info.value)


def test_load_rejects_dangling_symlinked_metadata(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir,
) -> None:
    """A dangling metadata symlink is not treated as a legacy index."""
    index = VectorIndex(dimension=384)
    index.add_documents(sample_documents, sample_embeddings)
    index_path = temp_dir / "dangling_metadata_index"
    index.save(index_path)

    meta_path = index_path / "index_meta.json"
    meta_path.unlink()
    os.symlink(temp_dir / "missing_index_meta.json", meta_path)

    with pytest.raises(IndexMetadataError, match="symlink") as exc_info:
        VectorIndex.load(index_path)

    assert "Re-run `lecture-rag ingest`" in str(exc_info.value)


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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
        ("  all-MiniLM-L6-v2  ", "all-MiniLM-L6-v2"),
        ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
        ("SENTENCE-TRANSFORMERS/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
        ("other/model", "other/model"),
    ],
)
def test_normalize_model_name(raw: str, expected: str) -> None:
    """Model names are stripped and lose a case-insensitive prefix."""
    assert normalize_model_name(raw) == expected


def test_shorten_model_name_keeps_short_names() -> None:
    """Names that fit within the limit are unchanged."""
    assert shorten_model_name("model-a") == "model-a"
    assert shorten_model_name("model-a", limit=7) == "model-a"


def test_shorten_model_name_truncates_long_names() -> None:
    """A 300-character name becomes a 200-character name ending in dots."""
    name = "x" * 300

    shortened = shorten_model_name(name)

    assert len(shortened) == 200
    assert shortened == name[:197] + "..."
    assert shortened.endswith("...")


def test_check_model_raises_when_names_differ() -> None:
    """A different model name with the same dimension is a hard error."""
    index = VectorIndex(dimension=384, model_name="model-a")

    with pytest.raises(IndexModelMismatchError) as exc_info:
        index.check_model("model-b", 384)

    message = str(exc_info.value)
    assert "does not match the model the index was built with" in message
    assert "Use --model model-a" in message
    assert "--allow-model-mismatch" in message


def test_check_model_allows_name_mismatch_without_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Allowing a name mismatch returns only a warning for the caller to print."""
    index = VectorIndex(dimension=384, model_name="model-a")

    with caplog.at_level(logging.WARNING):
        warning = index.check_model("model-b", 384, allow_model_mismatch=True)

    assert warning == (
        "Embedding model 'model-b' differs from the index model 'model-a'; "
        "results may be unreliable."
    )
    assert caplog.records == []


def test_check_model_normalizes_prefix(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Prefixed and unprefixed names match without an error or warning."""
    index = VectorIndex(dimension=384, model_name="all-MiniLM-L6-v2")

    with caplog.at_level(logging.WARNING):
        warning = index.check_model(
            "sentence-transformers/all-MiniLM-L6-v2",
            384,
        )

    assert warning is None
    assert "differs" not in caplog.text


def test_check_model_normalizes_prefixed_index_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unprefixed name matches a prefixed index name."""
    index = VectorIndex(
        dimension=384,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    with caplog.at_level(logging.WARNING):
        warning = index.check_model("all-MiniLM-L6-v2", 384)

    assert warning is None
    assert "differs" not in caplog.text


def test_check_model_legacy_index_returns_none() -> None:
    """A legacy index without a model name skips the name check."""
    index = VectorIndex(dimension=384)

    assert index.check_model("model-b", 384) is None


def test_check_model_dimension_mismatch_ignores_allow() -> None:
    """A dimension mismatch raises even when name mismatches are allowed."""
    index = VectorIndex(dimension=384, model_name="model-a")

    with pytest.raises(IndexModelMismatchError):
        index.check_model("model-b", 128, allow_model_mismatch=True)


def test_check_model_shortens_long_names_in_errors() -> None:
    """A very long model name is shortened in the error message."""
    index = VectorIndex(dimension=384, model_name="model-a")
    long_name = "x" * 1000

    with pytest.raises(IndexModelMismatchError) as exc_info:
        index.check_model(long_name, 128)

    message = str(exc_info.value)
    assert long_name not in message
    assert shorten_model_name(long_name) in message
    assert "..." in message


REBUILD_SUFFIX = "Re-run `lecture-rag ingest` to rebuild the index."


def save_test_index(
    temp_dir: Path,
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
) -> Path:
    """Build and save an index for invalid-index tests."""
    index = VectorIndex(dimension=384, model_name="model-a")
    index.add_documents(sample_documents, sample_embeddings)
    index_path = temp_dir / "saved_index"
    index.save(index_path)
    return index_path


def test_load_rejects_symlinked_index_directory(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir: Path,
) -> None:
    """A symlinked index directory is rejected even when it points at a valid index."""
    index_path = save_test_index(temp_dir, sample_documents, sample_embeddings)
    link_path = temp_dir / "linked_index"
    os.symlink(index_path, link_path)

    with pytest.raises(InvalidIndexError, match="symlink") as exc_info:
        VectorIndex.load(link_path)

    assert str(exc_info.value).endswith(REBUILD_SUFFIX)


def test_load_rejects_index_path_that_is_not_a_directory(
    temp_dir: Path,
) -> None:
    """An index path that is a regular file is rejected."""
    index_path = temp_dir / "index_file"
    index_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(InvalidIndexError, match="not a directory") as exc_info:
        VectorIndex.load(index_path)

    assert str(exc_info.value).endswith(REBUILD_SUFFIX)


def test_load_rejects_empty_index_directory(temp_dir: Path) -> None:
    """An empty index directory names the first missing index file."""
    index_path = temp_dir / "empty_index"
    index_path.mkdir()

    with pytest.raises(InvalidIndexError, match="index.faiss") as exc_info:
        VectorIndex.load(index_path)

    assert str(exc_info.value).endswith(REBUILD_SUFFIX)


@pytest.mark.parametrize("file_name", ["index.faiss", "documents.json"])
def test_load_rejects_missing_index_file(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir: Path,
    file_name: str,
) -> None:
    """A missing index file is named in the error."""
    index_path = save_test_index(temp_dir, sample_documents, sample_embeddings)
    (index_path / file_name).unlink()

    with pytest.raises(InvalidIndexError, match=file_name) as exc_info:
        VectorIndex.load(index_path)

    assert str(exc_info.value).endswith(REBUILD_SUFFIX)


@pytest.mark.parametrize("file_name", ["index.faiss", "documents.json"])
def test_load_rejects_symlinked_index_file(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir: Path,
    file_name: str,
) -> None:
    """A symlinked index file is rejected even when its target is valid."""
    index_path = save_test_index(temp_dir, sample_documents, sample_embeddings)
    file_path = index_path / file_name
    target = temp_dir / f"valid_{file_name}"
    file_path.rename(target)
    os.symlink(target, file_path)

    with pytest.raises(InvalidIndexError, match="symlink") as exc_info:
        VectorIndex.load(index_path)

    assert str(exc_info.value).endswith(REBUILD_SUFFIX)


@pytest.mark.parametrize("file_name", ["index.faiss", "documents.json"])
def test_load_rejects_dangling_symlinked_index_file(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir: Path,
    file_name: str,
) -> None:
    """A dangling index file symlink is rejected instead of treated as missing."""
    index_path = save_test_index(temp_dir, sample_documents, sample_embeddings)
    file_path = index_path / file_name
    file_path.unlink()
    os.symlink(temp_dir / f"missing_{file_name}", file_path)

    with pytest.raises(InvalidIndexError, match="symlink") as exc_info:
        VectorIndex.load(index_path)

    assert str(exc_info.value).endswith(REBUILD_SUFFIX)


def test_load_rejects_corrupt_faiss_index(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir: Path,
) -> None:
    """Garbage bytes in index.faiss are reported with the file path."""
    index_path = save_test_index(temp_dir, sample_documents, sample_embeddings)
    faiss_path = index_path / "index.faiss"
    faiss_path.write_bytes(b"not a faiss index")

    with pytest.raises(InvalidIndexError) as exc_info:
        VectorIndex.load(index_path)

    message = str(exc_info.value)
    assert str(faiss_path) in message
    assert message.endswith(REBUILD_SUFFIX)


@pytest.mark.parametrize(
    "content",
    [
        b"{not json",
        b"\xff\xfe\xfa",
        b"{}",
        b"[1]",
    ],
)
def test_load_rejects_invalid_documents_json(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir: Path,
    content: bytes,
) -> None:
    """Malformed, non-list or invalid document entries name documents.json."""
    index_path = save_test_index(temp_dir, sample_documents, sample_embeddings)
    docs_path = index_path / "documents.json"
    docs_path.write_bytes(content)

    with pytest.raises(InvalidIndexError) as exc_info:
        VectorIndex.load(index_path)

    message = str(exc_info.value)
    assert str(docs_path) in message
    assert message.endswith(REBUILD_SUFFIX)


def test_check_model_name_quotes_a_model_name_with_a_space() -> None:
    """A model name with a space is quoted in the runnable hint."""
    index = VectorIndex(dimension=384, model_name="my model")

    with pytest.raises(IndexModelMismatchError) as exc_info:
        index.check_model_name("other-model")

    assert "Use --model 'my model'" in str(exc_info.value)


def test_check_model_name_omits_hint_for_shortened_model_name() -> None:
    """A shortened model name is not shown as a runnable --model command."""
    long_name = "x" * 300
    index = VectorIndex(dimension=384, model_name=long_name)

    with pytest.raises(IndexModelMismatchError) as exc_info:
        index.check_model_name("other-model")

    message = str(exc_info.value)
    assert "--model " not in message
    assert "shortened" in message
    assert long_name not in message
