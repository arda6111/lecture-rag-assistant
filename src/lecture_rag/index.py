"""FAISS-based vector index for document retrieval."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from .models import Document

logger = logging.getLogger(__name__)


def normalize_model_name(name: str) -> str:
    """Normalize an embedding model name for comparison.

    Args:
        name: Embedding model name

    Returns:
        The name without surrounding whitespace or a leading
        ``sentence-transformers/`` prefix
    """
    normalized = name.strip()
    prefix = "sentence-transformers/"
    if normalized.lower().startswith(prefix):
        normalized = normalized[len(prefix):]
    return normalized


def shorten_model_name(name: str, limit: int = 200) -> str:
    """Shorten a model name for display in messages.

    Args:
        name: Embedding model name
        limit: Maximum length of the returned name

    Returns:
        The unchanged name when it fits, otherwise a truncated name ending
        in ``...``
    """
    if len(name) <= limit:
        return name
    return name[: limit - 3] + "..."


class IndexModelMismatchError(RuntimeError):
    """Raised when an embedding model does not match the index."""


class IndexMetadataError(ValueError):
    """Raised when index metadata is missing or invalid."""


def _metadata_error(path: Path, reason: str) -> IndexMetadataError:
    """Build a consistent index metadata error."""
    return IndexMetadataError(
        f"Invalid index metadata in {path}: {reason}. "
        "Re-run `lecture-rag ingest` to rebuild the index."
    )


class VectorIndex:
    """FAISS vector index for similarity search."""

    def __init__(self, dimension: int, model_name: str | None = None) -> None:
        """Initialize an empty index.

        Args:
            dimension: Dimensionality of the embeddings
            model_name: Name of the embedding model used to build the index
        """
        self.dimension = dimension
        self.model_name = model_name
        self.index: faiss.Index = faiss.IndexFlatIP(dimension)
        self.documents: list[Document] = []

    def check_model(
        self,
        model_name: str,
        dimension: int,
        allow_model_mismatch: bool = False,
    ) -> str | None:
        """Check that an embedding model can query this index.

        Args:
            model_name: Name of the embedding model that will query the index
            dimension: Dimensionality of vectors produced by the model
            allow_model_mismatch: Continue with a warning when the model name
                differs from the model the index was built with

        Returns:
            A warning message when the model names differ and mismatches are
            allowed, otherwise None

        Raises:
            IndexModelMismatchError: If the model dimension differs from the
                index dimension, or if the model name differs without
                permission to continue
        """
        shown = shorten_model_name(model_name)
        if dimension != self.dimension:
            if self.model_name is not None:
                shown_index = shorten_model_name(self.model_name)
                raise IndexModelMismatchError(
                    f"Embedding model {shown!r} produces {dimension}-dimensional "
                    f"vectors, but the index was built with {shown_index!r} "
                    f"({self.dimension} dimensions). Use --model {shown_index} "
                    "or re-run `lecture-rag ingest` with this model."
                )
            raise IndexModelMismatchError(
                f"Embedding model {shown!r} produces {dimension}-dimensional "
                f"vectors, but the index was built with a {self.dimension}-dimensional "
                "model. Use the model the index was built with or re-run "
                "`lecture-rag ingest` with this model."
            )

        if self.model_name is None:
            return None

        if normalize_model_name(model_name) == normalize_model_name(self.model_name):
            return None

        shown_index = shorten_model_name(self.model_name)
        if not allow_model_mismatch:
            raise IndexModelMismatchError(
                f"Embedding model {shown!r} does not match the model the index "
                f"was built with ({shown_index!r}). Use --model {shown_index} "
                "or re-run `lecture-rag ingest` with this model, or pass "
                "--allow-model-mismatch to continue anyway."
            )

        logger.warning(
            "Embedding model %r differs from the index model %r; "
            "results may be unreliable.",
            shown,
            shown_index,
        )
        return (
            f"Embedding model {shown!r} differs from the index model "
            f"{shown_index!r}; results may be unreliable."
        )

    def add_documents(self, documents: list[Document], embeddings: np.ndarray) -> None:
        """Add documents and their embeddings to the index.

        Args:
            documents: List of documents
            embeddings: Numpy array of embeddings, shape (len(documents), dimension)
        """
        if embeddings.shape[0] != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents to index, total: {len(self.documents)}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[Document, float]]:
        """Search for the k most similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of (document, score) tuples, sorted by score descending
        """
        if len(self.documents) == 0:
            return []

        k = min(k, len(self.documents))

        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))

        return results

    def save(self, path: Path) -> None:
        """Save the index and documents to disk.

        Args:
            path: Directory path to save index files
        """
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))

        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump([document.model_dump() for document in self.documents], f, ensure_ascii=False, indent=2)

        with open(path / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {"model_name": self.model_name, "dimension": self.dimension},
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"Saved index with {len(self.documents)} documents to {path}")

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        """Load an index from disk.

        Args:
            path: Directory path containing index files

        Returns:
            Loaded VectorIndex instance
        """
        index_path = path / "index.faiss"
        docs_path = path / "documents.json"

        if not index_path.exists() or not docs_path.exists():
            raise ValueError(f"Index files not found in {path}")

        faiss_index = faiss.read_index(str(index_path))

        with open(docs_path, encoding="utf-8") as f:
            documents = [Document.model_validate(item) for item in json.load(f)]

        if len(documents) != faiss_index.ntotal:
            raise ValueError(
                f"Document count {len(documents)} does not match index size "
                f"{faiss_index.ntotal} in {path}"
            )

        meta_path = path / "index_meta.json"
        if meta_path.is_symlink():
            raise _metadata_error(path, "index_meta.json is a symlink")

        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                raise _metadata_error(
                    path,
                    f"cannot read index_meta.json ({e})",
                ) from e

            if not isinstance(metadata, dict):
                raise _metadata_error(
                    path,
                    "index_meta.json must contain a JSON object",
                )

            dimension = metadata.get("dimension")
            model_name = metadata.get("model_name")
            if (
                "dimension" not in metadata
                or "model_name" not in metadata
                or not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or not (model_name is None or isinstance(model_name, str))
            ):
                raise _metadata_error(
                    path,
                    "index_meta.json must contain a model_name (string or null) and an "
                    "integer dimension",
                )

            if dimension != faiss_index.d:
                raise _metadata_error(
                    path,
                    f"metadata dimension {dimension} does not match the FAISS "
                    f"index dimension {faiss_index.d}",
                )
        else:
            dimension = faiss_index.d
            model_name = None

        vector_index = cls(dimension, model_name=model_name)
        vector_index.index = faiss_index
        vector_index.documents = documents

        logger.info(f"Loaded index with {len(documents)} documents from {path}")
        return vector_index
