"""FAISS-based vector index for document retrieval."""

import json
import logging
import os
import shlex
from pathlib import Path

import faiss
import numpy as np
from pydantic import ValidationError

from .models import Document

logger = logging.getLogger(__name__)

INDEX_NOT_FOUND_MESSAGE = (
    "Index directory not found. Run `lecture-rag ingest` to build the index."
)


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


class InvalidIndexError(ValueError):
    """Raised when an index file is missing, invalid or unsafe."""


class IndexMetadataError(InvalidIndexError):
    """Raised when index metadata is missing or invalid."""


def _invalid_error(path: Path, reason: str) -> InvalidIndexError:
    """Build a consistent invalid index error."""
    return InvalidIndexError(
        f"Invalid index at {path}: {reason}. "
        "Re-run `lecture-rag ingest` to rebuild the index."
    )


def _metadata_error(path: Path, reason: str) -> IndexMetadataError:
    """Build a consistent index metadata error."""
    return IndexMetadataError(
        f"Invalid index at {path}: {reason}. "
        "Re-run `lecture-rag ingest` to rebuild the index."
    )


def _model_hint(model_name: str) -> str:
    """Build a hint that only suggests a runnable model name when it fits."""
    if len(model_name) <= 200:
        return f"Use --model {shlex.quote(model_name)}"
    return "Use the model the index was built with (its name is shortened above)"


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

    def check_model_name(
        self,
        model_name: str,
        allow_model_mismatch: bool = False,
    ) -> str | None:
        """Check that an embedding model name can query this index.

        Args:
            model_name: Name of the embedding model that will query the index
            allow_model_mismatch: Continue with a warning when the model name
                differs from the model the index was built with

        Returns:
            A warning message when the model names differ and mismatches are
            allowed, otherwise None

        Raises:
            IndexModelMismatchError: If the model name differs without
                permission to continue
        """
        if self.model_name is None:
            return None

        if normalize_model_name(model_name) == normalize_model_name(self.model_name):
            return None

        shown = shorten_model_name(model_name)
        shown_index = shorten_model_name(self.model_name)
        if not allow_model_mismatch:
            raise IndexModelMismatchError(
                f"Embedding model '{shown}' does not match the model the index "
                f"was built with ('{shown_index}'). {_model_hint(self.model_name)} "
                "or re-run `lecture-rag ingest` with this model, or pass "
                "--allow-model-mismatch to continue anyway."
            )

        return (
            f"Embedding model '{shown}' differs from the index model "
            f"'{shown_index}'; results may be unreliable."
        )

    def check_dimension(self, model_name: str, dimension: int) -> None:
        """Check that an embedding model dimension can query this index.

        Args:
            model_name: Name of the embedding model that will query the index
            dimension: Dimensionality of vectors produced by the model

        Raises:
            IndexModelMismatchError: If the model dimension differs from the
                index dimension
        """
        if dimension == self.dimension:
            return

        shown = shorten_model_name(model_name)
        if self.model_name is not None:
            shown_index = shorten_model_name(self.model_name)
            raise IndexModelMismatchError(
                f"Embedding model '{shown}' produces {dimension}-dimensional "
                f"vectors, but the index was built with '{shown_index}' "
                f"({self.dimension} dimensions). {_model_hint(self.model_name)} "
                "or re-run `lecture-rag ingest` with this model."
            )
        raise IndexModelMismatchError(
            f"Embedding model '{shown}' produces {dimension}-dimensional "
            f"vectors, but the index was built with a {self.dimension}-dimensional "
            "model. Use the model the index was built with or re-run "
            "`lecture-rag ingest` with this model."
        )

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
        self.check_dimension(model_name, dimension)
        return self.check_model_name(
            model_name,
            allow_model_mismatch=allow_model_mismatch,
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

        Raises:
            ValueError: If the model name is empty or whitespace
        """
        if self.model_name is not None and not self.model_name.strip():
            raise ValueError("model_name must not be empty or whitespace")

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

        Raises:
            InvalidIndexError: If the index path or an index file is unsafe,
                missing or invalid
        """
        abs_path = Path(os.path.abspath(path))
        if abs_path.is_symlink():
            raise InvalidIndexError(
                f"Invalid index at {path}: the index directory is a symlink. "
                "Use the real (non-symlink) path of the index directory instead."
            )

        if not path.exists():
            raise InvalidIndexError(
                f"Index directory not found: {path}. "
                "Run `lecture-rag ingest` to build the index."
            )

        if not path.is_dir():
            raise _invalid_error(
                path,
                "the index path is not a directory",
            )

        index_path = path / "index.faiss"
        docs_path = path / "documents.json"

        for file_path in (index_path, docs_path):
            if file_path.is_symlink():
                raise _invalid_error(file_path, f"{file_path.name} is a symlink")

        for file_path in (index_path, docs_path):
            if not file_path.exists():
                raise _invalid_error(
                    file_path,
                    f"missing {file_path.name}",
                )

        try:
            faiss_index = faiss.read_index(str(index_path))
        except RuntimeError as e:
            logger.debug("Failed to read %s: %s", index_path, e)
            raise _invalid_error(
                index_path,
                "index.faiss is corrupt or not a FAISS index",
            ) from e

        try:
            with open(docs_path, encoding="utf-8") as f:
                raw_documents = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            raise _invalid_error(
                docs_path,
                f"cannot read documents.json ({e})",
            ) from e

        if not isinstance(raw_documents, list):
            raise _invalid_error(
                docs_path,
                "documents.json must contain a JSON list",
            )

        try:
            documents = [Document.model_validate(item) for item in raw_documents]
        except (ValidationError, TypeError) as e:
            raise _invalid_error(
                docs_path,
                f"invalid document entry ({e})",
            ) from e

        if len(documents) != faiss_index.ntotal:
            raise _invalid_error(
                docs_path,
                f"document count {len(documents)} does not match index size "
                f"{faiss_index.ntotal}",
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
            valid_model_name = model_name is None or (
                isinstance(model_name, str) and model_name.strip() != ""
            )
            if (
                "dimension" not in metadata
                or "model_name" not in metadata
                or not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or not valid_model_name
            ):
                raise _metadata_error(
                    path,
                    "index_meta.json must contain a model_name (non-empty string or null) and an "
                    "integer dimension",
                )

            if dimension != faiss_index.d:
                raise _metadata_error(
                    path,
                    f"index_meta.json dimension {dimension} does not match the "
                    f"FAISS index dimension {faiss_index.d}",
                )
        else:
            dimension = faiss_index.d
            model_name = None

        vector_index = cls(dimension, model_name=model_name)
        vector_index.index = faiss_index
        vector_index.documents = documents

        logger.info(f"Loaded index with {len(documents)} documents from {path}")
        return vector_index
