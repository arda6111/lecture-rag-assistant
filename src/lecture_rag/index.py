"""FAISS-based vector index for document retrieval."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from .models import Document

logger = logging.getLogger(__name__)


class IndexModelMismatchError(RuntimeError):
    """Raised when an embedding model does not match the index."""


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

    def check_model(self, model_name: str, dimension: int) -> None:
        """Check that an embedding model can query this index.

        Args:
            model_name: Name of the embedding model that will query the index
            dimension: Dimensionality of vectors produced by the model

        Raises:
            IndexModelMismatchError: If the model dimension differs from the
                index dimension
        """
        if dimension != self.dimension:
            if self.model_name is not None:
                raise IndexModelMismatchError(
                    f"Embedding model {model_name!r} produces {dimension}-dimensional "
                    f"vectors, but the index was built with {self.model_name!r} "
                    f"({self.dimension} dimensions). Use --model {self.model_name} "
                    "or re-run `lecture-rag ingest` with this model."
                )
            raise IndexModelMismatchError(
                f"Embedding model {model_name!r} produces {dimension}-dimensional "
                f"vectors, but the index was built with a {self.dimension}-dimensional "
                "model. Use the model the index was built with or re-run "
                "`lecture-rag ingest` with this model."
            )

        if self.model_name is not None and self.model_name != model_name:
            logger.warning(
                "Embedding model %r differs from the model used to build the index "
                "(%r); proceeding because both produce %d-dimensional vectors.",
                model_name,
                self.model_name,
                self.dimension,
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
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                raise ValueError(f"Invalid index metadata in {path}") from e

            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid index metadata in {path}")

            dimension = metadata.get("dimension")
            model_name = metadata.get("model_name")
            if (
                "dimension" not in metadata
                or "model_name" not in metadata
                or not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or not (model_name is None or isinstance(model_name, str))
            ):
                raise ValueError(f"Invalid index metadata in {path}")

            if dimension != faiss_index.d:
                raise ValueError(
                    f"Index metadata dimension {dimension} does not match "
                    f"the FAISS index in {path}"
                )
        else:
            dimension = faiss_index.d
            model_name = None

        vector_index = cls(dimension, model_name=model_name)
        vector_index.index = faiss_index
        vector_index.documents = documents

        logger.info(f"Loaded index with {len(documents)} documents from {path}")
        return vector_index
