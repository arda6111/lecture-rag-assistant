"""FAISS-based vector index for document retrieval."""

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from .models import Document

logger = logging.getLogger(__name__)


class VectorIndex:
    """FAISS vector index for similarity search."""

    def __init__(self, dimension: int) -> None:
        """Initialize an empty index.

        Args:
            dimension: Dimensionality of the embeddings
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[Document] = []

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

        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

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
        docs_path = path / "documents.pkl"

        if not index_path.exists() or not docs_path.exists():
            raise ValueError(f"Index files not found in {path}")

        faiss_index = faiss.read_index(str(index_path))

        with open(docs_path, "rb") as f:
            documents = pickle.load(f)

        vector_index = cls(faiss_index.d)
        vector_index.index = faiss_index
        vector_index.documents = documents

        logger.info(f"Loaded index with {len(documents)} documents from {path}")
        return vector_index
