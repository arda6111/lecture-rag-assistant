"""Retrieval logic for the RAG system."""

import logging
from pathlib import Path

from .embeddings import EmbeddingModel
from .index import VectorIndex
from .models import RetrievalResult

logger = logging.getLogger(__name__)


class Retriever:
    """Retriever combining embeddings and vector index."""

    def __init__(self, embedding_model: EmbeddingModel, index: VectorIndex) -> None:
        """Initialize the retriever.

        Args:
            embedding_model: Embedding model for encoding queries
            index: Vector index for similarity search
        """
        self.embedding_model = embedding_model
        self.index = index

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top k documents for a query.

        Args:
            query: Query string
            k: Number of documents to retrieve

        Returns:
            List of RetrievalResult objects, sorted by score descending
        """
        query_embedding = self.embedding_model.embed_query(query)
        results = self.index.search(query_embedding, k=k)

        retrieval_results = [
            RetrievalResult(document=doc, score=score)
            for doc, score in results
        ]

        return retrieval_results

    @classmethod
    def from_index_path(cls, index_path: Path, model_name: str = "all-MiniLM-L6-v2") -> "Retriever":
        """Load a retriever from a saved index.

        Args:
            index_path: Path to the saved index directory
            model_name: Name of the embedding model to use

        Returns:
            Initialized Retriever instance
        """
        embedding_model = EmbeddingModel(model_name)
        index = VectorIndex.load(index_path)
        return cls(embedding_model, index)
