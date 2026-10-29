"""Embedding generation using local sentence-transformers models."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Local embedding model using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded, dimension: {self.dimension}")

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of shape (len(texts), dimension)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query.

        Args:
            query: Query string

        Returns:
            Numpy array of shape (dimension,)
        """
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding
