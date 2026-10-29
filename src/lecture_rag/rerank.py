"""Optional cross-encoder reranking for retrieved documents."""

import os
from typing import Protocol

from .models import RetrievalResult

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def default_rerank_model() -> str:
    """Return the rerank model configured by LECTURE_RAG_RERANK_MODEL.

    Returns:
        The configured model name, or DEFAULT_RERANK_MODEL when the variable
        is unset or blank
    """
    raw = os.getenv("LECTURE_RAG_RERANK_MODEL")
    if raw is None or not raw.strip():
        return DEFAULT_RERANK_MODEL
    return raw.strip()


class RerankerError(RuntimeError):
    """Raised when a reranker model cannot be loaded."""


class Reranker(Protocol):
    """A reranker that scores documents for a query."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one score per document text."""
        ...


class CrossEncoderReranker:
    """Cross-encoder reranker backed by sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        """Initialize the reranker.

        Args:
            model_name: Name of the sentence-transformers cross-encoder model

        Raises:
            RerankerError: If the model cannot be loaded
        """
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
        except (OSError, ValueError) as e:
            raise RerankerError(f"Could not load reranker model '{model_name}': {e}") from e

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score document texts against a query with the cross-encoder.

        Args:
            query: Query string
            texts: Document texts to score

        Returns:
            One cross-encoder score per document text
        """
        if not texts:
            return []
        scores = self.model.predict([(query, text) for text in texts])
        return [float(score) for score in scores]


def rerank(
    query: str,
    results: list[RetrievalResult],
    reranker: Reranker,
    k: int,
) -> list[RetrievalResult]:
    """Rerank retrieval results and return the top k.

    Ties are broken by the original rank so the order is deterministic.

    Args:
        query: Query string
        results: First-stage retrieval results
        reranker: Reranker used to score document texts
        k: Number of results to return after reranking

    Returns:
        New RetrievalResult objects sorted by rerank score descending, with
        score_type set to "rerank"
    """
    if not results:
        return []

    scores = reranker.score(query, [result.document.text for result in results])
    ranked = sorted(
        enumerate(scores),
        key=lambda item: (-item[1], item[0]),
    )
    return [
        RetrievalResult(
            document=results[index].document,
            score=score,
            score_type="rerank",
        )
        for index, score in ranked[:k]
    ]
