"""Retrieval logic for the RAG system."""

import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel

from .bm25 import BM25Index
from .index import VectorIndex
from .models import Document, RetrievalResult, ScoreType

if TYPE_CHECKING:
    from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

RetrievalMode = Literal["dense", "bm25", "hybrid"]

VALID_MODES: frozenset[str] = frozenset({"dense", "bm25", "hybrid"})

DEFAULT_ABSTAIN_THRESHOLD = 0.35


def parse_abstain_threshold(raw: str | None) -> float:
    """Parse and round an abstain threshold.

    Args:
        raw: Raw threshold value from the environment

    Returns:
        Threshold rounded to the nearest 0.05 when it is in [0.0, 1.0],
        otherwise the default threshold
    """
    if raw is None or not raw.strip():
        return DEFAULT_ABSTAIN_THRESHOLD

    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid LECTURE_RAG_ABSTAIN_THRESHOLD=%r, using %.2f",
            raw,
            DEFAULT_ABSTAIN_THRESHOLD,
        )
        return DEFAULT_ABSTAIN_THRESHOLD

    if not math.isfinite(value):
        logger.warning(
            "Invalid LECTURE_RAG_ABSTAIN_THRESHOLD=%r, using %.2f",
            raw,
            DEFAULT_ABSTAIN_THRESHOLD,
        )
        return DEFAULT_ABSTAIN_THRESHOLD

    if value < 0.0 or value > 1.0:
        logger.warning(
            "LECTURE_RAG_ABSTAIN_THRESHOLD=%r is outside [0, 1], using %.2f",
            raw,
            DEFAULT_ABSTAIN_THRESHOLD,
        )
        return DEFAULT_ABSTAIN_THRESHOLD

    return round(round(value / 0.05) * 0.05, 2)


def default_retrieval_mode() -> RetrievalMode:
    """Return the retrieval mode configured by LECTURE_RAG_RETRIEVAL_MODE.

    Returns:
        The configured mode, or "dense" when the variable is unset

    Raises:
        ValueError: If the configured mode is not supported
    """
    raw = os.getenv("LECTURE_RAG_RETRIEVAL_MODE")
    if raw is None or not raw.strip():
        return "dense"
    return normalize_mode(raw)


def normalize_mode(mode: str) -> RetrievalMode:
    """Validate and normalize a retrieval mode.

    Args:
        mode: Retrieval mode name

    Returns:
        Normalized retrieval mode

    Raises:
        ValueError: If the mode is not supported
    """
    normalized = mode.strip().lower()
    if normalized not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"Invalid retrieval mode: {mode!r}. Expected one of: {valid}")
    return cast(RetrievalMode, normalized)


def reciprocal_rank_fusion(
    rankings: list[list[Document]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """Fuse ranked document lists using reciprocal rank fusion.

    Args:
        rankings: Ranked document lists to fuse
        k: RRF rank constant

    Returns:
        Fused (document, score) tuples sorted by score descending
    """
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranking in rankings:
        for rank, document in enumerate(ranking, start=1):
            chunk_id = document.chunk_id
            documents.setdefault(chunk_id, document)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    fused = [(documents[chunk_id], score) for chunk_id, score in scores.items()]
    fused.sort(key=lambda item: item[1], reverse=True)
    return fused


class RetrievalOutcome(BaseModel):
    """Results of one search together with the dense abstain signal."""

    results: list[RetrievalResult]
    max_dense_similarity: float
    mode: RetrievalMode

    def should_abstain(self, threshold: float) -> bool:
        """Return True if the best dense similarity is below the threshold."""
        return self.max_dense_similarity < threshold


class Retriever:
    """Retriever combining dense, sparse, and hybrid search."""

    def __init__(
        self,
        embedding_model: "EmbeddingModel",
        index: VectorIndex,
        mode: RetrievalMode | None = None,
        rrf_k: int = 60,
    ) -> None:
        """Initialize the retriever.

        Args:
            embedding_model: Embedding model for encoding queries
            index: Vector index for similarity search
            mode: Retrieval mode, defaults to LECTURE_RAG_RETRIEVAL_MODE or dense
            rrf_k: Reciprocal rank fusion constant
        """
        self.embedding_model = embedding_model
        self.index = index
        self.mode = normalize_mode(mode) if mode is not None else default_retrieval_mode()
        self.rrf_k = rrf_k
        self._bm25_index: BM25Index | None = None

    def set_mode(self, mode: RetrievalMode) -> None:
        """Change the retrieval mode.

        Args:
            mode: New retrieval mode
        """
        self.mode = normalize_mode(mode)

    def _get_bm25_index(self) -> BM25Index:
        """Build the BM25 index on first use.

        Returns:
            BM25 index over the vector index documents
        """
        if self._bm25_index is None:
            self._bm25_index = BM25Index(self.index.documents)
        return self._bm25_index

    def max_dense_similarity(self, query: str, k: int = 5) -> float:
        """Return the top dense cosine similarity for a query.

        The abstain signal always uses dense cosine similarity because BM25
        and RRF scores are not on a comparable scale.

        Args:
            query: Query string
            k: Number of dense results to consider

        Returns:
            Highest cosine similarity, or 0.0 if the index is empty
        """
        return self.search(query, k=k, mode="dense").max_dense_similarity

    def should_abstain(self, query: str, threshold: float) -> bool:
        """Decide whether to abstain from answering a query.

        Args:
            query: Query string
            threshold: Minimum dense cosine similarity required to answer

        Returns:
            True if the top dense similarity is below the threshold
        """
        return self.search(query, mode="dense").should_abstain(threshold)

    def search(
        self,
        query: str,
        k: int = 5,
        mode: RetrievalMode | None = None,
    ) -> RetrievalOutcome:
        """Search once and return results plus the dense abstain signal.

        The query is embedded once. The dense results are reused for the
        abstain signal, for dense mode and for hybrid fusion.

        Args:
            query: Query string
            k: Number of documents to retrieve
            mode: Optional retrieval mode that overrides the instance mode

        Returns:
            RetrievalOutcome with results sorted by score descending
        """
        active_mode = self.mode if mode is None else normalize_mode(mode)
        candidate_k = max(k * 4, 20)

        query_embedding = self.embedding_model.embed_query(query)
        dense_k = candidate_k if active_mode == "hybrid" else max(k, 1)
        dense = self.index.search(query_embedding, k=dense_k)
        max_similarity = max((float(score) for _, score in dense), default=0.0)

        if k <= 0:
            return RetrievalOutcome(
                results=[], max_dense_similarity=max_similarity, mode=active_mode
            )

        score_type: ScoreType
        if active_mode == "dense":
            results = dense[:k]
            score_type = "cosine"
        elif active_mode == "bm25":
            results = self._get_bm25_index().search(query, k)
            score_type = "bm25"
        else:
            dense_documents = [document for document, _ in dense]
            bm25_documents = [
                document for document, _ in self._get_bm25_index().search(query, candidate_k)
            ]
            results = reciprocal_rank_fusion(
                [dense_documents, bm25_documents],
                k=self.rrf_k,
            )[:k]
            score_type = "rrf"

        return RetrievalOutcome(
            results=[
                RetrievalResult(document=document, score=score, score_type=score_type)
                for document, score in results
            ],
            max_dense_similarity=max_similarity,
            mode=active_mode,
        )

    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: RetrievalMode | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the top k documents for a query.

        Args:
            query: Query string
            k: Number of documents to retrieve
            mode: Optional retrieval mode that overrides the instance mode

        Returns:
            List of RetrievalResult objects, sorted by score descending
        """
        return self.search(query, k=k, mode=mode).results

    @classmethod
    def from_index_path(
        cls,
        index_path: Path,
        model_name: str = "all-MiniLM-L6-v2",
        mode: RetrievalMode | None = None,
        rrf_k: int = 60,
    ) -> "Retriever":
        """Load a retriever from a saved index.

        Args:
            index_path: Path to the saved index directory
            model_name: Name of the embedding model to use
            mode: Retrieval mode, defaults to LECTURE_RAG_RETRIEVAL_MODE or dense
            rrf_k: Reciprocal rank fusion constant

        Returns:
            Initialized Retriever instance
        """
        from .embeddings import EmbeddingModel

        index = VectorIndex.load(index_path)
        embedding_model = EmbeddingModel(model_name)
        index.check_model(model_name, embedding_model.dimension)
        return cls(embedding_model, index, mode=mode, rrf_k=rrf_k)
