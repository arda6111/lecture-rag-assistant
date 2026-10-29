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
from .rerank import Reranker
from .rerank import rerank as rerank_results

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
    """Results of one search together with the dense abstain signal.

    The abstain signal is always the first-stage dense cosine similarity, also
    when reranking is enabled. Cross-encoder scores are unbounded logits on a
    different scale and are never compared with the abstain threshold.
    """

    results: list[RetrievalResult]
    max_dense_similarity: float
    mode: RetrievalMode
    reranked: bool = False

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
        reranker: Reranker | None = None,
        rerank_candidates: int | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            embedding_model: Embedding model for encoding queries
            index: Vector index for similarity search
            mode: Retrieval mode, defaults to LECTURE_RAG_RETRIEVAL_MODE or dense
            rrf_k: Reciprocal rank fusion constant
            reranker: Optional reranker used when rerank is requested
            rerank_candidates: Number of candidates retrieved before reranking,
                or None to use max(4*k, 20)
        """
        self.embedding_model = embedding_model
        self.index = index
        self.mode = normalize_mode(mode) if mode is not None else default_retrieval_mode()
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.model_warning: str | None = None
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

        The abstain signal always uses dense cosine similarity because BM25,
        RRF and cross-encoder scores are not on a comparable scale. It is the
        first-stage dense similarity and does not change when reranking is
        enabled.

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

    def _first_stage(self, query: str, k: int, mode: RetrievalMode) -> RetrievalOutcome:
        """Run one first-stage search in the given mode.

        The query is embedded once. The dense results are reused for the
        abstain signal, for dense mode and for hybrid fusion.

        Args:
            query: Query string
            k: Number of documents to retrieve
            mode: Retrieval mode for this search

        Returns:
            RetrievalOutcome with first-stage results sorted by score descending
        """
        candidate_k = max(k * 4, 20)

        query_embedding = self.embedding_model.embed_query(query)
        dense_k = candidate_k if mode == "hybrid" else max(k, 1)
        dense = self.index.search(query_embedding, k=dense_k)
        max_similarity = max((float(score) for _, score in dense), default=0.0)

        if k <= 0:
            return RetrievalOutcome(results=[], max_dense_similarity=max_similarity, mode=mode)

        score_type: ScoreType
        if mode == "dense":
            results = dense[:k]
            score_type = "cosine"
        elif mode == "bm25":
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
            mode=mode,
        )

    def search(
        self,
        query: str,
        k: int = 5,
        mode: RetrievalMode | None = None,
        rerank: bool = False,
    ) -> RetrievalOutcome:
        """Search once and return results plus the dense abstain signal.

        When reranking is enabled, a larger candidate pool is retrieved with
        the active first-stage mode and then reranked down to k. The abstain
        signal always remains the first-stage dense cosine similarity because
        cross-encoder logits are unbounded and on a different scale.

        Args:
            query: Query string
            k: Number of documents to retrieve
            mode: Optional retrieval mode that overrides the instance mode
            rerank: Whether to rerank the first-stage candidates

        Returns:
            RetrievalOutcome with results sorted by score descending

        Raises:
            ValueError: If rerank is requested without a configured reranker
        """
        active_mode = self.mode if mode is None else normalize_mode(mode)

        if not rerank:
            return self._first_stage(query, k, active_mode)

        if self.reranker is None:
            raise ValueError("Reranking requires a configured reranker")

        pool_size = self.rerank_candidates if self.rerank_candidates is not None else max(k * 4, 20)
        pool_size = max(pool_size, k)
        first_stage = self._first_stage(query, pool_size, active_mode)
        return RetrievalOutcome(
            results=rerank_results(query, first_stage.results, self.reranker, k),
            max_dense_similarity=first_stage.max_dense_similarity,
            mode=active_mode,
            reranked=True,
        )

    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: RetrievalMode | None = None,
        rerank: bool = False,
    ) -> list[RetrievalResult]:
        """Retrieve the top k documents for a query.

        Args:
            query: Query string
            k: Number of documents to retrieve
            mode: Optional retrieval mode that overrides the instance mode
            rerank: Whether to rerank the first-stage candidates

        Returns:
            List of RetrievalResult objects, sorted by score descending
        """
        return self.search(query, k=k, mode=mode, rerank=rerank).results

    @classmethod
    def from_index_path(
        cls,
        index_path: Path,
        model_name: str = "all-MiniLM-L6-v2",
        mode: RetrievalMode | None = None,
        rrf_k: int = 60,
        allow_model_mismatch: bool = False,
        reranker: Reranker | None = None,
        rerank_candidates: int | None = None,
    ) -> "Retriever":
        """Load a retriever from a saved index.

        Args:
            index_path: Path to the saved index directory
            model_name: Name of the embedding model to use
            mode: Retrieval mode, defaults to LECTURE_RAG_RETRIEVAL_MODE or dense
            rrf_k: Reciprocal rank fusion constant
            allow_model_mismatch: Continue with a warning when the model name
                differs from the model the index was built with
            reranker: Optional reranker used when rerank is requested
            rerank_candidates: Number of candidates retrieved before reranking,
                or None to use max(4*k, 20)

        Returns:
            Initialized Retriever instance
        """
        from .embeddings import EmbeddingModel

        index = VectorIndex.load(index_path)
        warning = index.check_model_name(
            model_name,
            allow_model_mismatch=allow_model_mismatch,
        )
        embedding_model = EmbeddingModel(model_name)
        index.check_dimension(model_name, embedding_model.dimension)
        retriever = cls(
            embedding_model,
            index,
            mode=mode,
            rrf_k=rrf_k,
            reranker=reranker,
            rerank_candidates=rerank_candidates,
        )
        retriever.model_warning = warning
        return retriever
