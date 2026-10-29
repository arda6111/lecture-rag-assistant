"""Data models for the RAG system."""

from typing import Literal

from pydantic import BaseModel

ScoreType = Literal["cosine", "bm25", "rrf"]


class Document(BaseModel):
    """A document chunk with metadata."""
    text: str
    file_name: str
    page_number: int
    chunk_id: str


class RetrievalResult(BaseModel):
    """A retrieved document with score."""
    document: Document
    score: float
    score_type: ScoreType = "cosine"

    @property
    def score_label(self) -> str:
        """Score formatted with its kind, for example 'rrf: 0.033'."""
        return f"{self.score_type}: {self.score:.3f}"


class Answer(BaseModel):
    """An answer with supporting documents."""
    answer_text: str
    sources: list[RetrievalResult]
    mode: str
