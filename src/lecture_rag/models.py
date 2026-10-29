"""Data models for the RAG system."""


from pydantic import BaseModel


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


class Answer(BaseModel):
    """An answer with supporting documents."""
    answer_text: str
    sources: list[RetrievalResult]
    mode: str
