"""Tests for text chunking functionality."""

from src.lecture_rag.chunking import chunk_documents
from src.lecture_rag.models import Document


def test_chunk_short_document() -> None:
    """Test chunking a document shorter than chunk size."""
    docs = [
        Document(
            text="Short text",
            file_name="test.pdf",
            page_number=1,
            chunk_id="test_1",
        )
    ]

    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text == "Short text"


def test_chunk_long_document() -> None:
    """Test chunking a long document."""
    long_text = "A" * 1000
    docs = [
        Document(
            text=long_text,
            file_name="test.pdf",
            page_number=1,
            chunk_id="test_1",
        )
    ]

    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)
    assert all(c.file_name == "test.pdf" for c in chunks)
    assert all(c.page_number == 1 for c in chunks)


def test_chunk_multiple_documents() -> None:
    """Test chunking multiple documents."""
    docs = [
        Document(
            text="First document " * 50,
            file_name="doc1.pdf",
            page_number=1,
            chunk_id="doc1_1",
        ),
        Document(
            text="Second document " * 50,
            file_name="doc2.pdf",
            page_number=2,
            chunk_id="doc2_2",
        ),
    ]

    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 2
    assert any(c.file_name == "doc1.pdf" for c in chunks)
    assert any(c.file_name == "doc2.pdf" for c in chunks)


def test_chunk_preserves_metadata() -> None:
    """Test that chunking preserves document metadata."""
    docs = [
        Document(
            text="A" * 200,
            file_name="test.pdf",
            page_number=5,
            chunk_id="original",
        )
    ]

    chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=5)

    assert all(c.file_name == "test.pdf" for c in chunks)
    assert all(c.page_number == 5 for c in chunks)
    assert all("test.pdf_page_5_chunk_" in c.chunk_id for c in chunks)
