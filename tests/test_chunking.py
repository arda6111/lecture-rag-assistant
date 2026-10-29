"""Tests for text chunking functionality."""

import pytest

from lecture_rag.chunking import chunk_documents
from lecture_rag.models import Document


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


def make_document(text: str) -> Document:
    """Create a document for chunking tests."""
    return Document(
        text=text,
        file_name="test.pdf",
        page_number=1,
        chunk_id="test_1",
    )


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 150),
    ],
)
def test_invalid_chunk_parameters_raise(chunk_size: int, chunk_overlap: int) -> None:
    """Invalid chunk size and overlap values raise ValueError."""
    with pytest.raises(ValueError):
        chunk_documents([make_document("A" * 200)], chunk_size, chunk_overlap)


def test_chunks_cover_text_without_trailing_overlap_only_chunk() -> None:
    """The final chunk ends at the end of the text and no chunk is contained."""
    text = "".join(chr(0x4E00 + index) for index in range(1000))

    chunks = chunk_documents(
        [make_document(text)],
        chunk_size=512,
        chunk_overlap=64,
    )

    expected_intervals = [(0, 512), (448, 960), (896, 1000)]

    assert len(chunks) == 3

    intervals = []
    for chunk, (start, end) in zip(chunks, expected_intervals):
        assert chunk.text == text[start:end]
        intervals.append((start, end))

    assert intervals == expected_intervals
    assert intervals[-1][1] == len(text)
    assert chunks[-1].text == text[-len(chunks[-1].text):]
    assert text[936:1000] not in [chunk.text for chunk in chunks]

    covered = set()
    for start, end in intervals:
        covered.update(range(start, end))
    assert covered == set(range(len(text)))

    for (prev_start, prev_end), (start, end) in zip(intervals, intervals[1:]):
        assert not (prev_start <= start and end <= prev_end)
