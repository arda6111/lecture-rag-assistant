"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import fitz
import numpy as np
import pytest

from src.lecture_rag.models import Document


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf(temp_dir: Path) -> Path:
    """Create a sample PDF file."""
    pdf_path = temp_dir / "test.pdf"
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((50, 50), "This is page one content.", fontsize=12)

    page2 = doc.new_page()
    page2.insert_text((50, 50), "This is page two content.", fontsize=12)

    doc.save(str(pdf_path))
    doc.close()

    return pdf_path


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents."""
    return [
        Document(
            text="Binary trees are hierarchical data structures.",
            file_name="test.pdf",
            page_number=1,
            chunk_id="test.pdf_page_1",
        ),
        Document(
            text="Hash tables provide O(1) lookup time.",
            file_name="test.pdf",
            page_number=2,
            chunk_id="test.pdf_page_2",
        ),
        Document(
            text="Dynamic programming solves optimization problems.",
            file_name="test.pdf",
            page_number=3,
            chunk_id="test.pdf_page_3",
        ),
    ]


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Create sample embeddings."""
    return np.random.rand(3, 384).astype(np.float32)
