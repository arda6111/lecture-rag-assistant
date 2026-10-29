"""Tests for PDF loading functionality."""

from pathlib import Path

import pytest

from lecture_rag.pdf_loader import load_directory, load_pdf


def test_load_pdf(sample_pdf: Path) -> None:
    """Test loading a single PDF."""
    documents = load_pdf(sample_pdf)

    assert len(documents) == 2
    assert all(doc.file_name == "test.pdf" for doc in documents)
    assert documents[0].page_number == 1
    assert documents[1].page_number == 2
    assert "page one" in documents[0].text.lower()
    assert "page two" in documents[1].text.lower()


def test_load_pdf_nonexistent(temp_dir: Path) -> None:
    """Test loading a nonexistent PDF raises error."""
    with pytest.raises(ValueError, match="not found"):
        load_pdf(temp_dir / "nonexistent.pdf")


def test_load_pdf_invalid_type(temp_dir: Path) -> None:
    """Test loading non-PDF file raises error."""
    txt_file = temp_dir / "test.txt"
    txt_file.write_text("Not a PDF")

    with pytest.raises(ValueError, match="not a PDF"):
        load_pdf(txt_file)


def test_load_directory(sample_pdf: Path, temp_dir: Path) -> None:
    """Test loading PDFs from directory."""
    documents = load_directory(sample_pdf.parent)

    assert len(documents) >= 2
    assert all(doc.file_name == "test.pdf" for doc in documents)


def test_load_directory_empty(temp_dir: Path) -> None:
    """Test loading from empty directory."""
    empty_dir = temp_dir / "empty"
    empty_dir.mkdir()

    documents = load_directory(empty_dir)
    assert len(documents) == 0


def test_load_directory_nonexistent(temp_dir: Path) -> None:
    """Test loading from nonexistent directory raises error."""
    with pytest.raises(ValueError, match="not found"):
        load_directory(temp_dir / "nonexistent")
