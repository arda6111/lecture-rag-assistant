"""Security-focused tests for ingest limits and index persistence."""

import shutil
from pathlib import Path

import fitz
import numpy as np
import pytest

from lecture_rag.index import VectorIndex
from lecture_rag.models import Document
from lecture_rag.pdf_loader import load_directory, load_pdf


def test_oversized_file_rejected(sample_pdf: Path) -> None:
    """Files above the configured size limit are rejected before parsing."""
    with pytest.raises(ValueError, match="too large"):
        load_pdf(sample_pdf, max_file_size_mb=0.0001)


def test_non_pdf_content_with_pdf_suffix_rejected(temp_dir: Path) -> None:
    """A .pdf suffix alone is not enough; magic bytes must match."""
    fake_pdf = temp_dir / "fake.pdf"
    fake_pdf.write_bytes(b"This is not a PDF document.")

    with pytest.raises(ValueError, match="not a PDF"):
        load_pdf(fake_pdf)


def test_symlink_rejected(sample_pdf: Path, temp_dir: Path) -> None:
    """Symlinks are rejected even when they point to a valid PDF."""
    link_path = temp_dir / "link.pdf"
    link_path.symlink_to(sample_pdf)

    with pytest.raises(ValueError, match="symlink"):
        load_pdf(link_path)


def test_too_many_pages_rejected(sample_pdf: Path) -> None:
    """PDFs with more pages than the configured limit are rejected."""
    with pytest.raises(ValueError, match="too many pages"):
        load_pdf(sample_pdf, max_pages=1)


def test_load_directory_skips_bad_files(sample_pdf: Path, temp_dir: Path) -> None:
    """Bad files are logged and skipped while good files still load."""
    mixed_dir = temp_dir / "mixed"
    mixed_dir.mkdir()

    shutil.copyfile(sample_pdf, mixed_dir / "good.pdf")
    (mixed_dir / "bad.pdf").write_bytes(b"not a real pdf")
    (mixed_dir / "link.pdf").symlink_to(mixed_dir / "good.pdf")

    documents = load_directory(mixed_dir)

    assert len(documents) == 2
    assert all(document.file_name == "good.pdf" for document in documents)


def test_save_writes_json_not_pickle(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir: Path,
) -> None:
    """Saving an index writes documents.json and never a pickle file."""
    index = VectorIndex(dimension=384)
    index.add_documents(sample_documents, sample_embeddings)

    index_path = temp_dir / "test_index"
    index.save(index_path)

    assert (index_path / "documents.json").exists()
    assert not (index_path / "documents.pkl").exists()
    assert not list(index_path.glob("*.pkl"))


def test_index_round_trip(
    sample_documents: list[Document],
    sample_embeddings: np.ndarray,
    temp_dir: Path,
) -> None:
    """An index saved as JSON loads back into equivalent documents."""
    index = VectorIndex(dimension=384)
    index.add_documents(sample_documents, sample_embeddings)

    index_path = temp_dir / "round_trip_index"
    index.save(index_path)

    loaded_index = VectorIndex.load(index_path)

    assert loaded_index.dimension == index.dimension
    assert loaded_index.documents == sample_documents
    assert all(isinstance(document, Document) for document in loaded_index.documents)


def test_encrypted_pdf_rejected(temp_dir: Path) -> None:
    """Encrypted PDFs are rejected with a clear message."""
    pdf_path = temp_dir / "encrypted.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Secret lecture content.", fontsize=12)
    doc.save(
        str(pdf_path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    doc.close()

    with pytest.raises(ValueError, match="encrypted"):
        load_pdf(pdf_path)


def test_owner_password_only_pdf_rejected(temp_dir: Path) -> None:
    """PDFs encrypted with only an owner password are rejected too."""
    pdf_path = temp_dir / "owner_only.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((50, 50), "Restricted lecture content.", fontsize=12)
    doc.save(
        str(pdf_path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="",
    )
    doc.close()

    with pytest.raises(ValueError, match="encrypted"):
        load_pdf(pdf_path)


def test_load_directory_accepts_uppercase_suffix(sample_pdf: Path, temp_dir: Path) -> None:
    """Files ending in .PDF are loaded like .pdf files."""
    upper_dir = temp_dir / "upper"
    upper_dir.mkdir()
    shutil.copy(sample_pdf, upper_dir / "ALGO.PDF")

    documents = load_directory(upper_dir)

    assert {document.file_name for document in documents} == {"ALGO.PDF"}
