"""PDF loading with page-level metadata extraction."""

import logging
from pathlib import Path

import fitz

from .models import Document

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: Path) -> list[Document]:
    """Load a PDF and extract text with page numbers.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of Document objects, one per page

    Raises:
        ValueError: If the path is invalid or file cannot be read
    """
    if not pdf_path.exists():
        raise ValueError(f"PDF file not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")

    documents = []

    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                if text.strip():
                    documents.append(
                        Document(
                            text=text,
                            file_name=pdf_path.name,
                            page_number=page_num + 1,
                            chunk_id=f"{pdf_path.name}_page_{page_num + 1}",
                        )
                    )
    except Exception as e:
        raise ValueError(f"Failed to read PDF {pdf_path}: {e}") from e

    logger.info(f"Loaded {len(documents)} pages from {pdf_path.name}")
    return documents


def load_directory(directory: Path) -> list[Document]:
    """Load all PDFs from a directory.

    Args:
        directory: Path to directory containing PDFs

    Returns:
        List of all Document objects from all PDFs

    Raises:
        ValueError: If directory does not exist or is not a directory
    """
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return []

    all_documents = []
    for pdf_path in pdf_files:
        try:
            documents = load_pdf(pdf_path)
            all_documents.extend(documents)
        except ValueError as e:
            logger.error(f"Skipping {pdf_path.name}: {e}")
            continue

    logger.info(f"Loaded {len(all_documents)} total pages from {len(pdf_files)} PDFs")
    return all_documents
