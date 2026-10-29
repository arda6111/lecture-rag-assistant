"""PDF loading with page-level metadata extraction."""

import logging
import os
from pathlib import Path

import fitz

from .models import Document

logger = logging.getLogger(__name__)

DEFAULT_MAX_FILE_SIZE_MB = 50
DEFAULT_MAX_PAGES = 500

MAX_FILE_SIZE_ENV = "LECTURE_RAG_MAX_FILE_SIZE_MB"
MAX_PAGES_ENV = "LECTURE_RAG_MAX_PAGES"


def _env_float(name: str, default: float) -> float:
    """Read a positive float from the environment, falling back to a default."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r, using %s", name, raw_value, default)
        return default
    if value <= 0:
        logger.warning("Invalid %s=%r, using %s", name, raw_value, default)
        return default
    return value


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment, falling back to a default."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r, using %s", name, raw_value, default)
        return default
    if value <= 0:
        logger.warning("Invalid %s=%r, using %s", name, raw_value, default)
        return default
    return value


def _resolve_limits(
    max_file_size_mb: float | None,
    max_pages: int | None,
) -> tuple[float, int]:
    """Resolve ingest limits from arguments, environment, or defaults."""
    if max_file_size_mb is None:
        max_file_size_mb = _env_float(MAX_FILE_SIZE_ENV, DEFAULT_MAX_FILE_SIZE_MB)
    if max_pages is None:
        max_pages = _env_int(MAX_PAGES_ENV, DEFAULT_MAX_PAGES)
    return max_file_size_mb, max_pages


def _check_pdf_header(pdf_path: Path) -> None:
    """Verify the PDF magic bytes before parsing the document."""
    with open(pdf_path, "rb") as file_handle:
        header = file_handle.read(5)
    if header != b"%PDF-":
        raise ValueError(f"File is not a PDF (invalid signature): {pdf_path}")


def load_pdf(
    pdf_path: Path,
    max_file_size_mb: float | None = None,
    max_pages: int | None = None,
) -> list[Document]:
    """Load a PDF and extract text with page numbers.

    Args:
        pdf_path: Path to the PDF file
        max_file_size_mb: Maximum allowed file size in MB
        max_pages: Maximum allowed number of pages per PDF

    Returns:
        List of Document objects, one per page

    Raises:
        ValueError: If the path is invalid or file cannot be read
    """
    max_file_size_mb, max_pages = _resolve_limits(max_file_size_mb, max_pages)

    if pdf_path.is_symlink():
        raise ValueError(f"Refusing to load symlink: {pdf_path}")

    if not pdf_path.exists():
        raise ValueError(f"PDF file not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")

    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    file_size = pdf_path.stat().st_size
    if file_size > max_file_size_bytes:
        raise ValueError(
            f"PDF file too large: {pdf_path} is {file_size} bytes, "
            f"limit is {max_file_size_mb} MB"
        )

    _check_pdf_header(pdf_path)

    documents = []

    try:
        with fitz.open(pdf_path) as doc:
            if doc.needs_pass:
                raise ValueError(f"PDF is encrypted and cannot be read: {pdf_path}")

            page_count = len(doc)
            if page_count > max_pages:
                raise ValueError(
                    f"PDF has too many pages: {pdf_path} has {page_count} pages, "
                    f"limit is {max_pages}"
                )

            for page_num in range(page_count):
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
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read PDF {pdf_path}: {e}") from e

    logger.info(f"Loaded {len(documents)} pages from {pdf_path.name}")
    return documents


def load_directory(
    directory: Path,
    max_file_size_mb: float | None = None,
    max_pages: int | None = None,
) -> list[Document]:
    """Load all PDFs from a directory.

    Args:
        directory: Path to directory containing PDFs
        max_file_size_mb: Maximum allowed file size in MB per PDF
        max_pages: Maximum allowed number of pages per PDF

    Returns:
        List of all Document objects from all PDFs

    Raises:
        ValueError: If directory does not exist or is not a directory
    """
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    max_file_size_mb, max_pages = _resolve_limits(max_file_size_mb, max_pages)

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return []

    all_documents = []
    for pdf_path in pdf_files:
        try:
            documents = load_pdf(
                pdf_path,
                max_file_size_mb=max_file_size_mb,
                max_pages=max_pages,
            )
            all_documents.extend(documents)
        except ValueError as e:
            logger.error(f"Skipping {pdf_path.name}: {e}")
            continue

    logger.info(f"Loaded {len(all_documents)} total pages from {len(pdf_files)} PDFs")
    return all_documents
