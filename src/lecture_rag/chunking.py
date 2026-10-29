"""Text chunking strategies for document processing."""

import logging

from .models import Document

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Split documents into smaller chunks with overlap.

    Args:
        documents: List of documents to chunk
        chunk_size: Maximum number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks

    Returns:
        List of chunked documents with updated chunk IDs

    Raises:
        ValueError: If the chunk size or overlap parameters are invalid
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunked_docs = []

    for doc in documents:
        text = doc.text
        text_length = len(text)

        if text_length <= chunk_size:
            chunked_docs.append(doc)
            continue

        start = 0
        chunk_idx = 0

        while start < text_length:
            end = min(start + chunk_size, text_length)

            chunk_text = text[start:end]

            if chunk_text.strip():
                chunked_docs.append(
                    Document(
                        text=chunk_text,
                        file_name=doc.file_name,
                        page_number=doc.page_number,
                        chunk_id=f"{doc.file_name}_page_{doc.page_number}_chunk_{chunk_idx}",
                    )
                )
                chunk_idx += 1

            if end >= text_length:
                break

            start = end - chunk_overlap

    logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
    return chunked_docs
