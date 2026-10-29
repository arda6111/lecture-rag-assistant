"""Build vector indexes from a directory of PDF documents."""

from collections.abc import Callable
from pathlib import Path

from .chunking import chunk_documents
from .embeddings import EmbeddingModel
from .index import VectorIndex
from .pdf_loader import load_directory

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


def build_index(
    input_dir: Path,
    index_dir: Path,
    embedding_model: EmbeddingModel,
    model_name: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    progress: Callable[[str], None] | None = None,
    max_file_size_mb: float | None = None,
    max_pages: int | None = None,
) -> int:
    """Load PDFs, embed their chunks, and save a vector index.

    Args:
        input_dir: Directory containing the PDF files to index
        index_dir: Directory where the index files are written
        embedding_model: Loaded embedding model used to embed the chunks
        model_name: Name recorded in the index metadata
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        progress: Optional callback that receives human-readable progress
            messages
        max_file_size_mb: Maximum allowed PDF file size in MB
        max_pages: Maximum allowed number of pages per PDF

    Returns:
        The number of indexed chunks

    Raises:
        ValueError: If no PDF documents can be loaded from input_dir
    """

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    report(f"Loading PDFs from {input_dir}...")
    documents = load_directory(
        input_dir,
        max_file_size_mb=max_file_size_mb,
        max_pages=max_pages,
    )

    if not documents:
        raise ValueError(f"No PDF documents could be loaded from {input_dir}")

    report(f"Chunking {len(documents)} documents...")
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    report(f"Loading embedding model: {model_name}...")
    report("Generating embeddings...")
    embeddings = embedding_model.embed_texts([doc.text for doc in chunks])

    report("Building index...")
    vector_index = VectorIndex(embedding_model.dimension, model_name=model_name)
    vector_index.add_documents(chunks, embeddings)

    report(f"Saving index to {index_dir}...")
    vector_index.save(index_dir)

    report(f"Indexed {len(chunks)} chunks from {len(documents)} documents")
    return len(chunks)
