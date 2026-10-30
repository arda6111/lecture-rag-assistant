"""Build vector indexes from a directory of PDF documents."""

import atexit
import hashlib
import logging
import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .chunking import chunk_documents
from .embeddings import EmbeddingModel
from .index import VectorIndex
from .pdf_loader import list_pdf_files, load_directory

logger = logging.getLogger(__name__)

CORPUS_RELATIVE_PATH = Path("data") / "corpus"
CORPUS_INDEX_NAME = "corpus-index"
_FINGERPRINT_CHUNK_SIZE = 1024 * 1024

_fallback_dir: Path | None = None


def resolve_corpus_dir(app_dir: Path | None = None) -> Path:
    """Resolve the bundled corpus directory without relying on the package path.

    The application directory is preferred when it contains the corpus. If it
    does not, the current working directory is checked. When neither candidate
    exists, the first candidate is returned so that callers report the expected
    location.

    Args:
        app_dir: Directory containing the demo application, if known

    Returns:
        The first existing candidate directory, or the first candidate
    """
    candidates: list[Path] = []
    if app_dir is not None:
        candidates.append(app_dir / CORPUS_RELATIVE_PATH)
    candidates.append(Path.cwd() / CORPUS_RELATIVE_PATH)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _fallback_cache_dir() -> Path:
    """Return the process-wide private fallback cache directory.

    The directory is created once and removed at interpreter exit. A new
    directory is created when the previous one no longer exists.

    Returns:
        The existing fallback directory, or a newly created one
    """
    global _fallback_dir
    if _fallback_dir is not None and _fallback_dir.is_dir() and not _fallback_dir.is_symlink():
        return _fallback_dir

    path = Path(tempfile.mkdtemp(prefix="lecture-rag-"))
    atexit.register(shutil.rmtree, path, True)
    _fallback_dir = path
    return path


def corpus_cache_dir() -> Path:
    """Return a private directory for the corpus index cache.

    The cache directory is ``$XDG_CACHE_HOME/lecture-rag`` when
    ``XDG_CACHE_HOME`` is an absolute path, otherwise ``~/.cache/lecture-rag``.
    The directory is created with mode 0700 and must remain a real directory
    owned by the current user without group or world access. If it cannot be
    created or fails those checks, a private temporary directory is used.

    Returns:
        A private directory that can hold the corpus index
    """
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home and Path(xdg_cache_home).is_absolute():
        base = Path(xdg_cache_home)
    else:
        base = Path.home() / ".cache"

    cache_dir = base / "lecture-rag"

    try:
        cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = os.lstat(cache_dir)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise OSError(f"{cache_dir} is not a real directory")
        getuid = getattr(os, "getuid", None)
        if getuid is not None and info.st_uid != getuid():
            raise OSError(f"{cache_dir} is owned by another user")
        if info.st_mode & 0o077 != 0:
            raise OSError(f"{cache_dir} is accessible by other users")
    except OSError as e:
        logger.warning("Cannot use %s as the corpus cache directory: %s", cache_dir, e)
        return _fallback_cache_dir()

    return cache_dir


def corpus_fingerprint(corpus_dir: Path) -> str:
    """Return a digest that identifies the corpus contents.

    The digest covers the sorted PDF files directly inside corpus_dir, using
    each file's name, size and content bytes.

    Args:
        corpus_dir: Directory containing the PDF files to fingerprint

    Returns:
        The SHA-256 hex digest of the corpus
    """
    digest = hashlib.sha256()
    for pdf_path in list_pdf_files(corpus_dir):
        digest.update(pdf_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(pdf_path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with open(pdf_path, "rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(_FINGERPRINT_CHUNK_SIZE), b""):
                digest.update(chunk)
    return digest.hexdigest()


def corpus_index_name(fingerprint: str) -> str:
    """Return the index directory name for a corpus fingerprint.

    Args:
        fingerprint: Corpus fingerprint returned by corpus_fingerprint

    Returns:
        A directory name derived from the fingerprint prefix
    """
    return f"corpus-index-{fingerprint[:16]}"


def _remove_stale_corpus_indexes(cache_dir: Path, target: Path) -> None:
    """Remove corpus index siblings other than the installed target.

    Args:
        cache_dir: Directory that holds corpus index directories
        target: The index directory that must be kept
    """
    for path in cache_dir.iterdir():
        if path == target:
            continue
        if path.name != CORPUS_INDEX_NAME and not path.name.startswith("corpus-index-"):
            continue
        try:
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            logger.warning("Cannot remove stale corpus index %s", path, exc_info=True)


def build_and_install_index(
    input_dir: Path,
    cache_dir: Path,
    embedding_model: EmbeddingModel,
    model_name: str,
    index_name: str = CORPUS_INDEX_NAME,
    **kwargs: Any,
) -> Path:
    """Build an index in a temporary cache directory and install it.

    The index is built in a temporary directory inside cache_dir. An existing
    index is renamed aside, the new one is moved into place with os.replace,
    and the old one is removed afterwards.

    Args:
        input_dir: Directory containing the PDF files to index
        cache_dir: Directory that holds the final corpus index
        embedding_model: Loaded embedding model used to embed the chunks
        model_name: Name recorded in the index metadata
        index_name: Name of the installed index directory inside cache_dir
        **kwargs: Additional arguments passed to ``build_index``

    Returns:
        The path to the installed corpus index

    Raises:
        ValueError: If no PDF documents can be loaded from input_dir
        OSError: If the temporary index cannot be moved into place
    """
    target = cache_dir / index_name
    temp_dir = Path(tempfile.mkdtemp(prefix=".corpus-index-new-", dir=cache_dir))

    try:
        build_index(input_dir, temp_dir, embedding_model, model_name, **kwargs)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    aside: Path | None = None
    try:
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            aside = cache_dir / f".corpus-index-old-{uuid.uuid4().hex}"
            os.replace(target, aside)

        try:
            os.replace(temp_dir, target)
        except OSError:
            if target.exists() or target.is_symlink():
                shutil.rmtree(temp_dir, ignore_errors=True)
                if aside is not None:
                    shutil.rmtree(aside, ignore_errors=True)
                return target
            if aside is not None:
                try:
                    os.replace(aside, target)
                except OSError:
                    logger.warning("Cannot restore the previous corpus index", exc_info=True)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    if aside is not None:
        shutil.rmtree(aside, ignore_errors=True)
    _remove_stale_corpus_indexes(cache_dir, target)
    return target


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
