"""Pytest configuration and fixtures."""

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import pytest

from lecture_rag.models import Document

TOKEN_PATTERN = re.compile(r"\w+")


@pytest.fixture(autouse=True)
def _isolate_environment(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove project environment variables and disable dotenv loading."""
    for name in list(os.environ):
        if name.startswith(("LECTURE_RAG_", "OPENAI_")):
            monkeypatch.delenv(name, raising=False)

    if request.node.get_closest_marker("uses_dotenv"):
        return

    def noop(*args: Any, **kwargs: Any) -> bool:
        """Do nothing and report that no environment file was loaded."""
        return False

    monkeypatch.setattr("dotenv.load_dotenv", noop)
    monkeypatch.setattr("lecture_rag.cli.load_dotenv", noop)


class HashingEmbeddingModel:
    """Deterministic offline embedding model for tests."""

    dimension = 384

    def __init__(self, model_name: str = "hashing") -> None:
        self.model_name = model_name

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dimension
            vector[bucket] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._embed(text) for text in texts]).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a single query."""
        return self._embed(query)


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
