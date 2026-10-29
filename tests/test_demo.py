"""Tests for the Streamlit demo configuration errors."""

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from lecture_rag.index import IndexModelMismatchError, VectorIndex  # noqa: E402
from lecture_rag.models import Document  # noqa: E402
from lecture_rag.retrieval import Retriever  # noqa: E402
from tests.conftest import HashingEmbeddingModel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_demo() -> AppTest:
    """Run the demo app and return the testing harness."""
    return AppTest.from_file(str(REPO_ROOT / "demo.py"), default_timeout=60).run()


def test_invalid_retrieval_mode_shows_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An invalid retrieval mode is reported as a configuration error."""
    monkeypatch.setenv("LECTURE_RAG_RETRIEVAL_MODE", "bogus")
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(tmp_path / "missing"))

    app = run_demo()
    errors = [str(element.value) for element in app.error]

    assert any("LECTURE_RAG_RETRIEVAL_MODE" in error for error in errors)
    assert not any("Failed to load index" in error for error in errors)
    assert app.selectbox[1].value == "dense"

    app.selectbox[1].set_value("bm25")
    app.run()

    assert not app.exception


def test_missing_index_shows_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A valid mode with a missing index reports that the index is missing."""
    monkeypatch.setenv("LECTURE_RAG_RETRIEVAL_MODE", "dense")
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(tmp_path / "missing"))

    app = run_demo()
    errors = [str(element.value) for element in app.error]

    assert any("not found" in error for error in errors)


def build_demo_index(path: Path) -> None:
    """Build a small index for the demo tests."""
    model = HashingEmbeddingModel()
    documents = [
        Document(
            text="Dynamic programming solves optimization problems.",
            file_name="algorithms.pdf",
            page_number=1,
            chunk_id="algorithms.pdf_page_1",
        ),
        Document(
            text="Hash tables provide constant-time lookup.",
            file_name="data_structures.pdf",
            page_number=2,
            chunk_id="data_structures.pdf_page_2",
        ),
    ]
    index = VectorIndex(model.dimension, model_name=model.model_name)
    index.add_documents(documents, model.embed_texts([doc.text for doc in documents]))
    index.save(path)


def test_retriever_cache_does_not_depend_on_retrieval_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Changing the retrieval mode reruns without reloading the retriever."""
    index_dir = tmp_path / "index"
    build_demo_index(index_dir)
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(index_dir))

    calls: list[Path] = []

    def fake_from_index_path(
        cls: type[Retriever],
        index_path: Path,
        model_name: str = "all-MiniLM-L6-v2",
        mode: str | None = None,
        rrf_k: int = 60,
    ) -> Retriever:
        calls.append(index_path)
        return Retriever(HashingEmbeddingModel(), VectorIndex.load(index_path), mode="dense")

    monkeypatch.setattr(
        Retriever,
        "from_index_path",
        classmethod(fake_from_index_path),
    )
    streamlit.cache_resource.clear()

    try:
        app = run_demo()
        assert not app.exception
        assert len(calls) == 1

        app.selectbox[1].set_value("bm25")
        app.run()
        assert not app.exception

        app.selectbox[1].set_value("hybrid")
        app.run()
        assert not app.exception

        assert len(calls) == 1
    finally:
        streamlit.cache_resource.clear()


def test_model_index_mismatch_shows_specific_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A model and index mismatch is reported distinctly from other failures."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(index_dir))

    def fake_from_index_path(
        cls: type[Retriever],
        index_path: Path,
        model_name: str = "all-MiniLM-L6-v2",
        mode: str | None = None,
        rrf_k: int = 60,
    ) -> Retriever:
        raise IndexModelMismatchError("x")

    monkeypatch.setattr(
        Retriever,
        "from_index_path",
        classmethod(fake_from_index_path),
    )
    streamlit.cache_resource.clear()

    try:
        app = run_demo()
        errors = [str(element.value) for element in app.error]

        assert any("Model and index do not match" in error for error in errors)
        assert not any("Failed to load index" in error for error in errors)
    finally:
        streamlit.cache_resource.clear()
