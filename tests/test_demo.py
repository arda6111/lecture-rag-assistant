"""Tests for the Streamlit demo configuration and safe rendering."""

import os
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from lecture_rag.generation import Generator  # noqa: E402
from lecture_rag.index import (  # noqa: E402
    IndexMetadataError,
    IndexModelMismatchError,
    InvalidIndexError,
    VectorIndex,
)
from lecture_rag.models import Answer, Document, RetrievalResult  # noqa: E402
from lecture_rag.retrieval import Retriever  # noqa: E402
from tests.conftest import HashingEmbeddingModel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
UNTRUSTED = "http://2130706433/login **bold** [x](http://e.com)"
UNSAFE_ELEMENTS = ("error", "warning", "info", "success", "markdown")


def run_demo() -> AppTest:
    """Run the demo app and return the testing harness."""
    return AppTest.from_file(str(REPO_ROOT / "demo.py"), default_timeout=60).run()


def element_values(app: AppTest, names: tuple[str, ...]) -> list[str]:
    """Return the string values of the named element kinds."""
    values: list[str] = []
    for name in names:
        values.extend(str(element.value) for element in getattr(app, name))
    return values


def all_rendered_values(app: AppTest) -> list[str]:
    """Return every rendered element value and expander label."""
    values = element_values(app, UNSAFE_ELEMENTS + ("code", "text"))
    values.extend(str(element.label) for element in app.expander)
    return values


def assert_rendered_safely(app: AppTest, text: str) -> None:
    """Assert untrusted text appears only in code or text elements."""
    assert any(text in value for value in element_values(app, ("code", "text")))
    assert not any(text in value for value in element_values(app, UNSAFE_ELEMENTS))
    assert not any(text in str(element.label) for element in app.expander)
    assert not any("\\*" in value or "\\[" in value for value in all_rendered_values(app))


def build_demo_index(
    path: Path,
    file_name: str = "algorithms.pdf",
) -> None:
    """Build a small index for the demo tests."""
    model = HashingEmbeddingModel()
    documents = [
        Document(
            text="Dynamic programming solves optimization problems.",
            file_name=file_name,
            page_number=1,
            chunk_id=f"{file_name}_page_1",
        ),
        Document(
            text="Hash tables provide constant-time lookup.",
            file_name="data_structures.pdf",
            page_number=2,
            chunk_id="data_structures.pdf_page_2",
        ),
        Document(
            text="Binary search trees keep keys in sorted order.",
            file_name="data_structures.pdf",
            page_number=3,
            chunk_id="data_structures.pdf_page_3",
        ),
        Document(
            text="Graphs are traversed with breadth-first or depth-first search.",
            file_name="graphs.pdf",
            page_number=1,
            chunk_id="graphs.pdf_page_1",
        ),
    ]
    index = VectorIndex(model.dimension, model_name=model.model_name)
    index.add_documents(documents, model.embed_texts([doc.text for doc in documents]))
    index.save(path)


def test_invalid_retrieval_mode_shows_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An invalid retrieval mode is reported as a fixed configuration error."""
    monkeypatch.setenv("LECTURE_RAG_RETRIEVAL_MODE", "bogus")
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(tmp_path / "missing"))

    app = run_demo()
    errors = element_values(app, ("error",))
    codes = element_values(app, ("code",))

    assert any("LECTURE_RAG_RETRIEVAL_MODE" in error for error in errors)
    assert any("dense, bm25 or hybrid" in error for error in errors)
    assert any("bogus" in code for code in codes)
    assert not any("Failed to load index" in error for error in errors)
    assert app.selectbox(key="retrieval_mode").value == "dense"

    app.selectbox(key="retrieval_mode").set_value("bm25")
    app.run()

    assert not app.exception


def test_missing_index_shows_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A valid mode with a missing index reports that the index is missing."""
    missing = tmp_path / "missing"
    monkeypatch.setenv("LECTURE_RAG_RETRIEVAL_MODE", "dense")
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(missing))

    app = run_demo()

    assert any(
        "Index directory not found" in error
        and "Run `lecture-rag ingest` to build the index." in error
        for error in element_values(app, ("error",))
    )
    assert any(str(missing) in code for code in element_values(app, ("code",)))


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
        allow_model_mismatch: bool = False,
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

        app.text_input(key="query").set_value("What is dynamic programming?")
        app.selectbox(key="retrieval_mode").set_value("bm25")
        app.run()
        assert not app.exception
        assert not app.error
        assert any("Sources" in str(element.value) for element in app.subheader)

        app.selectbox(key="retrieval_mode").set_value("hybrid")
        app.run()
        assert not app.exception
        assert not app.error
        assert any("Sources" in str(element.value) for element in app.subheader)

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
        allow_model_mismatch: bool = False,
    ) -> Retriever:
        raise IndexModelMismatchError("model **x**")

    monkeypatch.setattr(
        Retriever,
        "from_index_path",
        classmethod(fake_from_index_path),
    )
    streamlit.cache_resource.clear()

    try:
        app = run_demo()

        assert any(
            "Model and index do not match." in error for error in element_values(app, ("error",))
        )
        assert any("model **x**" in code for code in element_values(app, ("code",)))
        assert not any("Failed to load index" in error for error in element_values(app, ("error",)))
        assert_rendered_safely(app, "model **x**")
    finally:
        streamlit.cache_resource.clear()


def test_model_warning_requires_allow_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A model mismatch errors first and warns only after it is allowed."""
    index_dir = tmp_path / "index"
    build_demo_index(index_dir)
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(index_dir))

    def fake_from_index_path(
        cls: type[Retriever],
        index_path: Path,
        model_name: str = "all-MiniLM-L6-v2",
        mode: str | None = None,
        rrf_k: int = 60,
        allow_model_mismatch: bool = False,
    ) -> Retriever:
        if not allow_model_mismatch:
            raise IndexModelMismatchError("model mismatch for **x**")
        retriever = Retriever(
            HashingEmbeddingModel(),
            VectorIndex.load(index_path),
            mode="dense",
        )
        retriever.model_warning = "differs from **index**"
        return retriever

    monkeypatch.setattr(
        Retriever,
        "from_index_path",
        classmethod(fake_from_index_path),
    )
    streamlit.cache_resource.clear()

    try:
        app = run_demo()

        assert any(
            "Model and index do not match." in error for error in element_values(app, ("error",))
        )
        assert not app.warning

        app.checkbox(key="allow_model_mismatch").set_value(True)
        app.run()

        warnings = element_values(app, ("warning",))
        assert any(
            "The embedding model differs from the model the index was built with. "
            "Results may be unreliable." in warning
            for warning in warnings
        )
        assert any("differs from **index**" in code for code in element_values(app, ("code",)))
        assert not app.error
        assert_rendered_safely(app, "differs from **index**")
    finally:
        streamlit.cache_resource.clear()


def test_index_metadata_error_is_shown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid index metadata is reported distinctly from other failures."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(index_dir))
    message = (
        "Invalid index at x: index_meta.json is bad. "
        "Re-run `lecture-rag ingest` to rebuild the index."
    )

    def fake_from_index_path(
        cls: type[Retriever],
        index_path: Path,
        model_name: str = "all-MiniLM-L6-v2",
        mode: str | None = None,
        rrf_k: int = 60,
        allow_model_mismatch: bool = False,
    ) -> Retriever:
        raise IndexMetadataError(message)

    monkeypatch.setattr(
        Retriever,
        "from_index_path",
        classmethod(fake_from_index_path),
    )
    streamlit.cache_resource.clear()

    try:
        app = run_demo()

        assert any("Invalid index." in error for error in element_values(app, ("error",)))
        assert any(message in code for code in element_values(app, ("code",)))
        assert not any("Failed to load index" in error for error in element_values(app, ("error",)))
    finally:
        streamlit.cache_resource.clear()


def test_untrusted_index_model_error_is_rendered_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An untrusted model name in an error appears only in a code element."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(index_dir))

    def fake_from_index_path(
        cls: type[Retriever],
        index_path: Path,
        model_name: str = "all-MiniLM-L6-v2",
        mode: str | None = None,
        rrf_k: int = 60,
        allow_model_mismatch: bool = False,
    ) -> Retriever:
        raise IndexModelMismatchError(f"Embedding model {model_name!r} is wrong")

    monkeypatch.setattr(
        Retriever,
        "from_index_path",
        classmethod(fake_from_index_path),
    )
    streamlit.cache_resource.clear()

    try:
        app = run_demo()
        app.text_input(key="model_name").set_value(UNTRUSTED)
        app.run()

        assert_rendered_safely(app, UNTRUSTED)
    finally:
        streamlit.cache_resource.clear()


def test_untrusted_index_directory_is_rendered_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An untrusted index path in the not-found error appears only in code."""
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", UNTRUSTED)

    app = run_demo()

    assert any(
        "Index directory not found" in error
        and "Run `lecture-rag ingest` to build the index." in error
        for error in element_values(app, ("error",))
    )
    assert_rendered_safely(app, UNTRUSTED)


def test_untrusted_file_name_is_rendered_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An untrusted file name appears only in text, never in a label or message."""
    index_dir = tmp_path / "index"
    build_demo_index(index_dir, file_name=UNTRUSTED)
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("LECTURE_RAG_ABSTAIN_THRESHOLD", "0")

    def fake_from_index_path(
        cls: type[Retriever],
        index_path: Path,
        model_name: str = "all-MiniLM-L6-v2",
        mode: str | None = None,
        rrf_k: int = 60,
        allow_model_mismatch: bool = False,
    ) -> Retriever:
        return Retriever(
            HashingEmbeddingModel(),
            VectorIndex.load(index_path),
            mode="dense",
        )

    monkeypatch.setattr(
        Retriever,
        "from_index_path",
        classmethod(fake_from_index_path),
    )
    streamlit.cache_resource.clear()

    try:
        app = run_demo()
        app.text_input(key="query").set_value("dynamic programming")
        app.run()

        assert not app.exception
        assert_rendered_safely(app, UNTRUSTED)
        assert any(
            label.startswith("Source 1, page 1")
            for label in (str(element.label) for element in app.expander)
        )
    finally:
        streamlit.cache_resource.clear()


def test_invalid_index_error_is_shown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An invalid index file shows a fixed error plus the details in code."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(index_dir))
    message = "Invalid index at x: bad. Re-run `lecture-rag ingest` to rebuild the index."

    def fake_from_index_path(
        cls: type[Retriever],
        index_path: Path,
        model_name: str = "all-MiniLM-L6-v2",
        mode: str | None = None,
        rrf_k: int = 60,
        allow_model_mismatch: bool = False,
    ) -> Retriever:
        raise InvalidIndexError(message)

    monkeypatch.setattr(
        Retriever,
        "from_index_path",
        classmethod(fake_from_index_path),
    )
    streamlit.cache_resource.clear()

    try:
        app = run_demo()

        assert any("Invalid index." in error for error in element_values(app, ("error",)))
        assert any(message in code for code in element_values(app, ("code",)))
        assert not any("Failed to load index" in error for error in element_values(app, ("error",)))
    finally:
        streamlit.cache_resource.clear()


def test_symlinked_index_shows_real_path_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A symlinked index directory tells the user to use the real path."""
    index_dir = tmp_path / "index"
    build_demo_index(index_dir)
    link_path = tmp_path / "linked_index"
    os.symlink(index_dir, link_path)
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(link_path))

    app = run_demo()
    errors = element_values(app, ("error",))

    assert not app.exception
    assert any(
        "Invalid index: the index directory is a symlink. "
        "Use the real (non-symlink) path of the index directory instead." in error
        for error in errors
    )
    assert not any("Re-run" in error for error in errors)
    assert any(str(link_path) in code for code in element_values(app, ("code",)))


def test_dangling_symlinked_index_shows_symlink_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A dangling symlinked index directory is reported as a symlink."""
    link_path = tmp_path / "dangling_index"
    os.symlink(tmp_path / "missing_index", link_path)
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(link_path))

    app = run_demo()
    errors = element_values(app, ("error",))

    assert not app.exception
    assert any(
        "Invalid index: the index directory is a symlink. "
        "Use the real (non-symlink) path of the index directory instead." in error
        for error in errors
    )
    assert not any("Re-run" in error for error in errors)
    assert any(str(link_path) in code for code in element_values(app, ("code",)))


def test_index_under_symlinked_parent_loads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An index reached through a symlinked parent directory loads without error."""
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    build_demo_index(real_parent / "index")
    linked_parent = tmp_path / "linked_parent"
    os.symlink(real_parent, linked_parent)
    linked_index = linked_parent / "index"
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(linked_index))

    calls: list[Path] = []

    def fake_from_index_path(
        cls: type[Retriever],
        index_path: Path,
        model_name: str = "all-MiniLM-L6-v2",
        mode: str | None = None,
        rrf_k: int = 60,
        allow_model_mismatch: bool = False,
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
        assert not app.error
        assert calls == [linked_index]
    finally:
        streamlit.cache_resource.clear()


def test_hostile_chunk_and_answer_are_rendered_only_as_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Hostile chunk and answer text never reach labels or message elements."""
    hostile_chunk = "**bold** [x](http://e.com) http://2130706433/login <script>x</script>"
    hostile_answer = "**answer** [y](http://e.org) http://2130706433/admin <script>y</script>"
    index_dir = tmp_path / "index"
    model = HashingEmbeddingModel()
    documents = [
        Document(
            text=hostile_chunk,
            file_name="hostile.pdf",
            page_number=1,
            chunk_id="hostile.pdf_page_1",
        )
    ]
    index = VectorIndex(model.dimension, model_name=model.model_name)
    index.add_documents(documents, model.embed_texts([hostile_chunk]))
    index.save(index_dir)
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("LECTURE_RAG_ABSTAIN_THRESHOLD", "0")

    def fake_from_index_path(
        cls: type[Retriever],
        index_path: Path,
        model_name: str = "all-MiniLM-L6-v2",
        mode: str | None = None,
        rrf_k: int = 60,
        allow_model_mismatch: bool = False,
    ) -> Retriever:
        return Retriever(
            HashingEmbeddingModel(),
            VectorIndex.load(index_path),
            mode="dense",
        )

    class FakeGenerator:
        """Generator that returns a hostile answer and the given sources."""

        def generate_answer(
            self,
            query: str,
            sources: list[RetrievalResult],
        ) -> Answer:
            return Answer(
                answer_text=hostile_answer,
                sources=sources,
                mode="llm",
            )

    def fake_from_config(cls: type[Generator], mode: str = "llm") -> FakeGenerator:
        return FakeGenerator()

    monkeypatch.setattr(
        Retriever,
        "from_index_path",
        classmethod(fake_from_index_path),
    )
    monkeypatch.setattr(
        "lecture_rag.generation.Generator.from_config",
        classmethod(fake_from_config),
    )
    streamlit.cache_resource.clear()

    try:
        app = run_demo()
        app.text_input(key="query").set_value("bold login script")
        app.run()

        assert not app.exception
        assert_rendered_safely(app, hostile_chunk)
        assert_rendered_safely(app, hostile_answer)
    finally:
        streamlit.cache_resource.clear()
