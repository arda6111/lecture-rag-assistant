"""Streamlit demo application for the Lecture RAG Assistant."""

import os
import tempfile
from pathlib import Path
from typing import cast

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from lecture_rag import corpus_index, embeddings
from lecture_rag.corpus_index import CORPUS_DIR
from lecture_rag.embeddings import EmbeddingModelError
from lecture_rag.generation import (
    ExtractiveFallbackProvider,
    Generator,
    OpenAICompatibleProvider,
)
from lecture_rag.index import (
    INDEX_NOT_FOUND_MESSAGE,
    IndexModelMismatchError,
    InvalidIndexError,
    VectorIndex,
)
from lecture_rag.models import Answer, RetrievalResult
from lecture_rag.rerank import (
    CrossEncoderReranker,
    default_rerank_model,
)
from lecture_rag.retrieval import (
    RetrievalMode,
    Retriever,
    default_retrieval_mode,
    parse_abstain_threshold,
)

MAX_QUERY_CHARS = 500
MAX_SOURCES = 10
MAX_QUESTIONS_PER_SESSION = 30

CORPUS_MODEL_NAME = "all-MiniLM-L6-v2"
DEMO_LLM_BASE_URL = "https://api.openai.com/v1"
DEMO_LLM_MODEL = "gpt-4o-mini"

CORPUS_CAPTION = "Corpus: data/corpus, six CC BY 4.0 arXiv tutorials, see data/SOURCES.md."
TOO_LONG_QUERY_MESSAGE = "The question is too long. Please keep it to 500 characters or fewer."
SESSION_LIMIT_MESSAGE = (
    "This session has reached its limit of 30 searches. Reload the page to start a new session."
)
RERANK_DISABLED_MESSAGE = "Reranking is disabled on this deployment."
LLM_FALLBACK_MESSAGE = "No API key was entered, so the answer uses extractive mode."
LLM_ERROR_MESSAGE = "The language model request failed. Check the key or switch to extractive mode."


def demo_mode() -> str:
    """Return the demo mode selected by the environment or local index state.

    Returns:
        "corpus" for the bundled public corpus demo, otherwise "local"
    """
    raw_mode = os.getenv("LECTURE_RAG_DEMO_MODE")
    if raw_mode is not None:
        if raw_mode in ("corpus", "local"):
            return raw_mode
        st.error("Invalid LECTURE_RAG_DEMO_MODE. Set it to corpus or local, or leave it unset.")
        return "local"

    if os.getenv("LECTURE_RAG_INDEX_DIR") is None and not (Path("index") / "index.faiss").exists():
        return "corpus"
    return "local"


def rerank_disabled() -> bool:
    """Return True when reranking is disabled for this deployment."""
    value = os.getenv("LECTURE_RAG_DEMO_DISABLE_RERANK", "")
    return value.strip().lower() in ("1", "true", "yes")


@st.cache_resource
def load_retriever(
    index_path: str,
    model_name: str,
    allow_model_mismatch: bool,
) -> Retriever:
    """Load the retriever (cached)."""
    return Retriever.from_index_path(
        Path(index_path),
        model_name,
        mode="dense",
        allow_model_mismatch=allow_model_mismatch,
    )


@st.cache_resource
def load_reranker(model_name: str) -> CrossEncoderReranker:
    """Load the cross-encoder reranker (cached)."""
    return CrossEncoderReranker(model_name)


def _load_existing_corpus_index(
    index_dir: Path,
    model_name: str,
    dimension: int,
) -> VectorIndex | None:
    """Load a reusable corpus index, or return None when it must be rebuilt."""
    if not (index_dir / "index.faiss").exists() or not (index_dir / "documents.json").exists():
        return None

    try:
        index = VectorIndex.load(index_dir)
    except InvalidIndexError:
        return None

    if index.model_name != model_name or index.dimension != dimension:
        return None
    return index


# The retriever from st.cache_resource is shared by all sessions, so the
# embedding model and index are loaded only once per deployment.
@st.cache_resource(show_spinner=False)
def build_corpus_retriever(model_name: str) -> Retriever:
    """Build or load the corpus retriever (cached)."""
    index_dir = Path(tempfile.gettempdir()) / "lecture-rag-corpus-index"
    model = embeddings.EmbeddingModel(model_name)
    index = _load_existing_corpus_index(index_dir, model_name, model.dimension)

    if index is None:
        corpus_index.build_index(CORPUS_DIR, index_dir, model, model_name)
        index = VectorIndex.load(index_dir)

    return Retriever(model, index, mode="dense")


def _retrieval_mode_select() -> RetrievalMode:
    """Render the retrieval mode selector and return the selected mode."""
    retrieval_modes = ["dense", "bm25", "hybrid"]
    try:
        configured_mode = default_retrieval_mode()
    except ValueError as e:
        st.error(
            "Invalid LECTURE_RAG_RETRIEVAL_MODE configuration. "
            "Set LECTURE_RAG_RETRIEVAL_MODE to dense, bm25 or hybrid, "
            "or pick a mode in the sidebar."
        )
        st.code(str(e), language=None)
        configured_mode = "dense"

    return cast(
        RetrievalMode,
        st.sidebar.selectbox(
            "Retrieval mode",
            options=retrieval_modes,
            index=retrieval_modes.index(configured_mode),
            key="retrieval_mode",
        ),
    )


def _generate_local_answer(
    query: str,
    sources: list[RetrievalResult],
    answer_mode: str,
) -> Answer:
    """Generate an answer with the local environment configuration."""
    try:
        with st.spinner("Generating answer..."):
            generator = Generator.from_config(mode=answer_mode)
            return generator.generate_answer(query, sources)
    except ValueError as e:
        st.error("Could not generate an answer.")
        st.code(str(e), language=None)
        st.stop()


def _generate_corpus_answer(
    query: str,
    sources: list[RetrievalResult],
    answer_mode: str,
) -> Answer:
    """Generate an answer for the public corpus demo."""
    try:
        with st.spinner("Generating answer..."):
            if answer_mode == "llm":
                key = st.session_state.get("openai_api_key", "")
                if key:
                    provider = OpenAICompatibleProvider(
                        api_key=key,
                        base_url=DEMO_LLM_BASE_URL,
                        model=DEMO_LLM_MODEL,
                        timeout=30.0,
                    )
                    generator = Generator(provider)
                else:
                    st.info(LLM_FALLBACK_MESSAGE)
                    generator = Generator(ExtractiveFallbackProvider())
            else:
                generator = Generator(ExtractiveFallbackProvider())
            return generator.generate_answer(query, sources)
    except Exception as e:
        st.error(LLM_ERROR_MESSAGE)
        st.code(type(e).__name__, language=None)
        st.stop()


def _render_query(
    retriever: Retriever,
    k: int,
    answer_mode: str,
    retrieval_mode: RetrievalMode,
    abstain_threshold: float,
    rerank: bool,
    corpus_mode: bool,
) -> None:
    """Render the query input, run the search, and show the answer."""
    # The retriever comes from st.cache_resource and is shared by all sessions,
    # so per-session state such as a reranker must not be stored on it.
    query = st.text_input(
        "Ask a question:",
        placeholder="What is dynamic programming?",
        key="query",
        max_chars=MAX_QUERY_CHARS,
    )

    if not query:
        return

    if len(query) > MAX_QUERY_CHARS:
        st.warning(TOO_LONG_QUERY_MESSAGE)
        return

    # Every rerun with a question runs a search, including reruns caused by
    # changing a sidebar setting, so the cap counts searches.
    questions_asked = st.session_state.get("questions_asked", 0)
    if questions_asked >= MAX_QUESTIONS_PER_SESSION:
        st.warning(SESSION_LIMIT_MESSAGE)
        return

    with st.spinner("Searching..."):
        outcome = retriever.search(query, k=k, mode=retrieval_mode)

    st.session_state["questions_asked"] = questions_asked + 1

    if outcome.should_abstain(abstain_threshold):
        st.info("The lecture materials do not seem to cover this question.")
    else:
        if rerank:
            try:
                reranker = load_reranker(default_rerank_model())
            except Exception as e:
                # CrossEncoderReranker wraps load failures in RerankerError, but
                # anything else raised while loading gets the same friendly message.
                st.error(
                    "Could not load the reranking model. "
                    "Uncheck 'Rerank with a cross-encoder' to continue without it."
                )
                st.code(str(e), language=None)
                st.stop()

            outcome = retriever.search(
                query,
                k=k,
                mode=retrieval_mode,
                rerank=True,
                reranker=reranker,
            )

        sources = outcome.results

        if not sources:
            st.warning("No relevant documents found.")
        else:
            if corpus_mode:
                answer = _generate_corpus_answer(query, sources, answer_mode)
            else:
                answer = _generate_local_answer(query, sources, answer_mode)

            st.subheader("Answer")
            st.text(answer.answer_text)

            st.subheader("Sources")
            for i, result in enumerate(answer.sources, 1):
                doc = result.document
                with st.expander(f"Source {i}, page {doc.page_number} ({result.score_label})"):
                    st.text(f"File: {doc.file_name}")
                    st.text(doc.text)

    st.caption(f"Best dense similarity: {outcome.max_dense_similarity:.3f}")


def _render_local() -> None:
    """Render the local demo mode."""
    index_dir = st.sidebar.text_input(
        "Index Directory",
        value=os.getenv("LECTURE_RAG_INDEX_DIR", "index"),
        key="index_dir",
    )
    model_name = st.sidebar.text_input(
        "Embedding Model",
        value="all-MiniLM-L6-v2",
        key="model_name",
    )
    k = st.sidebar.slider(
        "Number of sources",
        min_value=1,
        max_value=MAX_SOURCES,
        value=5,
    )
    assert k <= MAX_SOURCES  # nosec B101
    answer_mode = st.sidebar.selectbox(
        "Answer Mode",
        options=["extractive", "llm"],
        key="answer_mode",
    )
    retrieval_mode = _retrieval_mode_select()
    abstain_threshold = st.sidebar.slider(
        "Abstain threshold",
        min_value=0.0,
        max_value=1.0,
        value=parse_abstain_threshold(os.getenv("LECTURE_RAG_ABSTAIN_THRESHOLD")),
        step=0.05,
    )
    allow_model_mismatch = st.sidebar.checkbox(
        "Allow model mismatch",
        value=False,
        key="allow_model_mismatch",
    )
    rerank = st.sidebar.checkbox(
        "Rerank with a cross-encoder",
        value=False,
        key="rerank",
    )

    index_path = Path(index_dir)
    abs_index = Path(os.path.abspath(index_dir))
    if abs_index.is_symlink():
        st.error(
            "Invalid index: the index directory is a symlink. "
            "Use the real (non-symlink) path of the index directory instead."
        )
        st.code(index_dir, language=None)
        st.stop()

    if not index_path.exists():
        st.error(INDEX_NOT_FOUND_MESSAGE)
        st.code(index_dir, language=None)
        st.stop()

    if not index_path.is_dir():
        st.error("Index path is not a directory.")
        st.code(index_dir, language=None)
        st.stop()

    try:
        retriever = load_retriever(index_dir, model_name, allow_model_mismatch)
        st.sidebar.success(f"Loaded index with {len(retriever.index.documents)} documents")
    except IndexModelMismatchError as e:
        st.error("Model and index do not match.")
        st.code(str(e), language=None)
        st.stop()
    except InvalidIndexError as e:
        st.error("Invalid index.")
        st.code(str(e), language=None)
        st.stop()
    except EmbeddingModelError as e:
        st.error("Failed to load the embedding model.")
        st.code(str(e), language=None)
        st.stop()
    except Exception as e:
        st.error("Failed to load index.")
        st.code(str(e), language=None)
        st.stop()

    if retriever.model_warning is not None:
        st.warning(
            "The embedding model differs from the model the index was built with. "
            "Results may be unreliable."
        )
        st.code(retriever.model_warning, language=None)

    _render_query(
        retriever,
        k,
        answer_mode,
        retrieval_mode,
        abstain_threshold,
        rerank,
        corpus_mode=False,
    )


def _render_corpus() -> None:
    """Render the public corpus demo mode."""
    st.caption(CORPUS_CAPTION)

    k = st.sidebar.slider(
        "Number of sources",
        min_value=1,
        max_value=MAX_SOURCES,
        value=5,
    )
    assert k <= MAX_SOURCES  # nosec B101
    answer_mode = st.sidebar.selectbox(
        "Answer Mode",
        options=["extractive", "llm"],
        key="answer_mode",
    )
    if answer_mode == "llm":
        st.sidebar.text_input(
            "OpenAI API key (optional, kept only in this browser session)",
            type="password",
            key="openai_api_key",
        )
    retrieval_mode = _retrieval_mode_select()
    abstain_threshold = st.sidebar.slider(
        "Abstain threshold",
        min_value=0.0,
        max_value=1.0,
        value=parse_abstain_threshold(os.getenv("LECTURE_RAG_ABSTAIN_THRESHOLD")),
        step=0.05,
    )

    if rerank_disabled():
        st.sidebar.caption(RERANK_DISABLED_MESSAGE)
        rerank = False
    else:
        rerank = st.sidebar.checkbox(
            "Rerank with a cross-encoder",
            value=False,
            key="rerank",
        )

    with st.spinner(
        "Building the search index from data/corpus on first start. This takes about a minute."
    ):
        try:
            retriever = build_corpus_retriever(CORPUS_MODEL_NAME)
            st.sidebar.success(f"Loaded index with {len(retriever.index.documents)} documents")
        except EmbeddingModelError as e:
            st.error("Failed to load the embedding model.")
            st.code(str(e), language=None)
            st.stop()
        except InvalidIndexError as e:
            st.error("Invalid index.")
            st.code(str(e), language=None)
            st.stop()
        except Exception as e:
            st.error("Failed to build the corpus index.")
            st.code(str(e), language=None)
            st.stop()

    _render_query(
        retriever,
        k,
        answer_mode,
        retrieval_mode,
        abstain_threshold,
        rerank,
        corpus_mode=True,
    )


def main() -> None:
    """Run the Streamlit demo application."""
    load_dotenv(find_dotenv(usecwd=True))

    st.set_page_config(
        page_title="Lecture RAG Assistant",
        layout="wide",
    )

    st.title("Lecture RAG Assistant")
    st.markdown("Ask questions about lecture materials and get answers with source citations.")

    if demo_mode() == "corpus":
        _render_corpus()
    else:
        _render_local()


if __name__ == "__main__":
    main()
