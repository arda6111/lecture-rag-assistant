"""Streamlit demo application for the Lecture RAG Assistant."""

import os
from pathlib import Path
from typing import cast

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from lecture_rag.embeddings import EmbeddingModelError
from lecture_rag.generation import Generator
from lecture_rag.index import (
    INDEX_NOT_FOUND_MESSAGE,
    IndexModelMismatchError,
    InvalidIndexError,
)
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

load_dotenv(find_dotenv(usecwd=True))

st.set_page_config(
    page_title="Lecture RAG Assistant",
    layout="wide",
)

st.title("Lecture RAG Assistant")
st.markdown("Ask questions about lecture materials and get answers with source citations.")


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
k = st.sidebar.slider("Number of sources", min_value=1, max_value=10, value=5)
mode = st.sidebar.selectbox(
    "Answer Mode",
    options=["extractive", "llm"],
    key="answer_mode",
)
RETRIEVAL_MODES = ["dense", "bm25", "hybrid"]
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
retrieval_mode = cast(
    RetrievalMode,
    st.sidebar.selectbox(
        "Retrieval mode",
        options=RETRIEVAL_MODES,
        index=RETRIEVAL_MODES.index(configured_mode),
        key="retrieval_mode",
    ),
)
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

# The retriever comes from st.cache_resource and is shared by all sessions, so
# per-session state such as a reranker must not be stored on it.
query = st.text_input(
    "Ask a question:",
    placeholder="What is dynamic programming?",
    key="query",
)

if query:
    with st.spinner("Searching..."):
        outcome = retriever.search(query, k=k, mode=retrieval_mode)

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
            try:
                with st.spinner("Generating answer..."):
                    generator = Generator.from_config(mode=mode)
                    answer = generator.generate_answer(query, sources)
            except ValueError as e:
                st.error("Could not generate an answer.")
                st.code(str(e), language=None)
                st.stop()

            st.subheader("Answer")
            st.text(answer.answer_text)

            st.subheader("Sources")
            for i, result in enumerate(answer.sources, 1):
                doc = result.document
                with st.expander(f"Source {i}, page {doc.page_number} ({result.score_label})"):
                    st.text(f"File: {doc.file_name}")
                    st.text(doc.text)

    st.caption(f"Best dense similarity: {outcome.max_dense_similarity:.3f}")
