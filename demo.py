"""Streamlit demo application for the Lecture RAG Assistant."""

import os
from pathlib import Path
from typing import cast

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from lecture_rag.embeddings import EmbeddingModelError
from lecture_rag.generation import Generator
from lecture_rag.index import IndexMetadataError, IndexModelMismatchError
from lecture_rag.retrieval import (
    RetrievalMode,
    Retriever,
    default_retrieval_mode,
    parse_abstain_threshold,
)

MARKDOWN_SPECIAL_CHARS = frozenset("\\`*_{}[]()#+-.!|<>~")

load_dotenv(find_dotenv(usecwd=True))


def escape_markdown(text: str) -> str:
    """Backslash-escape Markdown special characters in text."""
    return "".join(
        f"\\{character}" if character in MARKDOWN_SPECIAL_CHARS else character
        for character in text
    )

st.set_page_config(
    page_title="Lecture RAG Assistant",
    layout="wide",
)

st.title("Lecture RAG Assistant")
st.markdown(
    "Ask questions about lecture materials and get answers with source citations."
)


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
        escape_markdown(
            f"Configuration error: {e}. "
            "Set LECTURE_RAG_RETRIEVAL_MODE to dense, bm25 or hybrid, "
            "or pick a mode in the sidebar."
        )
    )
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

if not Path(index_dir).exists():
    st.error(f"Index directory '{index_dir}' not found. Please run `lecture-rag ingest` first.")
    st.stop()

try:
    retriever = load_retriever(index_dir, model_name, allow_model_mismatch)
    st.sidebar.success(f"Loaded index with {len(retriever.index.documents)} documents")
except IndexModelMismatchError as e:
    st.error(escape_markdown(f"Model and index do not match: {e}"))
    st.stop()
except IndexMetadataError as e:
    st.error(escape_markdown(str(e)))
    st.stop()
except EmbeddingModelError as e:
    st.error(escape_markdown(str(e)))
    st.stop()
except Exception as e:
    st.error(escape_markdown(f"Failed to load index: {e}"))
    st.stop()

if retriever.model_warning is not None:
    st.warning(escape_markdown(retriever.model_warning))

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
        sources = outcome.results

        if not sources:
            st.warning("No relevant documents found.")
        else:
            try:
                with st.spinner("Generating answer..."):
                    generator = Generator.from_config(mode=mode)
                    answer = generator.generate_answer(query, sources)
            except ValueError as e:
                st.error(escape_markdown(str(e)))
                st.stop()

            st.subheader("Answer")
            st.markdown(answer.answer_text)

            st.subheader("Sources")
            for i, result in enumerate(answer.sources, 1):
                doc = result.document
                with st.expander(
                    f"[{i}] {doc.file_name}, page {doc.page_number} ({result.score_label})"
                ):
                    st.text(doc.text)

    st.caption(f"Best dense similarity: {outcome.max_dense_similarity:.3f}")
