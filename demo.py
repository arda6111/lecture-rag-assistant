"""Streamlit demo application for the Lecture RAG Assistant."""

import os
from pathlib import Path

import streamlit as st

from lecture_rag.generation import Generator
from lecture_rag.retrieval import DEFAULT_ABSTAIN_THRESHOLD, Retriever

st.set_page_config(
    page_title="Lecture RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Lecture RAG Assistant")
st.markdown(
    "Ask questions about lecture materials and get answers with source citations."
)


@st.cache_resource
def load_retriever(index_path: str, model_name: str) -> Retriever:
    """Load the retriever (cached)."""
    return Retriever.from_index_path(Path(index_path), model_name)


def configured_abstain_threshold() -> float:
    """Return the abstain threshold from the environment or the default."""
    raw_value = os.getenv("LECTURE_RAG_ABSTAIN_THRESHOLD")
    if raw_value is None:
        return DEFAULT_ABSTAIN_THRESHOLD
    try:
        return float(raw_value)
    except ValueError:
        return DEFAULT_ABSTAIN_THRESHOLD


index_dir = st.sidebar.text_input(
    "Index Directory", value=os.getenv("LECTURE_RAG_INDEX_DIR", "index")
)
model_name = st.sidebar.text_input("Embedding Model", value="all-MiniLM-L6-v2")
k = st.sidebar.slider("Number of sources", min_value=1, max_value=10, value=5)
mode = st.sidebar.selectbox("Answer Mode", options=["extractive", "llm"])
retrieval_mode = st.sidebar.selectbox(
    "Retrieval mode",
    options=["dense", "bm25", "hybrid"],
    index=2,
)
abstain_threshold = st.sidebar.slider(
    "Abstain threshold",
    min_value=0.0,
    max_value=1.0,
    value=configured_abstain_threshold(),
    step=0.05,
)

if not Path(index_dir).exists():
    st.error(f"Index directory '{index_dir}' not found. Please run `lecture-rag ingest` first.")
    st.stop()

try:
    retriever = load_retriever(index_dir, model_name)
    st.sidebar.success(f"✓ Loaded index with {len(retriever.index.documents)} documents")
except Exception as e:
    st.error(f"Failed to load index: {e}")
    st.stop()

query = st.text_input("Ask a question:", placeholder="What is dynamic programming?")

if query:
    retriever.set_mode(retrieval_mode)

    if retriever.should_abstain(query, abstain_threshold):
        st.info("The lecture materials do not seem to cover this question.")
    else:
        with st.spinner("Searching..."):
            sources = retriever.retrieve(query, k=k)

        best_similarity = retriever.max_dense_similarity(query)

        if not sources:
            st.warning("No relevant documents found.")
        else:
            with st.spinner("Generating answer..."):
                generator = Generator.from_config(mode=mode)
                answer = generator.generate_answer(query, sources)

            st.subheader("Answer")
            st.markdown(answer.answer_text)

            st.subheader("Sources")
            for i, result in enumerate(answer.sources, 1):
                doc = result.document
                with st.expander(
                    f"[{i}] {doc.file_name}, page {doc.page_number} (score: {result.score:.3f})"
                ):
                    st.text(doc.text)

        st.caption(f"Best dense similarity: {best_similarity:.3f}")
