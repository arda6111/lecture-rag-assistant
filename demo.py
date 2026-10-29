"""Streamlit demo application for the Lecture RAG Assistant."""

from pathlib import Path

import streamlit as st

from src.lecture_rag.generation import Generator
from src.lecture_rag.retrieval import Retriever

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


index_dir = st.sidebar.text_input("Index Directory", value="index")
model_name = st.sidebar.text_input("Embedding Model", value="all-MiniLM-L6-v2")
k = st.sidebar.slider("Number of sources", min_value=1, max_value=10, value=5)
mode = st.sidebar.selectbox("Answer Mode", options=["extractive", "llm"])

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
    with st.spinner("Searching..."):
        sources = retriever.retrieve(query, k=k)
    
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
