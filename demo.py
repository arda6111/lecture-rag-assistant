"""Streamlit demo application for the Lecture RAG Assistant."""

import importlib
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"
SOURCE_STAMP_ATTR = "_demo_source_stamp"


def source_stamp(src_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """Return a fingerprint of the lecture_rag source files.

    Args:
        src_dir: The checkout's src directory that contains the lecture_rag package.

    Returns:
        A sorted tuple of relative path, modification time in nanoseconds, and size.
    """
    package_dir = src_dir / "lecture_rag"
    entries: list[tuple[str, int, int]] = []
    for path in package_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        stat = path.stat()
        entries.append((path.relative_to(src_dir).as_posix(), stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(entries))


def drop_stale_modules(src_dir: Path) -> bool:
    """Drop cached lecture_rag modules whose source files changed on disk.

    Streamlit Community Cloud updates the checkout in place and reruns the
    script in the same process, so modules imported by the previous version
    would otherwise stay in sys.modules.

    Args:
        src_dir: The checkout's src directory that contains the lecture_rag package.

    Returns:
        True when stale modules were removed, otherwise False.
    """
    if "lecture_rag" not in sys.modules or not (src_dir / "lecture_rag").is_dir():
        return False

    module = sys.modules["lecture_rag"]
    if getattr(module, SOURCE_STAMP_ATTR, None) == source_stamp(src_dir):
        return False

    for name in list(sys.modules):
        if name == "lecture_rag" or name.startswith("lecture_rag."):
            del sys.modules[name]

    importlib.invalidate_caches()
    return True


def use_repo_sources(src_dir: Path) -> bool:
    """Make imports resolve to the checkout's sources instead of installed copies.

    Args:
        src_dir: The checkout's src directory that contains the lecture_rag package.

    Returns:
        True when src_dir contains lecture_rag and was added to sys.path, otherwise False.
    """
    if not (src_dir / "lecture_rag").is_dir():
        return False

    resolved_src = src_dir.resolve()
    sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != resolved_src]
    sys.path.insert(0, str(src_dir))

    module = sys.modules.get("lecture_rag")
    if module is not None:
        module_file = getattr(module, "__file__", None)
        inside_src = module_file is not None and Path(module_file).resolve().is_relative_to(
            resolved_src
        )
        if not inside_src:
            for name in list(sys.modules):
                if name == "lecture_rag" or name.startswith("lecture_rag."):
                    del sys.modules[name]

    importlib.invalidate_caches()
    return True


# Streamlit Cloud can install a stale non-editable copy, so use the checkout's sources.
use_repo_sources(SRC_DIR)
# It also reruns the updated script in the same process, so drop modules from the old version.
_stale_modules_dropped = drop_stale_modules(SRC_DIR)

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError

import lecture_rag
from lecture_rag import corpus_index, embeddings
from lecture_rag.corpus_index import resolve_corpus_dir
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

_classes_token = id(sys.modules["lecture_rag.models"])

lecture_rag_file = getattr(lecture_rag, "__file__", None)
if (
    (SRC_DIR / "lecture_rag").is_dir()
    and lecture_rag_file is not None
    and Path(lecture_rag_file).resolve().is_relative_to(SRC_DIR.resolve())
):
    setattr(lecture_rag, SOURCE_STAMP_ATTR, source_stamp(SRC_DIR))


@st.cache_resource
def _loaded_classes() -> dict[str, int | None]:
    """Return the identity of the lecture_rag classes used by cached resources."""
    return {"token": None}  # nosec B105


# Streamlit's file watcher can re-import lecture_rag without drop_stale_modules
# seeing it, because the package may already be gone from sys.modules. Tie the
# cached resources to the loaded class module so old objects are never reused.
holder = _loaded_classes()
if (holder["token"] is not None and holder["token"] != _classes_token) or _stale_modules_dropped:
    st.cache_resource.clear()
    holder = _loaded_classes()
holder["token"] = _classes_token

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 500
MAX_SOURCES = 10
MAX_QUESTIONS_PER_SESSION = 30

CORPUS_MODEL_NAME = "all-MiniLM-L6-v2"
CORPUS_DEFAULT_RETRIEVAL_MODE: RetrievalMode = "hybrid"
CORPUS_DEFAULT_ABSTAIN_THRESHOLD = 0.45


@dataclass(frozen=True)
class LLMProvider:
    """A fixed LLM provider available in the public corpus demo.

    Attributes:
        name: Display name of the provider.
        base_url: Fixed OpenAI-compatible endpoint.
        model: Model name sent to the provider.
        key_label: Label for the API key input field.
    """

    name: str
    base_url: str
    model: str
    key_label: str


LLM_PROVIDERS: Mapping[str, LLMProvider] = MappingProxyType(
    {
        "OpenAI": LLMProvider(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            key_label="OpenAI API key",
        ),
        "Groq": LLMProvider(
            name="Groq",
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b",
            key_label="Groq API key",
        ),
        "OpenRouter": LLMProvider(
            name="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            key_label="OpenRouter API key",
        ),
    }
)
DEFAULT_LLM_PROVIDER = "OpenAI"

WELCOME_MESSAGE = (
    "Generated answers need your own API key. Switch Answer Mode to llm in the sidebar, "
    "pick OpenAI, Groq or OpenRouter, and paste your key there. Groq and OpenRouter offer "
    "free tiers. The key is kept only for this session and is never stored. Without a key, "
    "the demo shows the most relevant passages with their sources (extractive mode)."
)

CORPUS_CAPTION = (
    "Corpus: six CC BY 4.0 arXiv tutorials on transformers, autoencoders, reinforcement "
    "learning, testing neural networks, MLOps and clustering (see data/SOURCES.md). "
    "Questions outside these topics should get no answer."
)
EXAMPLE_QUESTIONS = (
    "How does DBSCAN identify core points and outliers?",
    "What does the bottleneck layer of an autoencoder learn?",
    "How does self-attention work in a transformer?",
    "How is neuron coverage used to test neural networks?",
)
TOO_LONG_QUERY_MESSAGE = "The question is too long. Please keep it to 500 characters or fewer."
SESSION_LIMIT_MESSAGE = (
    "This session has reached its limit of 30 searches. Reload the page to start a new session."
)
RERANK_DISABLED_MESSAGE = "Reranking is disabled on this deployment."
LLM_FALLBACK_MESSAGE = "No API key was entered, so the answer uses extractive mode."
CORPUS_INVALID_INDEX_MESSAGE = "The search index could not be loaded. Please try again later."
CORPUS_BUILD_ERROR_MESSAGE = "Failed to build the corpus index. Please try again later."


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


def rerank_enabled() -> bool:
    """Return True when reranking is enabled for this deployment."""
    value = os.getenv("LECTURE_RAG_DEMO_ENABLE_RERANK", "")
    return value.strip().lower() in ("1", "true", "yes")


def is_query_too_long(query: str) -> bool:
    """Return True when the query exceeds the maximum length."""
    return len(query) > MAX_QUERY_CHARS


def clamp_sources(k: int) -> int:
    """Clamp a source count to the supported range.

    Args:
        k: Requested number of sources

    Returns:
        The value limited to between 1 and MAX_SOURCES
    """
    return max(1, min(k, MAX_SOURCES))


@st.cache_resource
def load_retriever(
    index_path: str,
    model_name: str,
    allow_model_mismatch: bool,
    classes_token: int,
) -> Retriever:
    """Load the retriever (cached).

    Args:
        index_path: Path to the saved index directory.
        model_name: Name of the embedding model.
        allow_model_mismatch: Whether to allow a model and index mismatch.
        classes_token: Identity of the loaded lecture_rag.models module. It is
            only part of the cache key and is not used to load the retriever.

    Returns:
        The cached retriever.
    """
    return Retriever.from_index_path(
        Path(index_path),
        model_name,
        mode="dense",
        allow_model_mismatch=allow_model_mismatch,
    )


@st.cache_resource
def load_reranker(model_name: str, classes_token: int) -> CrossEncoderReranker:
    """Load the cross-encoder reranker (cached).

    Args:
        model_name: Name of the cross-encoder model.
        classes_token: Identity of the loaded lecture_rag.models module. It is
            only part of the cache key and is not used to load the reranker.

    Returns:
        The cached cross-encoder reranker.
    """
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
def build_corpus_retriever(
    model_name: str,
    corpus_dir: str,
    fingerprint: str,
    classes_token: int,
) -> Retriever:
    """Build or load the corpus retriever (cached).

    Args:
        model_name: Name of the embedding model.
        corpus_dir: Path to the corpus directory.
        fingerprint: Fingerprint that identifies the corpus contents.
        classes_token: Identity of the loaded lecture_rag.models module. It is
            only part of the cache key and is not used to build the retriever.

    Returns:
        The cached corpus retriever.
    """
    cache_dir = corpus_index.corpus_cache_dir()
    index_name = corpus_index.corpus_index_name(fingerprint)
    index_dir = cache_dir / index_name
    model = embeddings.EmbeddingModel(model_name)
    index = _load_existing_corpus_index(index_dir, model_name, model.dimension)

    if index is None:
        corpus_index.build_and_install_index(
            Path(corpus_dir),
            cache_dir,
            model,
            model_name,
            index_name=index_name,
        )
        index = VectorIndex.load(index_dir)

    return Retriever(model, index, mode="dense")


def _retrieval_mode_select(default_mode: RetrievalMode = "dense") -> RetrievalMode:
    """Render the retrieval mode selector and return the selected mode.

    Args:
        default_mode: Mode used when LECTURE_RAG_RETRIEVAL_MODE is unset or empty

    Returns:
        The retrieval mode selected in the sidebar
    """
    retrieval_modes = ["dense", "bm25", "hybrid"]
    raw_mode = os.getenv("LECTURE_RAG_RETRIEVAL_MODE")
    if raw_mode is None or not raw_mode.strip():
        configured_mode: RetrievalMode = default_mode
    else:
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


def _corpus_abstain_threshold() -> float:
    """Return the corpus-mode abstain threshold.

    Returns:
        The parsed environment value when it is set and non-empty, otherwise
        the corpus-mode default
    """
    raw = os.getenv("LECTURE_RAG_ABSTAIN_THRESHOLD")
    if raw is None or not raw.strip():
        return CORPUS_DEFAULT_ABSTAIN_THRESHOLD
    return parse_abstain_threshold(raw)


def _set_example_query(question: str) -> None:
    """Set the query input to an example question.

    Args:
        question: Example question selected by the button
    """
    st.session_state["query"] = question


@st.dialog("About answers in this demo")
def _welcome_dialog() -> None:
    """Show the one-time welcome dialog for the public corpus demo."""
    st.write(WELCOME_MESSAGE)
    if st.button("Got it", key="welcome_got_it"):
        st.session_state["welcome_seen"] = True
        st.rerun()


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


def describe_llm_error(exc: BaseException) -> str:
    """Return a short, safe description of an LLM request failure.

    Args:
        exc: Exception raised while generating an answer.

    Returns:
        A description naming the failure type and, for HTTP failures, the
        status code. It never includes the exception message, a URL, a path, or
        an API key.
    """
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__

    for error in chain:
        if isinstance(error, APIStatusError):
            detail = f"{type(error).__name__} (HTTP {error.status_code})"
            if error.status_code == 401:
                return f"{detail} - the key was rejected"
            if error.status_code == 429:
                return (
                    f"{detail} - rate limit or free-tier quota reached, "
                    "try again later or use another provider"
                )
            return detail
        if isinstance(error, APITimeoutError):
            return "APITimeoutError (no response within 30 s)"
        if isinstance(error, APIConnectionError):
            return "APIConnectionError"

    return type(chain[-1]).__name__


def _generate_corpus_answer(
    query: str,
    sources: list[RetrievalResult],
    answer_mode: str,
) -> Answer:
    """Generate an answer for the public corpus demo."""
    key = st.session_state.get("openai_api_key", "")
    if answer_mode != "llm" or not key:
        if answer_mode == "llm":
            st.info(LLM_FALLBACK_MESSAGE)
        try:
            with st.spinner("Generating answer..."):
                generator = Generator(ExtractiveFallbackProvider())
                return generator.generate_answer(query, sources)
        except Exception:
            logger.exception("Extractive answer generation failed")
            raise

    provider = LLM_PROVIDERS.get(
        st.session_state.get("llm_provider", DEFAULT_LLM_PROVIDER),
        LLM_PROVIDERS[DEFAULT_LLM_PROVIDER],
    )
    try:
        with st.spinner("Generating answer..."):
            generator = Generator(
                OpenAICompatibleProvider(
                    api_key=key,
                    base_url=provider.base_url,
                    model=provider.model,
                    timeout=30.0,
                )
            )
            return generator.generate_answer(query, sources)
    except Exception as exc:
        logger.warning(
            "The %s request failed with %s",
            provider.name,
            type(exc).__name__,
        )
        st.error(
            f"The {provider.name} request failed. Check the key, try another provider, "
            "or switch to extractive mode."
        )
        st.code(describe_llm_error(exc), language=None)
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
    if corpus_mode:
        st.caption("Try an example:")
        example_columns = st.columns(2)
        for index, question in enumerate(EXAMPLE_QUESTIONS):
            example_columns[index % len(example_columns)].button(
                question,
                key=f"example_{index}",
                on_click=_set_example_query,
                args=(question,),
            )

    query = st.text_input(
        "Ask a question:",
        placeholder="What is dynamic programming?",
        key="query",
        max_chars=MAX_QUERY_CHARS,
    )

    if not query:
        return

    if is_query_too_long(query):
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
                reranker = load_reranker(default_rerank_model(), _classes_token)
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
    k = clamp_sources(k)
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
        retriever = load_retriever(
            index_dir,
            model_name,
            allow_model_mismatch,
            _classes_token,
        )
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
    if not st.session_state.get("welcome_seen"):
        # Mark it as seen when it is shown, so closing it with the X or Escape
        # also keeps it from coming back in this session.
        st.session_state["welcome_seen"] = True
        _welcome_dialog()

    st.caption(CORPUS_CAPTION)

    k = st.sidebar.slider(
        "Number of sources",
        min_value=1,
        max_value=MAX_SOURCES,
        value=5,
    )
    k = clamp_sources(k)
    answer_mode = st.sidebar.selectbox(
        "Answer Mode",
        options=["extractive", "llm"],
        key="answer_mode",
    )
    if answer_mode == "llm":
        provider_name = st.sidebar.selectbox(
            "LLM provider",
            options=list(LLM_PROVIDERS),
            key="llm_provider",
        )
        provider = LLM_PROVIDERS[provider_name]
        st.sidebar.text_input(
            f"{provider.key_label} (optional, kept only for this session, never stored)",
            type="password",
            key="openai_api_key",
            help=(
                "Held only in this session's memory on the server, never written to disk, "
                f"logs or caches, and sent only to {provider.base_url}."
            ),
        )
    retrieval_mode = _retrieval_mode_select(CORPUS_DEFAULT_RETRIEVAL_MODE)
    abstain_threshold = st.sidebar.slider(
        "Abstain threshold",
        min_value=0.0,
        max_value=1.0,
        value=_corpus_abstain_threshold(),
        step=0.05,
    )

    if not rerank_enabled():
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
            corpus_dir = resolve_corpus_dir(APP_DIR)
            fingerprint = corpus_index.corpus_fingerprint(corpus_dir)
            retriever = build_corpus_retriever(
                CORPUS_MODEL_NAME,
                str(corpus_dir),
                fingerprint,
                _classes_token,
            )
            st.sidebar.success(f"Loaded index with {len(retriever.index.documents)} documents")
        except EmbeddingModelError:
            logger.exception("Corpus embedding model load failed")
            st.error("The embedding model could not be loaded. Please try again later.")
            st.stop()
        except InvalidIndexError:
            logger.exception("Corpus index load failed")
            st.error(CORPUS_INVALID_INDEX_MESSAGE)
            st.stop()
        except Exception:
            logger.exception("Corpus index build failed")
            st.error(CORPUS_BUILD_ERROR_MESSAGE)
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
