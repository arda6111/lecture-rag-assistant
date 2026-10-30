"""Offline tests for the public corpus demo and secret handling."""

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - newer openai packages use httpx2
    import httpx2 as httpx

streamlit = pytest.importorskip("streamlit")
from openai import (  # noqa: E402
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from streamlit.testing.v1 import AppTest  # noqa: E402

import demo  # noqa: E402
from lecture_rag import corpus_index  # noqa: E402
from lecture_rag.corpus_index import (  # noqa: E402
    build_and_install_index,
    build_index,
    corpus_fingerprint,
    corpus_index_name,
    resolve_corpus_dir,
)
from lecture_rag.embeddings import EmbeddingModelError  # noqa: E402
from lecture_rag.generation import (  # noqa: E402
    OpenAICompatibleProvider,
    redact_secrets,
)
from lecture_rag.index import InvalidIndexError, VectorIndex  # noqa: E402
from lecture_rag.models import Document, RetrievalResult  # noqa: E402
from tests.conftest import HashingEmbeddingModel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
SAMPLE_DIR = REPO_ROOT / "data" / "sample_lectures"
MODEL_NAME = "all-MiniLM-L6-v2"
# Built from parts so secret scanners do not flag this fake test value.
API_KEY = "-".join(["sk", "test", "SECRET1234567890"])
ON_TOPIC_QUERY = "How does DBSCAN identify core points and outliers?"
OUTPUT_ELEMENTS = (
    "error",
    "warning",
    "info",
    "success",
    "markdown",
    "code",
    "text",
    "caption",
    "subheader",
)


@pytest.fixture(autouse=True)
def _clear_streamlit_cache() -> None:
    """Clear the Streamlit resource cache before and after each test."""
    streamlit.cache_resource.clear()
    yield
    streamlit.cache_resource.clear()


@pytest.fixture(scope="session")
def corpus_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the corpus index once with the hashing test model."""
    base = tmp_path_factory.mktemp("corpus-cache")
    cache_dir = base / "lecture-rag"
    cache_dir.mkdir(mode=0o700)
    fingerprint = corpus_fingerprint(CORPUS_DIR)
    build_and_install_index(
        CORPUS_DIR,
        cache_dir,
        HashingEmbeddingModel(),
        MODEL_NAME,
        index_name=corpus_index_name(fingerprint),
    )
    return base


def element_values(app: AppTest, names: tuple[str, ...]) -> list[str]:
    """Return the string values of the named element kinds."""
    values: list[str] = []
    for name in names:
        values.extend(str(element.value) for element in getattr(app, name))
    return values


def all_output_values(app: AppTest) -> list[str]:
    """Return every output element value and expander label."""
    values = element_values(app, OUTPUT_ELEMENTS)
    values.extend(str(element.label) for element in app.expander)
    return values


def groq_request() -> httpx.Request:
    """Return a Groq request object without sending it."""
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def status_error(
    error_type: type[APIStatusError],
    status_code: int,
    message: str = "request failed",
) -> APIStatusError:
    """Build an OpenAI status error backed by a real HTTP response."""
    request = groq_request()
    return error_type(
        message,
        response=httpx.Response(status_code, request=request),
        body=None,
    )


def sample_source() -> RetrievalResult:
    """Return a single retrieval result for provider tests."""
    return RetrievalResult(
        document=Document(
            text="DBSCAN uses core points.",
            file_name="clustering.pdf",
            page_number=1,
            chunk_id="clustering.pdf_page_1",
        ),
        score=1.0,
    )


class FakeSentenceTransformer:
    """Offline stand-in for sentence-transformers used after a module reload."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = HashingEmbeddingModel(model_name)

    def get_sentence_embedding_dimension(self) -> int:
        """Return the deterministic embedding dimension."""
        return self._model.dimension

    def encode(
        self,
        texts: str | list[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = True,
    ) -> object:
        """Return deterministic embeddings for one text or a list of texts."""
        if isinstance(texts, str):
            return self._model.embed_query(texts)
        return self._model.embed_texts(texts, show_progress=show_progress_bar)


def run_corpus_app(
    monkeypatch: pytest.MonkeyPatch,
    cache_base: Path,
    embedding_model: type = HashingEmbeddingModel,
    abstain_threshold: str | None = "0",
    clear_cache: bool = True,
) -> AppTest:
    """Run the demo in corpus mode against a temporary index cache."""
    monkeypatch.setenv("LECTURE_RAG_DEMO_MODE", "corpus")
    if abstain_threshold is None:
        monkeypatch.delenv("LECTURE_RAG_ABSTAIN_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("LECTURE_RAG_ABSTAIN_THRESHOLD", abstain_threshold)
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_base))
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", embedding_model)
    if clear_cache:
        streamlit.cache_resource.clear()
    return AppTest.from_file(str(REPO_ROOT / "demo.py"), default_timeout=120).run()


def test_demo_app_dir_contains_the_corpus() -> None:
    """The app directory resolves to the repository corpus."""
    assert (demo.APP_DIR / "data" / "corpus").is_dir()
    assert resolve_corpus_dir(demo.APP_DIR) == demo.APP_DIR / "data" / "corpus"


def test_corpus_mode_loads_from_a_different_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
    tmp_path: Path,
) -> None:
    """Corpus mode loads the index even when the cwd has no corpus directory."""
    monkeypatch.chdir(tmp_path)

    app = run_corpus_app(monkeypatch, corpus_cache)

    assert not app.exception
    assert not app.error
    assert any("Loaded index with" in success for success in element_values(app, ("success",)))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 1), (-3, 1), (1, 1), (5, 5), (10, 10), (11, 10), (100, 10)],
)
def test_source_count_is_clamped(value: int, expected: int) -> None:
    """Out-of-range source counts are clamped to the slider bounds."""
    assert demo.clamp_sources(value) == expected


class FakeCompletions:
    """Fake chat completions endpoint."""

    def __init__(
        self,
        answer: str | None = None,
        error: Exception | None = None,
        calls: list[dict[str, object]] | None = None,
        response: object | None = None,
    ) -> None:
        self.answer = answer
        self.error = error
        self.calls = calls
        self.response = response

    def create(self, **kwargs: object) -> object:
        """Return a fake response or raise the configured error."""
        if self.calls is not None:
            self.calls.append({"create": kwargs})
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response
        message = SimpleNamespace(content=self.answer)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeClient:
    """Fake OpenAI client with a chat completions endpoint."""

    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    answer: str | None = None,
    error: Exception | None = None,
    calls: list[dict[str, object]] | None = None,
    response: object | None = None,
) -> None:
    """Install a fake OpenAI constructor and optionally record its arguments."""

    def fake_openai(**kwargs: object) -> FakeClient:
        if calls is not None:
            calls.append(kwargs)
        return FakeClient(
            FakeCompletions(answer=answer, error=error, calls=calls, response=response)
        )

    monkeypatch.setattr("lecture_rag.generation.OpenAI", fake_openai)


def ask_with_llm(app: AppTest, key: str, provider: str | None = None) -> None:
    """Select llm mode, optionally pick a provider, enter a key, and ask a question."""
    app.selectbox(key="answer_mode").set_value("llm")
    app.run()
    if provider is not None:
        app.selectbox(key="llm_provider").set_value(provider)
        app.run()
    app.text_input(key="openai_api_key").set_value(key)
    app.text_input(key="query").set_value(ON_TOPIC_QUERY)
    app.run()


def dialog_markdown_values(app: AppTest) -> list[str]:
    """Return the markdown values rendered inside open dialogs.

    AppTest keeps dialog content outside ``app.main``. The dialog itself is
    available through ``app.get("dialog")``, and its child elements are exposed
    as the usual typed collections on each dialog block.
    """
    values: list[str] = []
    for dialog in app.get("dialog"):
        values.extend(str(element.value) for element in dialog.markdown)
    return values


def test_corpus_mode_renders_sources_as_text(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """Corpus mode hides local index inputs and renders sources as text."""
    app = run_corpus_app(monkeypatch, corpus_cache)

    assert not app.exception
    assert not any(element.key == "index_dir" for element in app.text_input)
    assert not any(element.key == "model_name" for element in app.text_input)
    assert not any(element.key == "allow_model_mismatch" for element in app.checkbox)
    assert any(demo.CORPUS_CAPTION in caption for caption in element_values(app, ("caption",)))

    app.text_input(key="query").set_value(ON_TOPIC_QUERY)
    app.run()

    assert not app.exception
    assert not app.error
    assert any("Sources" in str(element.value) for element in app.subheader)
    assert any("clustering-intro" in value for value in element_values(app, ("text",)))
    assert not any("clustering-intro" in value for value in element_values(app, ("markdown",)))


def slider_value(app: AppTest, label: str) -> float:
    """Return the value of the slider with the given label."""
    return next(float(element.value) for element in app.slider if element.label == label)


def run_local_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AppTest:
    """Run the demo in local mode against a temporary index."""
    index_dir = tmp_path / "index"
    build_index(SAMPLE_DIR, index_dir, HashingEmbeddingModel(), MODEL_NAME)
    monkeypatch.setenv("LECTURE_RAG_DEMO_MODE", "local")
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(index_dir))
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", HashingEmbeddingModel)
    streamlit.cache_resource.clear()
    return AppTest.from_file(str(REPO_ROOT / "demo.py"), default_timeout=120).run()


def test_corpus_mode_uses_hybrid_and_the_corpus_threshold(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """Corpus mode defaults to hybrid retrieval and an abstain threshold of 0.45."""
    app = run_corpus_app(monkeypatch, corpus_cache, abstain_threshold=None)

    assert not app.exception
    assert app.selectbox(key="retrieval_mode").value == "hybrid"
    assert slider_value(app, "Abstain threshold") == 0.45


def test_corpus_mode_environment_overrides_win(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """Environment settings override the corpus-mode defaults."""
    monkeypatch.setenv("LECTURE_RAG_RETRIEVAL_MODE", "bm25")
    app = run_corpus_app(monkeypatch, corpus_cache, abstain_threshold="0.25")

    assert not app.exception
    assert app.selectbox(key="retrieval_mode").value == "bm25"
    assert slider_value(app, "Abstain threshold") == 0.25


def test_local_mode_keeps_dense_and_the_local_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Local mode keeps dense retrieval, a 0.35 threshold, and no examples."""
    app = run_local_app(monkeypatch, tmp_path)

    assert not app.exception
    assert app.selectbox(key="retrieval_mode").value == "dense"
    assert slider_value(app, "Abstain threshold") == 0.35
    assert not any(element.key and element.key.startswith("example_") for element in app.button)


def test_corpus_caption_describes_the_corpus() -> None:
    """The corpus caption names the topics and the source list."""
    assert demo.CORPUS_CAPTION == (
        "Corpus: six CC BY 4.0 arXiv tutorials on transformers, autoencoders, reinforcement "
        "learning, testing neural networks, MLOps and clustering (see data/SOURCES.md). "
        "Questions outside these topics should get no answer."
    )


def test_corpus_example_buttons_set_the_query_and_search(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """Each example button has the exact label and runs a search when clicked."""
    app = run_corpus_app(monkeypatch, corpus_cache)

    assert not app.exception
    assert any("Try an example:" in caption for caption in element_values(app, ("caption",)))
    for index, question in enumerate(demo.EXAMPLE_QUESTIONS):
        assert app.button(key=f"example_{index}").label == question

    app.button(key="example_0").click().run()

    assert not app.exception
    assert app.text_input(key="query").value == demo.EXAMPLE_QUESTIONS[0]
    assert app.session_state["questions_asked"] == 1
    assert any("Sources" in str(element.value) for element in app.subheader)


def test_corpus_shows_the_welcome_dialog_once(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """The welcome dialog appears on the first corpus visit and stays dismissed."""
    app = run_corpus_app(monkeypatch, corpus_cache)

    assert not app.exception
    assert app.get("dialog")
    assert any(demo.WELCOME_MESSAGE in value for value in dialog_markdown_values(app))
    assert any(button.label == "Got it" for button in app.button)

    app.button(key="welcome_got_it").click().run()

    assert not app.exception
    assert app.session_state["welcome_seen"] is True
    assert not app.get("dialog")
    assert all(demo.WELCOME_MESSAGE not in value for value in element_values(app, ("markdown",)))

    app.run()

    assert not app.get("dialog")
    assert all(demo.WELCOME_MESSAGE not in value for value in element_values(app, ("markdown",)))

    app.text_input(key="query").set_value(ON_TOPIC_QUERY)
    app.run()

    assert not app.exception
    assert not app.get("dialog")
    assert all(demo.WELCOME_MESSAGE not in value for value in element_values(app, ("markdown",)))
    assert any("Sources" in str(element.value) for element in app.subheader)


def test_local_mode_never_shows_the_welcome_dialog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The one-time welcome dialog is only rendered in corpus mode."""
    app = run_local_app(monkeypatch, tmp_path)

    assert not app.exception
    assert not app.get("dialog")
    assert all(demo.WELCOME_MESSAGE not in value for value in element_values(app, ("markdown",)))

    app.text_input(key="query").set_value(ON_TOPIC_QUERY)
    app.run()

    assert not app.exception
    assert not app.get("dialog")
    assert all(demo.WELCOME_MESSAGE not in value for value in element_values(app, ("markdown",)))


def test_session_cap_stops_searching(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """After the per-session question cap no further search is rendered."""
    app = run_corpus_app(monkeypatch, corpus_cache)
    app.text_input(key="query").set_value(ON_TOPIC_QUERY)

    for _ in range(demo.MAX_QUESTIONS_PER_SESSION):
        app.run()

    assert not app.exception
    assert app.session_state["questions_asked"] == demo.MAX_QUESTIONS_PER_SESSION

    app.run()

    assert not app.exception
    assert any(
        demo.SESSION_LIMIT_MESSAGE in warning for warning in element_values(app, ("warning",))
    )
    assert not any("Sources" in str(element.value) for element in app.subheader)


def test_query_input_is_capped_at_the_character_limit(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """The query input caps questions at MAX_QUERY_CHARS characters."""
    app = run_corpus_app(monkeypatch, corpus_cache)
    query_input = app.text_input(key="query")
    assert query_input.max_chars == demo.MAX_QUERY_CHARS == 500

    query_input.set_value("x" * (demo.MAX_QUERY_CHARS + 1))
    app.run()

    assert not app.exception
    assert len(app.text_input(key="query").value) == demo.MAX_QUERY_CHARS


class FakeReranker:
    """Reranker that records construction."""

    def __init__(self, model_name: str, instances: list[str]) -> None:
        instances.append(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return deterministic scores for the document texts."""
        return [float(index) for index in range(len(texts))]


def test_default_corpus_mode_disables_rerank(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """Corpus mode hides the checkbox and never constructs a reranker by default."""
    instances: list[str] = []

    def fake_reranker(model_name: str) -> FakeReranker:
        return FakeReranker(model_name, instances)

    monkeypatch.setattr("lecture_rag.rerank.CrossEncoderReranker", fake_reranker)
    app = run_corpus_app(monkeypatch, corpus_cache)

    assert not app.exception
    assert not any(element.key == "rerank" for element in app.checkbox)
    assert any(
        demo.RERANK_DISABLED_MESSAGE in caption for caption in element_values(app, ("caption",))
    )

    app.text_input(key="query").set_value(ON_TOPIC_QUERY)
    app.run()

    assert not app.exception
    assert instances == []


@pytest.mark.parametrize("value", ["0", ""])
def test_corpus_rerank_stays_disabled_for_other_values(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
    value: str,
) -> None:
    """Only explicit truthy values enable reranking in corpus mode."""
    monkeypatch.setenv("LECTURE_RAG_DEMO_ENABLE_RERANK", value)
    app = run_corpus_app(monkeypatch, corpus_cache)

    assert not app.exception
    assert not any(element.key == "rerank" for element in app.checkbox)
    assert any(
        demo.RERANK_DISABLED_MESSAGE in caption for caption in element_values(app, ("caption",))
    )


def test_corpus_rerank_is_enabled_with_the_environment_flag(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """The environment flag shows the checkbox and loads the reranker on demand."""
    instances: list[str] = []

    def fake_reranker(model_name: str) -> FakeReranker:
        return FakeReranker(model_name, instances)

    monkeypatch.setattr("lecture_rag.rerank.CrossEncoderReranker", fake_reranker)
    monkeypatch.setenv("LECTURE_RAG_DEMO_ENABLE_RERANK", "1")
    app = run_corpus_app(monkeypatch, corpus_cache)

    assert not app.exception
    rerank = app.checkbox(key="rerank")
    assert rerank.value is False
    assert not any(
        demo.RERANK_DISABLED_MESSAGE in caption for caption in element_values(app, ("caption",))
    )

    rerank.set_value(True)
    app.text_input(key="query").set_value(ON_TOPIC_QUERY)
    app.run()

    assert not app.exception
    assert instances


def test_corpus_index_is_reused_on_second_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A second run loads the existing index instead of rebuilding it."""
    calls: list[tuple[object, ...]] = []
    real_build = corpus_index.build_and_install_index

    def counting_build(*args: object, **kwargs: object) -> Path:
        calls.append(args)
        return real_build(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(corpus_index, "build_and_install_index", counting_build)

    first = run_corpus_app(monkeypatch, tmp_path)
    assert not first.exception
    assert len(calls) == 1

    second = run_corpus_app(monkeypatch, tmp_path)
    assert not second.exception
    assert len(calls) == 1


def test_corpus_answers_after_lecture_rag_is_reimported(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """Cached resources from old classes are replaced after a watcher reimport."""
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "lecture_rag" or name.startswith("lecture_rag.")
    }
    for name, module in original_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    try:
        first = run_corpus_app(monkeypatch, corpus_cache)
        first.session_state["welcome_seen"] = True
        first.text_input(key="query").set_value(ON_TOPIC_QUERY)
        first.run()

        assert not first.exception
        assert not first.error
        assert any("Sources" in str(element.value) for element in first.subheader)

        for name in list(sys.modules):
            if name == "lecture_rag" or name.startswith("lecture_rag."):
                del sys.modules[name]

        import sentence_transformers

        monkeypatch.setattr(
            sentence_transformers,
            "SentenceTransformer",
            FakeSentenceTransformer,
        )

        second = AppTest.from_file(str(REPO_ROOT / "demo.py"), default_timeout=120).run()
        second.session_state["welcome_seen"] = True
        second.text_input(key="query").set_value(ON_TOPIC_QUERY)
        second.run()

        assert not second.exception
        assert not second.error
        assert any("Sources" in str(element.value) for element in second.subheader)
    finally:
        for name in list(sys.modules):
            if name == "lecture_rag" or name.startswith("lecture_rag."):
                del sys.modules[name]


def test_corpus_build_failure_hides_filesystem_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Build failures show a fixed message and log the details server-side."""
    secret_path = "/secret/path/data/corpus"
    detail = f"Directory not found: {secret_path}"

    def failing_build(*args: object, **kwargs: object) -> Path:
        raise OSError(detail)

    monkeypatch.setattr(corpus_index, "build_and_install_index", failing_build)

    with caplog.at_level(logging.ERROR):
        app = run_corpus_app(monkeypatch, tmp_path)

    assert not app.exception
    assert any(
        demo.CORPUS_BUILD_ERROR_MESSAGE in error for error in element_values(app, ("error",))
    )

    values = all_output_values(app)
    assert all(secret_path not in value for value in values)
    assert all("Directory not found" not in value for value in values)

    codes = element_values(app, ("code",))
    assert all(secret_path not in code for code in codes)
    assert all("Directory not found" not in code for code in codes)

    assert secret_path in caplog.text
    assert "Directory not found" in caplog.text


class FailingEmbeddingModel:
    """Embedding model that fails to load with a secret-bearing message."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise EmbeddingModelError("model load failed for /secret/model")


def test_corpus_embedding_model_error_hides_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Embedding model failures show a fixed message and log the details."""
    detail = "model load failed for /secret/model"

    with caplog.at_level(logging.ERROR):
        app = run_corpus_app(monkeypatch, tmp_path, embedding_model=FailingEmbeddingModel)

    assert not app.exception
    assert any(
        "The embedding model could not be loaded. Please try again later." in error
        for error in element_values(app, ("error",))
    )
    values = all_output_values(app)
    assert all(detail not in value for value in values)
    assert detail in caplog.text


def test_corpus_invalid_index_error_hides_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invalid corpus indexes show a fixed message and log the details."""
    detail = "Invalid index at /secret/x"

    def failing_build(*args: object, **kwargs: object) -> Path:
        raise InvalidIndexError(detail)

    monkeypatch.setattr(corpus_index, "build_and_install_index", failing_build)

    with caplog.at_level(logging.ERROR):
        app = run_corpus_app(monkeypatch, tmp_path)

    assert not app.exception
    assert any(
        demo.CORPUS_INVALID_INDEX_MESSAGE in error for error in element_values(app, ("error",))
    )
    values = all_output_values(app)
    assert all(detail not in value for value in values)
    assert detail in caplog.text


@pytest.mark.parametrize("provider_name", list(demo.LLM_PROVIDERS))
def test_corpus_llm_error_never_exposes_key(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    provider_name: str,
) -> None:
    """A failing LLM request shows safe details and never renders or logs the key."""
    error = status_error(
        RateLimitError,
        429,
        f"quota exhausted for {API_KEY} at https://api.groq.com/openai/v1",
    )
    install_fake_openai(monkeypatch, error=error)

    class ExplodingSecrets:
        """Secrets object that fails if the app reads it."""

        def __getitem__(self, name: str) -> object:
            raise AssertionError("st.secrets must not be read")

        def get(self, name: str, default: object = None) -> object:
            raise AssertionError("st.secrets must not be read")

        def __contains__(self, name: object) -> bool:
            raise AssertionError("st.secrets must not be read")

        def __getattr__(self, name: str) -> object:
            raise AssertionError("st.secrets must not be read")

    monkeypatch.setattr(streamlit, "secrets", ExplodingSecrets())

    with caplog.at_level(logging.WARNING):
        app = run_corpus_app(monkeypatch, corpus_cache)
        ask_with_llm(app, API_KEY, provider=provider_name)

    provider = demo.LLM_PROVIDERS[provider_name]
    assert not app.exception
    assert any(
        f"The {provider.name} request failed. Check the key, try another provider, "
        "or switch to extractive mode." in error
        for error in element_values(app, ("error",))
    )
    assert any("RateLimitError (HTTP 429)" in code for code in element_values(app, ("code",)))
    assert any("rate limit" in code.lower() for code in element_values(app, ("code",)))

    values = all_output_values(app)
    assert all(API_KEY not in value for value in values)
    assert all(API_KEY[-8:] not in value for value in values)
    assert all("api.groq.com" not in value for value in values)

    assert API_KEY not in caplog.text
    assert API_KEY[-8:] not in caplog.text

    captured = capsys.readouterr()
    assert API_KEY not in captured.out
    assert API_KEY not in captured.err
    assert API_KEY[-8:] not in captured.out
    assert API_KEY[-8:] not in captured.err

    assert API_KEY not in os.environ.values()


def test_llm_providers_are_the_fixed_read_only_allowlist() -> None:
    """Only the three fixed providers are available and the mapping is read-only."""
    assert list(demo.LLM_PROVIDERS) == ["OpenAI", "Groq", "OpenRouter"]
    assert demo.DEFAULT_LLM_PROVIDER == "OpenAI"
    assert {provider.base_url for provider in demo.LLM_PROVIDERS.values()} == {
        "https://api.openai.com/v1",
        "https://api.groq.com/openai/v1",
        "https://openrouter.ai/api/v1",
    }
    assert all(provider.base_url.startswith("https://") for provider in demo.LLM_PROVIDERS.values())

    with pytest.raises(TypeError):
        demo.LLM_PROVIDERS["OpenAI"] = demo.LLM_PROVIDERS["Groq"]  # type: ignore[index]


def test_corpus_llm_mode_has_no_url_or_model_text_input(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """Corpus llm mode never offers a free-form URL or model input."""
    app = run_corpus_app(monkeypatch, corpus_cache)
    app.selectbox(key="answer_mode").set_value("llm")
    app.run()

    assert not app.exception
    assert {element.key for element in app.text_input} == {"query", "openai_api_key"}
    assert not any(
        "url" in element.label.lower() or "model" in element.label.lower()
        for element in app.text_input
    )


@pytest.mark.parametrize("provider_name", list(demo.LLM_PROVIDERS))
def test_corpus_llm_success_uses_the_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
    provider_name: str,
) -> None:
    """Each provider gets its fixed endpoint, model, timeout, and key label."""
    calls: list[dict[str, object]] = []
    install_fake_openai(
        monkeypatch,
        answer="DBSCAN identifies core points with a neighbourhood count.",
        calls=calls,
    )
    app = run_corpus_app(monkeypatch, corpus_cache)
    ask_with_llm(app, API_KEY, provider=provider_name)
    provider = demo.LLM_PROVIDERS[provider_name]

    assert not app.exception
    assert not app.error
    assert any("DBSCAN identifies core points" in value for value in element_values(app, ("text",)))
    assert app.selectbox(key="llm_provider").value == provider_name
    assert app.text_input(key="openai_api_key").label.startswith(provider.key_label)
    assert calls
    assert calls[0]["api_key"] == API_KEY
    assert calls[0]["base_url"] == provider.base_url
    assert calls[0]["timeout"] == 30.0
    create_calls = [call["create"] for call in calls if "create" in call]
    assert create_calls
    assert all(isinstance(call, dict) for call in create_calls)
    assert {call["model"] for call in create_calls if isinstance(call, dict)} == {provider.model}
    assert all(call["max_tokens"] == 1024 for call in create_calls if isinstance(call, dict))


def test_describe_llm_error_reports_only_safe_details() -> None:
    """Error descriptions use only types, status codes, and safe hints."""
    rate_limit = status_error(RateLimitError, 429, f"quota for {API_KEY} at {groq_request().url}")
    authentication = status_error(AuthenticationError, 401, f"bad key {API_KEY}")
    server = status_error(APIStatusError, 500, f"server failed at {groq_request().url}")
    timeout = APITimeoutError(request=groq_request())
    connection = APIConnectionError(request=groq_request())

    try:
        try:
            raise rate_limit
        except RateLimitError as e:
            raise ValueError(f"wrapped {API_KEY}") from e
    except ValueError as e:
        wrapped = e

    expected_rate_limit = (
        "RateLimitError (HTTP 429) - rate limit or free-tier quota reached, "
        "try again later or use another provider"
    )
    assert demo.describe_llm_error(rate_limit) == expected_rate_limit
    assert demo.describe_llm_error(wrapped) == expected_rate_limit
    assert (
        demo.describe_llm_error(authentication)
        == "AuthenticationError (HTTP 401) - the key was rejected"
    )
    assert demo.describe_llm_error(server) == "APIStatusError (HTTP 500)"
    assert demo.describe_llm_error(timeout) == "APITimeoutError (no response within 30 s)"
    assert demo.describe_llm_error(connection) == "APIConnectionError"
    assert demo.describe_llm_error(ValueError(f"bad {API_KEY}")) == "ValueError"

    descriptions = [
        demo.describe_llm_error(error)
        for error in (rate_limit, wrapped, authentication, server, timeout, connection)
    ]
    assert all("api.groq.com" not in description for description in descriptions)
    assert all(API_KEY not in description for description in descriptions)
    assert all(API_KEY[-8:] not in description for description in descriptions)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            SimpleNamespace(
                id="chatcmpl-groq",
                model="openai/gpt-oss-120b",
                choices=[
                    SimpleNamespace(
                        index=0,
                        message=SimpleNamespace(
                            role="assistant",
                            content="Groq gpt-oss answer.",
                            reasoning="hidden reasoning",
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=7,
                    total_tokens=18,
                    queue_time=0.1,
                    prompt_time=0.2,
                    completion_time=0.3,
                ),
            ),
            "Groq gpt-oss answer.",
        ),
        (
            SimpleNamespace(
                id="gen-openrouter",
                model="nvidia/nemotron-3-ultra-550b-a55b:free",
                provider="OpenRouter",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            role="assistant",
                            content="OpenRouter answer.",
                            reasoning=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
            ),
            "OpenRouter answer.",
        ),
        (
            SimpleNamespace(
                id="chatcmpl-openai",
                model="gpt-4o-mini",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            role="assistant",
                            content="OpenAI answer.",
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=8,
                    completion_tokens=4,
                    total_tokens=12,
                ),
            ),
            "OpenAI answer.",
        ),
    ],
    ids=["groq-gpt-oss", "openrouter", "openai"],
)
def test_provider_reads_recorded_response_shapes(
    monkeypatch: pytest.MonkeyPatch,
    response: SimpleNamespace,
    expected: str,
) -> None:
    """Recorded provider responses all yield their content as the answer text."""
    calls: list[dict[str, object]] = []
    install_fake_openai(monkeypatch, response=response, calls=calls)
    provider = OpenAICompatibleProvider(api_key=API_KEY, base_url="https://example.test/v1")

    answer = provider.generate("How does DBSCAN work?", [sample_source()])

    assert answer == expected
    create_calls = [call["create"] for call in calls if "create" in call]
    assert create_calls
    assert create_calls[0]["max_tokens"] == 1024
    assert create_calls[0]["temperature"] == 0.3


@pytest.mark.parametrize("content", [None, "", "   \n\t"])
def test_provider_rejects_missing_text_content(
    monkeypatch: pytest.MonkeyPatch,
    content: str | None,
) -> None:
    """A response without text raises instead of returning a placeholder."""
    calls: list[dict[str, object]] = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    install_fake_openai(monkeypatch, response=response, calls=calls)
    provider = OpenAICompatibleProvider(api_key=API_KEY, base_url="https://example.test/v1")

    with pytest.raises(ValueError, match="The model returned no text."):
        provider.generate("How does DBSCAN work?", [sample_source()])

    create_calls = [call["create"] for call in calls if "create" in call]
    assert create_calls
    assert create_calls[0]["max_tokens"] == 1024


def test_corpus_demo_shows_provider_failure_for_missing_text(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """A provider response without text shows the provider failure message."""
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
    install_fake_openai(monkeypatch, response=response)
    app = run_corpus_app(monkeypatch, corpus_cache)
    ask_with_llm(app, API_KEY, provider="Groq")

    assert not app.exception
    assert any(
        "The Groq request failed. Check the key, try another provider, "
        "or switch to extractive mode." in error
        for error in element_values(app, ("error",))
    )
    assert any("ValueError" in code for code in element_values(app, ("code",)))
    assert all(API_KEY not in value for value in all_output_values(app))


def test_corpus_unknown_provider_falls_back_to_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown provider in session state falls back to the default provider."""
    calls: list[dict[str, object]] = []
    install_fake_openai(
        monkeypatch,
        answer="DBSCAN identifies core points with a neighbourhood count.",
        calls=calls,
    )
    code = f"""
import streamlit as st

import demo
from lecture_rag.models import Document, RetrievalResult

source = RetrievalResult(
    document=Document(
        text="DBSCAN uses core points.",
        file_name="clustering.pdf",
        page_number=1,
        chunk_id="clustering.pdf_page_1",
    ),
    score=1.0,
)
st.session_state["llm_provider"] = "Unknown"
st.session_state["openai_api_key"] = {API_KEY!r}
answer = demo._generate_corpus_answer("How does DBSCAN work?", [source], "llm")
st.write(answer.answer_text)
"""
    app = AppTest.from_string(code).run()

    assert not app.exception
    assert not app.error
    assert calls
    assert calls[0]["base_url"] == demo.LLM_PROVIDERS[demo.DEFAULT_LLM_PROVIDER].base_url
    assert calls[0]["timeout"] == 30.0
    create_calls = [call["create"] for call in calls if "create" in call]
    assert {call["model"] for call in create_calls if isinstance(call, dict)} == {
        demo.LLM_PROVIDERS[demo.DEFAULT_LLM_PROVIDER].model
    }


def test_corpus_llm_without_key_falls_back_to_extractive(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """Without a key the corpus demo uses extractive answers and no client."""
    monkeypatch.setattr(
        "lecture_rag.generation.OpenAI",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("client must not be created")),
    )
    app = run_corpus_app(monkeypatch, corpus_cache)
    app.selectbox(key="answer_mode").set_value("llm")
    app.run()
    app.text_input(key="query").set_value(ON_TOPIC_QUERY)
    app.run()

    assert not app.exception
    assert any(demo.LLM_FALLBACK_MESSAGE in info for info in element_values(app, ("info",)))
    assert any("[Extractive mode" in value for value in element_values(app, ("text",)))
    assert not any("request failed" in error for error in element_values(app, ("error",)))


def test_redact_secrets_replaces_known_secrets() -> None:
    """Known secrets are replaced wherever they occur."""
    text = f"request failed for {API_KEY}"

    redacted = redact_secrets(text, [API_KEY, ""])

    assert API_KEY not in redacted
    assert "[redacted]" in redacted


def test_redact_secrets_replaces_secret_like_tokens() -> None:
    """Secret-like tokens are redacted even when they are not listed."""
    text = "keys sk-other1234 and sk-abc"

    redacted = redact_secrets(text, [])

    assert "sk-other1234" not in redacted
    assert "sk-abc" in redacted
    assert "[redacted]" in redacted


def test_provider_redacts_key_in_logs_and_errors(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Provider failures redact the key from logs, errors, and repr."""
    install_fake_openai(
        monkeypatch,
        error=RuntimeError(f"bad key {API_KEY} and sk-other1234"),
    )
    provider = OpenAICompatibleProvider(
        api_key=API_KEY,
        base_url="https://api.openai.com/v1",
    )
    source = RetrievalResult(
        document=Document(
            text="DBSCAN uses core points.",
            file_name="clustering.pdf",
            page_number=1,
            chunk_id="clustering.pdf_page_1",
        ),
        score=1.0,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError) as excinfo:
            provider.generate("How does DBSCAN work?", [source])

    message = str(excinfo.value)
    assert API_KEY not in message
    assert "sk-other1234" not in message
    assert "[redacted]" in message
    assert API_KEY not in caplog.text
    assert API_KEY not in repr(provider)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("x" * 500, False),
        ("x" * 501, True),
        ("", False),
    ],
)
def test_is_query_too_long(query: str, expected: bool) -> None:
    """The server-side length check matches the maximum query length."""
    assert demo.is_query_too_long(query) is expected


def test_build_index_on_sample_lectures(tmp_path: Path) -> None:
    """The shared build helper writes a loadable index for the sample PDFs."""
    index_dir = tmp_path / "index"

    count = build_index(
        SAMPLE_DIR,
        index_dir,
        HashingEmbeddingModel(),
        MODEL_NAME,
    )

    assert count > 0
    index = VectorIndex.load(index_dir)
    assert len(index.documents) == count
    assert index.model_name == MODEL_NAME
    assert index.dimension == HashingEmbeddingModel.dimension


@pytest.mark.parametrize(("value", "expected"), [("corpus", "corpus"), ("local", "local")])
def test_demo_mode_uses_valid_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: str,
) -> None:
    """Valid LECTURE_RAG_DEMO_MODE values select their mode."""
    monkeypatch.setenv("LECTURE_RAG_DEMO_MODE", value)

    assert demo.demo_mode() == expected


def test_demo_mode_reports_invalid_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid mode shows a fixed error and falls back to local."""
    errors: list[str] = []
    monkeypatch.setenv("LECTURE_RAG_DEMO_MODE", "bogus")
    monkeypatch.setattr(demo.st, "error", lambda message: errors.append(message))

    assert demo.demo_mode() == "local"
    assert errors
    assert "LECTURE_RAG_DEMO_MODE" in errors[0]


def test_demo_mode_defaults_to_corpus_without_a_local_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Corpus mode is the default when no local index file exists."""
    monkeypatch.delenv("LECTURE_RAG_DEMO_MODE", raising=False)
    monkeypatch.delenv("LECTURE_RAG_INDEX_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert demo.demo_mode() == "corpus"

    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "index.faiss").write_bytes(b"not a real index")

    assert demo.demo_mode() == "local"


def test_demo_mode_defaults_to_local_with_an_index_dir_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Setting LECTURE_RAG_INDEX_DIR selects local mode."""
    monkeypatch.delenv("LECTURE_RAG_DEMO_MODE", raising=False)
    monkeypatch.setenv("LECTURE_RAG_INDEX_DIR", str(tmp_path / "missing"))
    monkeypatch.chdir(tmp_path)

    assert demo.demo_mode() == "local"
