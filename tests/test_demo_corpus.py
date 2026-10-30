"""Offline tests for the public corpus demo and secret handling."""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import demo  # noqa: E402
from lecture_rag import corpus_index  # noqa: E402
from lecture_rag.corpus_index import (  # noqa: E402
    build_index,
    build_index_atomically,
    resolve_corpus_dir,
)
from lecture_rag.generation import (  # noqa: E402
    OpenAICompatibleProvider,
    redact_secrets,
)
from lecture_rag.index import VectorIndex  # noqa: E402
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
    build_index_atomically(
        CORPUS_DIR,
        cache_dir,
        HashingEmbeddingModel(),
        MODEL_NAME,
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


def run_corpus_app(
    monkeypatch: pytest.MonkeyPatch,
    cache_base: Path,
) -> AppTest:
    """Run the demo in corpus mode against a temporary index cache."""
    monkeypatch.setenv("LECTURE_RAG_DEMO_MODE", "corpus")
    monkeypatch.setenv("LECTURE_RAG_ABSTAIN_THRESHOLD", "0")
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_base))
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", HashingEmbeddingModel)
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
    [(-5, 1), (0, 1), (5, 5), (demo.MAX_SOURCES + 1, demo.MAX_SOURCES)],
)
def test_source_count_is_clamped(value: int, expected: int) -> None:
    """Out-of-range source counts are clamped to the slider bounds."""
    assert max(1, min(value, demo.MAX_SOURCES)) == expected


class FakeCompletions:
    """Fake chat completions endpoint."""

    def __init__(
        self,
        answer: str | None = None,
        error: Exception | None = None,
        calls: list[dict[str, object]] | None = None,
    ) -> None:
        self.answer = answer
        self.error = error
        self.calls = calls

    def create(self, **kwargs: object) -> object:
        """Return a fake response or raise the configured error."""
        if self.calls is not None:
            self.calls.append({"create": kwargs})
        if self.error is not None:
            raise self.error
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
) -> None:
    """Install a fake OpenAI constructor and optionally record its arguments."""

    def fake_openai(**kwargs: object) -> FakeClient:
        if calls is not None:
            calls.append(kwargs)
        return FakeClient(FakeCompletions(answer=answer, error=error, calls=calls))

    monkeypatch.setattr("lecture_rag.generation.OpenAI", fake_openai)


def ask_with_llm(app: AppTest, key: str) -> None:
    """Select llm mode, enter a key, and ask an on-topic question."""
    app.selectbox(key="answer_mode").set_value("llm")
    app.run()
    app.text_input(key="openai_api_key").set_value(key)
    app.text_input(key="query").set_value(ON_TOPIC_QUERY)
    app.run()


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
    assert any("data/corpus" in caption for caption in element_values(app, ("caption",)))

    app.text_input(key="query").set_value(ON_TOPIC_QUERY)
    app.run()

    assert not app.exception
    assert not app.error
    assert any("Sources" in str(element.value) for element in app.subheader)
    assert any("clustering-intro" in value for value in element_values(app, ("text",)))
    assert not any("clustering-intro" in value for value in element_values(app, ("markdown",)))


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
    real_build = corpus_index.build_index_atomically

    def counting_build(*args: object, **kwargs: object) -> Path:
        calls.append(args)
        return real_build(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(corpus_index, "build_index_atomically", counting_build)

    first = run_corpus_app(monkeypatch, tmp_path)
    assert not first.exception
    assert len(calls) == 1

    second = run_corpus_app(monkeypatch, tmp_path)
    assert not second.exception
    assert len(calls) == 1


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

    monkeypatch.setattr(corpus_index, "build_index_atomically", failing_build)

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


def test_corpus_llm_error_never_exposes_key(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failing LLM request never renders or logs the API key."""
    install_fake_openai(
        monkeypatch,
        error=RuntimeError(f"request failed for {API_KEY}"),
    )

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

    with caplog.at_level(logging.ERROR):
        app = run_corpus_app(monkeypatch, corpus_cache)
        ask_with_llm(app, API_KEY)

    assert not app.exception
    assert any(demo.LLM_ERROR_MESSAGE in error for error in element_values(app, ("error",)))
    assert any("ValueError" in code for code in element_values(app, ("code",)))

    values = all_output_values(app)
    assert all(API_KEY not in value for value in values)
    assert all(API_KEY[-8:] not in value for value in values)

    assert API_KEY not in caplog.text
    assert API_KEY[-8:] not in caplog.text

    captured = capsys.readouterr()
    assert API_KEY not in captured.out
    assert API_KEY not in captured.err
    assert API_KEY[-8:] not in captured.out
    assert API_KEY[-8:] not in captured.err

    assert API_KEY not in os.environ.values()


def test_corpus_llm_success_uses_fixed_endpoint_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    corpus_cache: Path,
) -> None:
    """A successful LLM answer is rendered as text and uses the fixed endpoint."""
    calls: list[dict[str, object]] = []
    install_fake_openai(
        monkeypatch,
        answer="DBSCAN identifies core points with a neighbourhood count.",
        calls=calls,
    )
    app = run_corpus_app(monkeypatch, corpus_cache)
    ask_with_llm(app, API_KEY)

    assert not app.exception
    assert not app.error
    assert any("DBSCAN identifies core points" in value for value in element_values(app, ("text",)))
    assert calls
    assert calls[0]["api_key"] == API_KEY
    assert calls[0]["base_url"] == demo.DEMO_LLM_BASE_URL
    assert calls[0]["timeout"] == 30.0
    create_calls = [call["create"] for call in calls if "create" in call]
    assert create_calls
    assert all(isinstance(call, dict) for call in create_calls)
    assert {call["model"] for call in create_calls if isinstance(call, dict)} == {
        demo.DEMO_LLM_MODEL
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
