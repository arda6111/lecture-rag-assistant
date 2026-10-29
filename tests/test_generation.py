"""Tests for answer generation functionality."""

import pytest

from lecture_rag.generation import ExtractiveFallbackProvider, Generator
from lecture_rag.models import Document, RetrievalResult


@pytest.fixture
def sample_retrieval_results():
    """Create sample retrieval results."""
    return [
        RetrievalResult(
            document=Document(
                text="Binary trees are hierarchical structures.",
                file_name="test.pdf",
                page_number=1,
                chunk_id="test_1",
            ),
            score=0.95,
        ),
        RetrievalResult(
            document=Document(
                text="Hash tables provide fast lookup.",
                file_name="test.pdf",
                page_number=2,
                chunk_id="test_2",
            ),
            score=0.80,
        ),
    ]


def test_extractive_fallback_provider(sample_retrieval_results) -> None:
    """Test extractive fallback provider."""
    provider = ExtractiveFallbackProvider()

    answer = provider.generate("What are binary trees?", sample_retrieval_results)

    assert "Binary trees" in answer
    assert "test.pdf" in answer
    assert "page 1" in answer
    assert "0.950" in answer


def test_extractive_fallback_empty_sources() -> None:
    """Test extractive fallback with no sources."""
    provider = ExtractiveFallbackProvider()

    answer = provider.generate("What are binary trees?", [])

    assert "No relevant sources" in answer


def test_generator_extractive_mode(sample_retrieval_results) -> None:
    """Test generator in extractive mode."""
    generator = Generator.from_config(mode="extractive")

    answer = generator.generate_answer("What are binary trees?", sample_retrieval_results)

    assert answer.mode == "extractive"
    assert len(answer.sources) == 2
    assert "Binary trees" in answer.answer_text


def test_generator_preserves_sources(sample_retrieval_results) -> None:
    """Test that generator preserves source citations."""
    generator = Generator.from_config(mode="extractive")

    answer = generator.generate_answer("What are binary trees?", sample_retrieval_results)

    assert len(answer.sources) == 2
    assert answer.sources[0].document.file_name == "test.pdf"
    assert answer.sources[0].document.page_number == 1
    assert answer.sources[1].document.page_number == 2


def test_llm_mode_without_key_fails_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM mode without an API key or base URL raises instead of calling the API."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Generator.from_config(mode="llm")


def test_llm_mode_with_local_base_url_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local OpenAI-compatible server can be used without a key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

    generator = Generator.from_config(mode="llm")

    assert generator.provider is not None


def test_unknown_generation_mode_raises() -> None:
    """Unknown generation modes are rejected."""
    with pytest.raises(ValueError, match="Unknown generation mode"):
        Generator.from_config(mode="magic")
