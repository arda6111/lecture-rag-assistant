"""Answer generation with multiple provider backends."""

import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI

from .models import Answer, RetrievalResult

logger = logging.getLogger(__name__)


class GenerationProvider(ABC):
    """Abstract base class for answer generation providers."""

    @abstractmethod
    def generate(self, query: str, sources: list[RetrievalResult]) -> str:
        """Generate an answer from query and retrieved sources."""
        pass


class OpenAICompatibleProvider(GenerationProvider):
    """OpenAI-compatible API provider (OpenAI, DeepSeek, Ollama, etc.)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-3.5-turbo",
    ) -> None:
        """Initialize the provider.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
            model: Model name to use
        """
        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or None
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or None

        if not resolved_key:
            if not resolved_base_url:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Set it in .env, point OPENAI_BASE_URL "
                    "at a local OpenAI-compatible server, or use extractive mode."
                )
            # Local OpenAI-compatible servers such as Ollama do not check the key.
            resolved_key = "not-needed"

        self.model = model
        self.client = OpenAI(api_key=resolved_key, base_url=resolved_base_url)

    def generate(self, query: str, sources: list[RetrievalResult]) -> str:
        """Generate an answer using the LLM.

        Args:
            query: User query
            sources: Retrieved source documents

        Returns:
            Generated answer text
        """
        context = self._build_context(sources)

        system_prompt = (
            "You are a helpful assistant that answers questions about lecture materials. "
            "Answer ONLY based on the provided context. "
            "If the answer is not in the context, say 'I cannot find this information in the provided materials.' "
            "Include specific references to source documents when answering."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            answer = response.choices[0].message.content or "No answer generated."
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise ValueError(f"Failed to generate answer: {e}") from e

    def _build_context(self, sources: list[RetrievalResult]) -> str:
        """Build context string from sources."""
        context_parts = []
        for i, result in enumerate(sources, 1):
            doc = result.document
            context_parts.append(
                f"[Source {i}] {doc.file_name} (page {doc.page_number}):\n{doc.text}\n"
            )
        return "\n".join(context_parts)


class ExtractiveFallbackProvider(GenerationProvider):
    """Extractive fallback that returns the most relevant chunk."""

    def generate(self, query: str, sources: list[RetrievalResult]) -> str:
        """Return an extractive answer from the top source.

        Args:
            query: User query
            sources: Retrieved source documents

        Returns:
            Extractive answer text
        """
        if not sources:
            return "No relevant sources found."

        top_result = sources[0]
        doc = top_result.document

        answer = (
            f"[Extractive mode - most relevant passage]:\n\n"
            f"{doc.text[:500]}...\n\n"
            f"(Source: {doc.file_name}, page {doc.page_number}, {top_result.score_label})"
        )

        return answer


class Generator:
    """Answer generator with provider abstraction."""

    def __init__(self, provider: GenerationProvider) -> None:
        """Initialize the generator.

        Args:
            provider: Generation provider to use
        """
        self.provider = provider

    def generate_answer(self, query: str, sources: list[RetrievalResult]) -> Answer:
        """Generate an answer with citations.

        Args:
            query: User query
            sources: Retrieved source documents

        Returns:
            Answer object with text and sources
        """
        answer_text = self.provider.generate(query, sources)

        mode = "extractive" if isinstance(self.provider, ExtractiveFallbackProvider) else "llm"

        return Answer(
            answer_text=answer_text,
            sources=sources,
            mode=mode,
        )

    @classmethod
    def from_config(cls, mode: str = "llm") -> "Generator":
        """Create a generator from configuration.

        Args:
            mode: Generation mode - "llm" or "extractive"

        Returns:
            Initialized Generator instance

        Raises:
            ValueError: If the mode is unknown or llm mode has no usable configuration
        """
        if mode not in ("llm", "extractive"):
            raise ValueError(f"Unknown generation mode: {mode!r}")

        if mode == "extractive":
            provider = ExtractiveFallbackProvider()
        else:
            provider = OpenAICompatibleProvider(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            )

        return cls(provider)
