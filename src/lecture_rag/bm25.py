"""BM25-based sparse retrieval."""

import re

from rank_bm25 import BM25Okapi

from .models import Document

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)

TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words without stopwords.

    Args:
        text: Text to tokenize

    Returns:
        List of normalized tokens
    """
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]


class BM25Index:
    """BM25 index over a collection of documents."""

    def __init__(self, documents: list[Document]) -> None:
        """Initialize the index.

        Args:
            documents: Documents to index
        """
        self.documents = list(documents)
        self._tokenized_documents = [tokenize(document.text) for document in self.documents]
        self._bm25: BM25Okapi | None = None
        if self.documents and any(self._tokenized_documents):
            self._bm25 = BM25Okapi(self._tokenized_documents)

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """Search for the k most relevant documents.

        Args:
            query: Query string
            k: Maximum number of results

        Returns:
            List of (document, score) tuples with positive scores, sorted descending
        """
        if self._bm25 is None or k <= 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(self.documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [(document, float(score)) for document, score in ranked if float(score) > 0][:k]
