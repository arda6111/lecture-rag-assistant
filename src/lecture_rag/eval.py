"""Evaluation harness for retrieval quality."""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .retrieval import Retriever

logger = logging.getLogger(__name__)


class EvalQuestion(BaseModel):
    """An evaluation question with expected source."""
    question: str
    expected_file: str
    expected_page: int


class EvalSet(BaseModel):
    """A set of evaluation questions."""
    questions: list[EvalQuestion]


def load_eval_set(path: Path) -> EvalSet:
    """Load an evaluation set from JSON.

    Args:
        path: Path to the JSON file

    Returns:
        Loaded EvalSet
    """
    with open(path) as f:
        data = json.load(f)
    return EvalSet(**data)


def evaluate_retrieval(
    retriever: Retriever,
    eval_set: EvalSet,
    k: int = 5,
) -> dict[str, Any]:
    """Evaluate retrieval quality.

    Args:
        retriever: Retriever to evaluate
        eval_set: Evaluation questions
        k: Number of documents to retrieve

    Returns:
        Dictionary of metrics
    """
    hits = 0
    reciprocal_ranks = []
    scores = []

    for question in eval_set.questions:
        results = retriever.retrieve(question.question, k=k)

        found_rank = None
        for rank, result in enumerate(results, 1):
            doc = result.document
            scores.append(result.score)

            if (
                doc.file_name == question.expected_file
                and doc.page_number == question.expected_page
            ):
                if found_rank is None:
                    found_rank = rank

        if found_rank:
            hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(eval_set.questions)
    hit_rate = hits / total if total > 0 else 0.0
    mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "total_questions": total,
        f"hit_at_{k}": hit_rate,
        f"mrr_at_{k}": mrr,
        "mean_score": mean_score,
    }
