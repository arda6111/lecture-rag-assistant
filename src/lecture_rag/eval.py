"""Evaluation harness for retrieval quality."""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .retrieval import RetrievalMode, Retriever

logger = logging.getLogger(__name__)


class EvalQuestion(BaseModel):
    """An evaluation question with an optional expected source."""

    question: str
    expected_file: str | None = None
    expected_page: int | None = None

    @property
    def answerable(self) -> bool:
        """Whether the question has a known answer in the corpus."""
        return self.expected_file is not None and self.expected_page is not None


class EvalSet(BaseModel):
    """A set of evaluation questions."""

    questions: list[EvalQuestion]


def load_eval_set(path: Path) -> EvalSet:
    """Load an evaluation set from JSON.

    Args:
        path: Path to the JSON file

    Returns:
        Loaded EvalSet

    Raises:
        ValueError: If the file is not valid JSON or does not match the schema
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return EvalSet.model_validate(data)
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as e:
        raise ValueError(f"Invalid eval file {path}: {e}") from e


def evaluate_retrieval(
    retriever: Retriever,
    eval_set: EvalSet,
    k: int = 5,
    abstain_threshold: float | None = None,
    mode: RetrievalMode | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval quality and, optionally, abstention decisions.

    Hit rate and MRR are computed over answerable questions only. Mean score
    keeps its original behaviour of averaging every retrieved result score.

    Args:
        retriever: Retriever to evaluate
        eval_set: Evaluation questions
        k: Number of documents to retrieve
        abstain_threshold: Optional dense similarity threshold for abstention
        mode: Optional retrieval mode that overrides the retriever mode

    Returns:
        Dictionary of metrics
    """
    if not eval_set.questions:
        raise ValueError("Eval set is empty")

    total = len(eval_set.questions)
    answerable_total = sum(1 for question in eval_set.questions if question.answerable)
    unanswerable_total = total - answerable_total

    hits = 0
    reciprocal_ranks: list[float] = []
    scores: list[float] = []
    answerable_hits_after_abstain = 0

    abstain_true_positives = 0
    abstain_false_positives = 0
    abstain_false_negatives = 0
    abstain_true_negatives = 0

    for question in eval_set.questions:
        if mode is None:
            results = retriever.retrieve(question.question, k=k)
        else:
            results = retriever.retrieve(question.question, k=k, mode=mode)

        found_rank = None
        for rank, result in enumerate(results, 1):
            doc = result.document
            scores.append(result.score)

            if (
                question.answerable
                and doc.file_name == question.expected_file
                and doc.page_number == question.expected_page
            ):
                if found_rank is None:
                    found_rank = rank

        abstained = False
        if abstain_threshold is not None:
            abstained = retriever.should_abstain(question.question, abstain_threshold)
            if question.answerable:
                if abstained:
                    abstain_false_positives += 1
                else:
                    abstain_true_negatives += 1
            else:
                if abstained:
                    abstain_true_positives += 1
                else:
                    abstain_false_negatives += 1

        if question.answerable:
            if found_rank:
                hits += 1
                reciprocal_ranks.append(1.0 / found_rank)
                if abstain_threshold is not None and not abstained:
                    answerable_hits_after_abstain += 1
            else:
                reciprocal_ranks.append(0.0)

    hit_rate = hits / answerable_total if answerable_total > 0 else 0.0
    mrr = (
        sum(reciprocal_ranks) / answerable_total
        if answerable_total > 0
        else 0.0
    )
    mean_score = sum(scores) / len(scores) if scores else 0.0

    metrics: dict[str, Any] = {
        "total_questions": total,
        "answerable_questions": answerable_total,
        "unanswerable_questions": unanswerable_total,
        f"hit_at_{k}": hit_rate,
        f"mrr_at_{k}": mrr,
        "mean_score": mean_score,
    }

    if abstain_threshold is not None:
        abstain_total = abstain_true_positives + abstain_false_positives
        unanswerable_abstained_total = (
            abstain_true_positives + abstain_false_negatives
        )

        metrics["abstain_accuracy"] = (
            (abstain_true_positives + abstain_true_negatives) / total
            if total > 0
            else 0.0
        )
        metrics["abstain_precision"] = (
            abstain_true_positives / abstain_total if abstain_total > 0 else 0.0
        )
        metrics["abstain_recall"] = (
            abstain_true_positives / unanswerable_abstained_total
            if unanswerable_abstained_total > 0
            else 0.0
        )
        metrics[f"answerable_hit_at_{k}_after_abstain"] = (
            answerable_hits_after_abstain / answerable_total
            if answerable_total > 0
            else 0.0
        )

    return metrics
