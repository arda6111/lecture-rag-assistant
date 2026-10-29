"""Evaluation harness for retrieval quality."""

import json
import logging
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field, StrictInt, ValidationError, model_validator

from .retrieval import RetrievalMode, Retriever

logger = logging.getLogger(__name__)


class EvalQuestion(BaseModel):
    """An evaluation question with an optional expected source."""

    question: str
    expected_file: Annotated[str, Field(min_length=1)] | None = None
    expected_page: StrictInt | None = None
    expected_pages: list[StrictInt] | None = None

    @model_validator(mode="after")
    def _validate_expected_pages(self) -> "EvalQuestion":
        """Validate that expected file and page fields are consistent."""
        if self.expected_page is not None and self.expected_pages is not None:
            raise ValueError("expected_page and expected_pages cannot both be set")
        if self.expected_pages is not None and not self.expected_pages:
            raise ValueError("expected_pages must not be empty")

        pages = self.pages
        if any(page < 1 for page in pages):
            raise ValueError("expected pages must be greater than or equal to 1")
        if self.expected_file is not None and not pages:
            raise ValueError("expected_file requires expected_page or expected_pages")
        if self.expected_file is None and pages:
            raise ValueError("expected pages require expected_file")
        return self

    @property
    def pages(self) -> frozenset[int]:
        """Expected page numbers, or an empty set for unanswerable questions."""
        if self.expected_pages is not None:
            return frozenset(self.expected_pages)
        if self.expected_page is not None:
            return frozenset({self.expected_page})
        return frozenset()

    @property
    def answerable(self) -> bool:
        """Whether the question has a known answer in the corpus."""
        return self.expected_file is not None and bool(self.pages)


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
    rerank: bool = False,
) -> dict[str, Any]:
    """Evaluate retrieval quality and, optionally, abstention decisions.

    Hit rate and MRR are computed over answerable questions only. Mean score
    keeps its original behaviour of averaging every retrieved result score.
    Abstention always uses the first-stage dense cosine similarity, also when
    reranking is enabled.

    Args:
        retriever: Retriever to evaluate
        eval_set: Evaluation questions
        k: Number of documents to retrieve
        abstain_threshold: Optional dense similarity threshold for abstention
        mode: Optional retrieval mode that overrides the retriever mode
        rerank: Whether to rerank the first-stage candidates

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
        if rerank:
            outcome = retriever.search(question.question, k=k, mode=mode, rerank=True)
        else:
            outcome = retriever.search(question.question, k=k, mode=mode)
        results = outcome.results

        found_rank = None
        for rank, result in enumerate(results, 1):
            doc = result.document
            scores.append(result.score)

            if (
                question.answerable
                and doc.file_name == question.expected_file
                and doc.page_number in question.pages
            ):
                if found_rank is None:
                    found_rank = rank

        abstained = False
        if abstain_threshold is not None:
            abstained = outcome.should_abstain(abstain_threshold)
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
    mrr = sum(reciprocal_ranks) / answerable_total if answerable_total > 0 else 0.0
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
        unanswerable_abstained_total = abstain_true_positives + abstain_false_negatives

        metrics["abstain_accuracy"] = (
            (abstain_true_positives + abstain_true_negatives) / total if total > 0 else 0.0
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
            answerable_hits_after_abstain / answerable_total if answerable_total > 0 else 0.0
        )

    return metrics


EVAL_MODES: tuple[RetrievalMode, ...] = ("dense", "bm25", "hybrid")
SWEEP_THRESHOLDS: tuple[float, ...] = tuple(round(0.20 + 0.05 * step, 2) for step in range(9))
SCORE_TYPES: dict[str, str] = {
    "dense": "cosine",
    "bm25": "bm25",
    "hybrid": "rrf",
    "rerank": "cross-encoder logit",
}


def run_full_evaluation(
    retriever: Retriever,
    eval_set: EvalSet,
    *,
    eval_file: str,
    k: int,
    abstain_threshold: float,
    model_name: str,
    indexed_chunks: int,
    command: str,
    rerank: bool = False,
) -> dict[str, Any]:
    """Evaluate every retrieval mode at k and at 1, plus a threshold sweep.

    Args:
        retriever: Retriever to evaluate
        eval_set: Evaluation questions
        eval_file: Eval file path as given on the command line
        k: Number of documents to retrieve
        abstain_threshold: Dense similarity threshold for abstention
        model_name: Embedding model name
        indexed_chunks: Number of chunks in the index
        command: Command that produced the results
        rerank: Whether to rerank the first-stage candidates

    Returns:
        JSON-serializable results without timestamps or absolute paths
    """
    modes: dict[str, dict[str, Any]] = {}
    for mode in EVAL_MODES:
        at_k = evaluate_retrieval(
            retriever,
            eval_set,
            k=k,
            abstain_threshold=abstain_threshold,
            mode=mode,
            rerank=rerank,
        )
        at_1 = evaluate_retrieval(
            retriever,
            eval_set,
            k=1,
            abstain_threshold=abstain_threshold,
            mode=mode,
            rerank=rerank,
        )
        modes[mode] = {
            "score_type": SCORE_TYPES["rerank" if rerank else mode],
            "hit_at_k": at_k[f"hit_at_{k}"],
            "mrr_at_k": at_k[f"mrr_at_{k}"],
            "hit_at_1": at_1["hit_at_1"],
            "mrr_at_1": at_1["mrr_at_1"],
            "abstain_accuracy": at_k["abstain_accuracy"],
            "abstain_precision": at_k["abstain_precision"],
            "abstain_recall": at_k["abstain_recall"],
            "hit_at_k_after_abstain": at_k[f"answerable_hit_at_{k}_after_abstain"],
            "mean_score": at_k["mean_score"],
        }

    sweep = [
        {
            "threshold": threshold,
            "abstain_accuracy": evaluate_retrieval(
                retriever,
                eval_set,
                k=k,
                abstain_threshold=threshold,
                mode="dense",
                rerank=rerank,
            )["abstain_accuracy"],
        }
        for threshold in SWEEP_THRESHOLDS
    ]

    answerable = sum(1 for question in eval_set.questions if question.answerable)
    return {
        "command": command,
        "eval_file": eval_file,
        "k": k,
        "abstain_threshold": abstain_threshold,
        "embedding_model": model_name,
        "indexed_chunks": indexed_chunks,
        "questions": {
            "total": len(eval_set.questions),
            "answerable": answerable,
            "unanswerable": len(eval_set.questions) - answerable,
        },
        "modes": modes,
        "sweep": sweep,
        "reranked": rerank,
    }


def render_markdown(results: dict[str, Any]) -> str:
    """Render full evaluation results as Markdown tables.

    Args:
        results: Output of run_full_evaluation

    Returns:
        Markdown text ending with a newline
    """
    k = results["k"]
    threshold = results["abstain_threshold"]
    questions = results["questions"]
    lines = [
        "# Evaluation results",
        "",
        f"Produced with `{results['command']}`.",
        "",
        f"- Eval file: `{results['eval_file']}`",
        f"- Questions: {questions['total']} ({questions['answerable']} answerable, "
        f"{questions['unanswerable']} unanswerable)",
        f"- Embedding model: `{results['embedding_model']}`",
        f"- Indexed chunks: {results['indexed_chunks']}",
        f"- Abstain threshold: {threshold:.2f} (dense cosine similarity)",
        "",
        "Hit and MRR are computed over answerable questions; abstain accuracy over all questions.",
        "",
        "## Retrieval mode comparison",
        "",
        f"| Mode | Hit@{k} | MRR@{k} | Hit@1 | MRR@1 | Abstain acc |",
        "|------|-------|-------|-------|-------|-------------|",
    ]
    for mode, metrics in results["modes"].items():
        lines.append(
            f"| {mode} | {metrics['hit_at_k']:.3f} | {metrics['mrr_at_k']:.3f} | "
            f"{metrics['hit_at_1']:.3f} | {metrics['mrr_at_1']:.3f} | "
            f"{metrics['abstain_accuracy']:.3f} |"
        )

    dense = results["modes"]["dense"]
    lines += [
        "",
        f"## Abstention at threshold {threshold:.2f} (dense)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Abstain accuracy | {dense['abstain_accuracy']:.3f} |",
        f"| Abstain precision | {dense['abstain_precision']:.3f} |",
        f"| Abstain recall | {dense['abstain_recall']:.3f} |",
        f"| Hit@{k} after abstain | {dense['hit_at_k_after_abstain']:.3f} |",
        f"| Mean cosine score (top {k}) | {dense['mean_score']:.3f} |",
        "",
        "## Threshold sweep (dense)",
        "",
        "| Threshold | Abstain acc |",
        "|-----------|-------------|",
    ]
    for row in results["sweep"]:
        lines.append(f"| {row['threshold']:.2f} | {row['abstain_accuracy']:.3f} |")

    return "\n".join(lines) + "\n"
