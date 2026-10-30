"""Report the corpus demo defaults and the abstain threshold trade-off.

This script needs the real embedding model and is not run in CI.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lecture_rag.chunking import chunk_documents
from lecture_rag.embeddings import EmbeddingModel
from lecture_rag.eval import EVAL_MODES, EvalQuestion, EvalSet, load_eval_set
from lecture_rag.models import RetrievalResult
from lecture_rag.pdf_loader import load_directory
from lecture_rag.retrieval import RetrievalMode, Retriever

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_corpus_eval import (  # noqa: E402
    CHUNK_OVERLAP,
    CORPUS_DIR,
    DEFAULT_CHUNK_SIZE,
    build_retriever,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = REPO_ROOT / "eval" / "corpus" / "eval_set.json"
NEGATIVES_FILE = REPO_ROOT / "eval" / "corpus" / "demo_negatives.json"
OUTPUT_PATH = REPO_ROOT / "eval" / "corpus" / "demo_defaults.md"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
K = 5
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55)

ANSWERABLE = "answerable"
UNANSWERABLE = "unanswerable"
OFF_CORPUS_ML = "off_corpus_ml"
INJECTION = "injection"
NEGATIVE_GROUPS = (OFF_CORPUS_ML, INJECTION)


@dataclass(frozen=True)
class DemoRecord:
    """One question with its dense similarity and per-mode hit result."""

    question: str
    group: str
    max_dense_similarity: float
    hits: dict[RetrievalMode, bool]


def load_negatives(path: Path = NEGATIVES_FILE) -> dict[str, list[str]]:
    """Load the demo negative questions.

    Args:
        path: Path to the demo negatives JSON file

    Returns:
        Mapping from negative group name to questions
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return {group: list(data.get(group, [])) for group in NEGATIVE_GROUPS}


def _is_hit(results: Sequence[RetrievalResult], question: EvalQuestion) -> bool:
    """Return True when a result comes from an expected file and page.

    Args:
        results: Retrieved results for one question
        question: Evaluation question with optional expected file and pages

    Returns:
        True when any result matches the expected source, otherwise False
    """
    if not question.answerable:
        return False
    return any(
        result.document.file_name == question.expected_file
        and result.document.page_number in question.pages
        for result in results
    )


def collect_eval_records(
    retriever: Retriever,
    eval_set: EvalSet,
    k: int = K,
) -> list[DemoRecord]:
    """Collect per-question records for the corpus evaluation set.

    Args:
        retriever: Retriever over the corpus
        eval_set: Evaluation questions
        k: Number of documents to retrieve

    Returns:
        One record per question with its dense cosine and per-mode hit result
    """
    records: list[DemoRecord] = []
    for question in eval_set.questions:
        dense_outcome = retriever.search(question.question, k=k, mode="dense")
        hits: dict[RetrievalMode, bool] = {}
        for mode in EVAL_MODES:
            if mode == "dense":
                outcome = dense_outcome
            else:
                outcome = retriever.search(question.question, k=k, mode=mode)
            hits[mode] = _is_hit(outcome.results, question)

        records.append(
            DemoRecord(
                question=question.question,
                group=ANSWERABLE if question.answerable else UNANSWERABLE,
                max_dense_similarity=dense_outcome.max_dense_similarity,
                hits=hits,
            )
        )
    return records


def collect_negative_records(
    retriever: Retriever,
    negatives: dict[str, list[str]],
    k: int = K,
) -> list[DemoRecord]:
    """Collect per-question records for the demo negative questions.

    Args:
        retriever: Retriever over the corpus
        negatives: Mapping from negative group name to questions
        k: Number of documents to retrieve

    Returns:
        One record per negative question with its dense cosine
    """
    records: list[DemoRecord] = []
    for group in NEGATIVE_GROUPS:
        for question in negatives.get(group, []):
            outcome = retriever.search(question, k=k, mode="dense")
            records.append(
                DemoRecord(
                    question=question,
                    group=group,
                    max_dense_similarity=outcome.max_dense_similarity,
                    hits={mode: False for mode in EVAL_MODES},
                )
            )
    return records


def answerable_count(records: Sequence[DemoRecord]) -> int:
    """Return the number of answerable questions in the records.

    Args:
        records: Per-question records

    Returns:
        Number of records in the answerable group
    """
    return sum(1 for record in records if record.group == ANSWERABLE)


def answered_count(records: Sequence[DemoRecord], threshold: float) -> int:
    """Return the number of answerable questions above the threshold.

    Args:
        records: Per-question records
        threshold: Minimum dense cosine similarity required to answer

    Returns:
        Number of answerable questions with a cosine at or above the threshold
    """
    return sum(
        1
        for record in records
        if record.group == ANSWERABLE and record.max_dense_similarity >= threshold
    )


def gated_hit_rate(
    records: Sequence[DemoRecord],
    mode: RetrievalMode,
    threshold: float,
) -> float:
    """Return Hit@5 over all answerable questions after the abstain gate.

    A question that is gated by the threshold or missed by retrieval counts as
    a miss, so the denominator is every answerable question.

    Args:
        records: Per-question records
        mode: Retrieval mode whose hit result is used
        threshold: Minimum dense cosine similarity required to answer

    Returns:
        Fraction of answerable questions that are answered and retrieved
    """
    answerable = [record for record in records if record.group == ANSWERABLE]
    if not answerable:
        return 0.0
    hits = sum(
        1
        for record in answerable
        if record.max_dense_similarity >= threshold and record.hits.get(mode, False)
    )
    return hits / len(answerable)


def abstention_rate(
    records: Sequence[DemoRecord],
    group: str,
    threshold: float,
) -> float:
    """Return the abstention rate for one group of questions.

    Args:
        records: Per-question records
        group: Group name to measure
        threshold: Minimum dense cosine similarity required to answer

    Returns:
        Fraction of the group with a cosine below the threshold
    """
    grouped = [record for record in records if record.group == group]
    if not grouped:
        return 0.0
    return sum(1 for record in grouped if record.max_dense_similarity < threshold) / len(grouped)


def build_summary_rows(
    records: Sequence[DemoRecord],
    thresholds: Sequence[float] = THRESHOLDS,
    modes: Sequence[RetrievalMode] = EVAL_MODES,
) -> list[dict[str, Any]]:
    """Build the threshold summary table rows.

    Args:
        records: Per-question records
        thresholds: Dense cosine thresholds to report
        modes: Retrieval modes to report

    Returns:
        One dictionary per threshold with answered counts and rates
    """
    total_answerable = answerable_count(records)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        row: dict[str, Any] = {
            "threshold": threshold,
            "answered": answered_count(records, threshold),
            "answerable": total_answerable,
            "abstain_unanswerable": abstention_rate(records, UNANSWERABLE, threshold),
            "abstain_off_corpus": abstention_rate(records, OFF_CORPUS_ML, threshold),
            "abstain_injection": abstention_rate(records, INJECTION, threshold),
        }
        for mode in modes:
            row[f"hit_{mode}"] = gated_hit_rate(records, mode, threshold)
        rows.append(row)
    return rows


def negative_records(records: Sequence[DemoRecord]) -> list[DemoRecord]:
    """Return the records for the demo negative groups.

    Args:
        records: Per-question records

    Returns:
        Records in the off-corpus and injection groups
    """
    return [record for record in records if record.group in NEGATIVE_GROUPS]


def render_markdown(
    rows: Sequence[dict[str, Any]],
    negatives: Sequence[DemoRecord],
) -> str:
    """Render the demo defaults report as Markdown.

    Args:
        rows: Output of build_summary_rows
        negatives: Records for the demo negative groups

    Returns:
        Markdown text ending with a newline
    """
    lines = [
        "# Demo defaults",
        "",
        "Corpus mode defaults to hybrid retrieval with an abstain threshold of 0.45 on the "
        "best dense cosine similarity. Local mode and the CLI keep dense retrieval and a "
        "threshold of 0.35.",
        "",
        "Hit@5 counts every answerable question in the denominator, so a question that is "
        "gated by the threshold or missed by retrieval counts as a miss.",
        "",
        "| Threshold | Answered | Dense Hit@5 | BM25 Hit@5 | Hybrid Hit@5 | "
        "Eval abstain | Off-corpus abstain | Injection abstain |",
        "|-----------|----------|-------------|------------|--------------|"
        "--------------|--------------------|-------------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']:.2f} | {row['answered']}/{row['answerable']} | "
            f"{row['hit_dense']:.3f} | {row['hit_bm25']:.3f} | {row['hit_hybrid']:.3f} | "
            f"{row['abstain_unanswerable']:.3f} | {row['abstain_off_corpus']:.3f} | "
            f"{row['abstain_injection']:.3f} |"
        )

    lines += [
        "",
        "## Negative questions",
        "",
        "| Group | Question | Max dense cosine |",
        "|-------|----------|------------------|",
    ]
    for record in negatives:
        lines.append(f"| {record.group} | {record.question} | {record.max_dense_similarity:.3f} |")

    return "\n".join(lines) + "\n"


def build_corpus_retriever(model_name: str) -> Retriever:
    """Build an in-memory corpus retriever with the real embedding model.

    Args:
        model_name: Sentence-transformers model name

    Returns:
        Retriever over the default corpus chunks
    """
    embedding_model = EmbeddingModel(model_name)
    documents = load_directory(CORPUS_DIR)
    chunks = chunk_documents(
        documents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return build_retriever(embedding_model, chunks, model_name)


def main() -> None:
    """Compute the demo defaults report and write it to disk."""
    parser = argparse.ArgumentParser(
        description="Report demo defaults and the abstain threshold trade-off."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Sentence-transformers model name",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Reuse an existing index directory instead of building one in memory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path for the Markdown report",
    )
    args = parser.parse_args()

    if args.index_dir is not None:
        retriever = Retriever.from_index_path(args.index_dir, args.model, mode="dense")
    else:
        retriever = build_corpus_retriever(args.model)

    eval_set = load_eval_set(EVAL_FILE)
    negatives = load_negatives(NEGATIVES_FILE)
    records = collect_eval_records(retriever, eval_set)
    records += collect_negative_records(retriever, negatives)
    rows = build_summary_rows(records)
    markdown = render_markdown(rows, negative_records(records))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
