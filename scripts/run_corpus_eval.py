"""Evaluate retrieval on the local corpus with a real embedding model."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from lecture_rag.chunking import chunk_documents
from lecture_rag.embeddings import EmbeddingModel
from lecture_rag.eval import (
    EVAL_MODES,
    EvalSet,
    evaluate_retrieval,
    load_eval_set,
    run_full_evaluation,
)
from lecture_rag.index import VectorIndex
from lecture_rag.models import Document
from lecture_rag.pdf_loader import load_directory
from lecture_rag.rerank import DEFAULT_RERANK_MODEL, CrossEncoderReranker, Reranker
from lecture_rag.retrieval import RetrievalMode, Retriever

CHUNK_SIZES = (256, 512, 1024)
CHUNK_OVERLAP = 64
DEFAULT_CHUNK_SIZE = 512
DEFAULT_K = 5
DEFAULT_ABSTAIN_THRESHOLD = 0.35
DEFAULT_MODEL = "all-MiniLM-L6-v2"
COMMAND = "python scripts/run_corpus_eval.py"
EVAL_FILE = "eval/corpus/eval_set.json"

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
EVAL_FILE_PATH = REPO_ROOT / EVAL_FILE
RESULTS_JSON = REPO_ROOT / "eval" / "corpus" / "results.json"
RESULTS_MD = REPO_ROOT / "eval" / "corpus" / "results.md"
README_PATH = REPO_ROOT / "README.md"

START_MARKER = "<!-- corpus-eval:start -->"
END_MARKER = "<!-- corpus-eval:end -->"


def build_retriever(
    embedding_model: EmbeddingModel,
    chunks: list[Document],
    model_name: str,
) -> Retriever:
    """Build an in-memory retriever over the given chunks.

    Args:
        embedding_model: Model used to embed chunks and queries
        chunks: Chunks to index
        model_name: Embedding model name stored in the index metadata

    Returns:
        Retriever backed by an in-memory vector index
    """
    embeddings = embedding_model.embed_texts([chunk.text for chunk in chunks])
    index = VectorIndex(embedding_model.dimension, model_name=model_name)
    index.add_documents(chunks, embeddings)
    return Retriever(embedding_model, index, mode="dense")


def _chunk_size_row(
    retriever: Retriever,
    eval_set: EvalSet,
    chunk_size: int,
    chunk_overlap: int,
    indexed_chunks: int,
    k: int,
    abstain_threshold: float,
) -> dict[str, Any]:
    """Evaluate one chunk size in dense mode and return a compact result row."""
    at_k = evaluate_retrieval(
        retriever,
        eval_set,
        k=k,
        abstain_threshold=abstain_threshold,
        mode="dense",
    )
    at_1 = evaluate_retrieval(retriever, eval_set, k=1, mode="dense")
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "indexed_chunks": indexed_chunks,
        "hit_at_k": round(at_k[f"hit_at_{k}"], 3),
        "mrr_at_k": round(at_k[f"mrr_at_{k}"], 3),
        "hit_at_1": round(at_1["hit_at_1"], 3),
        "abstain_accuracy": round(at_k["abstain_accuracy"], 3),
        "abstain_precision": round(at_k["abstain_precision"], 3),
        "abstain_recall": round(at_k["abstain_recall"], 3),
    }


def _mean_latency_ms(
    retriever: Retriever,
    questions: list[str],
    mode: RetrievalMode,
    k: int,
    rerank: bool,
) -> float:
    """Return mean per-query latency in milliseconds for one configuration."""
    if not questions:
        return 0.0

    retriever.search(questions[0], k=k, mode=mode, rerank=rerank)
    start = time.perf_counter()
    for question in questions:
        retriever.search(question, k=k, mode=mode, rerank=rerank)
    elapsed = time.perf_counter() - start
    return round(elapsed * 1000.0 / len(questions), 1)


def compute_corpus_results(
    embedding_model: EmbeddingModel,
    model_name: str | None = None,
    reranker: Reranker | None = None,
    rerank_model_name: str | None = None,
    *,
    corpus_dir: Path = CORPUS_DIR,
    eval_file: Path = EVAL_FILE_PATH,
    chunk_sizes: tuple[int, ...] = CHUNK_SIZES,
    chunk_overlap: int = CHUNK_OVERLAP,
    k: int = DEFAULT_K,
    abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
) -> dict[str, Any]:
    """Compute corpus evaluation results for several chunk sizes.

    Args:
        embedding_model: Model used to embed chunks and queries
        model_name: Embedding model name recorded in the results
        reranker: Optional reranker to evaluate on the default chunk size
        rerank_model_name: Reranker model name recorded in the results
        corpus_dir: Directory containing the corpus PDFs
        eval_file: Evaluation question file
        chunk_sizes: Chunk sizes to evaluate
        chunk_overlap: Chunk overlap used for every chunk size
        k: Number of documents to retrieve for the main metrics
        abstain_threshold: Dense similarity threshold for abstention

    Returns:
        JSON-serializable results without timestamps or absolute paths

    Raises:
        ValueError: If the default chunk size is not in chunk_sizes
    """
    if model_name is None:
        model_name = getattr(embedding_model, "model_name", DEFAULT_MODEL)

    documents = load_directory(corpus_dir)
    eval_set = load_eval_set(eval_file)

    rows: list[dict[str, Any]] = []
    built: dict[int, tuple[Retriever, list[Document]]] = {}
    for chunk_size in chunk_sizes:
        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        retriever = build_retriever(embedding_model, chunks, model_name)
        built[chunk_size] = (retriever, chunks)
        rows.append(
            _chunk_size_row(
                retriever,
                eval_set,
                chunk_size,
                chunk_overlap,
                len(chunks),
                k,
                abstain_threshold,
            )
        )

    if DEFAULT_CHUNK_SIZE not in built:
        raise ValueError(f"chunk_sizes must include {DEFAULT_CHUNK_SIZE}")
    default_retriever, default_chunks = built[DEFAULT_CHUNK_SIZE]

    results = run_full_evaluation(
        default_retriever,
        eval_set,
        eval_file=EVAL_FILE,
        k=k,
        abstain_threshold=abstain_threshold,
        model_name=model_name,
        indexed_chunks=len(default_chunks),
        command=COMMAND,
    )
    results["corpus"] = {
        "directory": "data/corpus",
        "files": len(list(corpus_dir.glob("*.pdf"))),
        "pages": len(documents),
    }
    results["chunk_size"] = DEFAULT_CHUNK_SIZE
    results["chunk_overlap"] = chunk_overlap
    results["chunk_sizes"] = rows

    if reranker is not None:
        default_retriever.reranker = reranker
        reranked = run_full_evaluation(
            default_retriever,
            eval_set,
            eval_file=EVAL_FILE,
            k=k,
            abstain_threshold=abstain_threshold,
            model_name=model_name,
            indexed_chunks=len(default_chunks),
            command=COMMAND,
            rerank=True,
        )
        candidates = default_retriever.rerank_candidates
        if candidates is None:
            candidates = max(k * 4, 20)
        results["reranker"] = {
            "model": rerank_model_name,
            "candidates": max(candidates, k),
            "modes": reranked["modes"],
        }

        questions = [question.question for question in eval_set.questions]
        results["latency_ms"] = {
            mode: {
                "base": _mean_latency_ms(default_retriever, questions, mode, k, rerank=False),
                "rerank": _mean_latency_ms(default_retriever, questions, mode, k, rerank=True),
            }
            for mode in EVAL_MODES
        }

    return results


def render_corpus_markdown(results: dict[str, Any]) -> str:
    """Render corpus evaluation results as a Markdown section.

    Args:
        results: Output of compute_corpus_results

    Returns:
        Markdown text ending with a newline, without a top-level heading
    """
    k = results["k"]
    threshold = results["abstain_threshold"]
    questions = results["questions"]
    corpus = results["corpus"]
    lines = [
        f"The corpus in `{corpus['directory']}` contains {corpus['files']} PDFs with "
        f"{corpus['pages']} loaded pages, indexed as {results['indexed_chunks']} chunks at "
        f"chunk size {results['chunk_size']} with `{results['embedding_model']}`, and evaluated "
        f"on {questions['total']} questions ({questions['answerable']} answerable, "
        f"{questions['unanswerable']} unanswerable) at k={k} with an abstain threshold of "
        f"{threshold:.2f}.",
        "",
        "#### Retrieval mode comparison",
        "",
    ]
    if results.get("reranker"):
        reranked_modes = results["reranker"]["modes"]
        lines += [
            f"| Mode | Rerank | Hit@{k} | MRR@{k} | Hit@1 | MRR@1 |",
            "|------|--------|-------|-------|-------|-------|",
        ]
        for mode, metrics in results["modes"].items():
            reranked = reranked_modes[mode]
            lines.append(
                f"| {mode} | no | {metrics['hit_at_k']:.3f} | "
                f"{metrics['mrr_at_k']:.3f} | {metrics['hit_at_1']:.3f} | "
                f"{metrics['mrr_at_1']:.3f} |"
            )
            lines.append(
                f"| {mode} | yes | {reranked['hit_at_k']:.3f} | "
                f"{reranked['mrr_at_k']:.3f} | {reranked['hit_at_1']:.3f} | "
                f"{reranked['mrr_at_1']:.3f} |"
            )
        lines += [
            "",
            "Abstention uses the first-stage dense similarity and is therefore "
            "the same with and without rerank.",
        ]
        if "latency_ms" in results:
            lines += [
                "",
                "| Mode | ms/query | ms/query with rerank |",
                "|------|----------|----------------------|",
            ]
            for mode, latency in results["latency_ms"].items():
                lines.append(f"| {mode} | {latency['base']:.1f} | {latency['rerank']:.1f} |")
            lines += [
                "",
                "Latency was measured on one CPU machine and varies between runs.",
            ]
    else:
        lines += [
            f"| Mode | Hit@{k} | MRR@{k} | Hit@1 | MRR@1 |",
            "|------|-------|-------|-------|-------|",
        ]
        for mode, metrics in results["modes"].items():
            lines.append(
                f"| {mode} | {metrics['hit_at_k']:.3f} | {metrics['mrr_at_k']:.3f} | "
                f"{metrics['hit_at_1']:.3f} | {metrics['mrr_at_1']:.3f} |"
            )

    dense = results["modes"]["dense"]
    lines += [
        "",
        f"#### Abstention at threshold {threshold:.2f} (dense)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Abstain accuracy | {dense['abstain_accuracy']:.3f} |",
        f"| Abstain precision | {dense['abstain_precision']:.3f} |",
        f"| Abstain recall | {dense['abstain_recall']:.3f} |",
        f"| Hit@{k} after abstain | {dense['hit_at_k_after_abstain']:.3f} |",
        "",
        "#### Threshold sweep (dense)",
        "",
        "| Threshold | Abstain acc |",
        "|-----------|-------------|",
    ]
    for row in results["sweep"]:
        lines.append(f"| {row['threshold']:.2f} | {row['abstain_accuracy']:.3f} |")

    lines += [
        "",
        "#### Chunk size comparison",
        "",
        f"| Chunk size | Chunks | Hit@{k} | MRR@{k} | Abstain acc |",
        "|------------|--------|-------|-------|-------------|",
    ]
    for row in results["chunk_sizes"]:
        lines.append(
            f"| {row['chunk_size']} | {row['indexed_chunks']} | {row['hit_at_k']:.3f} | "
            f"{row['mrr_at_k']:.3f} | {row['abstain_accuracy']:.3f} |"
        )

    return "\n".join(lines) + "\n"


def update_readme(readme_text: str, block: str) -> str:
    """Replace the text between the corpus evaluation markers in a README.

    Args:
        readme_text: Current README contents
        block: Markdown block to place between the markers

    Returns:
        README contents with the marked block replaced

    Raises:
        ValueError: If either marker is missing or the markers are out of order
    """
    start = readme_text.find(START_MARKER)
    end = readme_text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"README must contain {START_MARKER} before {END_MARKER}")

    stripped = block.strip()
    replacement = f"{START_MARKER}\n\n"
    if stripped:
        replacement += f"{stripped}\n\n"
    return readme_text[:start] + replacement + readme_text[end:]


def _write_outputs(results: dict[str, Any]) -> None:
    """Write results.json, results.md and the README block."""
    markdown = render_corpus_markdown(results)
    RESULTS_JSON.write_text(
        json.dumps(results, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    RESULTS_MD.write_text(
        "# Corpus evaluation results\n\n" + markdown,
        encoding="utf-8",
    )
    readme = README_PATH.read_text(encoding="utf-8")
    README_PATH.write_text(update_readme(readme, markdown), encoding="utf-8")


def main() -> None:
    """Run the corpus evaluation or re-render committed results."""
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval on data/corpus with a real embedding model."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Sentence-transformers model name",
    )
    parser.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        help="Cross-encoder reranker model name",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Re-render results.md and the README block from the committed results.json",
    )
    args = parser.parse_args()

    if args.render_only:
        results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        _write_outputs(results)
        return

    embedding_model = EmbeddingModel(args.model)
    reranker = None if args.no_rerank else CrossEncoderReranker(args.rerank_model)
    results = compute_corpus_results(
        embedding_model,
        args.model,
        reranker=reranker,
        rerank_model_name=None if args.no_rerank else args.rerank_model,
    )
    _write_outputs(results)


if __name__ == "__main__":
    main()
