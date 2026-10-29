"""Command-line interface for the RAG system."""

import json
import logging
import os
from pathlib import Path
from typing import Any

import click
from dotenv import find_dotenv, load_dotenv

from .chunking import chunk_documents
from .embeddings import EmbeddingModel, EmbeddingModelError
from .eval import (
    EVAL_MODES,
    evaluate_retrieval,
    load_eval_set,
    render_markdown,
    run_full_evaluation,
)
from .generation import Generator
from .index import VectorIndex
from .pdf_loader import load_directory
from .retrieval import (
    RetrievalMode,
    Retriever,
    default_retrieval_mode,
    normalize_mode,
    parse_abstain_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _default_abstain_threshold() -> float:
    """Read the default abstain threshold from the environment."""
    return parse_abstain_threshold(os.getenv("LECTURE_RAG_ABSTAIN_THRESHOLD"))


def _resolve_retrieval_mode(retrieval: str | None) -> RetrievalMode:
    """Resolve the retrieval mode from the CLI option or the environment."""
    try:
        return normalize_mode(retrieval) if retrieval else default_retrieval_mode()
    except ValueError as e:
        raise click.ClickException(f"{e} (check LECTURE_RAG_RETRIEVAL_MODE)") from e


def _load_retriever(index_dir: Path, model: str, retrieval: str | None) -> Retriever:
    """Validate the retrieval mode, then load the retriever."""
    mode = _resolve_retrieval_mode(retrieval)
    try:
        return Retriever.from_index_path(index_dir, model, mode=mode)
    except EmbeddingModelError as e:
        raise click.ClickException(str(e)) from e
    except ValueError as e:
        raise click.ClickException(f"{e}. Run `lecture-rag ingest` first.") from e


def _validate_output_dir(
    ctx: click.Context,
    param: click.Parameter,
    value: str | None,
) -> Path | None:
    """Validate the raw --output value and convert it to a path."""
    if value is None:
        return None
    if value.strip() == "":
        raise click.BadParameter("must not be empty")
    return Path(value)


def _prepare_output_dir(output: Path) -> None:
    """Create the output directory and verify that it is writable."""
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise click.ClickException(
            f"Cannot create output directory {output}: {e.strerror or e}"
        ) from e
    if not os.access(output, os.W_OK | os.X_OK):
        raise click.ClickException(f"Output directory is not writable: {output}")


class DotenvGroup(click.Group):
    """Click group that loads .env before parsing any command."""

    def main(self, *args: Any, **kwargs: Any) -> Any:
        load_dotenv(find_dotenv(usecwd=True))
        return super().main(*args, **kwargs)


@click.group(cls=DotenvGroup)
def main() -> None:
    """Lecture RAG Assistant - Retrieval-Augmented Generation for university lectures."""
    pass


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--index-dir",
    type=click.Path(path_type=Path),
    default="index",
    help="Directory to save the index",
)
@click.option(
    "--model",
    default="all-MiniLM-L6-v2",
    help="Sentence-transformers model name",
)
@click.option(
    "--chunk-size",
    type=click.IntRange(min=1),
    default=512,
    help="Maximum chunk size in characters",
)
@click.option(
    "--chunk-overlap",
    type=click.IntRange(min=0),
    default=64,
    help="Overlap between chunks in characters",
)
@click.option(
    "--max-file-size-mb",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Maximum PDF file size in MB (default: LECTURE_RAG_MAX_FILE_SIZE_MB or 50)",
)
@click.option(
    "--max-pages",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum pages per PDF (default: LECTURE_RAG_MAX_PAGES or 500)",
)
def ingest(
    input_dir: Path,
    index_dir: Path,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    max_file_size_mb: float | None,
    max_pages: int | None,
) -> None:
    """Ingest PDF files from INPUT_DIR and build the index."""
    if chunk_overlap >= chunk_size:
        raise click.BadParameter(
            f"must be smaller than --chunk-size ({chunk_size})",
            param_hint="--chunk-overlap",
        )

    click.echo(f"Loading PDFs from {input_dir}...")
    documents = load_directory(
        input_dir,
        max_file_size_mb=max_file_size_mb,
        max_pages=max_pages,
    )

    if not documents:
        raise click.ClickException(f"No PDF documents could be loaded from {input_dir}")

    click.echo(f"Chunking {len(documents)} documents...")
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    click.echo(f"Loading embedding model: {model}...")
    try:
        embedding_model = EmbeddingModel(model)
    except EmbeddingModelError as e:
        raise click.ClickException(str(e)) from e

    click.echo("Generating embeddings...")
    texts = [doc.text for doc in chunks]
    embeddings = embedding_model.embed_texts(texts)

    click.echo("Building index...")
    vector_index = VectorIndex(embedding_model.dimension)
    vector_index.add_documents(chunks, embeddings)

    click.echo(f"Saving index to {index_dir}...")
    vector_index.save(index_dir)

    click.echo(f"Indexed {len(chunks)} chunks from {len(documents)} documents")


@main.command()
@click.argument("query")
@click.option(
    "--index-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="index",
    help="Directory containing the index",
)
@click.option(
    "--model",
    default="all-MiniLM-L6-v2",
    help="Sentence-transformers model name (must match ingest model)",
)
@click.option(
    "--k",
    type=click.IntRange(min=1),
    default=5,
    help="Number of documents to retrieve",
)
@click.option(
    "--mode",
    type=click.Choice(["llm", "extractive"]),
    default="extractive",
    help="Answer generation mode",
)
@click.option(
    "--retrieval",
    type=click.Choice(["dense", "bm25", "hybrid"]),
    default=None,
    help="Retrieval mode (default: LECTURE_RAG_RETRIEVAL_MODE or dense)",
)
@click.option(
    "--abstain-threshold",
    type=click.FloatRange(0.0, 1.0),
    default=_default_abstain_threshold,
    show_default=True,
    help="Minimum dense similarity required to answer",
)
def ask(
    query: str,
    index_dir: Path,
    model: str,
    k: int,
    mode: str,
    retrieval: str | None,
    abstain_threshold: float,
) -> None:
    """Ask a QUERY and get an answer with citations."""
    try:
        generator = Generator.from_config(mode=mode)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Loading index from {index_dir}...")
    retriever = _load_retriever(index_dir, model, retrieval)

    click.echo("Retrieving relevant documents...")
    outcome = retriever.search(query, k=k)

    if outcome.should_abstain(abstain_threshold):
        click.echo("The lecture materials do not seem to cover this question.")
        return

    sources = outcome.results

    if not sources:
        click.echo("No relevant documents found.")
        return

    click.echo(f"\nGenerating answer (mode: {mode})...")
    try:
        answer = generator.generate_answer(query, sources)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    click.echo("\n" + "=" * 80)
    click.echo("ANSWER:")
    click.echo("=" * 80)
    click.echo(answer.answer_text)

    click.echo("\n" + "=" * 80)
    click.echo("SOURCES:")
    click.echo("=" * 80)
    for i, result in enumerate(answer.sources, 1):
        doc = result.document
        click.echo(f"\n[{i}] {doc.file_name}, page {doc.page_number} ({result.score_label})")
        click.echo(f"    {doc.text[:150]}...")


@main.command()
@click.option(
    "--eval-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="eval/eval_set.json",
    help="Path to evaluation JSON file",
)
@click.option(
    "--index-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="index",
    help="Directory containing the index",
)
@click.option(
    "--model",
    default="all-MiniLM-L6-v2",
    help="Sentence-transformers model name",
)
@click.option(
    "--k",
    type=click.IntRange(min=1),
    default=5,
    help="Number of documents to retrieve for metrics",
)
@click.option(
    "--retrieval",
    type=click.Choice(["dense", "bm25", "hybrid"]),
    default=None,
    help="Retrieval mode (default: LECTURE_RAG_RETRIEVAL_MODE or dense)",
)
@click.option(
    "--compare",
    is_flag=True,
    help="Compare dense, bm25, and hybrid retrieval",
)
@click.option(
    "--sweep",
    is_flag=True,
    help="Sweep abstain thresholds from 0.20 to 0.60 in dense mode",
)
@click.option(
    "--abstain-threshold",
    type=click.FloatRange(0.0, 1.0),
    default=_default_abstain_threshold,
    show_default=True,
    help="Minimum dense similarity required to answer",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    callback=_validate_output_dir,
    default=None,
    help=(
        "Write results.json and results.md for all modes to this directory "
        "(cannot be combined with --compare or --sweep)"
    ),
)
def eval(
    eval_file: Path,
    index_dir: Path,
    model: str,
    k: int,
    retrieval: str | None,
    compare: bool,
    sweep: bool,
    abstain_threshold: float,
    output: Path | None,
) -> None:
    """Run evaluation on the retrieval system."""
    if output is not None:
        if compare or sweep:
            raise click.UsageError(
                "--output evaluates all modes and cannot be combined with "
                "--compare or --sweep"
            )
        if retrieval is not None:
            click.echo(
                "Warning: --retrieval is ignored with --output; "
                "all retrieval modes are evaluated.",
                err=True,
            )

    click.echo(f"Loading evaluation set from {eval_file}...")
    try:
        eval_set = load_eval_set(eval_file)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    if not eval_set.questions:
        raise click.ClickException(f"Eval set is empty: {eval_file}")

    if output is not None:
        _prepare_output_dir(output)

    click.echo(f"Loading index from {index_dir}...")
    retriever = _load_retriever(index_dir, model, retrieval)

    if output is not None:
        command = (
            f"lecture-rag eval --eval-file {eval_file.as_posix()} --k {k} "
            f"--abstain-threshold {abstain_threshold:.2f} --output {output.as_posix()}"
        )
        click.echo(f"Evaluating all modes on {len(eval_set.questions)} questions...")
        results = run_full_evaluation(
            retriever,
            eval_set,
            eval_file=eval_file.as_posix(),
            k=k,
            abstain_threshold=abstain_threshold,
            model_name=model,
            indexed_chunks=len(retriever.index.documents),
            command=command,
        )
        try:
            (output / "results.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
            markdown = render_markdown(results)
            (output / "results.md").write_text(markdown, encoding="utf-8")
        except OSError as e:
            raise click.ClickException(
                f"Could not write results to {output}: {e.strerror or e}"
            ) from e
        click.echo(markdown)
        click.echo(f"Wrote {output / 'results.json'} and {output / 'results.md'}")
        return

    if sweep:
        sweep_mode: RetrievalMode = "dense"
        click.echo(f"Sweeping abstain thresholds on {len(eval_set.questions)} questions...")
        click.echo("")
        click.echo("| Threshold | Abstain acc |")
        click.echo("|-----------|-------------|")
        for step in range(9):
            threshold = round(0.20 + step * 0.05, 2)
            metrics = evaluate_retrieval(
                retriever,
                eval_set,
                k=k,
                abstain_threshold=threshold,
                mode=sweep_mode,
            )
            click.echo(f"| {threshold:.2f} | {metrics['abstain_accuracy']:.3f} |")
        return

    if compare:
        click.echo(f"Comparing retrieval modes on {len(eval_set.questions)} questions...")
        click.echo("")
        click.echo(f"| Mode | Hit@{k} | MRR@{k} | Abstain acc |")
        click.echo("|------|-------|-------|-------------|")
        for mode_name in EVAL_MODES:
            metrics = evaluate_retrieval(
                retriever,
                eval_set,
                k=k,
                abstain_threshold=abstain_threshold,
                mode=mode_name,
            )
            click.echo(
                f"| {mode_name} | {metrics[f'hit_at_{k}']:.3f} | "
                f"{metrics[f'mrr_at_{k}']:.3f} | "
                f"{metrics['abstain_accuracy']:.3f} |"
            )
        return

    click.echo(f"Evaluating {len(eval_set.questions)} questions...")
    metrics = evaluate_retrieval(
        retriever,
        eval_set,
        k=k,
        abstain_threshold=abstain_threshold,
    )

    click.echo("\n" + "=" * 80)
    click.echo("EVALUATION RESULTS:")
    click.echo("=" * 80)
    click.echo(f"Total questions: {metrics['total_questions']}")
    click.echo(f"Answerable questions: {metrics['answerable_questions']}")
    click.echo(f"Unanswerable questions: {metrics['unanswerable_questions']}")
    click.echo(f"Hit@{k}: {metrics[f'hit_at_{k}']:.3f}")
    click.echo(f"MRR@{k}: {metrics[f'mrr_at_{k}']:.3f}")
    click.echo(f"Mean retrieval score: {metrics['mean_score']:.3f}")
    click.echo(f"Abstain accuracy: {metrics['abstain_accuracy']:.3f}")
    click.echo(f"Abstain precision: {metrics['abstain_precision']:.3f}")
    click.echo(f"Abstain recall: {metrics['abstain_recall']:.3f}")
    click.echo(
        f"Hit@{k} after abstain: "
        f"{metrics[f'answerable_hit_at_{k}_after_abstain']:.3f}"
    )


if __name__ == "__main__":
    main()
