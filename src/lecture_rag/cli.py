"""Command-line interface for the RAG system."""

import logging
from pathlib import Path

import click

from .chunking import chunk_documents
from .embeddings import EmbeddingModel
from .eval import evaluate_retrieval, load_eval_set
from .generation import Generator
from .index import VectorIndex
from .pdf_loader import load_directory
from .retrieval import Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group()
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
    type=int,
    default=512,
    help="Maximum chunk size in characters",
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=64,
    help="Overlap between chunks in characters",
)
def ingest(
    input_dir: Path,
    index_dir: Path,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Ingest PDF files from INPUT_DIR and build the index."""
    click.echo(f"Loading PDFs from {input_dir}...")
    documents = load_directory(input_dir)

    if not documents:
        click.echo("No documents loaded. Exiting.")
        return

    click.echo(f"Chunking {len(documents)} documents...")
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    click.echo(f"Loading embedding model: {model}...")
    embedding_model = EmbeddingModel(model)

    click.echo("Generating embeddings...")
    texts = [doc.text for doc in chunks]
    embeddings = embedding_model.embed_texts(texts)

    click.echo("Building index...")
    vector_index = VectorIndex(embedding_model.dimension)
    vector_index.add_documents(chunks, embeddings)

    click.echo(f"Saving index to {index_dir}...")
    vector_index.save(index_dir)

    click.echo(f"✓ Indexed {len(chunks)} chunks from {len(documents)} documents")


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
    type=int,
    default=5,
    help="Number of documents to retrieve",
)
@click.option(
    "--mode",
    type=click.Choice(["llm", "extractive"]),
    default="extractive",
    help="Answer generation mode",
)
def ask(query: str, index_dir: Path, model: str, k: int, mode: str) -> None:
    """Ask a QUERY and get an answer with citations."""
    click.echo(f"Loading index from {index_dir}...")
    retriever = Retriever.from_index_path(index_dir, model)

    click.echo("Retrieving relevant documents...")
    sources = retriever.retrieve(query, k=k)

    if not sources:
        click.echo("No relevant documents found.")
        return

    click.echo(f"\nGenerating answer (mode: {mode})...")
    generator = Generator.from_config(mode=mode)
    answer = generator.generate_answer(query, sources)

    click.echo("\n" + "=" * 80)
    click.echo("ANSWER:")
    click.echo("=" * 80)
    click.echo(answer.answer_text)

    click.echo("\n" + "=" * 80)
    click.echo("SOURCES:")
    click.echo("=" * 80)
    for i, result in enumerate(answer.sources, 1):
        doc = result.document
        click.echo(f"\n[{i}] {doc.file_name}, page {doc.page_number} (score: {result.score:.3f})")
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
    type=int,
    default=5,
    help="Number of documents to retrieve for metrics",
)
def eval(eval_file: Path, index_dir: Path, model: str, k: int) -> None:
    """Run evaluation on the retrieval system."""
    click.echo(f"Loading evaluation set from {eval_file}...")
    eval_set = load_eval_set(eval_file)

    click.echo(f"Loading index from {index_dir}...")
    retriever = Retriever.from_index_path(index_dir, model)

    click.echo(f"Evaluating {len(eval_set.questions)} questions...")
    metrics = evaluate_retrieval(retriever, eval_set, k=k)

    click.echo("\n" + "=" * 80)
    click.echo("EVALUATION RESULTS:")
    click.echo("=" * 80)
    click.echo(f"Total questions: {metrics['total_questions']}")
    click.echo(f"Hit@{k}: {metrics[f'hit_at_{k}']:.3f}")
    click.echo(f"MRR@{k}: {metrics[f'mrr_at_{k}']:.3f}")
    click.echo(f"Mean retrieval score: {metrics['mean_score']:.3f}")


if __name__ == "__main__":
    main()
