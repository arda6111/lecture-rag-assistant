"""Tests for single-embedding search, score labels and eval output files."""

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from lecture_rag.chunking import chunk_documents
from lecture_rag.cli import main
from lecture_rag.eval import (
    EvalQuestion,
    EvalSet,
    evaluate_retrieval,
    render_markdown,
    run_full_evaluation,
)
from lecture_rag.generation import ExtractiveFallbackProvider
from lecture_rag.index import VectorIndex
from lecture_rag.pdf_loader import load_directory
from lecture_rag.retrieval import Retriever
from tests.conftest import HashingEmbeddingModel

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample_lectures"


class CountingEmbeddingModel(HashingEmbeddingModel):
    """Hashing embedding model that counts query embeddings."""

    def __init__(self, model_name: str = "counting") -> None:
        super().__init__(model_name)
        self.query_calls = 0

    def embed_query(self, query: str) -> np.ndarray:
        """Count and embed a query."""
        self.query_calls += 1
        return super().embed_query(query)


def build_retriever() -> tuple[CountingEmbeddingModel, Retriever]:
    """Build a retriever over the sample lectures with a counting model."""
    chunks = chunk_documents(load_directory(SAMPLE_DIR))
    model = CountingEmbeddingModel()
    index = VectorIndex(model.dimension)
    index.add_documents(chunks, model.embed_texts([chunk.text for chunk in chunks]))
    return model, Retriever(model, index, mode="dense")


SMALL_SET = EvalSet(
    questions=[
        EvalQuestion(
            question="Which collision resolution strategies do hash tables use?",
            expected_file="data_structures.pdf",
            expected_page=2,
        ),
        EvalQuestion(
            question="What is the Bellman equation in dynamic programming?",
            expected_file="algorithms.pdf",
            expected_page=1,
        ),
        EvalQuestion(question="How does TCP congestion control work?"),
    ]
)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_evaluate_embeds_each_query_once(mode: str) -> None:
    """Retrieval and the abstain decision share one query embedding."""
    model, retriever = build_retriever()

    evaluate_retrieval(retriever, SMALL_SET, k=5, abstain_threshold=0.35, mode=mode)

    assert model.query_calls == len(SMALL_SET.questions)


@pytest.mark.parametrize(
    ("mode", "score_type"),
    [("dense", "cosine"), ("bm25", "bm25"), ("hybrid", "rrf")],
)
def test_results_carry_score_type(mode: str, score_type: str) -> None:
    """Every result is labelled with the kind of score it carries."""
    _, retriever = build_retriever()

    outcome = retriever.search("What is the Bellman equation?", k=3, mode=mode)

    assert outcome.results
    assert {result.score_type for result in outcome.results} == {score_type}
    assert outcome.results[0].score_label.startswith(f"{score_type}: ")
    assert 0.0 < outcome.max_dense_similarity <= 1.0


def test_extractive_answer_labels_rrf_score() -> None:
    """The extractive answer names the score kind instead of 'score'."""
    _, retriever = build_retriever()
    sources = retriever.retrieve("What is the Bellman equation?", k=3, mode="hybrid")

    answer = ExtractiveFallbackProvider().generate("Bellman?", sources)

    assert ", rrf: " in answer
    assert "score:" not in answer


def test_run_full_evaluation_and_markdown() -> None:
    """Full evaluation covers every mode, k=1 and the sweep, and renders Markdown."""
    _, retriever = build_retriever()

    results = run_full_evaluation(
        retriever,
        SMALL_SET,
        eval_file="eval/small.json",
        k=5,
        abstain_threshold=0.35,
        model_name="hashing",
        indexed_chunks=len(retriever.index.documents),
        command="lecture-rag eval --output eval",
    )

    assert set(results["modes"]) == {"dense", "bm25", "hybrid"}
    assert results["questions"] == {"total": 3, "answerable": 2, "unanswerable": 1}
    assert results["modes"]["hybrid"]["score_type"] == "rrf"
    assert [row["threshold"] for row in results["sweep"]][0] == 0.2
    assert len(results["sweep"]) == 9
    json.dumps(results)

    markdown = render_markdown(results)
    assert "Produced with `lecture-rag eval --output eval`." in markdown
    assert "| Mode | Hit@5 | MRR@5 | Hit@1 | MRR@1 | Abstain acc |" in markdown
    assert "| 0.35 |" in markdown
    assert "/workspace" not in markdown


def test_cli_eval_output_writes_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """eval --output writes results.json and results.md."""
    monkeypatch.setattr("lecture_rag.cli.EmbeddingModel", HashingEmbeddingModel)
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", HashingEmbeddingModel)
    runner = CliRunner()
    index_dir = tmp_path / "index"
    assert (
        runner.invoke(main, ["ingest", str(SAMPLE_DIR), "--index-dir", str(index_dir)]).exit_code
        == 0
    )

    eval_file = tmp_path / "small.json"
    eval_file.write_text(SMALL_SET.model_dump_json(), encoding="utf-8")
    out_dir = tmp_path / "out"

    result = runner.invoke(
        main,
        [
            "eval",
            "--eval-file",
            str(eval_file),
            "--index-dir",
            str(index_dir),
            "--output",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert data["questions"]["total"] == 3
    assert data["indexed_chunks"] == len(json.loads((index_dir / "documents.json").read_text()))
    assert (out_dir / "results.md").read_text(encoding="utf-8").startswith("# Evaluation results")


def test_cli_ask_prints_score_kind(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ask labels source scores with their kind."""
    monkeypatch.setattr("lecture_rag.cli.EmbeddingModel", HashingEmbeddingModel)
    monkeypatch.setattr("lecture_rag.embeddings.EmbeddingModel", HashingEmbeddingModel)
    runner = CliRunner()
    index_dir = tmp_path / "index"
    runner.invoke(main, ["ingest", str(SAMPLE_DIR), "--index-dir", str(index_dir)])

    result = runner.invoke(
        main,
        [
            "ask",
            "What is the Bellman equation?",
            "--index-dir",
            str(index_dir),
            "--retrieval",
            "hybrid",
            "--abstain-threshold",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "(rrf: " in result.output
    assert "(score: " not in result.output
