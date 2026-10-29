"""Offline tests for the corpus evaluation script."""

import importlib.util
import json
from pathlib import Path

import fitz
import pytest

from lecture_rag.chunking import chunk_documents
from lecture_rag.eval import load_eval_set
from lecture_rag.pdf_loader import load_directory
from tests.conftest import HashingEmbeddingModel

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
EVAL_FILE = REPO_ROOT / "eval" / "corpus" / "eval_set.json"
RESULTS_JSON = REPO_ROOT / "eval" / "corpus" / "results.json"
RESULTS_MD = REPO_ROOT / "eval" / "corpus" / "results.md"
README_PATH = REPO_ROOT / "README.md"

SCRIPT_PATH = REPO_ROOT / "scripts" / "run_corpus_eval.py"
_SPEC = importlib.util.spec_from_file_location("run_corpus_eval", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
run_corpus_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_corpus_eval)


def _readme_block() -> str:
    """Return the stripped text between the corpus evaluation markers."""
    text = README_PATH.read_text(encoding="utf-8")
    start = text.index(run_corpus_eval.START_MARKER) + len(run_corpus_eval.START_MARKER)
    end = text.index(run_corpus_eval.END_MARKER)
    return text[start:end].strip()


def _corpus_chunks() -> list:
    """Return the corpus chunks at the default chunk size and overlap."""
    return chunk_documents(
        load_directory(CORPUS_DIR),
        chunk_size=512,
        chunk_overlap=64,
    )


def test_corpus_eval_set_is_valid() -> None:
    """The corpus eval set has unique questions and valid PDF page references."""
    eval_set = load_eval_set(EVAL_FILE)
    answerable = [question for question in eval_set.questions if question.answerable]
    unanswerable = [question for question in eval_set.questions if not question.answerable]

    assert 40 <= len(eval_set.questions) <= 60
    assert 8 <= len(unanswerable) <= 12
    assert len({question.question for question in eval_set.questions}) == len(eval_set.questions)

    for question in answerable:
        assert question.expected_file is not None
        pdf_path = CORPUS_DIR / question.expected_file
        assert pdf_path.is_file()
        with fitz.open(pdf_path) as pdf:
            page_count = len(pdf)
        assert all(1 <= page <= page_count for page in question.pages)


def test_compute_corpus_results_offline() -> None:
    """The compute function works with the offline hashing embedding model."""
    results = run_corpus_eval.compute_corpus_results(HashingEmbeddingModel(), "hashing")

    assert {
        "command",
        "eval_file",
        "k",
        "abstain_threshold",
        "embedding_model",
        "indexed_chunks",
        "questions",
        "modes",
        "sweep",
        "corpus",
        "chunk_size",
        "chunk_overlap",
        "chunk_sizes",
    } <= set(results)
    assert results["indexed_chunks"] == len(_corpus_chunks())
    assert results["corpus"] == {"directory": "data/corpus", "files": 6, "pages": 163}
    assert results["chunk_size"] == 512
    assert results["chunk_overlap"] == 64
    assert [row["chunk_size"] for row in results["chunk_sizes"]] == list(
        run_corpus_eval.CHUNK_SIZES
    )
    assert all(
        set(row)
        == {
            "chunk_size",
            "chunk_overlap",
            "indexed_chunks",
            "hit_at_k",
            "mrr_at_k",
            "hit_at_1",
            "abstain_accuracy",
            "abstain_precision",
            "abstain_recall",
        }
        for row in results["chunk_sizes"]
    )


def test_update_readme_replaces_only_the_marked_block() -> None:
    """update_readme replaces the marked block, is idempotent, and validates markers."""
    text = (
        f"Before\n{run_corpus_eval.START_MARKER}\nold block\n{run_corpus_eval.END_MARKER}\nAfter\n"
    )

    updated = run_corpus_eval.update_readme(text, "new block")

    assert updated == (
        f"Before\n{run_corpus_eval.START_MARKER}\n\nnew block\n\n{run_corpus_eval.END_MARKER}\nAfter\n"
    )
    assert run_corpus_eval.update_readme(updated, "new block") == updated

    with pytest.raises(ValueError):
        run_corpus_eval.update_readme("No markers here", "new block")

    with pytest.raises(ValueError):
        run_corpus_eval.update_readme(
            f"{run_corpus_eval.END_MARKER}\n{run_corpus_eval.START_MARKER}",
            "new block",
        )


@pytest.mark.skipif(not RESULTS_JSON.exists(), reason="committed corpus results are absent")
def test_committed_corpus_results_render_consistently() -> None:
    """The committed results render to the README block and results.md."""
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    markdown = run_corpus_eval.render_corpus_markdown(results)

    assert markdown.strip() == _readme_block()
    assert RESULTS_MD.read_text(encoding="utf-8") == "# Corpus evaluation results\n\n" + markdown


@pytest.mark.skipif(not RESULTS_JSON.exists(), reason="committed corpus results are absent")
def test_committed_corpus_results_metadata_matches_data() -> None:
    """The committed results describe the corpus, model, and questions they claim."""
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    eval_set = load_eval_set(EVAL_FILE)
    answerable = sum(1 for question in eval_set.questions if question.answerable)

    assert results["indexed_chunks"] == len(_corpus_chunks())
    assert results["embedding_model"] == "all-MiniLM-L6-v2"
    assert results["questions"] == {
        "total": len(eval_set.questions),
        "answerable": answerable,
        "unanswerable": len(eval_set.questions) - answerable,
    }


@pytest.mark.skipif(not RESULTS_JSON.exists(), reason="committed corpus results are absent")
def test_readme_chunk_count_matches_committed_results() -> None:
    """The chunk count stated in the README comes from the committed results."""
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    readme = README_PATH.read_text(encoding="utf-8")

    claim = f"{results['indexed_chunks']} chunks from {results['corpus']['pages']} pages"
    assert claim in readme
