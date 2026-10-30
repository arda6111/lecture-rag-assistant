"""Offline tests for the demo defaults report script."""

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_demo_defaults.py"
NEGATIVES_FILE = REPO_ROOT / "eval" / "corpus" / "demo_negatives.json"

_SPEC = importlib.util.spec_from_file_location("check_demo_defaults", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
check_demo_defaults = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_demo_defaults)


def make_record(
    question: str,
    group: str,
    cosine: float,
    hits: dict[str, bool],
) -> object:
    """Build a demo record for the pure function tests."""
    return check_demo_defaults.DemoRecord(
        question=question,
        group=group,
        max_dense_similarity=cosine,
        hits=hits,
    )


def test_build_summary_rows_uses_gated_hits() -> None:
    """The summary counts gated or missed answerable questions as misses."""
    records = [
        make_record("a", "answerable", 0.50, {"dense": True, "bm25": False, "hybrid": True}),
        make_record("b", "answerable", 0.40, {"dense": True, "bm25": True, "hybrid": True}),
        make_record("c", "unanswerable", 0.30, {"dense": False, "bm25": False, "hybrid": False}),
        make_record("d", "off_corpus_ml", 0.44, {"dense": False, "bm25": False, "hybrid": False}),
        make_record("e", "injection", 0.60, {"dense": False, "bm25": False, "hybrid": False}),
    ]

    rows = check_demo_defaults.build_summary_rows(records, thresholds=(0.45,))
    row = rows[0]

    assert row["answered"] == 1
    assert row["answerable"] == 2
    assert row["hit_dense"] == 0.5
    assert row["hit_bm25"] == 0.0
    assert row["hit_hybrid"] == 0.5
    assert row["abstain_unanswerable"] == 1.0
    assert row["abstain_off_corpus"] == 1.0
    assert row["abstain_injection"] == 0.0


def test_summary_handles_empty_groups() -> None:
    """Empty groups produce zero rates instead of dividing by zero."""
    records = [make_record("a", "answerable", 0.50, {"dense": True})]

    rows = check_demo_defaults.build_summary_rows(records, thresholds=(0.45,))
    row = rows[0]

    assert row["answerable"] == 1
    assert row["abstain_unanswerable"] == 0.0
    assert row["abstain_off_corpus"] == 0.0
    assert row["abstain_injection"] == 0.0


def test_render_markdown_lists_negatives_and_thresholds() -> None:
    """The report contains the threshold table and every negative cosine."""
    records = [
        make_record("a", "answerable", 0.50, {"dense": True, "bm25": True, "hybrid": True}),
        make_record("negative question", "injection", 0.12, {"dense": False}),
    ]
    rows = check_demo_defaults.build_summary_rows(records, thresholds=(0.35, 0.45))

    markdown = check_demo_defaults.render_markdown(
        rows,
        check_demo_defaults.negative_records(records),
    )

    assert "| 0.35 |" in markdown
    assert "| 0.45 |" in markdown
    assert "negative question" in markdown
    assert "0.120" in markdown


def test_demo_negatives_have_the_expected_groups_and_counts() -> None:
    """The negative file has six off-corpus questions and four injections."""
    data = json.loads(NEGATIVES_FILE.read_text(encoding="utf-8"))

    assert len(data["off_corpus_ml"]) == 6
    assert len(data["injection"]) == 4
    assert all(isinstance(question, str) for group in data.values() for question in group)


def test_load_negatives_reads_both_groups() -> None:
    """The loader returns both negative groups."""
    negatives = check_demo_defaults.load_negatives(NEGATIVES_FILE)

    assert list(negatives) == ["off_corpus_ml", "injection"]
    assert len(negatives["off_corpus_ml"]) == 6
    assert len(negatives["injection"]) == 4
