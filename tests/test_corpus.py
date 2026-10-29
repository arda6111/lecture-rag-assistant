"""Tests for the bundled arXiv corpus and its source metadata."""

import hashlib
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from lecture_rag.cli import main
from lecture_rag.pdf_loader import load_directory
from tests.conftest import HashingEmbeddingModel

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
SOURCES_FILE = REPO_ROOT / "data" / "SOURCES.md"


def _corpus_table_rows() -> list[list[str]]:
    """Return the cells of the corpus rows in the sources table."""
    rows = []
    for line in SOURCES_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("| `"):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def test_sources_cover_every_corpus_pdf() -> None:
    """Every corpus PDF has a table row, and every row points to a valid file."""
    rows = _corpus_table_rows()
    pdf_files = {path.name for path in CORPUS_DIR.glob("*.pdf")}
    row_files = {cells[0].strip("`") for cells in rows}

    assert len(rows) == 6
    assert row_files == pdf_files

    for cells in rows:
        pdf_path = CORPUS_DIR / cells[0].strip("`")
        expected_sha256 = cells[-1].strip("`")

        assert pdf_path.is_file()
        assert hashlib.sha256(pdf_path.read_bytes()).hexdigest() == expected_sha256


def test_sources_have_license_and_arxiv_url() -> None:
    """Every corpus row states the license and links to the arXiv abstract."""
    for cells in _corpus_table_rows():
        row = " | ".join(cells)

        assert "CC BY 4.0" in row
        assert "https://creativecommons.org/licenses/by/4.0/" in row
        assert re.search(r"https://arxiv\.org/abs/\S+", row)


def test_corpus_size_is_within_limit() -> None:
    """The corpus is smaller than 15 MB."""
    total_size = sum(path.stat().st_size for path in CORPUS_DIR.glob("*.pdf"))

    assert total_size < 15 * 1024 * 1024


def test_load_directory_reads_all_corpus_pages() -> None:
    """The corpus loads as page documents with text from all six PDFs."""
    documents = load_directory(CORPUS_DIR)

    assert len(documents) == 163
    assert len({document.file_name for document in documents}) == 6

    pages_with_text = sum(1 for document in documents if len(document.text) > 100)
    assert pages_with_text >= 0.9 * len(documents)


def test_ingest_corpus_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ingest the corpus with the offline test embedding model."""
    monkeypatch.setattr("lecture_rag.cli.EmbeddingModel", HashingEmbeddingModel)

    runner = CliRunner()
    index_dir = tmp_path / "index"
    result = runner.invoke(
        main,
        ["ingest", str(CORPUS_DIR), "--index-dir", str(index_dir)],
    )

    assert result.exit_code == 0, result.output
    assert "from 163 documents" in result.output

    match = re.search(r"Indexed (\d+) chunks", result.output)
    assert match is not None
    assert int(match.group(1)) > 500
