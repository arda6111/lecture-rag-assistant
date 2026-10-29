"""Checks on user-facing text in the repository."""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EMOJI_PATTERN = re.compile(
    "[\U0001F000-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]"
)
EMOJI_SHORTCODE = re.compile(r":[a-z0-9_+-]+:")
TEXT_FILES = sorted(
    [
        *REPO_ROOT.glob("*.md"),
        *REPO_ROOT.glob("*.py"),
        *(REPO_ROOT / "src").rglob("*.py"),
        *(REPO_ROOT / "scripts").rglob("*.py"),
        *(REPO_ROOT / "docs").rglob("*.md"),
        *(REPO_ROOT / "eval").rglob("*.md"),
    ]
)


@pytest.mark.parametrize("path", TEXT_FILES, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_no_emoji_in_app_and_docs(path: Path) -> None:
    """The app, CLI output and docs contain no emoji characters or shortcodes."""
    text = path.read_text(encoding="utf-8")

    assert not EMOJI_PATTERN.findall(text)
    if path.suffix == ".md":
        assert not EMOJI_SHORTCODE.findall(text)


@pytest.mark.parametrize(
    "path",
    [path for path in TEXT_FILES if path.suffix == ".md"],
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_no_em_dashes_in_docs(path: Path) -> None:
    """Docs use plain punctuation instead of em-dashes."""
    assert "\u2014" not in path.read_text(encoding="utf-8")
