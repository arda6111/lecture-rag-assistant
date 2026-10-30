"""Checks on user-facing text in the repository."""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EMOJI_PATTERN = re.compile("[\U0001f000-\U0001faff\u2300-\u23ff\u2600-\u27bf\u2b00-\u2bff\ufe0f]")
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


def test_requirements_file_for_streamlit_cloud() -> None:
    """The Cloud requirements file pins CPU torch and installs the demo extra."""
    lines = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()

    assert lines == [
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        "torch==2.15.0",
        "-e .[demo]",
    ]


def test_security_reporting_policy() -> None:
    """SECURITY.md points to private reporting and contains no email address."""
    text = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    joined = " ".join(text.split())

    assert "private vulnerability reporting" in text
    assert "once the repository is public" in joined
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)


def test_security_docs_warn_about_untrusted_faiss_files() -> None:
    """Both security docs tell readers to only load indexes they built."""
    phrase = "only load indexes you built yourself"
    for name in ("SECURITY.md", "README.md"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8").lower()
        assert phrase in text


def test_streamlit_client_config() -> None:
    """The Streamlit config hides the Deploy button, menu and error details."""
    import tomllib

    config = tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))

    assert config["client"]["toolbarMode"] == "minimal"
    assert config["client"]["showErrorDetails"] == "none"


def test_readme_states_corpus_license() -> None:
    """The README says the corpus PDFs are CC BY 4.0, not MIT."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "CC BY 4.0 by their original authors" in readme
    assert "(data/SOURCES.md)" in readme
