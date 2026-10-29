"""Checks that CI and development tooling stay configured."""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_FILE = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT_FILE = REPO_ROOT / "pyproject.toml"


def test_torch_installs_are_pinned() -> None:
    """Every torch install line pins an exact version."""
    lines = [
        line
        for line in CI_FILE.read_text(encoding="utf-8").splitlines()
        if "pip install torch" in line
    ]

    assert lines
    assert all("torch==" in line for line in lines)


def test_every_job_has_a_timeout() -> None:
    """Both CI jobs declare a timeout."""
    text = CI_FILE.read_text(encoding="utf-8")
    jobs_section = text.split("\njobs:", 1)[1]
    matches = list(re.finditer(r"(?m)^  ([A-Za-z0-9_-]+):\s*$", jobs_section))

    assert len(matches) >= 2
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(jobs_section)
        block = jobs_section[match.start() : end]
        assert "timeout-minutes:" in block


def test_ci_runs_on_macos() -> None:
    """At least one CI job runs on macOS."""
    assert "runs-on: macos-latest" in CI_FILE.read_text(encoding="utf-8")


def test_mypy_covers_demo_and_scripts() -> None:
    """CI type checks the package, the demo and the scripts."""
    assert "mypy src demo.py scripts" in CI_FILE.read_text(encoding="utf-8")


def test_pyproject_has_test_timeout_configuration() -> None:
    """The dev extra installs pytest-timeout and pytest has a global timeout."""
    data = tomllib.loads(PYPROJECT_FILE.read_text(encoding="utf-8"))

    dev_dependencies = data["project"]["optional-dependencies"]["dev"]
    assert any(
        dependency.startswith("pytest-timeout") for dependency in dev_dependencies
    )
    assert data["tool"]["pytest"]["ini_options"]["timeout"] == 300
