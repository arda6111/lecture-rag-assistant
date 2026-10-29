"""Checks that CI and development tooling stay configured."""

import re
import tomllib
from pathlib import Path

import pytest

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


def _job_block(name: str) -> str:
    """Return the text of one job in the CI workflow."""
    jobs_section = CI_FILE.read_text(encoding="utf-8").split("\njobs:", 1)[1]
    matches = list(re.finditer(r"(?m)^  ([A-Za-z0-9_-]+):\s*$", jobs_section))
    for index, match in enumerate(matches):
        if match.group(1) == name:
            end = matches[index + 1].start() if index + 1 < len(matches) else len(jobs_section)
            return jobs_section[match.start() : end]
    raise AssertionError(f"job {name} not found")


def test_macos_job_runs_only_on_demand_or_for_tags() -> None:
    """The macOS job exists but only runs on manual dispatch or release tags."""
    text = CI_FILE.read_text(encoding="utf-8")
    on_section = text.split("\njobs:", 1)[0]
    block = _job_block("test-macos")

    assert "workflow_dispatch:" in on_section
    assert "runs-on: macos-latest" in block
    assert "github.event_name == 'workflow_dispatch'" in block
    assert "startsWith(github.ref, 'refs/tags/')" in block


@pytest.mark.parametrize("name", ["lint-and-test", "security"])
def test_linux_jobs_run_on_every_push_and_pull_request(name: str) -> None:
    """The Linux jobs have no condition, so they run for every push and pull request."""
    block = _job_block(name)

    assert "runs-on: ubuntu-latest" in block
    assert "\n    if:" not in block


def test_security_job_scans_code_and_full_history() -> None:
    """The security job runs bandit on all code and gitleaks over the full history."""
    block = _job_block("security")

    assert "bandit -r src demo.py scripts" in block
    assert "fetch-depth: 0" in block
    assert 'gitleaks git --log-opts="--all"' in block
    assert "sha256sum -c" in block


def test_mypy_covers_demo_and_scripts() -> None:
    """CI type checks the package, the demo and the scripts."""
    assert "mypy src demo.py scripts" in CI_FILE.read_text(encoding="utf-8")


def test_pyproject_has_test_timeout_configuration() -> None:
    """The dev extra installs pytest-timeout and pytest has a global timeout."""
    data = tomllib.loads(PYPROJECT_FILE.read_text(encoding="utf-8"))

    dev_dependencies = data["project"]["optional-dependencies"]["dev"]
    assert any(dependency.startswith("pytest-timeout") for dependency in dev_dependencies)
    assert data["tool"]["pytest"]["ini_options"]["timeout"] == 300
