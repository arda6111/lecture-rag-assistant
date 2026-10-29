"""Capture a full-page screenshot of the Streamlit demo.

This script needs Playwright and its Chromium browser:

    pip install playwright
    playwright install chromium
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

DEFAULT_QUESTION = "Which traversal would you use to order tasks with dependencies?"
DEFAULT_URL = "http://localhost:8501"
DEFAULT_OUTPUT = "docs/demo.png"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Capture a full-page screenshot of the Streamlit demo."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Demo URL to open")
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Question to submit before capturing the screenshot",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path for the saved screenshot",
    )
    return parser.parse_args()


def main() -> None:
    """Open the demo, submit a question, and save a full-page screenshot."""
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto(args.url, wait_until="networkidle")

        question_input = page.get_by_placeholder("What is dynamic programming?")
        question_input.wait_for(state="visible", timeout=120_000)
        question_input.fill(args.question)
        question_input.press("Enter")

        sources_header = page.get_by_role("heading", name="Sources")
        sources_header.wait_for(state="visible", timeout=120_000)
        page.wait_for_timeout(2000)

        page.screenshot(path=str(output_path), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
