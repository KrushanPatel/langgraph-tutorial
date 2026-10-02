"""Guard the repository's purpose: use >= 90% of LangGraph's public API."""

from __future__ import annotations

from scripts.api_coverage import report


def test_public_api_coverage_stays_above_90_percent():
    covered, total, missing = report()
    assert covered / total >= 0.90, f"coverage dropped to {covered}/{total}: {missing}"
