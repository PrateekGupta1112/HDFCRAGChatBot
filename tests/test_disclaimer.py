"""Invariants on the user-facing copy.

These assertions are cheap and catch the drift that actually happens: a fourth
example question sneaking in, or the freshness stamp being reworded while the
UI and the README keep their old copies.
"""

from __future__ import annotations

from pathlib import Path

from app import disclaimer as d
from app.textutils import normalize_ws


def test_exactly_three_example_questions() -> None:
    assert len(d.EXAMPLE_QUESTIONS) == 3


def test_example_questions_cover_the_intended_three_topics() -> None:
    lowered = [q.lower() for q in d.EXAMPLE_QUESTIONS]
    assert any("expense ratio" in q for q in lowered)
    assert any("lock-in" in q for q in lowered)
    assert any("statement" in q for q in lowered)


def test_freshness_prefix_is_exact() -> None:
    assert d.FRESHNESS_PREFIX == "Last updated from sources:"


def test_disclaimer_opens_with_the_required_sentence() -> None:
    assert d.DISCLAIMER.startswith("Facts-only. No investment advice.")
    assert "market risks" in d.DISCLAIMER


def test_refusals_do_not_contain_scheme_specific_values() -> None:
    # Refusals must read the same regardless of scheme; a number here would
    # mean a partial answer leaked into a refusal (FR-7).
    assert "%" not in d.REFUSAL_ADVICE
    assert "%" not in d.REFUSAL_PERFORMANCE


def test_educational_links_are_https() -> None:
    for url in (d.AMFI_EDUCATION_URL, d.SEBI_URL, d.HDFC_AMC_URL):
        assert url.startswith("https://")


def test_readme_disclaimer_matches_the_constant() -> None:
    """Guards the copy against drift between the module and the README.

    Whitespace is normalised on both sides first, because the README wraps the
    paragraph across lines — a verbatim substring check would fail on the line
    break alone while missing a genuine wording change.
    """
    readme = Path(__file__).resolve().parents[1] / "README.md"
    if not readme.exists():
        return  # README is a P11 deliverable; nothing to check yet.
    text = normalize_ws(readme.read_text(encoding="utf-8"))
    assert normalize_ws(d.DISCLAIMER) in text
