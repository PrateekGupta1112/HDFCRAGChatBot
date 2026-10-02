from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import guardrails
from app.guardrails import GuardAction, classify, pii_detect, redact_pii

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def _golden(name: str) -> list[str]:
    """Load a golden query set."""
    return json.loads((EVAL_DIR / f"golden_{name}.json").read_text(encoding="utf-8"))


#: The literal personal value in each PII golden query. Every one of these must
#: be absent from ``redacted_query`` — that is the whole PII guarantee.
PII_LITERALS: dict[str, str] = {
    "My PAN is ABCDE1234F, can you check my returns?": "ABCDE1234F",
    "My Aadhaar is 2345 6789 0123, please update my details.": "2345 6789 0123",
    "Email me at ravi.sharma@gmail.com about the ELSS fund.": "ravi.sharma@gmail.com",
    "My account number is 30123456789, what is the NAV?": "30123456789",
    "The OTP is 482913, can you verify my folio?": "482913",
    "Call me on +91 98765 43210 with the details.": "98765 43210",
    "My phone number is 9876543210 and I want the exit load.": "9876543210",
    "Aadhaar XXXX XXXX 4567 - update my KYC status.": "XXXX XXXX 4567",
}


# --------------------------------------------------------------------------
# Golden gate sets
# --------------------------------------------------------------------------


@pytest.mark.parametrize("query", _golden("pii"))
def test_pii_queries_are_refused(query: str) -> None:
    decision = classify(query)
    assert decision.action is GuardAction.REFUSE_PII
    assert decision.reason == "pii_detected"
    assert decision.confidence == 1.0


@pytest.mark.parametrize("query", _golden("pii"))
def test_pii_value_absent_from_redacted_query(query: str) -> None:
    """The value must not survive anywhere in the redacted query."""
    literal = PII_LITERALS[query]
    assert literal not in classify(query).redacted_query
    assert "[REDACTED:" in classify(query).redacted_query


@pytest.mark.parametrize("query", _golden("advice"))
def test_advice_queries_are_refused(query: str) -> None:
    decision = classify(query)
    assert decision.action is GuardAction.REFUSE_ADVICE
    assert decision.reason == "advice_detected"
    assert decision.matched_pattern is not None
    assert decision.confidence == 0.9


@pytest.mark.parametrize("query", _golden("performance"))
def test_performance_queries_are_refused(query: str) -> None:
    decision = classify(query)
    assert decision.action is GuardAction.REFUSE_PERFORMANCE
    assert decision.reason == "performance_detected"
    assert decision.matched_pattern is not None
    assert decision.confidence == 0.9


@pytest.mark.parametrize("query", _golden("factual"))
def test_factual_queries_pass_through(query: str) -> None:
    """No false positives: every legitimate factual question is ANSWER."""
    decision = classify(query)
    assert decision.action is GuardAction.ANSWER
    assert decision.reason == "in_scope"
    assert decision.matched_pattern is None
    assert decision.confidence == 0.6


def test_offcorpus_is_not_decided_here() -> None:
    """A question that names nothing must not be refused on scope alone.

    Off-corpus needs retrieval evidence and is decided in P8 from
    ``RetrievalResult.is_empty``.
    """
    for query in ("What is the weather in Mumbai?", "Explain blockchain technology."):
        assert classify(query).action is GuardAction.ANSWER


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("query", _golden("factual"))
def test_redact_leaves_clean_query_byte_identical(query: str) -> None:
    assert redact_pii(query) == query


def test_redact_replaces_aadhaar_whole() -> None:
    """The phone rule also matches a 12-digit Aadhaar; the specific pattern
    must win so no digits are left behind."""
    redacted = redact_pii("My Aadhaar is 2345 6789 0123.")
    assert "2345" not in redacted
    assert "0123" not in redacted
    assert "[REDACTED:AADHAAR]" in redacted


def test_redact_marks_each_type() -> None:
    assert "[REDACTED:PAN]" in redact_pii("PAN ABCDE1234F")
    assert "[REDACTED:EMAIL]" in redact_pii("write to a.b@example.com")
    assert "[REDACTED:OTP]" in redact_pii("the OTP is 482913")


def test_redact_is_idempotent() -> None:
    once = redact_pii("PAN ABCDE1234F, mail a.b@example.com")
    assert redact_pii(once) == once


def test_bare_numbers_are_not_pii() -> None:
    """The corpus is full of figures. A naked digit run must not be flagged."""
    for query in (
        "What is the expense ratio?",
        "Is there an exit load?",
        "The lock-in period is 3 years.",
    ):
        assert classify(query).action is GuardAction.ANSWER


def test_pii_detect_reads_the_original_text() -> None:
    hits = dict(pii_detect("PAN ABCDE1234F"))
    assert hits["PAN"] == "ABCDE1234F"


def test_pii_detect_suppresses_overlapping_broad_matches() -> None:
    """A 12-digit Aadhaar is reported once, as AADHAAR, not again as a phone."""
    types = [t for t, _ in pii_detect("id 2345 6789 0123 end")]
    assert types.count("AADHAAR") == 1
    assert "PHONE_IN" not in types


def test_pii_detect_on_clean_text_is_empty() -> None:
    assert pii_detect("What is the benchmark of HDFC Equity Fund?") == []


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------


def test_pii_wins_over_advice() -> None:
    """An advice question carrying an e-mail must short-circuit at PII, so the
    address is redacted before the query is used anywhere."""
    decision = classify("Should I invest? Email me at ravi.sharma@gmail.com")
    assert decision.action is GuardAction.REFUSE_PII
    assert "ravi.sharma@gmail.com" not in decision.redacted_query


def test_advice_wins_over_performance() -> None:
    decision = classify("Should I sell after that return?")
    assert decision.action is GuardAction.REFUSE_ADVICE


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    ["", " ", "\n\t  ", "?", "?!", "1234567890", "a" * 10_000],
    ids=["empty", "spaces", "whitespace", "punct", "punct2", "digits", "10kb"],
)
def test_classify_never_raises(query: str) -> None:
    assert isinstance(classify(query), guardrails.GuardDecision)


def test_classify_on_empty_is_answer() -> None:
    decision = classify("")
    assert decision.action is GuardAction.ANSWER
    assert decision.redacted_query == ""


def test_classify_on_10kb_preserves_length() -> None:
    decision = classify("expense ratio " * 1_000)
    assert decision.action is GuardAction.ANSWER
    assert len(decision.redacted_query) == len("expense ratio " * 1_000)
