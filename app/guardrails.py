"""Guardrails — the refusal layer (architecture §9.4, PRD §6.5).

Three classes of question must never reach the LLM:

* **PII** — a PAN, Aadhaar, account number, OTP, e-mail or phone in the query.
  Refusing it is not a nicety: the P8 pipeline retrieves and generates on
  ``GuardDecision.redacted_query``, so a value that survived redaction would be
  embedded, sent to the model and rendered back to the user.
* **Advice** — "should I buy", "which is best", "is it right for me". The brief
  is facts-only; a suitability opinion is exactly what this bot must not do.
* **Performance** — forecasts, rankings and comparisons ("what will I make",
  "CAGR", "which performed best"). Scheme pages do carry historical return
  figures, and quoting one as a *prediction* is the failure mode FR-8 names.

:class:`classify` is a **pure function**. It opens no network, reads no corpus
and cannot raise, so P8 can call it before it knows whether a corpus exists.

**Ordering is load-bearing.** PII is tested first and unconditionally: an advice
question that also contains an e-mail must short-circuit at PII, because only
the PII branch guarantees the address is replaced before the query is used
anywhere. Advice is tested before performance because "should I sell after that
return?" is a refusal either way, and the more specific message wins.

**Off-corpus is deliberately not decided here.** A question can name a scheme
and still be unanswerable from the corpus, and a question can name nothing and
still be on-corpus ("minimum SIP"). Both need retrieval evidence, so
``NOT_IN_SOURCES`` is decided in P8 from ``RetrievalResult.is_empty`` and is
never returned by :func:`classify`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


class GuardAction(str, Enum):
    """What the pipeline should do with a query."""

    ANSWER = "answer"
    REFUSE_PII = "refuse_pii"
    REFUSE_ADVICE = "refuse_advice"
    REFUSE_PERFORMANCE = "refuse_performance"
    NOT_IN_SOURCES = "not_in_sources"


@dataclass(frozen=True)
class GuardDecision:
    """Outcome of classifying one query.

    Attributes:
        action: What the pipeline should do.
        reason: Stable machine-readable reason code.
        matched_pattern: Name of the pattern that decided it, or None.
        confidence: 1.0 for PII, 0.9 for advice/performance, 0.6 for in-scope.
        redacted_query: The query with every PII value replaced. **This, never
            the raw query, is what retrieval and generation receive.**
    """

    action: GuardAction
    reason: str
    matched_pattern: str | None
    confidence: float
    redacted_query: str


# --------------------------------------------------------------------------
# PII patterns (architecture §9.4)
# --------------------------------------------------------------------------

#: A PAN is 5 letters, 4 digits, 1 letter — the digits are what keep it from
#: matching an ordinary uppercase acronym, so the class is left case-sensitive.
_PAN = r"\b[A-Z]{5}\d{4}[A-Z]\b"

#: Aadhaar in plain, spaced and dashed form. The lookarounds are what stop a
#: 13-digit folio number from being read as its first 12 digits.
_AADHAAR = r"(?<![0-9X])\d{4}[ \-]?\d{4}[ \-]?\d{4}(?![0-9])"

#: Aadhaar as users actually type it once partially hidden. An ``X`` is
#: required, so a plain 12-digit number is AADHAAR and not AADHAAR_MASKED.
_AADHAAR_MASKED = r"\b(?:\d{4}[ \-]?X{4}[ \-]?\d{4}|X{4}[ \-]?X{4}[ \-]?\d{4})\b"

#: An account or folio number only counts when a keyword introduces it. A bare
#: digit run must not be flagged: the corpus is full of them, and every
#: factual question about a NAV or a fee is full of them.
_ACCOUNT_NUMBER = r"\b(?:a/c|acc(?:oun)?t|folio)\s*(?:no\.?|number|num)?\s*(?:is|was)?\s*[:#\-]?\s*\d{6,20}\b"

#: A one-time password, likewise keyword-gated for the same reason.
_OTP = r"\b(?:otp|one[ \-]?time[ \-]?password)\b\s*(?:is|was|:)?\s*\d{4,8}\b"

_EMAIL = r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"

#: A bare 10-digit Indian mobile number.
_PHONE_IN = r"(?<!\d)(?:\+?91[ \-]?)?[6-9]\d{4}[ \-]?\d{5}(?!\d)"

#: An international number written in groups, e.g. "+91 98765 43210".
_PHONE_INTL = r"(?<!\d)\+?\d{1,4}[ \-]\d{3,4}[ \-]\d{3,4}(?!\d)"

#: ``(name, pattern)`` in **redaction order**.
#:
#: The order is set here, not derived from pattern length. Structured ids are
#: redacted before the bare phone patterns on purpose: "2345 6789 0123" also
#: satisfies the 10-digit mobile rule, and applying that one first would leave
#: "0123" behind — a partial redaction that leaks the tail of the number it was
#: supposed to remove.
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AADHAAR_MASKED", re.compile(_AADHAAR_MASKED, re.IGNORECASE)),
    ("AADHAAR", re.compile(_AADHAAR)),
    ("ACCOUNT_NUMBER", re.compile(_ACCOUNT_NUMBER, re.IGNORECASE)),
    ("OTP", re.compile(_OTP, re.IGNORECASE)),
    ("PAN", re.compile(_PAN)),
    ("EMAIL", re.compile(_EMAIL, re.IGNORECASE)),
    ("PHONE_INTL", re.compile(_PHONE_INTL)),
    ("PHONE_IN", re.compile(_PHONE_IN)),
]


# --------------------------------------------------------------------------
# Advice patterns (architecture §9.4)
# --------------------------------------------------------------------------

#: Explicit requests for a decision: "should I", "can I buy", "worth buying".
#: Multi-word phrases only. A bare "should" or "best" would swallow factual
#: questions such as "What is the best lock-in period?" — and the golden
#: factual set is the test that catches exactly that.
ADVICE_INTENT: list[tuple[str, re.Pattern[str]]] = [
    ("ADVICE_INTENT", re.compile(r"\bshould\s+i\b", re.IGNORECASE)),
    ("ADVICE_INTENT", re.compile(r"\bshall\s+i\b", re.IGNORECASE)),
    ("ADVICE_INTENT", re.compile(r"\bcan\s+i\s+buy\b", re.IGNORECASE)),
    (
        "ADVICE_INTENT",
        re.compile(r"\bwould\s+you\s+(?:buy|recommend|suggest|pick)\b", re.IGNORECASE),
    ),
    ("ADVICE_INTENT", re.compile(r"\bis\s+it\s+(?:a\s+)?good\s+time\b", re.IGNORECASE)),
    ("ADVICE_INTENT", re.compile(r"\bis\s+now\s+a\s+good\s+time\b", re.IGNORECASE)),
    ("ADVICE_INTENT", re.compile(r"\bhow\s+much\s+should\s+i\b", re.IGNORECASE)),
    ("ADVICE_INTENT", re.compile(r"\bshould\s+i\s+(?:allocate|put)\b", re.IGNORECASE)),
    ("ADVICE_INTENT", re.compile(r"\btell\s+me\s+which\b", re.IGNORECASE)),
    (
        "ADVICE_INTENT",
        re.compile(r"\bmake\s+me\s+(?:rich|money)\b", re.IGNORECASE),
    ),
    ("ADVICE_INTENT", re.compile(r"\bworth\s+buying\b", re.IGNORECASE)),
]

#: Requests for a preferred option, or a bare "recommend".
ADVICE_PREFERENCE: list[tuple[str, re.Pattern[str]]] = [
    (
        "ADVICE_PREFERENCE",
        re.compile(
            r"\b(?:which|what)\s+is\s+the\s+best\b|\bbest\s+(?:fund|scheme|mutual\s+fund)\b",
            re.IGNORECASE,
        ),
    ),
    ("ADVICE_PREFERENCE", re.compile(r"\brecommend\b", re.IGNORECASE)),
    ("ADVICE_PREFERENCE", re.compile(r"\bsuggest\s+a\b", re.IGNORECASE)),
]

#: Suitability — the fund matched to a person's circumstances.
ADVICE_SUITABILITY: list[tuple[str, re.Pattern[str]]] = [
    ("ADVICE_SUITABILITY", re.compile(r"\bright\s+for\s+me\b", re.IGNORECASE)),
    ("ADVICE_SUITABILITY", re.compile(r"\bsafe\s+for\s+me\b", re.IGNORECASE)),
    ("ADVICE_SUITABILITY", re.compile(r"\bsuitable\s+for\b", re.IGNORECASE)),
    (
        "ADVICE_SUITABILITY",
        re.compile(r"\bright\s+for\s+a\s+\d+\s+year[\s-]?old\b", re.IGNORECASE),
    ),
]

ADVICE_PATTERNS: list[tuple[str, re.Pattern[str]]] = (
    ADVICE_INTENT + ADVICE_PREFERENCE + ADVICE_SUITABILITY
)


# --------------------------------------------------------------------------
# Performance patterns (architecture §9.4)
# --------------------------------------------------------------------------

#: Forward-looking money questions. A return **figure** the user asks to
#: compute ("how much will I make", "CAGR") is a forecast; a return **fact** on
#: the page is a description. Only the forecast is refused.
PERF_FORECAST: list[tuple[str, re.Pattern[str]]] = [
    (
        "PERF_FORECAST",
        re.compile(r"\bhow\s+much\s+will\s+i\s+(?:make|earn|get)\b", re.IGNORECASE),
    ),
    (
        "PERF_FORECAST",
        re.compile(r"\bwhat\s+will\s+i\s+(?:get|earn|make)\b", re.IGNORECASE),
    ),
    ("PERF_FORECAST", re.compile(r"\bexpected\s+returns?\b", re.IGNORECASE)),
    ("PERF_FORECAST", re.compile(r"\bprojected\s+returns?\b", re.IGNORECASE)),
    ("PERF_FORECAST", re.compile(r"\bcagr\b", re.IGNORECASE)),
    ("PERF_FORECAST", re.compile(r"\b\d+\s*year\s+returns?\b", re.IGNORECASE)),
    (
        "PERF_FORECAST",
        re.compile(r"\bwhat\s+is\s+the\s+(?:\d+\s*year\s+)?returns?\s+of\b", re.IGNORECASE),
    ),
]

#: Side-by-side comparison of two schemes' returns.
PERF_COMPARISON: list[tuple[str, re.Pattern[str]]] = [
    ("PERF_COMPARISON", re.compile(r"\bcompare\s+the\s+returns?\b", re.IGNORECASE)),
    (
        "PERF_COMPARISON",
        re.compile(r"\breturns?\s+(?:of|comparison)\b", re.IGNORECASE),
    ),
]

#: Ranking schemes by past or future performance.
PERF_RANKING: list[tuple[str, re.Pattern[str]]] = [
    (
        "PERF_RANKING",
        re.compile(
            r"\bwhich\s+(?:of\s+(?:these|my|the)\s+)?"
            r"(?:fund|funds|scheme|schemes)\s+(?:performed|performs|has\s+the\s+best)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PERF_RANKING",
        re.compile(
            r"\b(?:fund|funds|scheme|schemes)\s+(?:that\s+)?performed\s+best\b",
            re.IGNORECASE,
        ),
    ),
    ("PERF_RANKING", re.compile(r"\bbest\s+performing\b", re.IGNORECASE)),
    ("PERF_RANKING", re.compile(r"\bhighest\s+returns?\b", re.IGNORECASE)),
    ("PERF_RANKING", re.compile(r"\bperformance\s+comparison\b", re.IGNORECASE)),
]

PERFORMANCE_PATTERNS: list[tuple[str, re.Pattern[str]]] = (
    PERF_FORECAST + PERF_COMPARISON + PERF_RANKING
)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def redact_pii(text: str) -> str:
    """Replace every PII match with ``[REDACTED:<TYPE>]``.

    Patterns are applied in :data:`PII_PATTERNS` order, not longest-first by
    string length, so a structured id is never partially consumed by the
    generic phone rule that also matches it.

    Args:
        text: Raw query.

    Returns:
        The query with PII replaced. Text with no PII is returned unchanged.
    """
    out = text if isinstance(text, str) else ""
    for name, pattern in PII_PATTERNS:
        out = pattern.sub(f"[REDACTED:{name}]", out)
    return out


def pii_detect(text: str) -> list[tuple[str, str]]:
    """Find PII in the **original** text.

    Detection reads the raw string on purpose: redaction replaces the value, so
    a redaction-based detector would report nothing and the pipeline would not
    know to refuse.

    Args:
        text: Raw query.

    Returns:
        ``[(pii_type, matched_text)]`` in pattern order. Overlapping hits from
        a broad pattern are suppressed when a more specific one already
        reported the same characters.
    """
    if not isinstance(text, str):
        return []
    hits: list[tuple[str, str]] = []
    claimed: list[tuple[int, int]] = []
    for name, pattern in PII_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            # A generic pattern must not re-report characters a specific
            # pattern has already claimed.
            if any(span[0] < c[1] and c[0] < span[1] for c in claimed):
                continue
            claimed.append(span)
            hits.append((name, match.group(0)))
    return hits


def classify(query: str) -> GuardDecision:
    """Decide what to do with ``query``. Never raises.

    Order is PII, then advice, then performance, then in-scope. PII is first
    and unconditional so no personal value can reach retrieval, the model or
    the session log. Off-corpus is not decided here — it needs retrieval
    evidence and belongs to P8.

    Args:
        query: The user's raw question.

    Returns:
        A :class:`GuardDecision`. ``redacted_query`` is safe to use in place of
        the input.
    """
    if not isinstance(query, str) or not query.strip():
        return GuardDecision(
            action=GuardAction.ANSWER,
            reason="empty_query",
            matched_pattern=None,
            confidence=0.6,
            redacted_query="" if not isinstance(query, str) else query,
        )

    try:
        redacted = redact_pii(query)

        # 1. PII, decided on the RAW query.
        hits = pii_detect(query)
        if hits:
            return GuardDecision(
                action=GuardAction.REFUSE_PII,
                reason="pii_detected",
                matched_pattern=hits[0][0],
                confidence=1.0,
                redacted_query=redacted,
            )

        # 2. Advice.
        for name, pattern in ADVICE_PATTERNS:
            if pattern.search(redacted):
                return GuardDecision(
                    action=GuardAction.REFUSE_ADVICE,
                    reason="advice_detected",
                    matched_pattern=name,
                    confidence=0.9,
                    redacted_query=redacted,
                )

        # 3. Performance.
        for name, pattern in PERFORMANCE_PATTERNS:
            if pattern.search(redacted):
                return GuardDecision(
                    action=GuardAction.REFUSE_PERFORMANCE,
                    reason="performance_detected",
                    matched_pattern=name,
                    confidence=0.9,
                    redacted_query=redacted,
                )

        # 4. In scope.
        return GuardDecision(
            action=GuardAction.ANSWER,
            reason="in_scope",
            matched_pattern=None,
            confidence=0.6,
            redacted_query=redacted,
        )
    except Exception:  # noqa: BLE001 — a guardrail must never take the app down
        # Failing open here would be unsafe, so fall back to the PII refusal:
        # the least-permissive answer available without new information.
        logger.exception("guardrail classification failed; failing closed")
        return GuardDecision(
            action=GuardAction.REFUSE_PII,
            reason="classify_error",
            matched_pattern=None,
            confidence=1.0,
            redacted_query="[REDACTED]",
        )
