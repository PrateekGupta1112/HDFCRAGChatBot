"""Pure text helpers shared by ingestion, retrieval, guardrails and enforcement.

No third-party dependencies, no I/O, no global state — every function is a pure
string transformation. That is what makes them safe to use from the enforcement
layer, where a bug silently changes what a user is told.

The sentence splitter is regex-based on purpose. Adding `nltk` (or any
sentence-splitting package) would introduce a runtime data download, which
breaks the "runs offline after first install" property of the demo (NFR-5).
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Whitespace / unicode normalisation
# --------------------------------------------------------------------------

_ZERO_WIDTH = "‌‍﻿"
_QUOTES = {
    "‘": "'",
    "’": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "′": "'",
    "″": '"',
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
}


def normalize_ws(s: str) -> str:
    """Collapse every whitespace run to a single space and strip the result.

    Also folds non-breaking spaces, zero-width characters and curly quotes /
    dashes to their ASCII equivalents, so that text extracted from HTML
    compares cleanly against generated text.

    Note that this collapses newlines too: callers that need line structure
    preserved (table chunks, JSONL record boundaries) must not pre-normalise
    the text before splitting or serialising it.

    Args:
        s: Arbitrary input text.

    Returns:
        The normalised text, or ``""`` for empty/whitespace-only input.
    """
    if not s:
        return ""
    out = s.translate(str.maketrans({**_QUOTES, "\xa0": " "}))
    out = "".join(ch for ch in out if ch not in _ZERO_WIDTH)
    return re.sub(r"\s+", " ", out).strip()


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------

#: Tokens that end in a period without ending a sentence.
PROTECTED_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "rs",
        "no",
        "e.g",
        "i.e",
        "vs",
        "cf",
        "approx",
        "fy",
    }
)

# A sentence boundary is terminal punctuation followed by whitespace, or a
# newline. Requiring whitespace after the punctuation is what already prevents
# splitting on decimals ("1.23%"), ellipses mid-token and dotted hostnames
# ("groww.in") — none of those have a space after the dot.
_BOUNDARY_RE = re.compile(r"(?<=[.!?…])[ \t]+|\n+")

_ALPHA_TOKEN_BEFORE = re.compile(r"[A-Za-z][A-Za-z.]*$")
_DIGIT_RE = re.compile(r"\d")


def _token_before(text: str, boundary_start: int) -> str:
    """Return the alphabetic token immediately preceding a boundary.

    Args:
        text: The full text being split.
        boundary_start: Index of the whitespace character that begins the gap.

    Returns:
        The lowercased token with any trailing dots removed (so ``"e.g."``
        yields ``"e.g"``), or ``""`` if no alphabetic token precedes the gap.
    """
    m = _ALPHA_TOKEN_BEFORE.search(text[:boundary_start])
    return m.group(0).rstrip(".").lower() if m else ""


def _is_protected_boundary(text: str, boundary_start: int) -> bool:
    """Decide whether a candidate boundary should be suppressed.

    Suppressed when: the preceding token is a protected abbreviation
    (``Rs.``, ``Dr.``, ``e.g.``); a digit sits on both sides of a period
    (a decimal); or the following word starts lowercase, which indicates the
    period belongs to a token rather than terminating a sentence.

    Args:
        text: The full text being split.
        boundary_start: Index of the whitespace beginning the candidate gap.

    Returns:
        ``True`` if this boundary must not split the text.
    """
    # Walk back over any whitespace, then require terminal punctuation there.
    # A newline boundary has no punctuation before it, so it is never protected.
    i = boundary_start - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    if i < 0 or text[i] not in ".!?…":
        return False

    # Walk back over the punctuation run; i ends on the character before it.
    punct_end = i + 1
    while i >= 0 and text[i] in ".!?…":
        i -= 1

    # Walk forward to the next real character.
    j = boundary_start
    while j < len(text) and text[j] in " \t":
        j += 1

    prev_char = text[i] if i >= 0 else ""
    next_char = text[j] if j < len(text) else ""

    # Decimal such as "1. 23" — digits on both sides of the period.
    if prev_char.isdigit() and next_char.isdigit():
        return True
    # Protected abbreviation, e.g. "Rs. 500", "Dr. Smith", "e.g. the fee".
    if _token_before(text, punct_end) in PROTECTED_ABBREVIATIONS:
        return True
    # A lowercase word after the period means the period is intra-sentence.
    if next_char.islower():
        return True
    return False


def normalize_lines(s: str) -> str:
    """Normalise each line independently, preserving line structure.

    This is :func:`normalize_ws` applied *per line* rather than to the whole
    string, for callers that must keep the line boundaries — ingestion builds
    section bodies as one label:value pair per line, and P3's table chunker
    splits on those boundaries. Passing such text through
    :func:`normalize_ws` collapses every pair onto one line, which destroys the
    label/value adjacency that the atomicity rule exists to protect.

    Blank lines are dropped: ingestion emits them only where a table block had
    a gap, and a blank line would otherwise read as a label with no value.

    Args:
        s: Raw multi-line text.

    Returns:
        The same lines, each whitespace-normalised, joined by ``"\\n"``.
    """
    if not s:
        return ""
    return "\n".join(line for line in (normalize_ws(x) for x in s.splitlines()) if line)


def split_sentences(s: str) -> list[str]:
    """Split text into sentences using a regex, guarding abbreviations.

    Never splits on a decimal point, on a period inside a dotted token, or on a
    protected abbreviation. Terminal punctuation is retained on each piece.

    Newlines are treated as boundaries, so label/value lines in a table chunk
    become separate sentences — which is what the extractive fallback in P7
    needs in order to score them independently.

    Args:
        s: Text to split. Assumed whitespace-normalised, but tolerates raw.

    Returns:
        Sentences in document order, with empty pieces removed.
    """
    if not s or not s.strip():
        return []

    pieces: list[str] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(s):
        if _is_protected_boundary(s, m.start()):
            continue
        piece = s[start : m.start()].strip()
        if piece:
            pieces.append(piece)
        start = m.end()
    tail = s[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def truncate_sentences(s: str, n: int) -> tuple[str, int]:
    """Cut text to at most ``n`` sentences on a sentence boundary.

    Args:
        s: Text to truncate.
        n: Maximum number of sentences to keep. Values below 1 return ``""``.

    Returns:
        A ``(text, original_sentence_count)`` pair. When truncation was not
        needed the original string is returned unchanged.
    """
    sentences = split_sentences(s)
    original = len(sentences)
    if n < 1:
        return "", original
    if original <= n:
        return s, original
    return " ".join(sentences[:n]).strip(), original


# --------------------------------------------------------------------------
# URLs
# --------------------------------------------------------------------------

#: Full-width closing brackets. The system prompt numbers the context blocks
#: ``[1] … [n]`` and asks for one source link, and gpt-oss answers by wrapping
#: the link in the CJK pair ``【…】``. A bare-URL scan that stops only at ASCII
#: brackets therefore keeps the trailing ``】`` glued to the URL, the whitelist
#: comparison in P7's enforcement fails, and a correct answer is failed closed to
#: NOT_FOUND. Observed 2026-10-02 on a live run: the model returned the right
#: fact *and* the exact corpus URL and was still discarded, with
#: ``dropped_unverified_urls:1`` then ``fail_closed_no_citation``.
_FULLWIDTH_CLOSERS = "】］〉》」』〕"

_URL_RE = re.compile(r"https?://[^\s\)\]\}\"'<>`" + _FULLWIDTH_CLOSERS + r"]+")

#: Query parameters that identify a campaign/click, not a document.
_TRACKING_PARAMS = frozenset(
    {
        "ref",
        "referrer",
        "source",
        "fbclid",
        "gclid",
        "gbraid",
        "wbraid",
        "msclkid",
        "igshid",
        "mc_cid",
        "mc_eid",
    }
)

_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)]+)\)")


def normalize_url(u: str) -> str:
    """Canonicalise a URL so two spellings of the same page compare equal.

    Lowercases the host, drops the scheme, a leading ``www.``, tracking query
    parameters and any trailing slash or sentence punctuation. Path case is
    preserved because paths are case-sensitive.

    Used on **both sides** of every citation comparison — a model may return
    ``https://www.groww.in/x/`` for a corpus URL stored as
    ``https://groww.in/x``, and both must normalise to the same string.

    Args:
        u: A URL, possibly with surrounding whitespace or trailing punctuation.

    Returns:
        The canonical form, or ``""`` if the input is empty.
    """
    if not u:
        return ""
    # The same full-width closers the bare-URL scan stops at, so a URL that
    # arrived already stripped and one that did not normalise to the same
    # target. Comparing the two spellings is the whole point of this function.
    out = u.strip().rstrip(".,;:!?)" + _FULLWIDTH_CLOSERS)
    if not out:
        return ""

    # Scheme is not part of identity for our purposes.
    out = re.sub(r"^https?://", "", out, flags=re.IGNORECASE)

    head, sep, query = out.partition("?")
    head = head.split("#", 1)[0]
    host, slash, path = head.partition("/")
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]

    path = path.rstrip("/")

    if sep:
        kept = []
        for pair in query.split("&"):
            if not pair:
                continue
            key = pair.split("=", 1)[0].strip().lower()
            if key.startswith("utm_") or key in _TRACKING_PARAMS:
                continue
            kept.append(pair)
        if kept:
            path = f"{path}?{'&'.join(sorted(kept))}" if path else "?" + "&".join(sorted(kept))

    return f"{host}/{path}" if path else host


def extract_urls(s: str) -> list[str]:
    """Find every URL in text, in order, without duplicates.

    Markdown links are unwrapped to their target, and trailing punctuation is
    stripped so a URL ending a sentence does not absorb the full stop.

    Args:
        s: Text to scan.

    Returns:
        Distinct URLs in first-appearance order; ``[]`` when there are none.
    """
    if not s:
        return []
    # Keep only the target of markdown links, then scan for bare URLs.
    scanned = _MD_LINK_RE.sub(r"\1 \2", s)
    found: list[str] = []
    for raw in _URL_RE.findall(scanned):
        url = raw.rstrip(".,;:!?)]}'\"" + _FULLWIDTH_CLOSERS)
        if url and url not in found:
            found.append(url)
    return found


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

# Surface forms, not parsed values. The enforcement numeric audit (P7) asks
# "does this exact string occur in the retrieved text?", which is a substring
# test — parsing to float would lose the difference between 1.12% and 1.12.
_NUMBER_RE = re.compile(
    r"""
      (?:₹|Rs\.?|INR)\s*\d[\d,]*(?:\.\d+)?(?:[% ]*%)?      # Rs. 1,23,456 / ₹500 / ₹1.12%
    | \d[\d,]*\.\d+\s*%                                    # 1.23%
    | \d[\d,]*%                                            # 5%
    | \d{1,3}(?:,\d{2,3})+(?:\.\d+)?(?:\s*%)?              # 1,23,456
    | \d+\.\d+                                             # 3.5
    | \d+(?:\s*(?:years?|yrs?|months?|days?))?             # 3 / 3 years
    """,
    re.VERBOSE,
)

_DIGITS_ONLY_RE = re.compile(r"\d")

#: Whitespace immediately before a unit suffix. ``1.04 %`` and ``1.04%`` are the
#: same figure written two ways, and only one of them matches the page.
_SPACE_BEFORE_UNIT_RE = re.compile(r"\s+(?=[%)\]])")


def canonical_number(s: str) -> str:
    """Return the whitespace-tolerant form of a number or a chunk of text.

    The enforcement audit is a substring test, so two spellings of one figure
    have to collapse to one before it runs. gpt-oss writes ``1.04`` + U+202F +
    ``%`` in roughly one live run in four, while the corpus spells the same
    figure ``1.04%`` with nothing between the digits and the unit. Compared
    verbatim, that deletes a *correct* answer; the sentence then holds no figure
    at all, so it is dropped as mangled as well, and the reply collapses to a
    bare citation line with no body. Measured 2026-10-02, 1 of 4 runs of "what
    is the expense ratio of hdfc large cap fund".

    Only whitespace is touched. Digits, separators, decimal points and the unit
    itself are left exactly as written, so ``1.03`` still cannot pass on the
    strength of a ``1.0`` elsewhere — the property that keeps the audit a
    surface-form check rather than a value comparison.

    Args:
        s: A numeric surface form, or a whole block of text to canonicalise.

    Returns:
        The same text with whitespace before a unit suffix removed.
    """
    if not s:
        return ""
    return _SPACE_BEFORE_UNIT_RE.sub("", s)


def extract_numbers(s: str) -> list[str]:
    """Return the numeric literals in text as their original surface forms.

    A surface form may include a currency prefix and a unit suffix
    (``Rs. 1,23,456``, ``1.23%``, ``3 years``) so that a verbatim substring
    check against the retrieved context succeeds when the wording matches.

    Args:
        s: Text to scan.

    Returns:
        Surface forms in first-appearance order, without duplicates.
    """
    if not s:
        return []
    found: list[str] = []
    for m in _NUMBER_RE.finditer(s):
        form = m.group(0).strip()
        if form and form not in found:
            found.append(form)
    return found


def strip_number(s: str, num: str) -> str:
    """Remove one numeric surface form from a sentence.

    Tidies the punctuation left behind (a space before a comma or full stop) and
    collapses whitespace. If nothing legible remains — the sentence consisted
    only of that number — an empty string is returned so the caller can drop the
    sentence entirely rather than emit a stub.

    Args:
        s: The sentence to edit.
        num: A surface form previously returned by :func:`extract_numbers`.

    Returns:
        The edited sentence, or ``""`` when no alphanumeric content remains.
    """
    if not s or not num:
        return s
    out = s.replace(num, "", 1)
    out = re.sub(r"[ \t]+([.,;:!?])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    out = re.sub(r"^[\s,;:]+|[\s,;:]+$", "", out)
    if not _DIGITS_ONLY_RE.search(out) and not re.search(r"[A-Za-z]", out):
        return ""
    return out
