"""Loading — RAG data-ingestion stage 1 (architecture §9.1, PRD FR-1).

Fetches the 5 source pages listed in ``data/sources.csv`` and turns each into an
ordered tree of :class:`Section` records, which P3 chunks.

Extraction decisions come from the P1 spike (see ``SPIKE_NOTES.md``):

* The BS4 heading-tree walk is the **only** viable path. On all 5 pages
  ``trafilatura`` returns 4-27k characters containing **zero** of the target facts
  (``expense ratio``, ``exit load``, ``minimum sip`` …) because it drops the
  ``fundDetails_*`` divs that carry them. Trafilatura is kept only as a last-ditch
  fallback.
* ``<script>`` is decomposed **before** extraction. The Next.js ``__NEXT_DATA__``
  blob embeds return figures and the riskometer value; leaving it in would pollute
  chunks and risk leaking performance numbers into answers (FR-8).
* Fee and minimum-investment facts are key:value pairs in nested ``<div>``s, not
  ``<table>``s. Section text is therefore built newline-joined so
  :func:`is_table_block` can see one label:value pair per line — collapsing to a
  single line would destroy the signal P3's table-atomicity rule depends on.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path

import httpx
import trafilatura
from bs4 import BeautifulSoup, NavigableString, Tag

from app.config import Settings, get_settings
from app.errors import ExtractionError, FetchError
from app.textutils import normalize_lines, normalize_ws

logger = logging.getLogger(__name__)

SOURCES_CSV = Path("data/sources.csv")
RAW_DIR = Path("data/raw")
DOCUMENTS_JSONL = Path("data/processed/documents.jsonl")
INGEST_REPORT_JSON = Path("data/processed/ingest_report.json")

#: P4 added a chunk/embed/store step here. It used to call ``write_chunks()``
#: with no path, which silently wrote to the *real* ``chunks.jsonl`` even when
#: every other output had been redirected — so a unit test with a synthetic page
#: overwrote the committed artifact. Every destination in this module is
#: therefore a module-level constant that tests can monkeypatch.
CHUNKS_JSONL = Path("data/processed/chunks.jsonl")

#: Minimum usable main content, in characters, before extraction is considered to
#: have failed and the trafilatura fallback is attempted.
MIN_MAIN_CHARS = 500

#: Extraction modes recorded in ``Document.extraction_mode``. See
#: :func:`extract` for why these differ from the names in implementation.md §P2.
MODE_HEADING_WALK = "bs4_heading_walk"
MODE_TRAFILATURA = "trafilatura"

#: Headings that are chrome rather than content. Groww's pages carry a nav that
#: would otherwise become 10-15 junk "sections".
BOILERPLATE_HEADINGS = frozenset(
    {
        "home",
        "login",
        "log in",
        "sign up",
        "signup",
        "sign in",
        "download app",
        "contact us",
        "careers",
        "privacy policy",
        "terms and conditions",
        "disclaimer",
        "cookie",
        "cookies",
        "skip to content",
    }
)

#: Elements whose text is never content.
NON_CONTENT_TAGS = ("script", "style", "noscript", "svg", "template", "iframe")

#: CSS-module base names identifying site chrome. Groww ships **no** ``<nav>``,
#: ``<footer>`` or ``<main>`` tags (verified in P1), so chrome is only
#: identifiable by its generated class names. Without this filter the footer is
#: absorbed into whichever heading happens to be last — on ``hdfc_large_cap``
#: that buried 60+ lines of "Contact Us / GROWW / Careers" and two e-mail
#: addresses inside the ``Fund house`` section.
#:
#: Matched against the *base name* of a hashed CSS-module class only, never as a
#: prefix. Prefix matching was tried first and is wrong: ``header_`` also
#: matches ``header_schemeName__zL6RN``, the class on the page's own ``<h1>``,
#: and decomposing that deleted every heading on the page.
BOILERPLATE_CLASS_BASES = frozenset(
    {
        "footer",
        "footerTopSection",
        "footer_gap",
        "hamburger",
        "searchBar",
        "appDownload",
        "cookieBanner",
        "stickyAd",
        "companyLogo",
    }
)

#: CSS-module classes look like ``baseName_hash__AbCd``; the hash and hash
#: suffix vary per build, so the base name is extracted and compared.
_CSS_MODULE_CLASS = re.compile(r"^(?P<base>[A-Za-z_][A-Za-z0-9_]*?)_[A-Za-z0-9]{4,}__[A-Za-z0-9]{2,}$")

#: Tags whose text is collected when reconstructing a section's body.
BLOCK_TAGS = ("div", "p", "li", "span", "td", "th", "tr", "h1", "h2", "h3", "h4", "h5", "h6")

#: Candidate roots for the main content, in order of preference.
_CONTENT_ROOTS = ("main", "article", "div")

REQUIRED_COLUMNS = ("scheme_id", "scheme_name", "category", "plan", "url")


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """One row of ``data/sources.csv``."""

    scheme_id: str
    scheme_name: str
    category: str
    plan: str
    url: str


@dataclass(frozen=True)
class Section:
    """A heading and the text beneath it."""

    section_title: str
    heading_level: int
    text: str
    is_table: bool
    order: int


@dataclass(frozen=True)
class Document:
    """A fully ingested source page."""

    scheme_id: str
    scheme_name: str
    category: str
    plan: str
    source_url: str
    fetched_at: str
    content_sha256: str
    extraction_mode: str
    sections: list[Section] = field(default_factory=list)


@dataclass(frozen=True)
class IngestReport:
    """Outcome of one ingestion run."""

    sources_ok: int
    sources_failed: list[str]
    documents: list[Document]
    max_fetched_at: str | None
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def read_sources(path: Path | None = None) -> list[SourceSpec]:
    """Load and validate ``data/sources.csv``.

    Args:
        path: CSV file to read; defaults to :data:`SOURCES_CSV`, resolved at
            call time so tests can redirect it.

    Returns:
        One :class:`SourceSpec` per data row, in file order.

    Raises:
        ValueError: If a required column is missing or a row is incomplete.
    """
    path = Path(path) if path is not None else SOURCES_CSV
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
        specs = []
        for lineno, row in enumerate(reader, start=2):
            if not any((v or "").strip() for v in row.values()):
                continue
            empty = [c for c in REQUIRED_COLUMNS if not (row.get(c) or "").strip()]
            if empty:
                raise ValueError(f"{path} line {lineno}: empty field(s): {', '.join(empty)}")
            specs.append(
                SourceSpec(
                    scheme_id=row["scheme_id"].strip(),
                    scheme_name=row["scheme_name"].strip(),
                    category=row["category"].strip(),
                    plan=row["plan"].strip(),
                    url=row["url"].strip(),
                )
            )
    return specs


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------


def fetch(url: str, s: Settings | None = None) -> tuple[str, str, int]:
    """Fetch one page, following redirects, with bounded retries.

    Args:
        url: URL to fetch.
        s: Settings; defaults to the process singleton.

    Returns:
        ``(final_url, html, status)``. ``final_url`` is what the user will be
        sent to by a citation, so a redirect must never be discarded.

    Raises:
        FetchError: If the request fails or returns a non-2xx status after
            exhausting retries.
    """
    s = s or get_settings()
    headers = {
        "User-Agent": s.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
    }
    backoff = 1.0
    last: str = "unknown error"
    with httpx.Client(timeout=float(s.fetch_timeout_s), follow_redirects=True) as client:
        for attempt in range(1, s.fetch_retries + 1):
            try:
                resp = client.get(url, headers=headers)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if 200 <= resp.status_code < 300:
                    return str(resp.url), resp.text, resp.status_code
                last = f"HTTP {resp.status_code}"
            if attempt < s.fetch_retries:
                logger.warning("fetch attempt %d/%d failed for %s (%s), retrying",
                               attempt, s.fetch_retries, url, last)
                time.sleep(backoff)
                backoff *= 2
    raise FetchError(f"{url}: {last} (after {s.fetch_retries} attempts)")


# --------------------------------------------------------------------------
# Table detection
# --------------------------------------------------------------------------

#: "Label: value" / "Label - value". This is the only form implementation.md
#: §P2 names, and on the real pages it never fires — see the two patterns below.
_LABEL_COLON_VALUE = re.compile(r"^\s*[\w ()%/-]{2,40}\s*[:\-]\s*\S+")

#: "Label Rs.500" / "Label 1.23%" — label and value on the same line.
_LABEL_TRAILING_NUMBER = re.compile(
    r"^.{2,60}?\s+(?:₹|rs\.?\s*)?[\d,]+(?:\.\d+)?\s*%?$", re.IGNORECASE
)

#: A line that is *only* a figure: "₹100", "1.23%", "Rs. 500", "+7.6%", "-0.35 %".
#: The optional sign matters: Groww renders every return with an explicit +/-
#: ("Fund returns" / "+7.6%"), and without it those lines read as prose labels,
#: the pair run never reaches :data:`MIN_TABLE_PAIRS`, and the whole returns
#: block is left on the prose path where P3's atomicity rule does not apply to
#: it — a figure free to drift away from the label that names it.
_VALUE_ONLY = re.compile(r"^(?:[+-]|₹|rs\.?)?\s*[\d,]+(?:\.\d+)?\s*%?$", re.IGNORECASE)

#: How many consecutive label:value pairs make a block "table-like".
MIN_TABLE_PAIRS = 3


def _is_value_only(line: str) -> bool:
    """Whether a line is a bare figure, i.e. the value half of a pair."""
    line = line.strip()
    return bool(line) and " " not in line.replace("₹", "").replace("Rs.", "").replace("rs.", "") \
        and bool(_VALUE_ONLY.match(line))


def _pair_runs(lines: list[str]) -> int:
    """Count the longest run of consecutive label:value pairs.

    Three shapes are accepted, because the corpus uses all three:

    * ``"Expense ratio: 1.03%"`` — label and value on one line.
    * ``"Min. for SIP ₹500"`` — label and value on one line, no colon.
    * ``"Min. for SIP"`` then ``"₹500"`` — label and value on adjacent lines.
      This is what Groww's nested ``fundDetails_*`` divs actually produce, and
      the only reason section text is newline-joined rather than collapsed.

    A prose line interrupts the run: a table interrupted by a paragraph is not
    one atomic block.

    Returns:
        The largest number of unbroken pairs found. Three or more means
        table-like.
    """
    pairs = best = 0
    pending_label = False
    for raw in lines:
        line = raw.strip()
        if not line:
            pending_label = False
            continue
        if _LABEL_COLON_VALUE.match(line) or _LABEL_TRAILING_NUMBER.match(line):
            pairs += 1  # self-contained pair
            pending_label = False
        elif _is_value_only(line):
            if pending_label:
                pairs += 1  # completes the pair opened by the previous line
            pending_label = False
        else:
            pending_label = True
        if not pending_label:
            best = max(best, pairs)
    return best


def _in_table(el: Tag | None) -> bool:
    """Whether ``el`` is a ``<table>`` or sits inside one.

    The element handed in is a *leaf* block — a ``<td>`` — so a descendant
    search alone would always miss the enclosing ``<table>``.
    """
    node = el
    while isinstance(node, Tag):
        if node.name == "table":
            return True
        node = node.parent
    return False


def is_table_block(text: str, soup_section: Tag | None = None) -> bool:
    """Decide whether a section body is a table-like label:value block.

    True when the source element was inside a real ``<table>``, or when the text
    holds at least ``MIN_TABLE_PAIRS`` consecutive label:value pairs.

    Args:
        text: Section body, newline-separated (see the module docstring).
        soup_section: A representative originating element, if available.

    Returns:
        Whether the block must be kept atomic by P3.
    """
    if _in_table(soup_section):
        return True
    if not text:
        return False
    return _pair_runs(text.splitlines()) >= MIN_TABLE_PAIRS


# --------------------------------------------------------------------------
# Extract
# --------------------------------------------------------------------------


def _content_root(soup: BeautifulSoup) -> Tag:
    """Locate the subtree that holds the scheme page's own content.

    Groww serves **no** ``<main>`` and no ``<article>`` (verified in P1), so
    this normally lands on ``<body>``. An earlier version fell back to the
    first ``<div>``, which is arbitrary — it happened to be the outermost
    wrapper on these pages, but nothing guarantees that.

    Returns:
        The root element, never ``None``.
    """
    for name in ("main", "article"):
        found = soup.find(name)
        if isinstance(found, Tag):
            return found
    for attr in ({"attrs": {"role": "main"}}, {"attrs": {"id": "__next"}}):
        found = soup.find(**attr)
        if isinstance(found, Tag):
            return found
    return soup.body or soup


def _is_boilerplate(el: Tag) -> bool:
    """Whether an element is site chrome rather than scheme content.

    Args:
        el: The element to test. Already known to be a :class:`Tag`, but its
            attribute dict may be ``None`` after an earlier ``decompose()``,
            since decomposing a node clears the attributes of its children.

    Returns:
        Whether the element is site chrome.
    """
    attrs = el.attrs
    if not attrs:
        return False
    classes = attrs.get("class") or []
    if isinstance(classes, str):
        classes = classes.split()
    for cls in classes:
        m = _CSS_MODULE_CLASS.match(cls)
        if m and m.group("base") in BOILERPLATE_CLASS_BASES:
            return True
    return False


def _is_leaf_block(el: Tag) -> bool:
    """Whether ``el`` contains no further block-level descendants."""
    return not el.find(BLOCK_TAGS)


def _leaf_line(el: Tag) -> str:
    """The line this block contributes, excluding text owned by nested blocks.

    A leaf contributes its whole subtree — nothing else can claim that text.
    A block that nests other blocks contributes **only its own direct text**,
    because the walk visits those nested blocks separately and a full-subtree
    read would count them twice.

    The second case is what keeps a label alive when it shares a div with a
    child block. Groww renders every labelled figure as
    ``<div>Expense ratio<div class="...infoIcon"><svg/></div></div>``, and
    requiring strict leaf-ness dropped the label while its sibling value
    survived — leaving a bare ``1.03%`` with nothing to say what it was, which
    is precisely the label-without-value failure P3's atomicity rule exists to
    prevent.

    Args:
        el: A block element from the walk.

    Returns:
        The normalised text, or ``""`` when the block holds nothing of its own
        (an icon-only wrapper, for instance).
    """
    if _is_leaf_block(el):
        return normalize_ws(el.get_text(" ", strip=True))
    own = " ".join(str(c) for c in el.children if isinstance(c, NavigableString))
    return normalize_ws(own)


HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def _heading_level(tag: Tag) -> int | None:
    """Heading depth of ``tag``, or ``None`` if it is not a heading."""
    if isinstance(tag, Tag) and tag.name in HEADING_TAGS:
        return int(tag.name[1])
    return None


def _walk_sections(root: Tag) -> tuple[list[tuple[Tag, list[str], list[Tag]]], int]:
    """Split a content root into sections by walking it in document order.

    Every leaf block is attributed to the nearest preceding content heading.
    Walking descendants rather than a heading's siblings is what makes this
    work: on these pages the fee block, NAV and exit-load figures sit inside
    ``fundDetails_fundDetailsContainer__*`` divs nested deep under a wrapper,
    with no heading of their own. A sibling walk never sees them — measured on
    ``hdfc_large_cap``, it recovered 0 of 13 target keywords against 10 of 13
    for this walk.

    A heading's own text is not counted as body text, so the title is never
    duplicated into its section. Content appearing *before* the first heading
    is the site nav on these pages; it is counted, not kept, and the caller is
    told how much was dropped rather than losing it silently.

    Args:
        root: The content subtree.

    Returns:
        ``(sections, dropped_preamble_chars)`` where each section is
        ``(heading, lines, containers)`` in document order.
    """
    # Prune chrome up front. Doing this inside the walk corrupts it:
    # decomposing a node invalidates the generator being iterated, and the
    # walk then returns 0 sections for every page.
    for el in root.find_all(True):
        if _is_boilerplate(el):
            el.decompose()

    heading_by_id: dict[int, Tag] = {}
    for h in root.find_all(HEADING_TAGS[:4]):
        title = normalize_ws(h.get_text(" ", strip=True))
        if not title or title.lower() in BOILERPLATE_HEADINGS:
            continue
        heading_by_id.setdefault(id(h), h)

    order: list[Tag] = []
    buckets: dict[int, tuple[list[str], list[Tag]]] = {}
    preamble = 0
    current: int | None = None

    for el in root.descendants:
        if not isinstance(el, Tag):
            continue
        if id(el) in heading_by_id:
            current = id(el)
            if current not in buckets:
                buckets[current] = ([], [])
                order.append(heading_by_id[current])
            continue
        if el.name not in BLOCK_TAGS:
            continue
        # Every block is visited, not just leaves: _leaf_line yields a nested
        # block's *own* text and leaves its children to be visited in turn.
        line = _leaf_line(el)
        if not line:
            continue
        if current is None:
            preamble += len(line)
            continue
        lines, containers = buckets[current]
        lines.append(line)
        containers.append(el)

    return [(h, *buckets[id(h)]) for h in order], preamble


def _trafilatura_lines(html: str) -> list[str]:
    """Last-ditch fallback extraction (see P1: weak, kept only for safety)."""
    try:
        text = trafilatura.extract(html, include_comments=False, include_tables=True)
    except Exception:  # noqa: BLE001 — a fallback must never crash the run
        logger.warning("trafilatura extraction failed", exc_info=True)
        return []
    return [ln for ln in (normalize_ws(x) for x in (text or "").splitlines()) if ln]


def extract(
    html: str, s: Settings | None = None, min_chars: int = MIN_MAIN_CHARS
) -> tuple[list[Section], str]:
    """Turn a fetched page into an ordered list of sections.

    Args:
        html: Raw page HTML.
        s: Settings; defaults to the process singleton.
        min_chars: Character floor below which the trafilatura fallback is
            attempted. Lowered by tests so a small fixture need not be padded
            to production size.

    Returns:
        ``(sections, extraction_mode)`` where mode is ``"bs4_heading_walk"`` for
        the primary path or ``"trafilatura"`` if the fallback rescued it.

        The mode names differ from implementation.md §P2, which listed them
        as ``"trafilatura" | "bs4_fallback"`` on the assumption that
        trafilatura was primary. P1 measured the opposite — trafilatura
        returns **0 of 13** target keywords on all 5 pages — so the names now
        describe the path that actually ran.

    Raises:
        ExtractionError: If neither path reaches ``min_chars``.
    """
    s = s or get_settings()
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(NON_CONTENT_TAGS):
        tag.decompose()

    root = _content_root(soup)
    walked, preamble_chars = _walk_sections(root)
    if preamble_chars:
        # Site nav, on these pages. Counted, not kept — but never silently.
        logger.info("dropped %d chars of pre-heading content (site nav)", preamble_chars)

    sections: list[Section] = []
    seen: set[tuple[str, str]] = set()

    for heading, lines, containers in walked:
        if not lines:
            continue
        body = "\n".join(lines)
        key = (normalize_ws(heading.get_text(" ", strip=True)).lower(), body)
        # Groww renders desktop and mobile DOM together, duplicating sections.
        if key in seen:
            continue
        seen.add(key)
        sections.append(
            Section(
                section_title=normalize_ws(heading.get_text(" ", strip=True)),
                heading_level=int(heading.name[1]),
                text=body,
                is_table=is_table_block(body, containers[0] if containers else None),
                order=len(sections),
            )
        )

    total = sum(len(x.text) for x in sections)
    if total >= min_chars:
        return sections, MODE_HEADING_WALK

    logger.warning(
        "heading walk yielded %d chars (< %d); trying trafilatura fallback",
        total, min_chars,
    )
    fallback = _trafilatura_lines(html)
    if sum(len(x) for x in fallback) < min_chars:
        raise ExtractionError(
            f"extraction produced {total} chars via heading walk and "
            f"{sum(len(x) for x in fallback)} via trafilatura, "
            f"both below the {min_chars} minimum"
        )
    first = root.find(["h1", "h2", "h3"])
    title = normalize_ws(first.get_text(" ", strip=True)) if first else "Page content"
    body = "\n".join(fallback)
    return [Section(title, 1, body, is_table_block(body), 0)], MODE_TRAFILATURA


# --------------------------------------------------------------------------
# AUM mislabelling repair
# --------------------------------------------------------------------------

#: The **scheme's** own AUM, in the summary strip beside NAV, expense ratio and
#: rating. This is the authoritative figure for "how big is this fund".
_SCHEME_AUM = re.compile(
    r"Fund\s+size\s*\(\s*AUM\s*\)\s*(?P<amt>[₹]?\s*[\d,]+(?:\.\d+)?\s*Cr)",
    re.IGNORECASE,
)

#: The **AMC's** aggregate AUM, in the "Fund house" section beside the AMC's
#: "#2 in India by total assets" rank. Never the scheme's figure.
_AMC_AUM = re.compile(
    r"Total\s+AUM\s*(?P<amt>[₹]?\s*[\d,]+(?:\.\d+)?\s*Cr)",
    re.IGNORECASE,
)

#: The prose sentence in the "About" block. On all five HDFC pages this carries
#: the **AMC's** total, not the scheme's — Groww renders one AMC-level template
#: string into every scheme page it hosts for that AMC. Measured across the five
#: pages in ``data/sources.csv``: 5/5 prose figures equal the AMC total and 0/5
#: equal the scheme's own, so this is a site-wide source defect, not a one-off.
_PROSE_AUM = re.compile(
    r"(?P<pre>Asset\s+Under\s+Management\s*\(?\s*AUM\s*\)?\s*(?:is|of)\s*)"
    r"(?P<amt>[₹]?\s*[\d,]+(?:\.\d+)?\s*Cr)",
    re.IGNORECASE,
)

#: Anything that is not a digit or a decimal point in an amount string.
_NON_NUMERIC = re.compile(r"[^\d.]")

#: Two amounts are "the same figure" when they agree to within one whole crore
#: **or** one part in a thousand, whichever is larger. The prose figure is
#: rendered to 0 decimal places while the summary strip carries 2, so an exact
#: string or float comparison never matches and the defect would survive.
AUM_MATCH_ABS_TOL_CR = 1.0
AUM_MATCH_REL_TOL = 0.001


def _cr_amount(text: str) -> float:
    """Parse a crore-denominated amount string into a float."""
    return float(_NON_NUMERIC.sub("", text))


def _amounts_match(left: float, right: float) -> bool:
    """Whether two crore amounts denote the same figure within rounding."""
    return abs(left - right) <= max(AUM_MATCH_ABS_TOL_CR, abs(right) * AUM_MATCH_REL_TOL)


def _repair_mislabeled_aum(
    sections: list[Section],
) -> tuple[list[Section], list[str]]:
    """Point prose AUM figures at the scheme's own AUM instead of the AMC's.

    A source page can contradict itself: the summary strip says
    ``Fund size (AUM) <scheme>``, the ``Fund house`` block says
    ``Total AUM <amc>``, and the ``About`` prose attributes ``<amc>`` to the
    scheme. Left alone, the bot answers "how big is this fund" with the AMC's
    assets under management — on this corpus 24.7x too large for
    ``hdfc_large_cap`` — and cites it confidently, because P7's numeric audit
    only checks that a figure appears in the retrieved context, and it does.

    The repair substitutes the figure the **same page** already publishes under
    the more specific label. It invents nothing and drops nothing: the sentence,
    its ``Latest NAV`` clause and its section all survive intact. A page whose
    prose already cites the scheme's own AUM is left byte-for-byte alone, so the
    function is idempotent and safe to re-run over cached documents.

    Args:
        sections: The document's sections.

    Returns:
        ``(sections, repaired_titles)``. ``sections`` is a new list only when
        something was repaired; ``repaired_titles`` names the sections touched,
        for the caller to log.
    """
    scheme_pairs: list[tuple[float, str]] = [
        (_cr_amount(m.group("amt")), m.group("amt"))
        for sec in sections
        for m in _SCHEME_AUM.finditer(sec.text)
    ]
    amc_values = [
        _cr_amount(m.group("amt"))
        for sec in sections
        for m in _AMC_AUM.finditer(sec.text)
    ]
    # Both labels must be present to prove a contradiction exists. Without them
    # there is nothing to compare against, and rewriting on a guess is worse
    # than reporting the source as written.
    if not scheme_pairs or not amc_values:
        return sections, []

    repaired: list[str] = []
    out: list[Section] = []

    for sec in sections:
        seen = [False]

        def _rewrite(match: re.Match[str]) -> str:
            amount = _cr_amount(match.group("amt"))
            # Already correct, or unrelated to either known figure.
            if any(_amounts_match(amount, s) for s, _ in scheme_pairs):
                return match.group(0)
            if not any(_amounts_match(amount, a) for a in amc_values):
                return match.group(0)
            seen[0] = True
            # Reuse the summary strip's own rendering, so precision and
            # formatting come from the source rather than from a reformat here.
            return match.group("pre") + scheme_pairs[0][1]

        text = _PROSE_AUM.sub(_rewrite, sec.text)
        if seen[0]:
            repaired.append(sec.section_title)
            out.append(replace(sec, text=text))
        else:
            out.append(sec)

    return out, repaired


def repair_document_aum(doc: Document) -> Document:
    """Apply :func:`_repair_mislabeled_aum` to a document, refreshing its hash.

    ``content_sha256`` covers the section text, so a repaired document must be
    re-hashed or the fingerprint would attest to content the corpus no longer
    holds.

    Args:
        doc: The document, freshly built or replayed from cache.

    Returns:
        The repaired document, or ``doc`` itself when nothing needed repairing.
    """
    sections, repaired = _repair_mislabeled_aum(doc.sections)
    if not repaired:
        return doc
    logger.warning(
        "%s: repaired mislabeled AUM in section(s) %s",
        doc.scheme_id,
        ", ".join(repaired),
    )
    all_text = " ".join(s.text for s in sections)
    return replace(
        doc,
        sections=sections,
        content_sha256=hashlib.sha256(normalize_ws(all_text).encode("utf-8")).hexdigest(),
    )


# --------------------------------------------------------------------------
# Build / serialise
# --------------------------------------------------------------------------


def build_document(
    spec: SourceSpec,
    html: str,
    final_url: str,
    mode: str,
    sections: list[Section] | None = None,
) -> Document:
    """Assemble a :class:`Document` from an extracted section list.

    Args:
        spec: The source row this page came from.
        html: Raw page HTML, used only if ``sections`` is not supplied.
        final_url: The URL after redirects — the citation target.
        mode: Extraction mode reported by :func:`extract`.
        sections: Pre-extracted sections, to avoid extracting twice.

    Returns:
        The document, with ``order`` renumbered from 0 and empty sections dropped.
    """
    if sections is None:
        sections, mode = extract(html)

    kept: list[Section] = []
    for sec in sections:
        # normalize_lines, NOT normalize_ws: a section body is one label:value
        # pair per line (see the module docstring), and collapsing the newlines
        # here glued every label to its neighbours. That silently defeated
        # P3's table-atomicity rule — with no line boundaries there was nothing
        # left to split on, and "Expense ratio" sat in one unbroken run with
        # seven other figures and no way to tell which number was the fee.
        body = normalize_lines(sec.text)
        if not body:
            continue
        kept.append(
            Section(
                section_title=normalize_ws(sec.section_title) or "Untitled",
                heading_level=sec.heading_level,
                text=body,
                is_table=sec.is_table,
                order=len(kept),
            )
        )

    all_text = " ".join(s.text for s in kept)
    doc = Document(
        scheme_id=spec.scheme_id,
        scheme_name=spec.scheme_name,
        category=spec.category,
        plan=spec.plan,
        # Store the *final* URL, not the one in the CSV: it is what a citation
        # link must point at once redirects have been followed.
        source_url=final_url,
        # date, not datetime: re-running on the same day must be idempotent.
        fetched_at=date.today().isoformat(),
        content_sha256=hashlib.sha256(normalize_ws(all_text).encode("utf-8")).hexdigest(),
        extraction_mode=mode,
        sections=kept,
    )
    return repair_document_aum(doc)


def write_documents(docs: list[Document], path: Path | None = None) -> None:
    """Write documents as JSONL, one compact object per line.

    Args:
        docs: Documents to persist, in order.
        path: Destination; defaults to :data:`DOCUMENTS_JSONL`. Resolved at
            call time, not import time, so tests can redirect it.
    """
    path = Path(path) if path is not None else DOCUMENTS_JSONL
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")


def load_documents(path: Path | None = None) -> list[Document]:
    """Read documents back from JSONL. Used by P3.

    Args:
        path: Source file; defaults to :data:`DOCUMENTS_JSONL`. A missing file
            yields an empty list, so a first run is not an error.

    Returns:
        The documents, with ``sections`` rebuilt into :class:`Section` objects.
    """
    path = Path(path) if path is not None else DOCUMENTS_JSONL
    if not path.exists():
        return []
    out: list[Document] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw["sections"] = [Section(**s) for s in raw.get("sections", [])]
        out.append(Document(**raw))
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_ingest(
    force: bool = False,
    fetch_only: bool = False,
    s: Settings | None = None,
    sources: list[SourceSpec] | None = None,
) -> IngestReport:
    """Fetch and extract every source, continuing past individual failures.

    Args:
        force: Re-fetch even when a cached document exists.
        fetch_only: Stop after writing ``documents.jsonl`` (P3 adds chunking).
        s: Settings; defaults to the process singleton.
        sources: Override the source list, for tests.

    Returns:
        An :class:`IngestReport`. A failed source is recorded in
        ``sources_failed``; it never aborts the run.
    """
    s = s or get_settings()
    sources = sources if sources is not None else read_sources()

    cached: dict[str, Document] = {} if force else {d.scheme_id: d for d in load_documents()}
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    docs: list[Document] = []
    failed: list[str] = []
    warnings: list[str] = []

    for i, spec in enumerate(sources):
        if spec.scheme_id in cached:
            # Reuse the fetched text, but take the metadata from the current
            # CSV so editing a scheme name does not require a re-fetch.
            old = cached[spec.scheme_id]
            # The AUM repair is applied here too, not only in build_document.
            # Cached documents skip the build path entirely, so a corpus written
            # before the repair existed would otherwise keep serving the defect
            # forever without any re-fetch. Idempotent, so re-running is safe.
            docs.append(
                repair_document_aum(
                    replace(
                        old,
                        scheme_name=spec.scheme_name,
                        category=spec.category,
                        plan=spec.plan,
                    )
                )
            )
            logger.info("%s: reusing cached document", spec.scheme_id)
            continue
        if i:
            time.sleep(s.fetch_delay_s)
        try:
            final_url, html, status = fetch(spec.url, s)
            (RAW_DIR / f"{spec.scheme_id}.html").write_text(html, encoding="utf-8")
            sections, mode = extract(html, s)
            doc = build_document(spec, html, final_url, mode, sections)
            if final_url != spec.url:
                warnings.append(f"{spec.scheme_id}: redirected {spec.url} -> {final_url}")
            docs.append(doc)
            logger.info("%s: HTTP %s, %d sections, %s", spec.scheme_id, status, len(doc.sections), mode)
        except (FetchError, ExtractionError) as exc:
            # Reported, never swallowed — and never fatal to the other sources.
            failed.append(spec.scheme_id)
            warnings.append(f"{spec.scheme_id}: {exc}")
            logger.error("ingest failed for %s: %s", spec.scheme_id, exc)

    write_documents(docs)
    INGEST_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report = IngestReport(
        sources_ok=len(docs),
        sources_failed=failed,
        documents=docs,
        max_fetched_at=max((d.fetched_at for d in docs), default=None),
        warnings=warnings,
    )
    INGEST_REPORT_JSON.write_text(
        json.dumps(
            {
                "sources_ok": report.sources_ok,
                "sources_failed": report.sources_failed,
                "max_fetched_at": report.max_fetched_at,
                "warnings": report.warnings,
                "documents": [
                    {
                        "scheme_id": d.scheme_id,
                        "source_url": d.source_url,
                        "fetched_at": d.fetched_at,
                        "sections": len(d.sections),
                        "extraction_mode": d.extraction_mode,
                        "content_sha256": d.content_sha256[:12],
                    }
                    for d in docs
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not fetch_only:
        from app.chunking import chunk_all, write_chunks
        from app.embedding import embed_texts
        from app.store import upsert_chunks
        # load documents we just wrote
        docs_written = load_documents()
        chunks, violations = chunk_all(docs_written)
        write_chunks(chunks, CHUNKS_JSONL)
        if violations:
            warnings.append(f"chunking violations: {len(violations)}")
            logger.warning("chunking produced %d violations", len(violations))
        embed_texts_list = [c.embed_text for c in chunks]
        embeddings = embed_texts(embed_texts_list)
        written = upsert_chunks(chunks, embeddings, force=force)
        logger.info("embedded %d chunks, upserted %d to Chroma", len(chunks), written)

    return report


def main() -> None:
    """CLI entry point: ``python -m app.ingest [--force] [--fetch-only]``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Ingest the 5 source scheme pages.")
    ap.add_argument("--force", action="store_true", help="re-fetch even if cached")
    ap.add_argument(
        "--fetch-only",
        action="store_true",
        help="stop after writing documents.jsonl (no chunking/embedding yet)",
    )
    args = ap.parse_args()

    report = run_ingest(force=args.force, fetch_only=args.fetch_only)

    print(f"\nsources ok    : {report.sources_ok}")
    print(f"sources failed: {report.sources_failed or 'none'}")
    print(f"fetched_at    : {report.max_fetched_at}")
    for w in report.warnings:
        print(f"  WARNING {w}")
    print(f"\n{'scheme_id':<26}{'sections':>9}{'tables':>8}{'chars':>9}  mode")
    for d in report.documents:
        chars = sum(len(x.text) for x in d.sections)
        print(
            f"{d.scheme_id:<26}{len(d.sections):>9}"
            f"{sum(x.is_table for x in d.sections):>8}{chars:>9}  {d.extraction_mode}"
        )
    if args.fetch_only:
        print(f"\nwrote {DOCUMENTS_JSONL}  (chunking arrives in P3)")
    return 0 if not report.sources_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
