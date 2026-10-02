"""Chunking — RAG data-ingestion stage 2 (architecture §9.1, PRD FR-2).

Turns each ingested :class:`~app.ingest.Document` into the retrieval units the
vector store holds. One function, :func:`chunk_all`, and every invariant below is
asserted by ``tests/test_chunking.py``.

**The rule that outranks all others: a label must never be separated from its
value.** A chunk holding ``Expense ratio`` and a different chunk holding ``1.12%``
is not a smaller answer, it is a confidently wrong one. So table blocks are
atomic: they are never split on a row boundary, never given overlap, and a part
is only ever drawn at a *line* boundary, where a label and its value are
already on the same line or on two lines that were adjacent in the source.

Everything else is subordinate to that. Overlap exists to stop a sentence being
cut in half — a sentence is not a label/value pair — so it is applied to prose
only. Where the two rules conflict, atomicity wins and the overlap is dropped.

Sizing is measured in **word-pieces of the embedding model itself**, not
characters (``app/tokenization.py``): all-MiniLM-L6-v2 truncates at 256 silently,
so an over-long chunk is not an error but a vector that fails to represent the
text it claims to. The context header is counted against the budget, which is
why the body is sized to ``chunk_max_tokens - len(header)`` rather than to
``chunk_max_tokens`` outright.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from app.config import Settings, get_settings
from app.ingest import Document, Section, load_documents
from app.textutils import normalize_lines, normalize_ws, split_sentences
from app.tokenization import count_tokens

logger = logging.getLogger(__name__)

CHUNKS_JSONL = Path("data/processed/chunks.jsonl")

#: Word-pieces consumed by the newline that joins the context header to the body.
HEADER_SEPARATOR_TOKENS = 1

#: Shortest a section may be before it is merged forward, in tokens. From
#: ``settings.min_section_tokens``.
MIN_SECTION_TOKENS = 40

#: Target and ceiling for one chunk's body, in tokens. From
#: ``settings.chunk_target_tokens`` / ``settings.chunk_max_tokens``.
CHUNK_TARGET_TOKENS = 200
CHUNK_MAX_TOKENS = 240

#: Carry-over between consecutive prose windows, in tokens. Prose only.
CHUNK_OVERLAP_TOKENS = 60

#: Hard ceiling on one table part, in characters. From ``settings.max_table_chars``.
MAX_TABLE_CHARS = 1800

#: Rows a single table part may carry. From ``settings.max_table_rows``.
#:
#: NOT enforced by :func:`_split_table`, deliberately. Splitting there is driven
#: by the character and token budgets, which are what actually bound a chunk and
#: what keep a label with its value. A row ceiling was tried as an extra bound
#: and had to be removed: it re-cut tables that already fit inside the budget,
#: blindly and without the dangling-label check, so it split a 124-token block
#: into 12/12/1 lines and left ``Rating`` in one chunk with ``5`` alone in the
#: next. ``max_table_rows`` remains a setting because architecture §9.1 names it;
#: the line-packed splitter has no use for it, because a "row" is not a unit this
#: function is permitted to cut on.
MAX_TABLE_ROWS = 12

#: Separators tried in order by :func:`_recursive_split`, coarsest first.
_SEPARATORS = ("\n\n", "\n", " ")


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit.

    All twelve fields are populated for every chunk written to
    ``chunks.jsonl``; ``is_table`` is the only one allowed to be ``False``.
    ``embed_text`` is what the encoder sees (header included, so the vector
    carries scheme and section identity — architecture §9.1) and ``display_text``
    is the body alone, which is what the LLM and the UI are shown.
    """

    chunk_id: str
    scheme_id: str
    scheme_name: str
    category: str
    plan: str
    section: str
    source_url: str
    fetched_at: str
    chunk_index: int
    is_table: bool
    content_sha256: str
    embed_text: str
    display_text: str


def build_context_header(doc: Document, section: str) -> str:
    """Build the identity line prefixed to every chunk body.

    Embedded, not merely stored. ``Expense ratio 1.12%`` on its own has no
    scheme identity in it, so the vector it produces cannot answer a question
    that names a fund; with the header the vector carries the scheme name and
    section alongside the fact. This is the single highest-leverage trick in the
    pipeline.

    Args:
        doc: The document the chunk came from.
        section: Section title, as it should appear in a citation.

    Returns:
        A single bracketed line, e.g.
        ``"[Scheme: HDFC Large Cap Fund Direct Growth | Category: Equity Large
        Cap | Plan: Direct Growth | Section: Minimum investments]"``.
    """
    return (
        f"[Scheme: {doc.scheme_name} | Category: {doc.category} | "
        f"Plan: {doc.plan} | Section: {section}]"
    )


def _body_budget(header: str, s: Settings) -> int:
    """Tokens available to a body once its header is accounted for.

    The header is 40-50 word-pieces on this corpus — well over the 25-35 that
    implementation.md §P3 assumed — so budgeting ``chunk_max_tokens`` for the
    body alone would overrun the encoder on nearly every chunk.

    Args:
        header: The context header that will precede the body.
        s: Settings.

    Returns:
        The body budget in tokens, never below 1.
    """
    return max(1, s.chunk_max_tokens - count_tokens(header) - HEADER_SEPARATOR_TOKENS)


def _sha256(text: str) -> str:
    """SHA-256 hex digest of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Splitting primitives
# --------------------------------------------------------------------------


def _is_bare_label(line: str) -> bool:
    """Whether a line names something without carrying a figure itself.

    Groww's labelled figures are a label line followed by a value line
    (``Rating`` / ``4``), so a line with no digit in it is a label and a line
    with one is a value.

    Args:
        line: One line of a table block.

    Returns:
        ``True`` if the line holds no figure.
    """
    return not any(ch.isdigit() for ch in line)


def _avoid_dangling_label(lines: list[str], start: int, end: int) -> int:
    """Move the split point so a label and its value cannot fall apart.

    Packing lines inside a budget is free to choose a cut that lands between a
    label and the value beneath it, which is the one boundary this module exists
    to protect. Observed on ``hdfc_elss``: ``Rating`` closed one chunk and the
    ``5`` that gives it meaning opened the next, leaving a chunk that retrieves
    perfectly on "what is the fund's rating" and answers with a bare word.

    Two adjustments, because a bad cut can strand a label at either edge:

    * **Extend** when the part would be a lone label whose value is next. Here
      the budget yields to adjacency: an over-budget part is reported by
      :func:`chunk_all`, whereas an orphaned label is a silent wrong answer.
    * **Shrink** when the part would end on a label whose value is next. Giving
      up one line is always safe — the part shrinks, so it stays inside its
      budget — and the label travels to the next part, which is where its value
      already is.

    Args:
        lines: Every line of the block.
        start: First line of the part being cut.
        end: Proposed cut, exclusive.

    Returns:
        A cut at, just before, or just after ``end``. Never shrinks a part to
        nothing.
    """
    if end >= len(lines):
        return end
    next_is_value = not _is_bare_label(lines[end])
    if end - start == 1 and _is_bare_label(lines[start]) and next_is_value:
        return end + 1
    if end - start > 1 and _is_bare_label(lines[end - 1]) and next_is_value:
        return end - 1
    return end


def _pack_lines(lines: list[str], max_chars: int, max_tokens: int) -> list[str]:
    """Greedily pack lines into parts inside both budgets, never splitting one.

    Each part is the longest run of consecutive lines whose joined text stays
    within ``max_chars`` **and** ``max_tokens``, shortened if needed so that no
    part ends on a label whose value sits in the next one. A single line that
    exceeds a budget on its own is emitted whole rather than broken, because a
    line is where the source put a label and its value together and cutting it is
    the one thing this module must not do. Such a part is over budget by
    construction; :func:`chunk_all` reports it rather than hiding it.

    Args:
        lines: Lines to pack, in order.
        max_chars: Character ceiling per part.
        max_tokens: Word-piece ceiling per part.

    Returns:
        The packed parts, in order. At least one, whenever ``lines`` is
        non-empty.
    """

    def fits(candidate: list[str]) -> bool:
        joined = "\n".join(candidate)
        return len(joined) <= max_chars and count_tokens(joined) <= max_tokens

    parts: list[str] = []
    start = 0
    while start < len(lines):
        # Binary search the longest run starting here that fits. `lo` starts one
        # line in so an over-budget single line still makes progress.
        lo, hi, best = start + 1, len(lines), start + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits(lines[start:mid]):
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        best = _avoid_dangling_label(lines, start, best)
        parts.append("\n".join(lines[start:best]))
        start = best
    return parts


def _split_table(
    text: str, max_chars: int = MAX_TABLE_CHARS, max_tokens: int | None = None
) -> list[str]:
    """Split a table block into atomic parts.

    Tables are never split on a row boundary and never overlapped. Splitting is
    only ever drawn at a line boundary, so a label and the value sitting on its
    line — or on the adjacent line Groww put it on — stay in the same part.

    ``max_chars`` alone cannot keep a chunk inside the encoder's limit: at this
    corpus's ~3.55 characters per word-piece, 1800 characters is roughly 507
    word-pieces, more than twice ``chunk_max_tokens``. ``max_tokens`` therefore
    also constrains the result, and the character cap remains a hard ceiling on
    top of it.

    Args:
        text: A table block, newline-separated.
        max_chars: Character ceiling per part. Defaults to ``MAX_TABLE_CHARS``.
        max_tokens: Word-piece ceiling per part. ``None`` means the character
            cap alone applies.

    Returns:
        One or more parts, in order, with no overlap between them.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    if max_tokens is None:
        if len(text) <= max_chars:
            return [text]
        max_tokens = count_tokens(text)

    return _pack_lines(lines, max_chars, max_tokens)


def _greedy_group(pieces: list[str], target: int, maximum: int, joiner: str) -> list[str]:
    """Pack ``pieces`` into runs of at most ``maximum`` tokens.

    Args:
        pieces: Units to pack, in order.
        target: Preferred size, used to end a run early.
        maximum: Hard ceiling.
        joiner: Separator used when re-joining a run.

    Returns:
        The packed runs, in order.
    """
    runs: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for piece in pieces:
        piece_tokens = count_tokens(piece)
        if current and current_tokens + piece_tokens > maximum:
            runs.append(joiner.join(current))
            current, current_tokens = [], 0
        current.append(piece)
        current_tokens += piece_tokens
        if current_tokens >= target:
            runs.append(joiner.join(current))
            current, current_tokens = [], 0
    if current:
        runs.append(joiner.join(current))
    return runs


def _word_runs(text: str, target: int, maximum: int) -> list[str]:
    """Last-resort split of text that no separator divides cleanly."""
    return _greedy_group(text.split(), target, maximum, " ")


def _recursive_split(text: str, target: int, maximum: int) -> list[str]:
    """Split prose into pieces of at most ``maximum`` tokens.

    Tries ``"\\n\\n"``, then ``"\\n"``, then ``" "``, stopping at the first level
    that actually divides the text into pieces small enough to pack. At that
    level the pieces are re-grouped up to ``target`` tokens, because the point of
    this stage is to find a *split point*, not to emit one chunk per line — a
    52-line fund-manager section must not become 52 chunks.

    Args:
        text: Prose body.
        target: Preferred piece size, in tokens.
        maximum: Hard ceiling per piece, in tokens.

    Returns:
        The pieces, in order.
    """
    text = normalize_ws(text)
    if not text:
        return []
    if count_tokens(text) <= maximum:
        return [text]
    for separator in _SEPARATORS:
        parts = [p for p in (normalize_ws(x) for x in text.split(separator)) if p]
        if len(parts) <= 1:
            continue
        if all(count_tokens(p) <= maximum for p in parts):
            return _greedy_group(parts, target, maximum, "\n")
    return _word_runs(text, target, maximum)


def _sliding_windows(text: str, target: int, overlap: int) -> list[str]:
    """Pack a piece into overlapping windows of roughly ``target`` tokens.

    Windows break on sentence boundaries and carry up to ``overlap`` tokens of
    the previous window's tail into the next one, so a sentence is never cut in
    half. A trailing window that adds nothing beyond pure overlap is dropped
    rather than emitted as a duplicate.

    This is applied to **prose only**. Tables go through :func:`_split_table`,
    which never overlaps: on a table, an overlap is not redundant context, it is
    a second chance to strand a value away from its label.

    Args:
        text: One piece from :func:`_recursive_split`.
        target: Preferred window size, in tokens.
        overlap: Carry-over from the previous window, in tokens.

    Returns:
        The windows, in order.
    """
    text = normalize_ws(text)
    if not text:
        return []
    sentences = split_sentences(text)
    if not sentences:
        return [text]

    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current and current_tokens + sentence_tokens > target:
            windows.append(" ".join(current))
            carry: list[str] = []
            carry_tokens = 0
            for previous in reversed(current):
                previous_tokens = count_tokens(previous)
                if carry_tokens + previous_tokens > overlap:
                    break
                carry.insert(0, previous)
                carry_tokens += previous_tokens
            current, current_tokens = carry, carry_tokens
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        tail = " ".join(current)
        if not windows or tail != windows[-1]:
            windows.append(tail)
    return windows


# --------------------------------------------------------------------------
# Section merging
# --------------------------------------------------------------------------


def _merge_small_sections(sections: list[Section], s: Settings | None = None) -> list[Section]:
    """Merge sections too short to stand alone into the one that follows.

    A 11-token ``Exit load`` heading produces a chunk that retrieves well and
    answers nothing. Merging it forward into the next section keeps it
    retrievable *with* the content it labels, at the cost of a slightly longer
    citation title — ``"Exit load / Tax implication"`` — which is exactly the
    trade the spec asks for.

    Two boundaries are respected. Merging never crosses a document, because
    :func:`chunk_document` is called per document and never sees another. And
    merging never crosses an ``is_table`` boundary, because a table block is
    atomic: folding prose into it would put it on the prose path, where the
    label/value protection no longer applies.

    Args:
        sections: One document's sections, in order.
        s: Settings; defaults to the process singleton.

    Returns:
        Sections with the same ``order`` values, some now holding joined text
        and a joined title.
    """
    s = s or get_settings()
    merged: list[Section] = []
    index = 0
    while index < len(sections):
        current = sections[index]
        title = current.section_title
        body = current.text
        is_table = current.is_table
        # Absorb forward while this section is still too small to stand alone.
        while count_tokens(body) < s.min_section_tokens and index + 1 < len(sections):
            following = sections[index + 1]
            if following.is_table != is_table:
                break
            if count_tokens(f"{body}\n{following.text}") > s.chunk_max_tokens:
                break
            title = f"{title} / {following.section_title}"
            body = f"{body}\n{following.text}"
            index += 1
        merged.append(replace(current, section_title=title, text=body))
        index += 1
    return merged


# --------------------------------------------------------------------------
# Chunk construction
# --------------------------------------------------------------------------


def _make_chunk(
    doc: Document, sec: Section, text: str, is_table: bool, index: int, s: Settings
) -> Chunk:
    """Build one :class:`Chunk` from a piece of section text.

    Args:
        doc: Source document.
        sec: The section the piece came from.
        text: The body. Whitespace-normalised, line structure preserved.
        is_table: Whether the piece came from a table block.
        index: Document-global running index (architecture §9.1).
        s: Settings.

    Returns:
        The assembled chunk.
    """
    header = build_context_header(doc, sec.section_title)
    body = normalize_lines(text)
    return Chunk(
        # implementation.md §P3 includes chunk_index here; architecture §9.1
        # omits it. Included, so that the two identical bodies a large holdings
        # table inevitably produces get distinct ids and the upsert in P4 does
        # not collapse them into one.
        chunk_id=_sha256(f"{doc.source_url}|{sec.order}|{index}|{body}")[:16],
        scheme_id=doc.scheme_id,
        scheme_name=doc.scheme_name,
        category=doc.category,
        plan=doc.plan,
        section=sec.section_title,
        source_url=doc.source_url,
        fetched_at=doc.fetched_at,
        chunk_index=index,
        is_table=is_table,
        content_sha256=_sha256(body),
        embed_text=f"{header}\n{body}",
        display_text=body,
    )


def _dedupe(pieces: list[tuple[Section, str, bool]]) -> list[tuple[Section, str, bool]]:
    """Drop repeated bodies within one document, keeping the first.

    Applied to *pieces* rather than to finished chunks, which is what keeps
    ``chunk_index`` gapless: de-duplicating after ids were assigned would leave
    a hole in the counter at every dropped duplicate, and implementation.md §P3
    requires it document-global **and** gapless. Removing the piece before the
    index is allocated avoids renumbering anything afterwards.

    Two guards, both deliberate:

    * Only *within* a document. The holdings tables legitimately repeat short
      bodies ("Nifty 50"/"NIFTY 100 Total Return Index") across funds, and each
      occurrence is a separate fact about a separate scheme.
    * Never a table. A table piece that repeats a body is a row that repeats —
      dropping it silently deletes a holding from the answer's evidence.

    Args:
        pieces: ``(section, text, is_table)`` triples from one document.

    Returns:
        The de-duplicated pieces, in order.
    """
    seen: set[str] = set()
    kept: list[tuple[Section, str, bool]] = []
    for section, text, is_table in pieces:
        if is_table:
            kept.append((section, text, is_table))
            continue
        digest = _sha256(normalize_lines(text))
        if digest in seen:
            logger.debug("dropped duplicate piece from section %s", section.section_title)
            continue
        seen.add(digest)
        kept.append((section, text, is_table))
    return kept


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def chunk_document(doc: Document, s: Settings | None = None) -> list[Chunk]:
    """Chunk one document.

    Merge small sections forward, then take one of two paths per section.
    Tables are packed whole and split only at line boundaries. Prose is split
    into pieces and windowed with overlap. Either way ``chunk_index`` is a single
    counter running across the **whole document** — architecture §9.1 makes it
    document-global, so a chunk's index identifies its position in the document
    rather than in its section.

    Args:
        doc: An ingested document.
        s: Settings; defaults to the process singleton.

    Returns:
        The document's chunks, ordered by ``chunk_index`` and gapless from 0.
    """
    s = s or get_settings()
    sections = _merge_small_sections(doc.sections, s)
    pieces: list[tuple[Section, str, bool]] = []

    for sec in sections:
        header = build_context_header(doc, sec.section_title)
        budget = _body_budget(header, s)

        if sec.is_table:
            pieces.extend((sec, part, True) for part in _split_table(sec.text, s.max_table_chars, budget))
            continue

        # The body budget, not chunk_max_tokens, bounds the prose path too:
        # the header is prepended to every chunk regardless of its kind, so
        # splitting prose to the full ceiling and then adding ~45 tokens of
        # header is how a 240-token ceiling becomes a 285-token chunk.
        window_target = min(s.chunk_target_tokens, budget)
        for part in _recursive_split(sec.text, window_target, budget):
            pieces.extend((sec, window, False) for window in _sliding_windows(part, window_target, s.chunk_overlap_tokens))

    # De-duplicate before the index is allocated, so the counter stays gapless.
    return [
        _make_chunk(doc, section, text, is_table, index, s)
        for index, (section, text, is_table) in enumerate(_dedupe(pieces))
    ]


def _find_violations(chunks: list[Chunk], s: Settings) -> list[str]:
    """Check every invariant that must hold for a chunk to be usable.

    Reported rather than asserted. A chunker that raises on an over-budget table
    row would take down the whole build over one pathological source line; the
    pipeline needs to finish and *say* which chunks are oversized, so the
    problem is visible and countable instead of fatal.

    Args:
        chunks: Every chunk produced.
        s: Settings.

    Returns:
        One message per violation; empty when the corpus is clean.
    """
    violations: list[str] = []
    for chunk in chunks:
        tokens = count_tokens(chunk.embed_text)
        if tokens > s.chunk_max_tokens:
            violations.append(
                f"{chunk.chunk_id} [{chunk.scheme_id} §{chunk.section}] "
                f"is {tokens} tokens, over the {s.chunk_max_tokens} limit "
                f"(is_table={chunk.is_table})"
            )
        if not chunk.embed_text.startswith("[Scheme:"):
            violations.append(f"{chunk.chunk_id} embed_text does not start with [Scheme:")
        if "[Scheme:" in chunk.display_text:
            violations.append(f"{chunk.chunk_id} display_text leaks the context header")
        for field in (
            "chunk_id",
            "scheme_id",
            "scheme_name",
            "category",
            "plan",
            "section",
            "source_url",
            "fetched_at",
            "content_sha256",
            "embed_text",
            "display_text",
        ):
            if not str(getattr(chunk, field)).strip():
                violations.append(f"{chunk.chunk_id} has an empty {field}")
    return violations


def chunk_all(
    docs: list[Document], s: Settings | None = None
) -> tuple[list[Chunk], list[str]]:
    """Chunk every document and verify the result.

    Args:
        docs: Ingested documents.
        s: Settings; defaults to the process singleton.

    Returns:
        ``(chunks, violations)``. ``violations`` is empty when every chunk
        satisfies the token ceiling, carries all twelve fields, embeds under a
        ``[Scheme:`` header and shows the body alone in ``display_text``.
    """
    s = s or get_settings()
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, s))
    violations = _find_violations(chunks, s)
    if violations:
        logger.error("chunking produced %d violations", len(violations))
    return chunks, violations


def write_chunks(chunks: list[Chunk], path: Path | None = None) -> None:
    """Write chunks as JSONL, one compact object per line.

    Args:
        chunks: Chunks to persist, in order.
        path: Destination; defaults to :data:`CHUNKS_JSONL`. Resolved at call
            time, not import time, so tests can redirect it.
    """
    path = Path(path) if path is not None else CHUNKS_JSONL
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path | None = None) -> list[Chunk]:
    """Read chunks back from JSONL. Used by P4.

    Args:
        path: Source file; defaults to :data:`CHUNKS_JSONL`. A missing file
            yields an empty list, so a first run is not an error.

    Returns:
        The chunks, in file order.
    """
    path = Path(path) if path is not None else CHUNKS_JSONL
    if not path.exists():
        return []
    return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    """``python -m app.chunking`` — chunk the corpus and report on it."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Chunk the ingested corpus.")
    parser.add_argument("--documents", type=Path, default=None, help="Input documents JSONL.")
    parser.add_argument("--out", type=Path, default=None, help="Output chunks JSONL.")
    args = parser.parse_args()

    s = get_settings()
    docs = load_documents(args.documents)
    if not docs:
        raise SystemExit("No documents found. Run `python -m app.ingest` first.")

    chunks, violations = chunk_all(docs, s)
    write_chunks(chunks, args.out)

    print(f"documents: {len(docs)}   chunks: {len(chunks)}   violations: {len(violations)}")
    for doc in docs:
        mine = [c for c in chunks if c.scheme_id == doc.scheme_id]
        if not mine:
            print(f"  {doc.scheme_id:<24}   0 chunks  <-- WARNING: nothing chunked")
            continue
        tables = sum(c.is_table for c in mine)
        sizes = [count_tokens(c.embed_text) for c in mine]
        print(
            f"  {doc.scheme_id:<24} {len(mine):>4} chunks  ({tables} table)  "
            f"tokens min={min(sizes)} max={max(sizes)} mean={sum(sizes) // len(sizes)}"
        )
    if violations:
        print(f"\n{len(violations)} VIOLATIONS:")
        for message in violations:
            print(f"  - {message}")
        raise SystemExit(1)
    print(f"\nwrote {len(chunks)} chunks to {args.out or CHUNKS_JSONL}")


if __name__ == "__main__":  # pragma: no cover
    main()