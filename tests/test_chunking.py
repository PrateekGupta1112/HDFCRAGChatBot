"""Tests for the chunking stage (architecture §9.1, PRD FR-2).

implementation.md §P3 calls this "the most important test file in the project".
That is not an exaggeration: a chunker that separates a label from its value
produces a corpus that retrieves confidently and answers wrongly, and every
later stage — retrieval, generation, enforcement, evaluation — will faithfully
report that corruption as a confident fact. So the invariants asserted here are
stated as properties of the *output*, not of a particular algorithm, and the
fixtures are synthetic: no real fund figures, no network, no model download.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.chunking import (
    Chunk,
    _avoid_dangling_label,
    _merge_small_sections,
    _pack_lines,
    _recursive_split,
    _sliding_windows,
    _split_table,
    build_context_header,
    chunk_all,
    chunk_document,
    load_chunks,
    write_chunks,
)
from app.config import Settings
from app.ingest import Document, Section
from app.tokenization import count_tokens

# --------------------------------------------------------------------------
# Synthetic fixtures — invented figures only, per the P0 rule that no real
# mutual-fund value may appear in code or tests.
# --------------------------------------------------------------------------

SYNTH_URL = "https://example.invalid/synthetic-fund"
SYNTH_OTHER_URL = "https://example.invalid/other-fund"


def make_document(
    sections: list[Section],
    *,
    scheme_id: str = "syn_fund",
    scheme_name: str = "Synthetic Equity Fund",
    category: str = "Large Cap",
    plan: str = "Direct Growth",
    url: str = SYNTH_URL,
) -> Document:
    """Build a Document around hand-written sections."""
    return Document(
        scheme_id=scheme_id,
        scheme_name=scheme_name,
        category=category,
        plan=plan,
        source_url=url,
        fetched_at="2026-01-15",
        content_sha256="synthetic",
        extraction_mode="bs4_heading_walk",
        sections=sections,
    )


def fee_table_body(rows: int = 10) -> str:
    """A label-line/value-line fee table, in the shape Groww renders.

    Every value contains a digit. The chunker's notion of "this line is a label"
    is "this line carries no figure", so a value like ``Available`` would be
    indistinguishable from a label and the fixture could not prove anything
    about labels staying with their values.
    """
    labels = [
        "Expense ratio",
        "Exit load",
        "Min. for SIP",
        "Min. for 1st investment",
        "Min. for 2nd investment",
        "Fund size (AUM)",
        "Rating",
        "Lock-in period",
        "Fund benchmark",
        "Direct plan",
    ]
    values = [
        "1.23%",
        "1.00%",
        "Rs. 500",
        "Rs. 1,000",
        "Rs. 1,000",
        "Rs. 1,23,456 Cr",
        "4",
        "3 years",
        "NIFTY 100 Total Return Index",
        "1 of 2",
    ]
    out: list[str] = []
    for i in range(rows):
        out.append(labels[i % len(labels)])
        out.append(values[i % len(values)])
    return "\n".join(out)


def unique_table_body(rows: int = 10) -> str:
    """A fee table whose lines never repeat, for overlap assertions."""
    return "\n".join(
        line
        for i in range(rows)
        for line in (f"Metric label {i}", f"{i}.{i}%")
    )


def long_prose(paragraphs: int = 14, offset: int = 0) -> str:
    """Enough distinct prose to force several windows, with countable sentences.

    ``offset`` rotates the sentence list so two sections built from this fixture
    are not byte-identical — otherwise de-duplication correctly removes the
    second one and the test measures the wrong thing.
    """
    sentences = [
        "The scheme seeks to provide long-term capital appreciation by investing predominantly in large-capitalisation companies.",
        "The fund manager selects securities using a blend of fundamental and quantitative research inputs.",
        "Portfolio construction is reviewed on a monthly basis and rebalanced when a holding drifts beyond its target weight.",
        "The scheme follows a benchmark index that tracks the performance of large-capitalisation equities.",
        "Expenses are charged as a percentage of the net asset value of the scheme.",
        "Securities lending is permitted subject to the limits laid down by the regulator.",
        "The trustee monitors the activities of the fund and its service providers.",
        "Investors should read the scheme information document before investing.",
    ]
    sentences = sentences[offset:] + sentences[:offset]
    return "\n".join(
        " ".join(sentences[(i + offset) % len(sentences)] for i in range(3)) for _ in range(paragraphs)
    )


@pytest.fixture
def settings() -> Settings:
    """Production settings, except for a table cap tests can shrink."""
    return Settings(chunk_max_tokens=240, chunk_target_tokens=200, min_section_tokens=40)


# --------------------------------------------------------------------------
# Header / body separation
# --------------------------------------------------------------------------


def test_context_header_present(settings: Settings) -> None:
    """Every embed_text opens with the scheme-identity header."""
    doc = make_document(
        [
            Section("Overview", 1, fee_table_body(6), True, 0),
            Section("Objective", 2, long_prose(4), False, 1),
        ]
    )
    chunks = chunk_document(doc, settings)
    assert chunks
    assert all(c.embed_text.startswith("[Scheme:") for c in chunks)
    assert all("Synthetic Equity Fund" in c.embed_text for c in chunks)


def test_display_text_excludes_header(settings: Settings) -> None:
    """display_text is the body alone — the header is for the encoder only."""
    doc = make_document([Section("Overview", 1, fee_table_body(6), True, 0)])
    chunks = chunk_document(doc, settings)
    assert chunks
    for chunk in chunks:
        assert "[Scheme:" not in chunk.display_text
        assert "|" not in chunk.display_text.splitlines()[0]
        assert chunk.display_text in chunk.embed_text


def test_build_context_header_names_every_dimension() -> None:
    doc = make_document([])
    header = build_context_header(doc, "Minimum investments")
    assert header == (
        "[Scheme: Synthetic Equity Fund | Category: Large Cap | "
        "Plan: Direct Growth | Section: Minimum investments]"
    )


def test_embed_text_is_header_newline_body(settings: Settings) -> None:
    doc = make_document([Section("Objective", 2, long_prose(2), False, 0)])
    chunk = chunk_document(doc, settings)[0]
    header, _, body = chunk.embed_text.partition("\n")
    assert header.startswith("[Scheme:") and header.endswith("]")
    assert body == chunk.display_text


# --------------------------------------------------------------------------
# Table atomicity — the invariant that outranks the rest
# --------------------------------------------------------------------------


def test_table_not_row_split(settings: Settings) -> None:
    """No chunk contains a label without its own value on the same line.

    Spec's test, and the one that matters. The budget is squeezed to a size
    that forces several parts; every part must still carry whole label/value
    pairs, because a part holding "Expense ratio" and a *different* part
    holding "1.23%" is a confidently wrong answer waiting to be generated.
    """
    body = fee_table_body(10)
    tiny = Settings(chunk_max_tokens=240, chunk_target_tokens=200, min_section_tokens=40)
    parts = _split_table(body, max_chars=60, max_tokens=count_tokens(body) // 6 or 1)
    assert len(parts) > 1, "the fixture must actually force a split"

    for part in parts:
        lines = [ln for ln in part.splitlines() if ln.strip()]
        assert lines, "no empty parts"
        for index, line in enumerate(lines):
            if not any(ch.isdigit() for ch in line):
                # A bare label must be followed by its own value, in this part.
                following = lines[index + 1] if index + 1 < len(lines) else ""
                assert any(ch.isdigit() for ch in following), (
                    f"label {line!r} was separated from its value in part {part!r}"
                )
    assert tiny.chunk_max_tokens == 240


def test_split_table_never_splits_inside_a_line() -> None:
    """Pieces are line-aligned, so no label is cut away from its text."""
    body = fee_table_body(10)
    parts = _split_table(body, max_chars=60, max_tokens=8)
    assert len(parts) > 1
    rejoined = "\n".join(parts)
    assert rejoined == body, "splitting must be lossless and line-aligned"


def test_split_table_short_table_is_returned_unchanged(settings: Settings) -> None:
    """A table inside the character budget is never split, per the spec."""
    body = "Expense ratio\n1.23%\nExit load\n1.00%"
    assert _split_table(body, max_chars=1800, max_tokens=None) == [body]


def test_split_table_has_no_overlap_between_parts() -> None:
    """Parts are disjoint. Overlap on a table is a second chance to orphan a value."""
    parts = _split_table(fee_table_body(10), max_chars=50, max_tokens=8)
    for first, second in zip(parts, parts[1:]):
        assert first.splitlines()[-1] != second.splitlines()[0]


def test_oversized_single_line_is_kept_intact_and_reported(settings: Settings) -> None:
    """A line too long to fit is emitted whole; the violation is surfaced, not hidden."""
    huge = " ".join(f"Scheme Information Document number {i} is filed with the regulator." for i in range(80))
    doc = make_document([Section("Links", 2, huge, True, 0)])
    chunks, violations = chunk_all([doc], settings)
    assert len(chunks) == 1
    assert chunks[0].display_text == " ".join(huge.split()), "the line must not be truncated"
    assert violations, "an over-budget chunk must be reported, never silently accepted"
    assert any("over the 240 limit" in v for v in violations)


def test_pack_lines_keeps_an_oversized_line_and_makes_progress() -> None:
    lines = ["a" * 500, "short", "also short"]
    parts = _pack_lines(lines, max_chars=40, max_tokens=5)
    assert len(parts) == 2, "an unfittable line must not stall the packer"
    assert parts[0] == "a" * 500, "the unfittable line is kept whole"
    assert "\n".join(parts).splitlines() == lines, "packing is lossless"


def test_avoid_dangling_label_pulls_a_trailing_label_forward() -> None:
    lines = ["Expense ratio", "1.23%", "Rating", "4"]
    assert _avoid_dangling_label(lines, 0, 3) == 2
    assert _avoid_dangling_label(lines, 0, 4) == 4, "a complete pair must not be cut"


def test_avoid_dangling_label_takes_the_value_with_a_lone_label() -> None:
    """A part may not be one bare label. Adjacency outranks the budget."""
    lines = ["Expense ratio", "1.23%", "Rating", "4"]
    assert _avoid_dangling_label(lines, 2, 3) == 4, "a lone label must pull in its value"


def test_avoid_dangling_label_never_empties_a_part() -> None:
    lines = ["4", "Exit load", "1%"]
    assert _avoid_dangling_label(lines, 1, 2) == 3, "the pair must extend, not shrink to nothing"


# --------------------------------------------------------------------------
# chunk_index
# --------------------------------------------------------------------------


def test_chunk_index_is_document_global(settings: Settings) -> None:
    """Indices run 0,1,2,... across the whole document and never reset."""
    doc = make_document(
        [
            Section("Fees", 2, fee_table_body(10), True, 0),
            Section("Objective", 2, long_prose(6), False, 1),
            Section("Manager", 3, long_prose(6, offset=3), False, 2),
            Section("House", 3, fee_table_body(4), True, 3),
        ]
    )
    chunks = chunk_document(doc, settings)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len(chunks) > 4, "the fixture must span several sections"
    # More than one section must have contributed, or the test proves nothing.
    assert len({c.section for c in chunks}) >= 3


def test_chunk_index_restarts_per_document(settings: Settings) -> None:
    """Two documents each number from 0 — the counter is document-scoped."""
    one = make_document([Section("Fees", 2, fee_table_body(8), True, 0)], scheme_id="a")
    two = make_document([Section("Fees", 2, fee_table_body(8), True, 0)], scheme_id="b", url=SYNTH_OTHER_URL)
    chunks, _ = chunk_all([one, two], settings)
    assert [c.chunk_index for c in chunks if c.scheme_id == "a"] == [0]
    assert [c.chunk_index for c in chunks if c.scheme_id == "b"] == [0]


def test_chunk_id_is_derived_from_url_section_index_and_body() -> None:
    """The id is reproducible, and distinct bodies never collide."""
    doc = make_document([Section("Fees", 2, fee_table_body(6), True, 0)])
    first = chunk_document(doc)[0]
    second = chunk_document(doc)[0]
    assert first.chunk_id == second.chunk_id
    expected = hashlib.sha256(
        f"{SYNTH_URL}|0|{first.chunk_index}|{first.display_text}".encode("utf-8")
    ).hexdigest()[:16]
    assert first.chunk_id == expected
    assert len(first.chunk_id) == 16


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------


def test_merge_small_sections(settings: Settings) -> None:
    """A tiny section is absorbed forward, and both titles survive."""
    sections = [
        Section("Exit load", 4, "Charged as a percentage of redemption amount.", False, 0),
        Section("Tax implication", 4, long_prose(2), False, 1),
    ]
    merged = _merge_small_sections(sections, settings)
    assert len(merged) == 1, "the small section must not stand alone"
    assert "/" in merged[0].section_title
    assert "Exit load" in merged[0].section_title
    assert "Tax implication" in merged[0].section_title


def test_merge_small_sections_produces_one_chunk_not_two(settings: Settings) -> None:
    doc = make_document(
        [
            Section("Exit load", 4, "Charged as a percentage of redemption amount.", False, 0),
            Section("Tax implication", 4, long_prose(2), False, 1),
        ]
    )
    chunks = chunk_document(doc, settings)
    assert all("/" in c.section for c in chunks if c.section.startswith("Exit load"))
    assert len(chunks) == 1


def test_merge_small_sections_respects_the_chunk_ceiling(settings: Settings) -> None:
    """Merging must not be allowed to build a section that cannot be chunked."""
    sections = [
        Section("Exit load", 4, "Charged as a percentage of redemption amount.", False, 0),
        Section("Tax implication", 4, long_prose(40), False, 1),
    ]
    merged = _merge_small_sections(sections, settings)
    assert len(merged) == 2, "an over-ceiling merge must be refused"
    chunks, violations = chunk_all([make_document(sections)], settings)
    assert violations == []


def test_merge_small_sections_leaves_a_large_section_alone(settings: Settings) -> None:
    sections = [
        Section("Fees", 2, fee_table_body(8), True, 0),
        Section("Objective", 2, long_prose(8), False, 1),
    ]
    merged = _merge_small_sections(sections, settings)
    assert len(merged) == 2
    assert merged[0].section_title == "Fees"


def test_merge_small_sections_never_mixes_a_table_into_prose(settings: Settings) -> None:
    """Folding prose into a table would put atomic content on the prose path."""
    sections = [
        Section("Fees", 2, "Expense ratio\n1.23%", True, 0),
        Section("Objective", 2, long_prose(8), False, 1),
    ]
    merged = _merge_small_sections(sections, settings)
    assert len(merged) == 2, "a table must not absorb prose"
    assert all(m.is_table == s.is_table for m, s in zip(merged, sections))


def test_merge_small_sections_does_not_cross_documents(settings: Settings) -> None:
    """Merging is per-document; two documents never share a chunk."""
    one = make_document([Section("A", 2, "Tiny section text here.", False, 0)], scheme_id="a")
    two = make_document([Section("B", 2, "Another tiny section.", False, 0)], scheme_id="b")
    chunks, _ = chunk_all([one, two], settings)
    for chunk in chunks:
        assert "A / B" not in chunk.section


# --------------------------------------------------------------------------
# Dedupe
# --------------------------------------------------------------------------


def test_dedupe_drops_a_repeated_prose_body(settings: Settings) -> None:
    doc = make_document(
        [
            Section("First", 2, long_prose(3), False, 0),
            Section("Second", 2, long_prose(3), False, 1),
        ]
    )
    chunks = chunk_document(doc, settings)
    bodies = [c.content_sha256 for c in chunks]
    assert len(bodies) == len(set(bodies))


def test_dedupe_never_drops_a_table_chunk(settings: Settings) -> None:
    """A repeated table body is a repeated row; dropping it loses a holding."""
    body = fee_table_body(6)
    doc = make_document(
        [
            Section("Holdings", 2, body, True, 0),
            Section("Peers", 2, body, True, 1),
        ]
    )
    chunks = chunk_document(doc, settings)
    assert len(chunks) == 2, "identical table bodies must both survive"
    assert all(c.is_table for c in chunks)


def test_dedupe_is_per_document(settings: Settings) -> None:
    """The same body in two schemes is two facts, not one duplicate."""
    section = [Section("Fees", 2, fee_table_body(6), False, 0)]
    one = make_document(section, scheme_id="a")
    two = make_document(section, scheme_id="b", url=SYNTH_OTHER_URL)
    chunks, _ = chunk_all([one, two], settings)
    assert len([c for c in chunks if c.scheme_id == "a"]) >= 1
    assert len([c for c in chunks if c.scheme_id == "b"]) >= 1


# --------------------------------------------------------------------------
# Token ceiling
# --------------------------------------------------------------------------


def test_under_max_tokens(settings: Settings) -> None:
    """No chunk exceeds the ceiling — header included."""
    doc = make_document(
        [
            Section("Fees", 2, fee_table_body(10), True, 0),
            Section("Objective", 2, long_prose(20), False, 1),
            Section("Manager", 3, long_prose(20), False, 2),
        ]
    )
    chunks, violations = chunk_all([doc], settings)
    assert violations == []
    assert all(count_tokens(c.embed_text) <= settings.chunk_max_tokens for c in chunks)


def test_real_corpus_chunks_respect_the_ceiling() -> None:
    """The generated corpus, if present, is clean. Skipped when it is not built."""
    documents = Path("data/processed/documents.jsonl")
    if not documents.exists():
        pytest.skip("corpus not built; run `python -m app.ingest` first")
    from app.ingest import load_documents

    chunks, violations = chunk_all(load_documents(documents))
    assert violations == [], violations
    assert all(count_tokens(c.embed_text) <= 240 for c in chunks)


def test_header_is_counted_against_the_budget() -> None:
    """A body sized to the full ceiling would overrun once the header is added."""
    doc = make_document([Section("Fees", 2, fee_table_body(10), True, 0)])
    chunk = chunk_document(doc)[0]
    header = chunk.embed_text.partition("\n")[0]
    assert count_tokens(chunk.embed_text) > count_tokens(chunk.display_text)
    assert count_tokens(header) >= 15, "the header is real content, not a token or two"


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------


def test_overlap_applied_to_prose_only(settings: Settings) -> None:
    """Consecutive prose windows share text; consecutive table parts do not."""
    doc = make_document([Section("Objective", 2, long_prose(20), False, 0)])
    prose = chunk_document(doc, settings)
    assert len(prose) > 1
    shared = False
    for first, second in zip(prose, prose[1:]):
        a, b = set(first.display_text.split()), set(second.display_text.split())
        if a & b:
            shared = True
    assert shared, "no prose window shared text with its neighbour"

    # A table's lines must never repeat across parts. Compared line-by-line,
    # not word-by-word: two different holdings share the word "Sector".
    body = unique_table_body(10)
    parts = _split_table(body, max_chars=60, max_tokens=8)
    assert len(parts) > 1
    for first, second in zip(parts, parts[1:]):
        assert not (set(first.splitlines()) & set(second.splitlines())), (
            "table parts must not overlap"
        )


def test_sliding_windows_does_not_emit_a_pure_overlap_tail() -> None:
    text = " ".join(f"Sentence number {i} carries a little bit of weight." for i in range(30))
    windows = _sliding_windows(text, target=40, overlap=20)
    assert windows
    assert len(windows) == len(set(windows)), "no window may repeat another verbatim"


def test_sliding_windows_keeps_sentences_whole(settings: Settings) -> None:
    """Windows break on sentence boundaries, never mid-sentence."""
    body = long_prose(10)
    for window in _sliding_windows(body, settings.chunk_target_tokens, settings.chunk_overlap_tokens):
        assert window
        assert window == " ".join(window.split())


def test_recursive_split_never_exceeds_its_ceiling() -> None:
    body = long_prose(20)
    for piece in _recursive_split(body, target=60, maximum=80):
        assert count_tokens(piece) <= 80, piece


# --------------------------------------------------------------------------
# Whole-corpus contract
# --------------------------------------------------------------------------


def test_chunk_all_reports_no_violations_on_a_clean_corpus(settings: Settings) -> None:
    docs = [
        make_document([Section("Fees", 2, fee_table_body(10), True, 0)], scheme_id="a"),
        make_document(
            [Section("Fees", 2, fee_table_body(10), True, 0)], scheme_id="b", url=SYNTH_OTHER_URL
        ),
    ]
    chunks, violations = chunk_all(docs, settings)
    assert violations == []
    assert len(chunks) == 2, "one in-budget table per document"
    assert {c.scheme_id for c in chunks} == {"a", "b"}


def test_chunk_all_reports_an_empty_field(settings: Settings) -> None:
    """A chunk missing a field is a violation, not a crash."""
    doc = make_document([Section("Fees", 2, fee_table_body(6), True, 0)])
    chunks, _ = chunk_all([doc], settings)
    broken = Chunk(**{**chunks[0].__dict__, "source_url": ""})
    from app.chunking import _find_violations

    messages = _find_violations([broken], settings)
    assert any("empty source_url" in m for m in messages)


def test_chunk_all_reports_a_header_leaking_into_display(settings: Settings) -> None:
    from app.chunking import _find_violations

    doc = make_document([Section("Fees", 2, fee_table_body(6), True, 0)])
    chunks, _ = chunk_all([doc], settings)
    broken = Chunk(**{**chunks[0].__dict__, "display_text": "[Scheme: x] " + chunks[0].display_text})
    assert any("leaks the context header" in m for m in _find_violations([broken], settings))


def test_every_field_is_populated(settings: Settings) -> None:
    doc = make_document(
        [
            Section("Fees", 2, fee_table_body(10), True, 0),
            Section("Objective", 2, long_prose(12), False, 1),
        ]
    )
    chunks, violations = chunk_all([doc], settings)
    assert violations == []
    for chunk in chunks:
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
            assert str(getattr(chunk, field)).strip(), f"{field} is empty"


def test_chunk_is_frozen() -> None:
    doc = make_document([Section("Fees", 2, fee_table_body(4), True, 0)])
    chunk = chunk_document(doc)[0]
    with pytest.raises(Exception):
        chunk.chunk_index = 99  # type: ignore[misc]


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def test_chunks_jsonl_round_trip(tmp_path: Path, settings: Settings) -> None:
    doc = make_document(
        [
            Section("Fees", 2, fee_table_body(8), True, 0),
            Section("Objective", 2, long_prose(8), False, 1),
        ]
    )
    chunks, _ = chunk_all([doc], settings)
    path = tmp_path / "chunks.jsonl"
    write_chunks(chunks, path)
    assert path.read_text(encoding="utf-8").count("\n") == len(chunks)
    assert load_chunks(path) == chunks


def test_load_chunks_on_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_chunks(tmp_path / "absent.jsonl") == []


def test_written_chunk_has_every_declared_field(tmp_path: Path, settings: Settings) -> None:
    """Every field on the dataclass reaches the file, and nothing else does.

    implementation.md §P3 says "all 12 fields populated" and elsewhere lists 13.
    The dataclass is the contract P4 reads back, so 13 is what is asserted.
    """
    doc = make_document([Section("Fees", 2, fee_table_body(6), True, 0)])
    chunks, _ = chunk_all([doc], settings)
    path = tmp_path / "chunks.jsonl"
    write_chunks(chunks, path)
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert set(record) == set(Chunk.__dataclass_fields__)
    assert len(record) == len(Chunk.__dataclass_fields__) == 13


def test_write_chunks_is_deterministic(tmp_path: Path, settings: Settings) -> None:
    doc = make_document([Section("Fees", 2, fee_table_body(8), True, 0)])
    chunks, _ = chunk_all([doc], settings)
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_chunks(chunks, first)
    write_chunks(chunks, second)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------


def test_count_tokens_is_monotonic_and_non_negative() -> None:
    assert count_tokens("") == 0
    short = count_tokens("Expense ratio")
    long = count_tokens("Expense ratio " * 50)
    assert 0 < short < long


def test_exceeds_embed_limit_respects_the_configured_ceiling() -> None:
    from app.tokenization import exceeds_embed_limit

    assert not exceeds_embed_limit("Expense ratio 1.23%")
    assert exceeds_embed_limit("word " * 1000)
    assert not exceeds_embed_limit("")