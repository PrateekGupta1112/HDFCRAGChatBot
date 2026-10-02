"""Regression tests for citation fidelity — P4 store identity, P7 enforcement.

Every test here corresponds to a defect observed on a live Groq run on
2026-10-02, where the bot held a correct fact in context and then either threw
it away or served the wrong number:

1. ``upsert_chunks`` keyed on ``content_sha256``, a hash of the chunk **body
   alone**. Boilerplate that scheme pages share verbatim collided, and 18 of 131
   chunks were silently dropped — including the ``Understand terms``
   expense-ratio definition for four of the five schemes. Retrieval filters on
   ``scheme_id``, so that text had been chunked and then made unreachable.

2. ``_URL_SCAN_RE`` stopped only at ASCII brackets, so the full-width ``】``
   that gpt-oss wraps citations in stayed glued to the URL. The whitelist
   comparison failed and a correct answer was failed closed to NOT_FOUND.

3. The prompt numbers context blocks ``[1] … [n]`` while also asking for one
   source link. gpt-oss resolved that ambiguity by citing ``【5】``, which no
   part of enforcement understood, so 3 of 4 live runs of a working question
   were discarded.

Fixtures use obviously synthetic values. No real scheme names, no real figures.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.chunking import Chunk
from app.generation import enforce
from app.retrieval import RetrievedChunk, RetrievalResult
from app.store import upsert_chunks
from app.textutils import extract_urls, normalize_url

SYNTHETIC_URL_A = "https://example.invalid/synthetic-fund-a-direct-growth"
SYNTHETIC_URL_B = "https://example.invalid/synthetic-fund-b-direct-growth"

#: Shared boilerplate, byte-identical across schemes. This is what collided.
SHARED_BODY = "Expense ratio A fee payable to a mutual fund house for managing the assets."
#: Distinct per scheme, so it never collided.
UNIQUE_BODY = "Fund size (AUM) 11,111.11 Cr"


def _chunk(chunk_id: str, scheme_id: str, body: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        scheme_id=scheme_id,
        scheme_name=f"Synthetic Fund {scheme_id[-1].upper()}",
        category="Large Cap",
        plan="Direct Growth",
        source_url=SYNTHETIC_URL_A if scheme_id.endswith("a") else SYNTHETIC_URL_B,
        fetched_at="2026-01-01",
        section="Understand terms",
        chunk_index=0,
        content_sha256=chunk_id,
        display_text=body,
        embed_text=f"[Scheme: {scheme_id}]\n{body}",
        is_table=False,
    )


def _result(texts: list[str], url: str = SYNTHETIC_URL_A) -> RetrievalResult:
    chunks = [
        RetrievedChunk(
            rank=i,
            chunk_id=f"c{i}",
            text=t,
            section="Understand terms",
            scheme_name="Synthetic Fund",
            source_url=url,
            fetched_at="2026-01-01",
            similarity=0.9 - i * 0.1,
            mmr_score=None,
        )
        for i, t in enumerate(texts, start=1)
    ]
    return RetrievalResult(
        query="q", filtered_scheme_id="synthetic_a", chunks=chunks,
        max_similarity=chunks[0].similarity if chunks else 0.0, is_empty=False,
    )


def _embed(n: int) -> np.ndarray:
    v = np.zeros((n, 384), dtype=np.float32)
    v[:, 0] = 1.0
    return v


def _fake_settings(tmp_path):
    """Settings stand-in for store tests. Chroma rejects names under 3 chars."""
    return type(
        "S",
        (),
        {"chroma_path": str(tmp_path / "chroma"), "chroma_collection": "synthetic_collection"},
    )()


# --------------------------------------------------------------------------
# 1. Store identity: nothing may be dropped
# --------------------------------------------------------------------------


def test_chunks_sharing_a_body_are_all_stored(tmp_path, monkeypatch) -> None:
    """Identical boilerplate from five schemes must yield five stored chunks.

    Keying on the body hash stored one and discarded four. The discarded four
    were then unreachable, because retrieval filters on scheme_id.
    """
    monkeypatch.setattr("app.store.get_settings", lambda: _fake_settings(tmp_path))
    chunks = [_chunk(f"id_{i}", f"synthetic_{i}", SHARED_BODY) for i in range(5)]

    assert upsert_chunks(chunks, _embed(5), force=True) == 5

    from app.store import get_collection
    got = get_collection().get(include=["metadatas"])
    assert len(got["ids"]) == 5
    assert {m["scheme_id"] for m in got["metadatas"]} == {f"synthetic_{i}" for i in range(5)}


def test_distinct_bodies_across_schemes_are_all_stored(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.store.get_settings", lambda: _fake_settings(tmp_path)
    )
    chunks = [_chunk(f"id_{i}", f"synthetic_{i}", f"{UNIQUE_BODY} scheme {i}") for i in range(5)]
    assert upsert_chunks(chunks, _embed(5), force=True) == 5


def test_every_schemes_expense_ratio_chunk_is_retrievable(tmp_path, monkeypatch) -> None:
    """The end-to-end property: a scheme-filtered query must find its own chunk."""
    monkeypatch.setattr(
        "app.store.get_settings", lambda: _fake_settings(tmp_path)
    )
    chunks = [_chunk(f"id_{i}", f"synthetic_{i}", SHARED_BODY) for i in range(5)]
    upsert_chunks(chunks, _embed(5), force=True)

    from app.store import get_collection
    coll = get_collection()
    for i in range(5):
        found = coll.get(where={"scheme_id": f"synthetic_{i}"}, include=["metadatas"])
        assert len(found["ids"]) == 1, f"synthetic_{i} lost its chunk"


def test_stale_chunk_ids_are_removed(tmp_path, monkeypatch) -> None:
    """Identity-based staleness: a removed chunk_id must not linger."""
    monkeypatch.setattr(
        "app.store.get_settings", lambda: _fake_settings(tmp_path)
    )
    chunks = [_chunk(f"id_{i}", f"synthetic_{i}", f"body {i}") for i in range(3)]
    upsert_chunks(chunks, _embed(3), force=True)

    upsert_chunks(chunks[:2], _embed(2))

    from app.store import get_collection
    got = get_collection().get(include=["metadatas"])
    assert len(got["ids"]) == 2


def test_reupsert_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.store.get_settings", lambda: _fake_settings(tmp_path)
    )
    chunks = [_chunk(f"id_{i}", f"synthetic_{i}", SHARED_BODY) for i in range(3)]
    upsert_chunks(chunks, _embed(3), force=True)
    upsert_chunks(chunks, _embed(3))
    from app.store import get_collection
    assert len(get_collection().get(include=[])["ids"]) == 3


# --------------------------------------------------------------------------
# 2. Full-width closing bracket in a URL
# --------------------------------------------------------------------------


def test_url_wrapped_in_fullwidth_brackets_is_extracted_cleanly() -> None:
    text = f"the fee is 1.23%. 【{SYNTHETIC_URL_A}】"
    assert extract_urls(text) == [SYNTHETIC_URL_A]


@pytest.mark.parametrize("closer", ["】", "］", "〉", "》", "」", "』", "〕"])
def test_every_fullwidth_closer_is_stripped(closer: str) -> None:
    text = f"see {SYNTHETIC_URL_A}{closer}"
    assert extract_urls(text) == [SYNTHETIC_URL_A]


def test_normalize_url_tolerates_a_fullwidth_closer() -> None:
    assert normalize_url(f"{SYNTHETIC_URL_A}】") == normalize_url(SYNTHETIC_URL_A)


def test_bracketed_url_survives_enforcement() -> None:
    """The live failure: right fact, right URL, discarded anyway."""
    retr = _result(["Expense ratio 1.23%. Fund size 11,111.11 Cr"])
    raw = f"The expense ratio is 1.23%. 【{SYNTHETIC_URL_A}】"
    ans = enforce(raw, retr, "2026-01-01", "llm")
    assert ans.kind.value == "factual"
    assert ans.citations and ans.citations[0].url == SYNTHETIC_URL_A


# --------------------------------------------------------------------------
# 3. Chunk-index citation markers
# --------------------------------------------------------------------------


def test_chunk_marker_is_resolved_to_that_chunks_url() -> None:
    retr = _result(["Expense ratio 1.23%. Fund size 11,111.11 Cr"])
    ans = enforce("The expense ratio is 1.23%. 【1】", retr, "2026-01-01", "llm")
    assert ans.kind.value == "factual"
    assert ans.citations and ans.citations[0].url == SYNTHETIC_URL_A
    assert "resolved_chunk_markers:1" in ans.trace.postprocess_actions


def test_ascii_square_bracket_marker_is_resolved() -> None:
    retr = _result(["Expense ratio 1.23%. Fund size 11,111.11 Cr"])
    ans = enforce("The expense ratio is 1.23%. [1]", retr, "2026-01-01", "llm")
    assert ans.kind.value == "factual"


def test_marker_picks_the_right_chunk_not_always_the_first() -> None:
    retr = _result(["body one", "body two", "body three"], url=SYNTHETIC_URL_B)
    ans = enforce("The ratio is 1.23%. 【3】", retr, "2026-01-01", "llm")
    assert ans.kind.value == "factual"
    assert ans.citations and ans.citations[0].url == SYNTHETIC_URL_B
    assert "resolved_chunk_markers:3" in ans.trace.postprocess_actions


def test_marker_substitution_is_spaced_from_the_preceding_token() -> None:
    """Live output read "1.04%https://…" without this.

    The live model wrote the marker flush against the figure
    (``…is 1.04%【5】``), so the substituted URL has to bring its own leading
    space. Written here without a sentence-final full stop first, because a
    marker after ``.`` is spaced by that full stop instead and would test
    nothing.
    """
    retr = _result(["Expense ratio 1.23%. Fund size 11,111.11 Cr"])
    ans = enforce("The expense ratio is 1.23%【1】", retr, "2026-01-01", "llm")
    assert "%http" not in ans.text
    assert "1.23% http" in ans.text


def test_marker_after_a_full_stop_does_not_double_the_space() -> None:
    """`.【1】` must not become two spaces before the URL."""
    retr = _result(["Expense ratio 1.23%. Fund size 11,111.11 Cr"])
    ans = enforce("The expense ratio is 1.23%.【1】", retr, "2026-01-01", "llm")
    assert "  http" not in ans.text
    assert ans.text.count("  ") == 0


def test_out_of_range_marker_still_fails_closed() -> None:
    """An unresolvable marker must not silently become another chunk's citation."""
    retr = _result(["Expense ratio 1.23%. Fund size 11,111.11 Cr"])
    ans = enforce("The expense ratio is 1.23%. 【9】", retr, "2026-01-01", "llm")
    assert ans.kind.value == "not_found"
    assert not ans.citations


def test_marker_cannot_cite_an_unretrieved_chunk() -> None:
    """Marker resolution cannot invent a citation: the pool is retr.chunks."""
    retr = _result(["Expense ratio 1.23%. Fund size 11,111.11 Cr"])
    ans = enforce("The expense ratio is 1.23%. 【7】", retr, "2026-01-01", "llm")
    assert ans.citations == []


def test_explicit_url_wins_over_a_marker() -> None:
    retr = _result(["Expense ratio 1.23%. Fund size 11,111.11 Cr"])
    raw = f"The expense ratio is 1.23%. {SYNTHETIC_URL_A} 【1】"
    ans = enforce(raw, retr, "2026-01-01", "llm")
    assert ans.kind.value == "factual"
    assert ans.citations[0].url == SYNTHETIC_URL_A
    assert not any(a.startswith("resolved_chunk_markers") for a in ans.trace.postprocess_actions)


# --------------------------------------------------------------------------
# 4. One figure, two spellings (narrow no-break space before the unit)
# --------------------------------------------------------------------------


def test_narrow_no_break_space_before_percent_is_not_a_mismatch() -> None:
    """The live failure: right figure, discarded, answer body lost entirely.

    gpt-oss writes ``1.04`` + U+202F + ``%``; the corpus spells it ``1.04%``.
    The audit compared them verbatim, deleted the figure, and then deleted the
    sentence for being left mangled — so the reply was a citation line with no
    answer on it. Measured 2026-10-02, 1 of 4 live runs.
    """
    retr = _result(["Expense ratio 1.23%. Fund size 11,111.11 Cr"])
    raw = "The expense ratio is 1.23 %. {}".format(SYNTHETIC_URL_A)
    ans = enforce(raw, retr, "2026-01-01", "llm")
    assert ans.kind.value == "factual"
    assert "1.23" in ans.text
    assert not any(a.startswith("dropped_unsupported_number") for a in ans.trace.postprocess_actions)


def test_plain_space_before_percent_is_not_a_mismatch() -> None:
    retr = _result(["Expense ratio 1.23%. Fund size 11,111.11 Cr"])
    raw = "The expense ratio is 1.23 %. {}".format(SYNTHETIC_URL_A)
    ans = enforce(raw, retr, "2026-01-01", "llm")
    assert "1.23" in ans.text


def test_a_genuinely_different_figure_is_still_dropped() -> None:
    """Whitespace tolerance must not become numeric tolerance.

    The audit is a surface-form check; collapsing every space would let a
    fabricated ``99.99 %`` ride in on the strength of a real ``99.99`` elsewhere
    in a *different* scheme's chunk, which is the exact failure mode the audit
    exists to prevent.
    """
    retr = _result(["Expense ratio 1.23%. Fund size 11,111.11 Cr"])
    raw = "The expense ratio is 99.99 %. {}".format(SYNTHETIC_URL_A)
    ans = enforce(raw, retr, "2026-01-01", "llm")
    assert "99.99" not in ans.text
    assert any(a.startswith("dropped_unsupported_number") for a in ans.trace.postprocess_actions)


def test_canonical_number_preserves_the_unit_and_the_digits() -> None:
    from app.textutils import canonical_number

    assert canonical_number("1.04 %") == "1.04%"
    assert canonical_number("1.04 %") == "1.04%"
    assert canonical_number("₹39,933.36 Cr") == "₹39,933.36 Cr"
    assert canonical_number("1.03%") == "1.03%"
    assert canonical_number("") == ""


def test_marker_with_no_retrieved_chunks_fails_closed() -> None:
    retr = RetrievalResult(query="q", filtered_scheme_id=None, chunks=[],
                           max_similarity=0.0, is_empty=True)
    ans = enforce("The ratio is 1.23%. 【1】", retr, "2026-01-01", "llm")
    assert ans.kind.value == "not_found"


def test_marker_answer_still_passes_the_numeric_audit() -> None:
    """Resolving a marker must not smuggle an unverified figure past step 4."""
    retr = _result(["Expense ratio 1.23%. Fund size 11,111.11 Cr"])
    ans = enforce("The expense ratio is 99.99%. 【1】", retr, "2026-01-01", "llm")
    assert "99.99" not in ans.text
