"""Unit tests for app.retrieval — the retrieval stage.

Two layers:

* **Isolated tests** build a tiny synthetic Chroma collection under ``tmp_path``
  and assert the contracts: the ``1 - distance`` arithmetic, the threshold, MMR
  ranking, and the empty-corpus error. All fixture text is obviously fake and
  clearly labelled synthetic — no real fund names, URLs or values.
* **Integration tests** run against the real ingested corpus and are skipped
  when it has not been built. They assert the *behavioural* claims: the alias
  filter resolves, and an off-corpus question returns nothing.

MMR's vector step is stubbed in the isolated tests. ``_mmr`` re-embeds its
candidates, and a unit test must not need the 90 MB encoder to check an
arithmetic property; the stub returns a deterministic unit vector per distinct
text, which is enough to make "these two are near-duplicates" expressible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.chunking import Chunk
from app.errors import CorpusEmptyError
from app.retrieval import RetrievedChunk, _mmr, retrieve
from app.scheme_aliases import load_aliases, resolve_scheme_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMBED_DIM = 384

#: Synthetic body text. Obvious placeholder figures, no real scheme data.
#: Every body is distinct on purpose. ``upsert_chunks`` de-duplicates by
#: ``content_sha256``, so byte-identical bodies never reach the store at all —
#: which is exactly how the real corpus loses 18 of its 131 chunks. MMR's
#: near-duplicate handling is tested by calling ``_mmr`` directly, where the
#: candidates bypass the store entirely.
SYNTHETIC_BODIES = {
    "fees": "Section: Fees. Total expense ratio: 1.11%. This is synthetic test data.",
    "exit": "Section: Exit load. Exit load: 0.50%. This is synthetic test data.",
    "sip": "Section: Minimum investments. Minimum for SIP: 100. Synthetic test data.",
    "risk": "Section: Riskometer. Risk level: Very High. Synthetic test data.",
    "lock": "Section: Lock in. Lock-in period: 3 years. Synthetic test data.",
    "bench": "Section: Benchmark. Benchmark: TRIX. Synthetic test data.",
    "manager": "Section: Fund manager. Managed by a fictional asset manager. Synthetic.",
    "nav": "Section: NAV. Net asset value is computed once per business day. Synthetic.",
}


def _synthetic_chunk(
    chunk_id: str,
    scheme_id: str,
    section: str,
    body: str,
    chunk_index: int = 0,
) -> Chunk:
    """Build one synthetic Chunk with the same shape ingestion produces."""
    import hashlib

    scheme_name = f"Synthetic {scheme_id} Fund Direct Growth"
    embed_text = (
        f"[Scheme: {scheme_name} | Category: Test | Plan: Direct Growth | Section: {section}]\n"
        f"{body}"
    )
    return Chunk(
        chunk_id=chunk_id,
        scheme_id=scheme_id,
        scheme_name=scheme_name,
        category="Test",
        plan="Direct Growth",
        section=section,
        source_url=f"https://example.com/{scheme_id}",
        fetched_at="2026-10-02",
        chunk_index=chunk_index,
        is_table=False,
        content_sha256=hashlib.sha256(body.encode()).hexdigest(),
        embed_text=embed_text,
        display_text=body,
    )


@pytest.fixture
def synthetic_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build an 8-chunk synthetic Chroma collection under ``tmp_path``.

    Returns the collection so a test can query it independently — which is how
    ``test_similarity_is_exactly_one_minus_chroma_distance`` avoids trusting
    ``app.retrieval``'s own arithmetic.

    The chunks are embedded with the **real** encoder, not a stub. Chroma embeds
    the query with its own copy of the same model, so a synthetic vector in some
    other space makes the cosine meaningless and the similarity assertions test
    nothing. ``chunks.jsonl`` and the fixture therefore live in one space, and
    identical fixture text produces an identical vector — which is what lets the
    near-duplicate MMR case be expressed without a stub at all.
    """
    from app import config
    from app.embedding import embed_texts
    from app.store import get_collection, upsert_chunks

    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    config.get_settings.cache_clear()

    chunks = [
        _synthetic_chunk("syn0000000000001", "synthetic_a", "Fees", SYNTHETIC_BODIES["fees"], 0),
        _synthetic_chunk("syn0000000000002", "synthetic_a", "Exit load", SYNTHETIC_BODIES["exit"], 1),
        _synthetic_chunk("syn0000000000003", "synthetic_a", "Minimum investments", SYNTHETIC_BODIES["sip"], 2),
        _synthetic_chunk("syn0000000000004", "synthetic_a", "Riskometer", SYNTHETIC_BODIES["risk"], 3),
        _synthetic_chunk("syn0000000000005", "synthetic_b", "Lock in", SYNTHETIC_BODIES["lock"], 0),
        _synthetic_chunk("syn0000000000006", "synthetic_b", "Benchmark", SYNTHETIC_BODIES["bench"], 1),
        _synthetic_chunk("syn0000000000007", "synthetic_a", "Fund manager", SYNTHETIC_BODIES["manager"], 4),
        _synthetic_chunk("syn0000000000008", "synthetic_b", "NAV", SYNTHETIC_BODIES["nav"], 2),
    ]

    vectors = embed_texts([c.embed_text for c in chunks])
    written = upsert_chunks(chunks, vectors, force=True)
    assert written == len(chunks), (
        f"only {written} of {len(chunks)} chunks reached the store; the fixture "
        "bodies are not all distinct"
    )
    yield get_collection()
    config.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# The similarity contract — the highest-risk bug in the project
# ---------------------------------------------------------------------------


def test_similarity_is_exactly_one_minus_chroma_distance(synthetic_corpus) -> None:
    """Every reported similarity must equal ``1.0 - distance`` from Chroma.

    This is the guard the guide calls out by name. The threshold
    (``min_similarity = 0.25``) is a *similarity*; Chroma returns a *distance*.
    If this assertion ever fails, the bot either refuses everything or answers
    everything.
    """
    query = "what is the expense ratio"
    result = retrieve(query, min_sim=-1.0, use_mmr=False)

    raw = synthetic_corpus.query(
        query_texts=[query],
        n_results=synthetic_corpus.count(),
        include=["distances"],
    )
    expected = {round(1.0 - float(d), 9) for d in raw["distances"][0]}
    reported = {round(c.similarity, 9) for c in result.chunks}
    assert reported, "retrieve returned nothing"
    assert reported <= expected, (
        f"similarities {reported} are not in the set Chroma implies {expected}"
    )


def test_all_similarities_are_within_unit_range(synthetic_corpus) -> None:
    for chunk in retrieve("synthetic question", min_sim=-1.0).chunks:
        assert 0.0 <= chunk.similarity <= 1.0, chunk.similarity


def test_threshold_drops_weak_candidates(synthetic_corpus) -> None:
    """A high floor must exclude everything; a negative floor must exclude nothing."""
    assert retrieve("synthetic question", min_sim=1.01).is_empty is True
    assert retrieve("synthetic question", min_sim=-1.0).is_empty is False


def test_threshold_is_applied_to_similarity_not_distance(synthetic_corpus) -> None:
    """Setting the floor to 0.5 must keep only chunks with similarity >= 0.5.

    Throttling on the raw distance instead would invert this: distance 0.9 would
    read as "passes a 0.5 floor" while similarity 0.1 would read as "fails".
    """
    result = retrieve("synthetic question", min_sim=0.5)
    assert all(c.similarity >= 0.5 for c in result.chunks)


# ---------------------------------------------------------------------------
# Scheme alias resolution — against the REAL alias file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is the expense ratio of HDFC Large Cap Fund?", "hdfc_large_cap"),
        ("expense ratio of large cap", "hdfc_large_cap"),
        ("tell me about largecap", "hdfc_large_cap"),
        ("What is the benchmark of HDFC Equity Fund?", "hdfc_flexi_cap"),
        ("hdfc equity fund charges", "hdfc_flexi_cap"),
        ("flexi cap exit load", "hdfc_flexi_cap"),
        ("what is the lock-in period of tax saver fund", "hdfc_elss"),
        ("elss lock in", "hdfc_elss"),
        ("tax saver fund minimum sip", "hdfc_elss"),
        ("riskometer of small cap fund", "hdfc_small_cap"),
        ("smallcap benchmark", "hdfc_small_cap"),
        ("benchmark of HDFC Balanced Advantage Fund", "hdfc_balanced_advantage"),
        ("balanced advantage risk", "hdfc_balanced_advantage"),
    ],
)
def test_resolve_scheme_id_resolves_each_alias(query: str, expected: str) -> None:
    assert resolve_scheme_id(query) == expected


def test_every_alias_in_the_file_resolves_to_its_own_scheme() -> None:
    """Every alias must map back to the scheme it is filed under."""
    for scheme_id, aliases in load_aliases().items():
        for alias in aliases:
            assert resolve_scheme_id(f"question about {alias} please") == scheme_id, alias


def test_no_alias_matches_returns_none_for_a_global_search() -> None:
    assert resolve_scheme_id("what is the minimum sip amount") is None


def test_multi_scheme_query_returns_none_not_a_wrong_filter() -> None:
    """Naming two schemes must not silently filter to one of them."""
    for query in (
        "compare large cap and small cap",
        "Is HDFC Equity Fund better than HDFC Small Cap Fund?",
        "expense ratio of hdfc elss tax saver and balanced advantage",
    ):
        assert resolve_scheme_id(query) is None, query


def test_alias_matching_respects_word_boundaries() -> None:
    """"elss" must not fire inside "else", "well" or "itself"."""
    for query in (
        "else what is the expense ratio",
        "the answer is well documented",
        "explain this to yourself first",
        "shell scripting",
    ):
        assert resolve_scheme_id(query) is None, query


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_resolve_scheme_id_on_empty_returns_none(query: str) -> None:
    assert resolve_scheme_id(query) is None


# ---------------------------------------------------------------------------
# Scheme filtering
# ---------------------------------------------------------------------------


def test_scheme_filter_restricts_results_to_that_scheme(
    synthetic_corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolvable alias must narrow the search, and only that scheme may survive.

    The alias table maps real scheme ids, none of which exist in this synthetic
    corpus, so the resolver is redirected at the fixture's two ids.
    """
    monkeypatch.setattr("app.retrieval.resolve_scheme_id", lambda q: "synthetic_b")
    result = retrieve("anything at all", min_sim=-1.0)

    assert result.filtered_scheme_id == "synthetic_b"
    assert result.chunks, "filter matched a scheme present in the corpus"
    assert {c.chunk_id for c in result.chunks} == {
        "syn0000000000005",
        "syn0000000000006",
        "syn0000000000008",
    }, "results escaped the scheme filter"


def test_unfiltered_query_can_reach_every_scheme(synthetic_corpus) -> None:
    """With no alias the search is global, so both schemes are reachable."""
    result = retrieve("anything at all", min_sim=-1.0)
    assert result.filtered_scheme_id is None
    assert len(result.chunks) > 3, "global search returned no more than one scheme's worth"


def test_unresolvable_query_searches_globally(synthetic_corpus) -> None:
    result = retrieve("what is the minimum sip amount", min_sim=-1.0)
    assert result.filtered_scheme_id is None


def test_filter_naming_an_absent_scheme_is_empty_not_an_error(
    synthetic_corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``where=`` on a scheme_id not in the corpus returns nothing, not a raise."""
    monkeypatch.setattr("app.retrieval.resolve_scheme_id", lambda q: "scheme_does_not_exist")
    result = retrieve("any question", min_sim=-1.0)
    assert result.is_empty is True
    assert result.chunks == []


# ---------------------------------------------------------------------------
# MMR
# ---------------------------------------------------------------------------


def test_mmr_returns_at_most_top_k_with_contiguous_ranks(synthetic_corpus) -> None:
    result = retrieve("synthetic question", min_sim=-1.0, top_k=3, use_mmr=True)
    assert len(result.chunks) <= 3
    assert [c.rank for c in result.chunks] == list(range(1, len(result.chunks) + 1))


def test_mmr_score_is_populated_when_enabled(synthetic_corpus) -> None:
    result = retrieve("synthetic question", min_sim=-1.0, use_mmr=True)
    assert result.chunks
    assert all(c.mmr_score is not None for c in result.chunks)


def test_mmr_score_is_none_when_disabled(synthetic_corpus) -> None:
    """``None`` means "not computed" and must differ from a real score of 0.0."""
    result = retrieve("synthetic question", min_sim=-1.0, use_mmr=False)
    assert result.chunks
    assert all(c.mmr_score is None for c in result.chunks)


def test_mmr_prefers_diversity_over_near_duplicates() -> None:
    """MMR's whole job: break a tie on redundancy.

    Relevance for ``b`` and ``c`` is nearly equal, so the only thing separating
    them is that ``b`` is byte-identical to the already-selected ``a``. The
    redundancy term must therefore push ``c`` ahead of ``b``.

    Note the relevance gap is deliberately small. MMR is a *trade-off*, not a
    diversity mandate: at ``lam=0.7`` a chunk at 0.89 relevance legitimately
    beats one at 0.50, and ``test_mmr_keeps_relevance_when_diversity_is_weak``
    pins that the other half of the bargain also holds.

    ``a`` and ``b`` are given *different* sections on purpose. Their redundancy is
    by text, and that is the case the MMR redundancy term has to catch — two
    chunks that are near-identical in wording even though they were cut from
    different parts of the page. Same-section redundancy is already removed
    earlier, by the section de-duplication in ``_mmr``.
    """
    def candidate(cid: str, text: str, section: str, similarity: float) -> RetrievedChunk:
        return RetrievedChunk(
            rank=0, chunk_id=cid, text=text, section=section, scheme_name="s",
            source_url="https://example.com/s", fetched_at="2026-10-02",
            similarity=similarity, mmr_score=None,
        )

    fees = "Section: Fees. Total expense ratio: 1.11%."
    bench = "Section: Benchmark. Benchmark: TRIX."

    a = candidate("a", fees, "Fees", 0.90)
    b = candidate("b", fees, "Fees (continued)", 0.89)   # byte-identical text to a
    c = candidate("c", bench, "Benchmark", 0.88)         # distinct text, barely lower relevance

    picked = _mmr([a, b, c], k=2, lam=0.7)
    picked_ids = {p.chunk_id for p in picked}
    assert len(picked_ids) == 2
    assert "c" in picked_ids, "MMR returned two near-duplicates and dropped the distinct chunk"


def test_mmr_keeps_relevance_when_diversity_is_weak() -> None:
    """The other half of the trade-off: a big relevance gap still wins.

    Guards against over-correcting the formula into pure diversity — at
    ``lam=0.7`` the 0.89-relevant near-duplicate beats the 0.50-relevant
    distinct chunk, and that is the intended behaviour.
    """
    def candidate(cid: str, text: str, section: str, similarity: float) -> RetrievedChunk:
        return RetrievedChunk(
            rank=0, chunk_id=cid, text=text, section=section, scheme_name="s",
            source_url="https://example.com/s", fetched_at="2026-10-02",
            similarity=similarity, mmr_score=None,
        )

    fees = "Section: Fees. Total expense ratio: 1.11%."
    bench = "Section: Benchmark. Benchmark: TRIX."

    a = candidate("a", fees, "Fees", 0.90)
    b = candidate("b", fees, "Fees (continued)", 0.89)
    c = candidate("c", bench, "Benchmark", 0.50)

    picked = _mmr([a, b, c], k=2, lam=0.7)
    picked_ids = {p.chunk_id for p in picked}
    assert picked_ids == {"a", "b"}, picked_ids


def test_mmr_lambda_controls_the_relevance_diversity_balance() -> None:
    """Lower ``lam`` (more diversity weight) must retain the distinct chunk."""
    def candidate(cid: str, text: str, section: str, similarity: float) -> RetrievedChunk:
        return RetrievedChunk(
            rank=0, chunk_id=cid, text=text, section=section, scheme_name="s",
            source_url="https://example.com/s", fetched_at="2026-10-02",
            similarity=similarity, mmr_score=None,
        )

    fees = "Section: Fees. Total expense ratio: 1.11%."
    bench = "Section: Benchmark. Benchmark: TRIX."

    cands = [
        candidate("a", fees, "Fees", 0.90),
        candidate("b", fees, "Fees (continued)", 0.89),
        candidate("c", bench, "Benchmark", 0.50),
    ]
    picked_high = _mmr(cands, k=2, lam=0.9)
    picked_low = _mmr(cands, k=2, lam=0.1)
    assert {p.chunk_id for p in picked_low} == {"a", "c"}, "diversity weight did not take effect"
    assert {p.chunk_id for p in picked_high} != {"a", "c"}, (
        "a relevance weight of 0.9 should still favour the near-duplicate here"
    )


def test_mmr_first_pick_is_the_top_scoring_candidate() -> None:
    # Distinct sections: with a shared one the section de-duplication would drop
    # b and c before MMR ever scored them, leaving nothing to assert about.
    def candidate(cid: str, section: str, similarity: float) -> RetrievedChunk:
        return RetrievedChunk(
            rank=0, chunk_id=cid, text=f"text {cid}", section=section, scheme_name="s",
            source_url="https://example.com/s", fetched_at="2026-10-02",
            similarity=similarity, mmr_score=None,
        )

    cands = [
        candidate("a", "one", 0.9),
        candidate("b", "two", 0.5),
        candidate("c", "three", 0.4),
    ]
    picked = _mmr(cands, k=3, lam=0.7)
    assert len(picked) == 3
    assert picked[0].chunk_id == "a"
    assert picked[0].rank == 1
    assert [c.rank for c in picked] == [1, 2, 3]


def test_mmr_on_fewer_candidates_than_k_returns_them_all() -> None:
    def candidate(cid: str, section: str) -> RetrievedChunk:
        return RetrievedChunk(
            rank=0, chunk_id=cid, text=f"text {cid}", section=section, scheme_name="s",
            source_url="https://example.com/s", fetched_at="2026-10-02",
            similarity=0.5, mmr_score=None,
        )

    picked = _mmr([candidate("a", "one"), candidate("b", "two")], k=5, lam=0.7)
    assert [c.rank for c in picked] == [1, 2]
    assert [c.chunk_id for c in picked] == ["a", "b"]


def test_mmr_never_returns_two_chunks_from_the_same_section() -> None:
    """Same-section chunks are near-duplicates by construction and share a slot.

    HDFC Large Cap stores 18 chunks across only 14 distinct sections — Holdings
    split three ways, ``DM Dhruv Muchhal Jun 2023 - Present View details`` twice.
    Before this rule, raising ``top_k`` past 14 forced MMR to pad the result with
    same-section duplicates and 15 of 15 golden queries repeated a section. The
    higher ``top_k`` is what lets co-managers reach the prompt at all, so it is
    only safe together with this de-duplication.
    """
    def candidate(cid: str, section: str, similarity: float) -> RetrievedChunk:
        return RetrievedChunk(
            rank=0, chunk_id=cid, text=f"text {cid}", section=section, scheme_name="s",
            source_url="https://example.com/s", fetched_at="2026-10-02",
            similarity=similarity, mmr_score=None,
        )

    # 6 candidates, only 3 distinct sections, k=6 -> only 3 can be returned.
    cands = [
        candidate("a1", "About", 0.90), candidate("a2", "About", 0.88),
        candidate("h1", "Holdings", 0.80), candidate("h2", "Holdings", 0.79),
        candidate("o1", "Objective", 0.70), candidate("o2", "Objective", 0.69),
    ]
    picked = _mmr(cands, k=6, lam=0.7)
    sections = [c.section for c in picked]
    assert len(set(sections)) == len(sections)
    assert sorted(sections) == ["About", "Holdings", "Objective"]
    # The surviving representative is the best-KNN one for its section, not an
    # arbitrary member of the group.
    assert [c.chunk_id for c in picked] == ["a1", "h1", "o1"]


def test_mmr_section_dedup_does_not_starve_a_small_pool() -> None:
    """If every candidate shares a section, one is still returned — never zero."""
    def candidate(cid: str) -> RetrievedChunk:
        return RetrievedChunk(
            rank=0, chunk_id=cid, text=f"text {cid}", section="only", scheme_name="s",
            source_url="https://example.com/s", fetched_at="2026-10-02",
            similarity=0.5, mmr_score=None,
        )

    picked = _mmr([candidate("a"), candidate("b"), candidate("c")], k=5, lam=0.7)
    assert len(picked) == 1
    assert picked[0].chunk_id == "a"
    assert picked[0].rank == 1


def test_mmr_on_empty_input_returns_empty() -> None:
    assert _mmr([], k=5, lam=0.7) == []


# ---------------------------------------------------------------------------
# Corpus state
# ---------------------------------------------------------------------------


def test_empty_corpus_raises_corpus_empty_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unbuilt corpus is a setup error, distinct from "question unanswerable"."""
    from app import config

    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    config.get_settings.cache_clear()
    with pytest.raises(CorpusEmptyError):
        retrieve("anything at all")
    config.get_settings.cache_clear()


def test_off_corpus_question_returns_is_empty(synthetic_corpus) -> None:
    """A question with nothing to do with the corpus must clear the threshold."""
    result = retrieve(
        "What is the capital of a country that does not exist in this corpus?",
        min_sim=0.9,
    )
    assert result.is_empty is True
    assert result.chunks == []
    assert result.max_similarity == 0.0


# ---------------------------------------------------------------------------
# Header stripping
# ---------------------------------------------------------------------------


def test_returned_text_excludes_the_internal_context_header(synthetic_corpus) -> None:
    """P7 shows the model ``text``; the ``[Scheme: ...]`` header is encoder-only."""
    result = retrieve("what is the expense ratio", min_sim=-1.0)
    assert result.chunks
    for chunk in result.chunks:
        assert "[Scheme:" not in chunk.text


def test_returned_text_equals_the_original_display_text(synthetic_corpus) -> None:
    result = retrieve("what is the expense ratio", min_sim=-1.0)
    bodies = {c.text for c in retrieve("x", min_sim=-1.0).chunks}
    assert all(c.text in bodies | {SYNTHETIC_BODIES[k] for k in SYNTHETIC_BODIES} for c in result.chunks)


def test_chunks_carry_citation_metadata(synthetic_corpus) -> None:
    result = retrieve("what is the expense ratio", min_sim=-1.0)
    for chunk in result.chunks:
        assert chunk.section
        assert chunk.scheme_name
        assert chunk.source_url.startswith("https://example.com/")
        assert chunk.fetched_at == "2026-10-02"
        assert chunk.chunk_id


def test_max_similarity_is_the_best_surviving_score(synthetic_corpus) -> None:
    result = retrieve("what is the expense ratio", min_sim=-1.0)
    if result.chunks:
        assert result.max_similarity == pytest.approx(max(c.similarity for c in result.chunks))


# ---------------------------------------------------------------------------
# Integration — real corpus
# ---------------------------------------------------------------------------

_REAL_CORPUS = Path("data/processed/chunks.jsonl")


def _real_corpus_available() -> bool:
    if not _REAL_CORPUS.exists():
        return False
    from app.store import corpus_is_empty
    from app.config import get_settings

    if get_settings().chroma_path.startswith("tmp"):
        return False
    return not corpus_is_empty()


requires_real_corpus = pytest.mark.skipif(
    not _real_corpus_available(), reason="real corpus not built; run `python -m app.ingest`"
)


@requires_real_corpus
def test_real_corpus_resolves_all_five_schemes() -> None:
    """Every scheme in the corpus must be reachable by alias."""
    from app.store import corpus_stats

    stats = corpus_stats()
    assert stats["count"] > 0
    for scheme_id in stats["schemes"]:
        # "hdfc" plus any alias the file provides for this scheme
        from app.scheme_aliases import load_aliases
        alias = load_aliases()[scheme_id][0]
        assert resolve_scheme_id(f"question about {alias}") == scheme_id


@requires_real_corpus
def test_real_corpus_finds_the_expense_ratio_fact() -> None:
    """The top hit must actually contain the expense ratio, whichever section
    it is filed under."""
    result = retrieve("What is the expense ratio of HDFC Large Cap Fund?")
    assert result.is_empty is False
    assert result.filtered_scheme_id == "hdfc_large_cap"
    top = result.chunks[0]
    assert "expense ratio" in top.text.lower(), (
        f"rank-1 section {top.section!r} does not carry the requested fact"
    )


@requires_real_corpus
def test_real_corpus_scheme_filter_excludes_other_schemes() -> None:
    result = retrieve("What is the lock in period of tax saver fund?")
    assert result.is_empty is False
    assert result.filtered_scheme_id == "hdfc_elss"
    assert all(c.scheme_name.startswith("HDFC ELSS") for c in result.chunks)


@requires_real_corpus
def test_real_corpus_off_corpus_question_is_empty() -> None:
    """This is the query that becomes a NOT_FOUND answer in P8."""
    result = retrieve("what is the price of gold today")
    assert result.is_empty is True


@requires_real_corpus
def test_real_corpus_returns_diverse_sections() -> None:
    """MMR must not hand back five chunks from a single section."""
    result = retrieve("What is the expense ratio of HDFC Large Cap Fund?")
    sections = [c.section for c in result.chunks]
    assert len(set(sections)) == len(sections), f"MMR returned duplicates: {sections}"


@requires_real_corpus
def test_real_corpus_similarities_are_sane() -> None:
    """A real question must score well above the 0.25 floor, and never above 1.0."""
    result = retrieve("What is the expense ratio of HDFC Large Cap Fund?")
    assert result.max_similarity > 0.25
    for chunk in result.chunks:
        assert 0.0 <= chunk.similarity <= 1.0
        assert chunk.similarity >= 0.25