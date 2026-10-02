"""Retrieval — filter → KNN → threshold → MMR → assemble (architecture §9.3).

This is RAG stage ⑤. It turns a question into at most five chunks that are
(a) about the right scheme, (b) above a relevance floor, and (c) not five
restatements of the same paragraph.

**The single highest-risk bug in this project lives in this file.**
Chroma returns ``distances`` as **cosine distance**, where smaller is closer. The
configured floor (``settings.min_similarity = 0.25``) is a **cosine similarity**,
where larger is closer. Thresholding the raw distance inverts the test: a
distance of ``0.8`` — a genuinely good match — reads as "0.8, above 0.25, keep
it", while a distance of ``0.20`` — near-identical — reads as "0.20, below 0.25,
drop it". The bot then answers nonsense questions and refuses real ones. Every
value compared against ``min_similarity`` in this module is therefore computed as
``1.0 - distance`` first, and ``tests/test_retrieval.py`` recomputes it from
Chroma's raw output rather than trusting this module's arithmetic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.embedding import embed_texts
from app.errors import CorpusEmptyError
from app.scheme_aliases import resolve_scheme_id
from app.store import corpus_is_empty, get_collection
from app.textutils import normalize_url

logger = logging.getLogger(__name__)

#: Chroma truncates the embedding of every stored chunk at this marker. The
#: stored ``documents`` field is ``embed_text`` (header + body) because the
#: header is what makes a chunk self-describing for the *encoder*; the header is
#: stripped back off on read, because P7's prompt and numeric audit must see only
#: what the page actually says.
_CONTEXT_HEADER_MARKER = "[Scheme:"

#: Chroma rejects ``n_results`` larger than the collection, so the over-fetch is
#: clamped rather than trusted. A corpus smaller than ``over_fetch`` is a real
#: situation in tests and in a freshly-seeded collection.
_MIN_CHROMA_N_RESULTS = 1


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned to the generator, with everything P7 needs to cite it.

    Attributes:
        rank: 1-based position after MMR, not after KNN.
        chunk_id: Stable id from ingestion, for trace/debugging.
        text: The chunk body **without** the internal ``[Scheme: ...]`` context
            header. This is what the model is shown and what the numeric audit
            compares against.
        section: Section title, used verbatim in the citation.
        scheme_name: Human-readable scheme name for the citation label.
        source_url: The page the text came from. Compared with
            ``normalize_url`` on both sides everywhere.
        fetched_at: ISO date the source page was fetched.
        similarity: Cosine **similarity** in ``[0.0, 1.0]``, i.e. ``1 - distance``.
        mmr_score: The score MMR used to select this chunk, or ``None`` when MMR
            is disabled. ``None`` and ``0.0`` are different states: ``None`` means
            "not computed", and P9 renders them differently.
    """

    rank: int
    chunk_id: str
    text: str
    section: str
    scheme_name: str
    source_url: str
    fetched_at: str
    similarity: float
    mmr_score: float | None


@dataclass(frozen=True)
class RetrievalResult:
    """The outcome of one retrieval call.

    Attributes:
        query: The query as passed in. Note that P8 passes
            ``GuardDecision.redacted_query`` here, never the raw user input.
        filtered_scheme_id: The scheme the search was restricted to, or ``None``
            for a global search.
        chunks: Ranked, thresholded, MMR-diversified. Empty when ``is_empty``.
        max_similarity: Best cosine similarity among candidates that **survived**
            the threshold — ``0.0`` when nothing did. Used by P10's AC8
            diagnostics to distinguish "no candidate at all" from "candidates
            existed but were all too weak".
        is_empty: True when no candidate cleared ``min_similarity``. P8 turns
            this into a NOT_FOUND answer; P6 deliberately does not decide it.
    """

    query: str
    filtered_scheme_id: str | None
    chunks: list[RetrievedChunk]
    max_similarity: float
    is_empty: bool


def _strip_context_header(document: str) -> str:
    """Recover ``display_text`` from a stored ``embed_text``.

    ``embed_text`` is ``"[Scheme: ...\\n" + body``. The header is metadata for the
    encoder and must not reach the model or the numeric audit.

    Args:
        document: The document as stored in Chroma.

    Returns:
        The body, or the document unchanged if it does not start with the header.
    """
    if not document.startswith(_CONTEXT_HEADER_MARKER):
        return document
    _, newline, body = document.partition("\n")
    return body if newline else document


def _empty_result(query: str, scheme_id: str | None, max_similarity: float = 0.0) -> RetrievalResult:
    """Build a no-results :class:`RetrievalResult`."""
    return RetrievalResult(
        query=query,
        filtered_scheme_id=scheme_id,
        chunks=[],
        max_similarity=max_similarity,
        is_empty=True,
    )


def _candidates_from_response(
    response: dict[str, Any],
    scheme_id: str | None,
) -> list[RetrievedChunk]:
    """Turn one Chroma query response into unranked candidates.

    Rank is filled in by :func:`_mmr` (or left sequential if MMR is off), because
    the rank that matters is the post-MMR position, not the KNN position.
    """
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]

    candidates: list[RetrievedChunk] = []
    for idx, distance in enumerate(distances):
        meta = dict(metadatas[idx] or {}) if idx < len(metadatas) else {}
        document = documents[idx] if idx < len(documents) else ""
        # THE LINE THAT MATTERS. Chroma gives cosine distance; the threshold and
        # every displayed score are cosine similarity.
        similarity = 1.0 - float(distance)
        candidates.append(
            RetrievedChunk(
                rank=0,
                chunk_id=str(meta.get("chunk_id", "")),
                text=_strip_context_header(document),
                section=str(meta.get("section", "")),
                scheme_name=str(meta.get("scheme_name", "")),
                source_url=str(meta.get("source_url", "")),
                fetched_at=str(meta.get("fetched_at", "")),
                similarity=similarity,
                mmr_score=None,
            )
        )
    return candidates


def _mmr(
    cands: list[RetrievedChunk],
    k: int,
    lam: float,
) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance selection (architecture §9.3).

    Without this, five near-identical chunks from one long section — which is
    what table-heavy pages produce — crowd out the single chunk that actually
    contains the answer.

    Only the surviving candidates (at most ``over_fetch``, so at most 8) are
    re-embedded, and only to measure how similar they are *to each other*. The
    query is not re-embedded: Chroma already scored every candidate against it,
    and that score is on ``c.similarity``.

    **Deviation from the guide's pseudo-code, and it is a bug fix.** The guide
    gives ``lam * sim(c, selected[-1]) - (1 - lam) * max_sim(c, selected)`` —
    relevance measured against the *last selected chunk*. That does not work. On
    the first greedy step only one chunk is selected, so ``max_sim(c, selected)``
    and ``sim(c, selected[-1])`` are the same number ``s``, and the score
    collapses to ``lam*s - (1-lam)*s = (2*lam - 1)*s``. With ``lam = 0.7`` that is
    ``0.4 * s``: strictly increasing in ``s``, so a "diversifying" score always
    picks the chunk *most similar to the one already chosen*. It is the exact
    opposite of its stated purpose, and it fails the phase's own exit criterion,
    "MMR returns diverse chunks (not 5 from one section)".

    The relevance term is therefore measured against the **query**, which is what
    MMR means and what makes the formula work::

        mmr(c) = lam * sim(c, query) - (1 - lam) * max_{s in selected} sim(c, s)

    ``sim(c, query)`` is ``c.similarity``, already computed by Chroma. The
    redundancy term is unchanged.
    ``test_mmr_prefers_diversity_over_near_duplicates`` pins this behaviour.

    **Candidates are de-duplicated by section before selection.** Two chunks with
    the same ``section`` come from the same region of the same page and are
    near-duplicates by construction — ``Holdings ( 87 )`` split across three
    chunks, or ``DM Dhruv Muchhal Jun 2023 - Present View details`` stored twice.
    Scoring them against each other is wasted work and admitting both wastes a
    prompt slot. Collapsing them to the best-KNN representative first means every
    slot in the prompt carries a different part of the page.

    This is what makes a wider ``top_k`` safe. Without it, raising ``top_k`` past
    the number of *distinct* sections a page has (HDFC Large Cap: 14 distinct
    sections across 18 chunks) forces MMR to pad the result with same-section
    duplicates, which is the exact failure MMR exists to prevent — measured at
    top_k=16 before this change, 15 of 15 golden queries returned a repeated
    section. With it, ``top_k`` can grow to cover co-managers and other
    multi-value fields without the diversity invariant regressing.

    Args:
        cands: Candidates in descending KNN order. Must be non-empty.
        k: Maximum number to return.
        lam: Trade-off in ``[0, 1]``. Higher keeps more relevance, lower keeps
            more diversity.

    Returns:
        At most ``k`` chunks with ``rank`` reassigned to ``1..k`` and
        ``mmr_score`` populated. Every returned chunk has a distinct ``section``,
        unless the candidate pool itself held only one.
    """
    if not cands:
        return []
    if k <= 0:
        return []

    # Candidates arrive in descending KNN order, so the first chunk seen for a
    # section is its most query-relevant representative and the one to keep.
    seen_sections: set[str] = set()
    distinct: list[RetrievedChunk] = []
    for c in cands:
        if c.section in seen_sections:
            continue
        seen_sections.add(c.section)
        distinct.append(c)

    if len(distinct) <= k:
        return [
            replace(c, rank=i + 1, mmr_score=c.similarity)
            for i, c in enumerate(distinct)
        ]

    cands = distinct

    # Vectors are L2-normalised, so the inner product is cosine similarity and no
    # separate normalisation step is needed.
    vectors = embed_texts([c.text for c in cands])
    sim_matrix: np.ndarray = np.asarray(vectors) @ np.asarray(vectors).T

    selected: list[int] = [0]
    pool: list[int] = list(range(1, len(cands)))
    # Relevance is Chroma's own score against the query, not a re-derived one.
    relevance: dict[int, float] = {i: cands[i].similarity for i in range(len(cands))}
    scores: dict[int, float] = {0: cands[0].similarity}

    while len(selected) < k and pool:
        def mmr_score(index: int) -> float:
            # Similarities from candidate `index` to everything chosen so far.
            redundancy = float(np.max(sim_matrix[index, selected]))
            return lam * relevance[index] - (1.0 - lam) * redundancy

        best = max(pool, key=mmr_score)
        pool.remove(best)
        selected.append(best)
        scores[best] = mmr_score(best)

    return [
        replace(cands[i], rank=pos + 1, mmr_score=scores[i])
        for pos, i in enumerate(selected)
    ]


def retrieve(
    query: str,
    top_k: int | None = None,
    over_fetch: int | None = None,
    min_sim: float | None = None,
    use_mmr: bool | None = None,
    mmr_lambda: float | None = None,
    settings: Settings | None = None,
) -> RetrievalResult:
    """Retrieve chunks for ``query``: alias filter → KNN → threshold → MMR.

    Args:
        query: The query text. Callers pass ``GuardDecision.redacted_query``.
        top_k: How many chunks to return. ``None`` reads
            ``settings.retrieval_top_k``.
        over_fetch: How many candidates to pull from Chroma before thresholding.
            ``None`` reads ``settings.retrieval_over_fetch``. This deliberately
            exceeds ``top_k``: thresholding first and *then* topping up leaves no
            room for MMR to choose between alternatives.
        min_sim: Cosine **similarity** floor. ``None`` reads
            ``settings.min_similarity``.
        use_mmr: ``None`` reads ``settings.use_mmr``.
        mmr_lambda: ``None`` reads ``settings.mmr_lambda``.
        settings: Injected settings, for tests. ``None`` uses the process-wide
            singleton.

    Returns:
        A :class:`RetrievalResult`. ``is_empty`` is True when the corpus is
        empty, when the alias filter matched nothing, or when every candidate
        fell below the floor.

    Raises:
        CorpusEmptyError: If the vector store holds no chunks. Callers treat this
            as "setup not run" rather than as "question unanswerable" — P8 turns
            it into an ERROR answer with setup instructions.
    """
    s = settings or get_settings()
    k = s.retrieval_top_k if top_k is None else top_k
    n_fetch = s.retrieval_over_fetch if over_fetch is None else over_fetch
    floor = s.min_similarity if min_sim is None else min_sim
    mmr_on = s.use_mmr if use_mmr is None else use_mmr
    lam = s.mmr_lambda if mmr_lambda is None else mmr_lambda

    scheme_id = resolve_scheme_id(query)

    if corpus_is_empty():
        raise CorpusEmptyError(
            "Vector store is empty. Run `python -m app.ingest` to build the corpus."
        )

    collection = get_collection()
    try:
        available = int(collection.count())
    except Exception:  # pragma: no cover - Chroma internal failure
        available = 0
    if available == 0:
        raise CorpusEmptyError(
            "Vector store is empty. Run `python -m app.ingest` to build the corpus."
        )

    # Chroma errors when n_results exceeds the collection size, and a corpus of 3
    # chunks is a legitimate state during development.
    n_results = max(_MIN_CHROMA_N_RESULTS, min(n_fetch, available))

    where = {"scheme_id": scheme_id} if scheme_id else None
    try:
        response = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        # A `where` filter naming a scheme_id that is not in the corpus returns
        # nothing rather than raising, but a malformed filter does raise. Either
        # way the honest answer is "nothing retrieved", not an exception to the
        # caller — P8 turns is_empty into a NOT_FOUND answer.
        logger.warning("Chroma query failed (%s); treating as no results", type(exc).__name__)
        return _empty_result(query, scheme_id)

    cands = _candidates_from_response(response, scheme_id)
    if not cands:
        return _empty_result(query, scheme_id)

    kept = [c for c in cands if c.similarity >= floor]
    if not kept:
        # Report how close the best miss was. Never the query — logging policy is
        # architecture §11.
        logger.debug(
            "all %d candidates below floor %.3f (best %.3f)",
            len(cands), floor, max(c.similarity for c in cands),
        )
        return _empty_result(query, scheme_id, max_similarity=0.0)

    max_similarity = kept[0].similarity

    if mmr_on:
        ranked = _mmr(kept, k, lam)
    else:
        ranked = [
            replace(c, rank=i + 1, mmr_score=None)
            for i, c in enumerate(kept[:k])
        ]

    return RetrievalResult(
        query=query,
        filtered_scheme_id=scheme_id,
        chunks=ranked,
        max_similarity=max_similarity,
        is_empty=False,
    )


def citation_url(chunk: RetrievedChunk) -> str:
    """Return ``chunk.source_url`` normalised for comparison, not for display.

    P7 compares a model's emitted URL against this value with
    ``normalize_url`` on both sides, and renders the *original* URL to the user.
    """
    return normalize_url(chunk.source_url)