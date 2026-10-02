"""Unit tests for app.config — the settings contract every later phase reads."""

from __future__ import annotations

from app.config import Settings, get_settings


def test_defaults_match_the_specified_values() -> None:
    s = Settings(_env_file=None)
    assert s.chunk_target_tokens == 200
    assert s.chunk_max_tokens == 240
    assert s.embed_max_seq_length == 256
    assert s.min_similarity == 0.25
    assert s.retrieval_top_k == 12  # DEVIATION from architecture.md:779, see below
    assert s.retrieval_over_fetch == 24  # DEVIATION, see below
    assert s.mmr_lambda == 0.7
    assert s.max_sentences == 3
    assert s.env == "demo"


def test_top_k_deviation_from_architecture_is_deliberate_and_costless() -> None:
    """architecture.md §10 pins ``RETRIEVAL_TOP_K`` to 5; the shipped default is 12.

    Documented so the assertion above cannot be "fixed" back to 5 without also
    reading why. Three separately measured steps, all on corpus fingerprint
    449d9e77a656.

    **5 -> 8.** Each scheme page's key-value header (``Very High Risk ...
    Expense ratio 1.04% Rating 4``) is one chunk scoring ~0.78 for a short
    attribute question — above ``min_similarity``, but MMR ranks the "About" prose
    chunk above it because that prose paraphrases the question. At top_k=5 no
    retrieved chunk contained the string ``Rating``, so "what is the rating of hdfc
    large cap fund" was answered from the neighbouring sentence "The HDFC Large
    Cap Fund Direct Growth is rated Very High risk" (Groww's own wording, so
    grounded, but the wrong facet — the page's star rating is 4).

    **8 -> 12.** Enumeration questions ("who *else* manages this fund?") could not
    be answered from 8 chunks: within the alias-filtered scheme the co-managers'
    sections rank 9th and 11th-23rd, so they were never fetched and the model
    reported only the manager it could see. Raising ``over_fetch`` alone is not
    enough and can hurt — with top_k fixed, a wider pool lets generic
    Holdings/Understand-terms chunks crowd the manager sections back out.

    **16 was tried and rejected** because it broke the MMR diversity invariant:
    HDFC Large Cap has only 14 distinct sections across 18 chunks, so a 16-chunk
    result must pad with same-section duplicates, and 15 of 15 golden queries came
    back repeating a section. The real fix was de-duplicating candidates by
    ``section`` inside :func:`app.retrieval._mmr`; after that, top_k=12 gives both
    2-3 co-managers in context and 0 of 15 golden queries repeating a section.

    Cost: unchanged from (5, 8) — AC1/AC2/AC3/AC7 100%, AC4/AC5/AC6 100%, AC8
    15/15, AC9 4/5. The lone AC9 miss ("How do I open a demat account?") is
    Groww's own help article and fails at every setting.
    """
    assert Settings(_env_file=None).retrieval_top_k == 12
    assert Settings(_env_file=None).retrieval_over_fetch == 24
    # The spec values remain reachable, so a caller can still ask for them.
    assert Settings(retrieval_top_k=5, _env_file=None).retrieval_top_k == 5


def test_over_fetch_is_wide_enough_for_mmr_to_have_choices() -> None:
    """``over_fetch`` must exceed ``top_k``.

    architecture.md §P10 documents the ordering: threshold-then-top-up leaves MMR
    nothing to choose between, so candidates are fetched first and trimmed after.
    If the two ever collapse to the same value the retrieval stage silently
    degenerates into plain top-k KNN.
    """
    s = Settings(_env_file=None)
    assert s.retrieval_over_fetch > s.retrieval_top_k


def test_chunk_budget_stays_inside_the_embedding_limit() -> None:
    """architecture §18.1: a chunk larger than 256 word-pieces is silently
    truncated by all-MiniLM-L6-v2, so the budget must remain below it."""
    s = Settings(_env_file=None)
    assert s.chunk_max_tokens <= s.embed_max_seq_length


def test_llm_enabled_requires_a_key_and_a_demo_environment() -> None:
    assert Settings(groq_api_key=None, env="demo", _env_file=None).llm_enabled is False
    assert Settings(groq_api_key="gsk-test", env="demo", _env_file=None).llm_enabled is True
    # env="eval" forces the deterministic, network-free extractive path.
    assert Settings(groq_api_key="gsk-test", env="eval", _env_file=None).llm_enabled is False


def test_llm_endpoint_is_openai_compatible_and_not_openai() -> None:
    """PRD C8 names OpenAI; §16 permits any OpenAI-compatible endpoint. P7 points
    the client at Groq, so the base URL must never be left at the default."""
    s = Settings(_env_file=None)
    assert "groq.com" in s.llm_base_url
    assert s.llm_base_url.startswith("https://")
    # A model id OpenAI serves would 404 on this endpoint.
    assert "gpt-4o" not in s.llm_model


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
