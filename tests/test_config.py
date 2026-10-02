"""Unit tests for app.config — the settings contract every later phase reads."""

from __future__ import annotations

from app.config import Settings, get_settings


def test_defaults_match_the_specified_values() -> None:
    s = Settings(_env_file=None)
    assert s.chunk_target_tokens == 200
    assert s.chunk_max_tokens == 240
    assert s.embed_max_seq_length == 256
    assert s.min_similarity == 0.25
    assert s.retrieval_top_k == 5
    assert s.retrieval_over_fetch == 8
    assert s.mmr_lambda == 0.7
    assert s.max_sentences == 3
    assert s.env == "demo"


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
