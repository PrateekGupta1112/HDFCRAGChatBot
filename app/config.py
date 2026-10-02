"""All tunables for the application (architecture §10).

Every magic number lives here so that later phases never hardcode a value in
logic. Field names map to UPPERCASE environment variables automatically via
pydantic-settings (e.g. `min_similarity` <- `MIN_SIMILARITY`).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables. See architecture.md §10."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    #
    # DEVIATION from PRD C8, which names OpenAI `gpt-4o-mini`. PRD §16 permits
    # "an OpenAI-compatible LLM endpoint", and Groq serves one, so the `openai`
    # client is simply pointed at `llm_base_url` instead of api.openai.com. No
    # extra dependency: `openai` was already a requirement.
    groq_api_key: str | None = None
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    """Must be a model id Groq actually serves. Verified against
    `GET {llm_base_url}/models` on 2026-10-02, which returned 11 ids, of which
    these are general-purpose generators: `openai/gpt-oss-120b`,
    `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`. The `meta-llama/llama-prompt-guard-*`
    entries are classifiers, not generators. `llama-3.3-70b-versatile` is NOT
    served and fails with a NotFoundError, falling back to the extractive path."""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    """DEVIATION from PRD §6.6's "max_tokens ~= 220".

    That figure is calibrated for a non-reasoning model. `openai/gpt-oss-120b`
    emits ~850 characters of reasoning *before* the answer, so a 220-token budget
    is spent entirely on reasoning and the completion arrives with
    `finish_reason="length"` and no usable content — measured on 2026-10-02:
    220 -> 11 characters of answer, 512 -> full answer, 1024 -> full answer.

    Raising this does NOT weaken the 3-sentence limit. The sentence budget is
    enforced in code (`generation.enforce` -> `_fit_budget`), not by the token
    cap, so `llm_max_tokens` is only a cost and latency control."""
    llm_timeout_s: int = 20

    # Embedding / store
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_max_seq_length: int = 256
    """Hard limit of all-MiniLM-L6-v2 in WORD-PIECES. Longer input is silently
    truncated by the encoder, so chunk sizing (architecture §18.1) must stay
    below this value."""

    chroma_path: str = "./chroma_db"
    chroma_collection: str = "mf_faq_v1"

    # Retrieval
    retrieval_top_k: int = 12
    """Raised 5 -> 8 -> 16 -> 12 on 2026-10-02. The final value is the smallest
    that fixes both symptoms; the path there is the evidence.

    **5 -> 8.** Each scheme page's key-value header (``Very High Risk ...
    Expense ratio 1.04% Rating 4``) is a single chunk scoring ~0.78 for a short
    attribute question — above ``min_similarity``, but MMR ranks the "About" prose
    chunk above it because that prose paraphrases the question. At top_k=5 no
    retrieved chunk contained the string ``Rating``, so "what is the rating of hdfc
    large cap fund" was answered from the neighbouring sentence "The HDFC Large
    Cap Fund Direct Growth is rated Very High risk" — Groww's own wording, so
    grounded, but the wrong facet; the page's star rating is 4.

    **8 -> 12.** Enumeration questions ("who *else* manages this fund?") could not
    be answered from 8 chunks. Within the alias-filtered scheme the co-managers'
    sections rank 9th and 11th-23rd, so they were never fetched and the model
    correctly reported only the manager it could see.

    Raising ``over_fetch`` *alone* is not sufficient and can even hurt: with top_k
    fixed, a wider candidate pool lets generic Holdings/Understand-terms chunks
    crowd the manager sections back out of the final k. The two move together.

    **16 was tried and rejected.** It broke the MMR diversity invariant — 15 of 15
    golden queries came back with a repeated section, because HDFC Large Cap only
    has 14 distinct sections across its 18 chunks, so a 16-chunk result must pad
    with duplicates. The fix was to de-duplicate candidates by ``section`` in
    :func:`app.retrieval._mmr` rather than to pick a smaller k; after that,
    top_k=12 satisfies both goals at once (2-3 co-managers in context, 0 of 15
    golden queries repeating a section).

    Cost, measured on corpus fingerprint 449d9e77a656 with ENV=eval: AC1/AC2/AC3/
    AC7 100%, AC4/AC5/AC6 100%, AC8 15/15, AC9 4/5 — unchanged from (5, 8). The
    lone AC9 miss ("How do I open a demat account?") is Groww's own help article
    and fails at every setting.
    """
    retrieval_over_fetch: int = 24
    min_similarity: float = 0.25
    """COSINE SIMILARITY, not cosine distance. Chroma returns distance, so
    `similarity = 1.0 - distance` must be computed before thresholding."""

    use_mmr: bool = True
    mmr_lambda: float = 0.7

    # Chunking (architecture §18.1)
    chunk_target_tokens: int = 200
    chunk_max_tokens: int = 240
    chunk_overlap_tokens: int = 60
    min_section_tokens: int = 40
    max_table_rows: int = 12
    max_table_chars: int = 1800

    # Ingestion
    fetch_delay_s: float = 1.5
    fetch_timeout_s: int = 30
    fetch_retries: int = 3
    user_agent: str = "MF-Facts-Bot/0.1 (academic RAG demo; contact: student@example.edu)"

    # Answer / UI
    max_sentences: int = 3
    max_chat_history_turns: int = 3
    env: str = "demo"  # demo | eval

    @property
    def llm_enabled(self) -> bool:
        """Whether to call the LLM.

        False when no API key is configured, and also when `env == "eval"`,
        which is what makes the evaluation harness deterministic and
        network-free (NFR-5, architecture §16).
        """
        return bool(self.groq_api_key) and self.env != "eval"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that every module observes the same values and the `.env` file is
    read exactly once.
    """
    return Settings()
