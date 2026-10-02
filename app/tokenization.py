"""Word-piece counting against the embedding model's own tokenizer.

architecture §18.1 turns on one number: all-MiniLM-L6-v2 truncates its input at
256 **word-pieces**, silently, with no warning. A chunk that overruns that limit
is not an error at encode time — the tail is simply dropped, so the stored
vector never represented the tail of the text it claims to describe. Counting
with a character heuristic instead (roughly four characters per word-piece)
makes the limit merely aspirational; counting with the model's tokenizer makes
it enforceable.

That tokenizer ships inside the ~90 MB model download, which is unacceptable
for a unit-test run and for ``pytest`` in CI. So the count is authoritative when
the model is present and *estimated* when it is not, and
:func:`tokenization_is_authoritative` reports which of the two a given process
is using so no caller has to guess.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.errors import ModelUnavailableError

logger = logging.getLogger(__name__)

#: Average characters per word-piece for English financial prose. Used only as
#: the offline fallback; see the module docstring.
CHARS_PER_TOKEN = 4

#: Tri-state for the module-level fallback flag.
_UNKNOWN = 0
_ESTIMATED = 1
_AUTHORITATIVE = 2

_token_state = _UNKNOWN


@lru_cache(maxsize=1)
def get_tokenizer():
    """Return the embedding model's own tokenizer, loading it once.

    implementation.md §P3 specifies ``SentenceTransformer(model).tokenizer``.
    That path is deliberately *not* used: importing ``sentence_transformers``
    pulls in the full torch runtime, which is seconds of import cost and a hard
    dependency on a working torch/NumPy ABI pair for what is only a
    ``WordPiece`` vocabulary lookup. ``AutoTokenizer`` returns the same
    ``all-MiniLM-L6-v2`` tokenizer — it is the very object
    ``SentenceTransformer`` delegates to — without either cost. P4 still loads
    ``SentenceTransformer`` itself; this module is counting, not encoding.

    If the model files are not already in the local Hugging Face cache the
    tokenizer is *not* fetched: chunking and the test suite must not trigger a
    90 MB download (or hang on one), which is the whole reason the fallback in
    :func:`count_tokens` exists.

    Returns:
        A ``transformers`` tokenizer for ``settings.embed_model``.

    Raises:
        ModelUnavailableError: If the model is not cached locally or the
            tokenizer cannot be constructed from it.
    """
    if not _model_is_cached():
        raise ModelUnavailableError(
            f"{get_settings().embed_model} is not in the local Hugging Face "
            "cache; refusing to download it during chunking"
        )
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415 — lazy, heavy import

        return AutoTokenizer.from_pretrained(get_settings().embed_model)
    except Exception as exc:  # noqa: BLE001 — reported as unavailable, not raised raw
        raise ModelUnavailableError(
            f"could not load a tokenizer for {get_settings().embed_model}: {exc}"
        ) from exc


def _model_is_cached() -> bool:
    """Whether the embedding model's files are already on disk.

    Checked rather than assumed so that a chunking run on a fresh machine does
    not silently begin a large download mid-build.

    Returns:
        ``True`` if a snapshot of the configured model exists in the HF cache.
    """
    model = get_settings().embed_model
    slug = "models--" + model.replace("/", "--")
    roots = [
        Path(os.environ["HF_HUB_CACHE"]) if os.environ.get("HF_HUB_CACHE") else None,
        Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    return any(root is not None and (root / slug).is_dir() for root in roots)


def _resolve_tokenizer():
    """Load the tokenizer once, remembering whether that succeeded."""
    global _token_state
    if _token_state == _UNKNOWN:
        try:
            get_tokenizer()
            _token_state = _AUTHORITATIVE
            logger.info("token counts are AUTHORITATIVE (model tokenizer loaded)")
        except Exception:  # noqa: BLE001 — the whole point is to tolerate this
            _token_state = _ESTIMATED
            logger.info(
                "embedding model unavailable; token counts fall back to an "
                "ESTIMATE of ceil(chars/%d). Chunk sizing is therefore a "
                "lower bound on safety, not a guarantee.",
                CHARS_PER_TOKEN,
            )
    return _token_state


def count_tokens(text: str) -> int:
    """Count the word-pieces in ``text``.

    Authoritative when the model's tokenizer is loadable, estimated otherwise.
    Never raises: chunking, tests and the evaluation harness must all run on a
    machine that has never downloaded the model.

    The estimate is ``ceil(len(text) / 4)``, which *under*-counts English
    word-pieces slightly (typical English runs nearer 4.2-4.5 chars/piece), so
    a chunk sized to the estimate can still land a few word-pieces over the
    limit. Sizing against it is therefore safe in the direction that matters: a
    chunk that measures 240 estimated tokens holds at most ~240 real ones plus
    a small margin, and :func:`exceeds_embed_limit` is re-checked with the real
    tokenizer once P4 has loaded the model.

    Args:
        text: Any string. Empty input counts as 0.

    Returns:
        The word-piece count, never negative.
    """
    if not text:
        return 0
    if _resolve_tokenizer() == _AUTHORITATIVE:
        try:
            with warnings.catch_warnings():
                # Encoding past the limit is the point here — we are counting,
                # not feeding the encoder — so this warning is expected noise.
                # Its "512" is also a red herring: it is the tokenizer's own
                # model_max_length, not the 256 budget the chunker honours.
                warnings.filterwarnings("ignore", message=".*longer than the specified maximum.*")
                return len(get_tokenizer().encode(text, add_special_tokens=False))
        except Exception:  # noqa: BLE001 — degrade, never crash a chunking run
            logger.warning("tokenizer failed mid-run; falling back to estimate", exc_info=True)
            _set_state(_ESTIMATED)
            return math.ceil(len(text) / CHARS_PER_TOKEN)
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _set_state(state: int) -> None:
    """Overwrite the fallback flag (used when a mid-run tokenizer failure demotes it)."""
    global _token_state
    _token_state = state


def tokenization_is_authoritative() -> bool:
    """Whether :func:`count_tokens` is using the real word-piece tokenizer.

    P4 re-checks every chunk with the authoritative tokenizer once the model is
    loaded, precisely because an estimated sizing pass can only approximate.

    Returns:
        ``True`` if the model tokenizer is in use.
    """
    return _resolve_tokenizer() == _AUTHORITATIVE


def exceeds_embed_limit(embed_text: str) -> bool:
    """Whether ``embed_text`` would be silently truncated by the encoder.

    Args:
        embed_text: The text that would be handed to the encoder, header included.

    Returns:
        ``True`` when its word-piece count exceeds ``settings.embed_max_seq_length``.
    """
    if not embed_text:
        return False
    return count_tokens(embed_text) > get_settings().embed_max_seq_length