from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from app.config import get_settings
from app.errors import ModelUnavailableError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model() -> "SentenceTransformer":
    """Return a singleton SentenceTransformer model.

    Raises:
        ModelUnavailableError: If the model cannot be loaded/constructed.

    Note:
        The embedding model (~90MB) downloads on first run and needs network.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - import guard
        raise ModelUnavailableError(
            "sentence-transformers is not available. Install requirements first."
        ) from exc

    settings = get_settings()
    try:
        model = SentenceTransformer(
            settings.embed_model,
            device=None,
        )
    except Exception as exc:
        raise ModelUnavailableError(
            f"Failed to load embedding model '{settings.embed_model}'. "
            "The model downloads on first run and needs network access."
        ) from exc

    logger.debug("Loaded embedding model: %s", settings.embed_model)
    return model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts and return L2-normalized float32 vectors.

    Args:
        texts: List of strings to embed.

    Returns:
        A numpy array of shape (N, 384) with float32 dtype and L2-normalized rows.

    Raises:
        AssertionError: If the resulting embedding dimension is not 384.
    """
    if not texts:
        settings = get_settings()
        return np.empty((0, settings.embed_max_seq_length), dtype=np.float32)

    model = get_model()
    settings = get_settings()

    try:
        embeddings = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        raise ModelUnavailableError(
            "Failed to encode texts with the embedding model."
        ) from exc

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    expected_dim = getattr(get_model(), "get_sentence_embedding_dimension", lambda: 384)()
    if expected_dim is None:
        expected_dim = 384
    assert embeddings.shape[1] == expected_dim, (
        f"Expected embedding dim {expected_dim}, got {embeddings.shape[1]}"
    )
    return embeddings


def embed_one(text: str) -> np.ndarray:
    """Embed a single text and return a 1-D L2-normalized vector of length 384.

    Args:
        text: The string to embed.

    Returns:
        A numpy array of shape (384,) with float32 dtype.
    """
    vec = embed_texts([text])
    if vec.size == 0:
        settings = get_settings()
        return np.zeros((settings.embed_max_seq_length,), dtype=np.float32)
    return vec[0]
