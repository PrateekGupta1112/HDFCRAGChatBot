from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

from app.chunking import Chunk, load_chunks
from app.config import get_settings
from app.errors import CorpusEmptyError

logger = logging.getLogger(__name__)


def get_client() -> "chromadb.Client":
    """Return a Chroma PersistentClient for the configured path."""
    try:
        import chromadb
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("chromadb is not available. Install requirements first.") from exc

    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client


def get_collection() -> "chromadb.Collection":
    """Get or create the configured Chroma collection with cosine space."""
    try:
        from chromadb.api import Collection  # type: ignore
    except Exception:  # pragma: no cover
        pass

    client = get_client()
    settings = get_settings()
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """Convert a Chunk to a dict (only fields defined on the dataclass)."""
    d = asdict(chunk)
    # ensure we only include dataclass fields
    allowed = {f.name for f in fields(Chunk)}
    return {k: v for k, v in d.items() if k in allowed}


def to_chroma_metadata(chunk: Chunk) -> dict[str, Any]:
    """Return flat scalar metadata for Chroma (only str/int; bools as 'true'/'false')."""
    c = _chunk_to_dict(chunk)
    meta: dict[str, Any] = {}
    meta["chunk_id"] = str(c.get("chunk_id", ""))
    meta["scheme_id"] = str(c.get("scheme_id", ""))
    meta["scheme_name"] = str(c.get("scheme_name", ""))
    meta["category"] = str(c.get("category", ""))
    meta["plan"] = str(c.get("plan", ""))
    meta["section"] = str(c.get("section", ""))
    meta["source_url"] = str(c.get("source_url", ""))
    meta["fetched_at"] = str(c.get("fetched_at", ""))
    idx = c.get("chunk_index", 0)
    meta["chunk_index"] = int(idx) if isinstance(idx, (int, float)) else 0
    is_table = c.get("is_table", False)
    meta["is_table"] = "true" if bool(is_table) else "false"
    meta["content_sha256"] = str(c.get("content_sha256", ""))
    # Return only scalar str/int values as required
    return meta


def upsert_chunks(chunks: list[Chunk], embeddings: np.ndarray, force: bool = False) -> int:
    """Upsert chunks into Chroma, keyed by ``chunk_id``.

    Every chunk passed in is stored: the returned count equals ``len(chunks)``
    unless two chunks share a ``chunk_id``, which is a chunking bug and is
    logged as an error rather than absorbed.

    Args:
        chunks: List of Chunk objects.
        embeddings: Numpy array of shape (N, D) corresponding to chunks.
        force: If True, delete and recreate the collection.

    Returns:
        Number of chunks written (upserted).
    """
    settings = get_settings()
    client = get_client()
    collection = get_collection()

    if force:
        try:
            client.delete_collection(name=settings.chroma_collection)
        except Exception:
            pass
        collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    if len(chunks) == 0:
        return 0

    if embeddings.ndim == 1:
        embeddings_arr = embeddings.reshape(len(chunks), -1)
    else:
        embeddings_arr = embeddings
    if len(embeddings_arr) != len(chunks):
        raise ValueError(f"Embeddings count {len(embeddings_arr)} != chunks count {len(chunks)}")

    # Identity is **chunk_id**, never content_sha256.
    #
    # content_sha256 is a hash of the chunk *body alone* (app/chunking.py), so
    # boilerplate that several scheme pages share verbatim collides: the
    # "Expense ratio — A fee payable to a mutual fund house…" definition, the
    # exit-load line, the minimum-investment block, the AMC's "Fund house"
    # block. Keying on it dropped 18 of 131 chunks on this corpus, and the old
    # `continue  # should not happen` hid it: the comment was wrong, it fired
    # 18 times.
    #
    # The loss was not merely 18 rows. The surviving copy belonged to whichever
    # scheme was written first, so `hdfc_elss`, `hdfc_small_cap`,
    # `hdfc_flexi_cap` and `hdfc_balanced_advantage` ended up with **no
    # "Understand terms" chunk at all**. Retrieval filters on scheme_id, so a
    # question about the expense ratio of ELSS or Small Cap could never reach an
    # expense-ratio definition — the text had been chunked, and then thrown away
    # where nothing could find it. chunk_id is unique per chunk (verified: 131
    # chunks, 131 distinct ids), so keying on it is lossless.
    existing: dict[str, str] = {}
    try:
        res = collection.get(include=["metadatas"])
        metas = res.get("metadatas") or []
        ids_list = res.get("ids") or []
        for cid in ids_list:
            existing[str(cid)] = str(cid)
    except Exception:
        existing = {}

    to_upsert_ids: list[str] = []
    to_upsert_embeds: list[Any] = []
    to_upsert_metas: list[dict[str, Any]] = []
    to_upsert_docs: list[str] = []

    # Stale = a chunk_id no longer produced by the current chunking run. Compare
    # on identity, not content, so an edited body is an update rather than a
    # delete-then-orphan.
    new_ids = {c.chunk_id for c in chunks}
    stale_ids = [cid for cid in existing if cid not in new_ids]

    if stale_ids:
        try:
            collection.delete(ids=stale_ids)
        except Exception:
            logger.debug("Failed to delete stale ids: %s", stale_ids)

    # Upsert every chunk. The previous "skip unchanged" branch saved a Chroma
    # write but not an embedding — embedding already happened upstream in
    # run_ingest — while making the return value understate what is stored.
    seen_in_batch: set[str] = set()
    for i, c in enumerate(chunks):
        if c.chunk_id in seen_in_batch:
            # Two chunks sharing an id would overwrite each other. That is a
            # chunking bug, so it is reported rather than absorbed.
            logger.error("duplicate chunk_id %s; the later copy is not stored", c.chunk_id)
            continue
        seen_in_batch.add(c.chunk_id)
        meta = to_chroma_metadata(c)
        to_upsert_ids.append(c.chunk_id)
        emb = embeddings_arr[i]
        # Convert to list/float list for Chroma
        if isinstance(emb, np.ndarray):
            to_upsert_embeds.append(emb.tolist())
        else:
            to_upsert_embeds.append(list(emb))
        to_upsert_metas.append(meta)
        to_upsert_docs.append(c.embed_text)

    if to_upsert_ids:
        collection.upsert(
            ids=to_upsert_ids,
            embeddings=to_upsert_embeds,
            metadatas=to_upsert_metas,
            documents=to_upsert_docs,
        )
    return len(to_upsert_ids)


def _compute_fingerprint(chunk_ids: list[str]) -> str:
    m = hashlib.sha256()
    for cid in chunk_ids:
        m.update(str(cid).encode("utf-8"))
    return m.hexdigest()[:12]


def corpus_fingerprint(chunks: list[Chunk] | None = None) -> str:
    """Return a 12-char sha256 fingerprint of sorted chunk_ids.

    If chunks is None, read from Chroma collection metadata/documents? Or recompute
    by reading chunks.jsonl is not required; better to read from collection if available.
    """
    if chunks is None:
        try:
            collection = get_collection()
            res = collection.get(include=["metadatas"])
            ids_list = res.get("ids") or []
            if ids_list:
                return _compute_fingerprint(sorted(ids_list))
        except Exception:
            pass
        # fallback: read from chunks file if exists
        try:
            from pathlib import Path

            from app.chunking import load_chunks

            path = Path(get_settings().chroma_path).parent / "data" / "processed" / "chunks.jsonl"
            # but chunks.jsonl lives under repo root? Repo root is cwd
            path2 = Path("data/processed/chunks.jsonl")
            chs = load_chunks(path2)
            return _compute_fingerprint(sorted(c.chunk_id for c in chs))
        except Exception:
            return ""
    return _compute_fingerprint(sorted(c.chunk_id for c in chunks))


def corpus_stats() -> dict[str, Any]:
    """Return corpus statistics."""
    try:
        collection = get_collection()
        res = collection.get(include=["metadatas"])
        ids_list = res.get("ids") or []
        metas = res.get("metadatas") or []
        count = len(ids_list)
    except Exception:
        count = 0
        metas = []

    max_fetched_at = None
    schemes: set[str] = set()
    for m in metas:
        if m and m.get("fetched_at"):
            fa = str(m["fetched_at"])
            if max_fetched_at is None or fa > max_fetched_at:
                max_fetched_at = fa
        if m and m.get("scheme_id"):
            schemes.add(str(m["scheme_id"]))
    return {
        "count": count,
        "max_fetched_at": max_fetched_at,
        "schemes": sorted(schemes),
    }


def corpus_is_empty() -> bool:
    """Return True if the corpus is empty."""
    try:
        collection = get_collection()
        try:
            count = collection.count()
        except Exception:
            res = collection.get()
            count = len(res.get("ids") or [])
        return count == 0
    except Exception:
        return True
