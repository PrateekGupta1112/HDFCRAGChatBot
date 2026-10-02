from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.chunking import Chunk
from app.config import get_settings


@pytest.fixture
def tmp_chroma(tmp_path, monkeypatch):
    # Set a unique chroma path for each test
    chroma_dir = tmp_path / "chroma"
    monkeypatch.setenv("CHROMA_PATH", str(chroma_dir))
    # Reset cached settings if any? get_settings is cached; we can rely on env or re-read
    from app import config

    config.get_settings.cache_clear()
    return chroma_dir


def make_synthetic_chunk(
    chunk_id: str,
    scheme_id: str,
    scheme_name: str,
    category: str,
    plan: str,
    section: str,
    body: str,
    is_table: bool = False,
) -> Chunk:
    from app.tokenization import count_tokens

    embed_text = f"[Scheme: {scheme_name} | Category: {category} | Plan: {plan} | Section: {section}]\n{body}"
    return Chunk(
        chunk_id=chunk_id,
        scheme_id=scheme_id,
        scheme_name=scheme_name,
        category=category,
        plan=plan,
        section=section,
        source_url=f"https://example.com/synthetic/{scheme_id}",
        fetched_at="2026-10-02",
        chunk_index=0,
        is_table=is_table,
        content_sha256="0" * 64,  # will be recomputed conceptually; tests use real data
        embed_text=embed_text,
        display_text=body,
    )


def test_upsert_idempotent(tmp_chroma):
    from app.store import to_chroma_metadata, upsert_chunks, corpus_stats

    body = "Section: Expense ratio. Value: 1.11%."
    c1 = Chunk(
        chunk_id="abc1234567890ab",
        scheme_id="synthetic_a",
        scheme_name="Synthetic A",
        category="Test",
        plan="Direct",
        section="Fees",
        source_url="https://example.com/synthetic-1",
        fetched_at="2026-10-02",
        chunk_index=0,
        is_table=False,
        content_sha256="1111111111111111111111111111111111111111111111111111111111111111",
        embed_text=f"[Scheme: Synthetic A | Category: Test | Plan: Direct | Section: Fees]\n{body}",
        display_text=body,
    )
    emb = np.array([[0.1, 0.2, 0.3, 0.4] * 96], dtype=np.float32)  # 384 dims approx
    # pad/truncate to 384
    emb = emb[:, :384]

    n1 = upsert_chunks([c1], emb, force=True)
    assert n1 == 1
    s1 = corpus_stats()["count"]
    n2 = upsert_chunks([c1], emb, force=False)
    s2 = corpus_stats()["count"]
    assert s2 == s1 == 1


def test_upsert_force_rebuild(tmp_chroma):
    from app.store import upsert_chunks, corpus_stats

    body = "Value: 1.11%."
    c = Chunk(
        chunk_id="id1",
        scheme_id="synthetic_a",
        scheme_name="Synthetic A",
        category="Test",
        plan="Direct",
        section="Fees",
        source_url="https://example.com/synthetic-1",
        fetched_at="2026-10-02",
        chunk_index=0,
        is_table=False,
        content_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        embed_text=f"[Scheme: Synthetic A | Category: Test | Plan: Direct | Section: Fees]\n{body}",
        display_text=body,
    )
    emb = np.zeros((1, 384), dtype=np.float32)
    upsert_chunks([c], emb, force=True)
    assert corpus_stats()["count"] == 1
    # add another with force
    c2 = Chunk(
        chunk_id="id2",
        scheme_id="synthetic_b",
        scheme_name="Synthetic B",
        category="Test",
        plan="Direct",
        section="Fees",
        source_url="https://example.com/synthetic-2",
        fetched_at="2026-10-02",
        chunk_index=0,
        is_table=False,
        content_sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        embed_text=f"[Scheme: Synthetic B | Category: Test | Plan: Direct | Section: Fees]\n{body}",
        display_text=body,
    )
    emb2 = np.zeros((1, 384), dtype=np.float32)
    upsert_chunks([c2], emb2, force=True)
    assert corpus_stats()["count"] == 1  # rebuilt with just c2


def test_metadata_flat_scalars(tmp_chroma):
    from app.store import to_chroma_metadata

    c = Chunk(
        chunk_id="id1",
        scheme_id="synthetic_a",
        scheme_name="Synthetic A",
        category="Test",
        plan="Direct",
        section="Fees",
        source_url="https://example.com/synthetic-1",
        fetched_at="2026-10-02",
        chunk_index=5,
        is_table=True,
        content_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        embed_text="[Scheme: ...]\nbody",
        display_text="body",
    )
    meta = to_chroma_metadata(c)
    for v in meta.values():
        assert isinstance(v, (str, int)), f"Non-scalar in metadata: {type(v)}"
    assert meta["is_table"] == "true"


def test_corpus_fingerprint_stable_and_differs(tmp_chroma):
    from app.store import upsert_chunks, corpus_fingerprint

    body = "Value: 1.11%."
    c1 = Chunk(
        chunk_id="id1",
        scheme_id="synthetic_a",
        scheme_name="Synthetic A",
        category="Test",
        plan="Direct",
        section="Fees",
        source_url="https://example.com/s1",
        fetched_at="2026-10-02",
        chunk_index=0,
        is_table=False,
        content_sha256="1111111111111111111111111111111111111111111111111111111111111111",
        embed_text=f"[Scheme: A | Cat: T | Plan: D | Section: F]\n{body}",
        display_text=body,
    )
    c2 = Chunk(
        chunk_id="id2",
        scheme_id="synthetic_b",
        scheme_name="Synthetic B",
        category="Test",
        plan="Direct",
        section="Fees",
        source_url="https://example.com/s2",
        fetched_at="2026-10-02",
        chunk_index=0,
        is_table=False,
        content_sha256="2222222222222222222222222222222222222222222222222222222222222222",
        embed_text=f"[Scheme: B | Cat: T | Plan: D | Section: F]\n{body}",
        display_text=body,
    )
    emb = np.zeros((1, 384), dtype=np.float32)
    upsert_chunks([c1], emb[:1], force=True)
    fp1 = corpus_fingerprint([c1])
    fp1b = corpus_fingerprint([c1])
    assert fp1 == fp1b
    fp2 = corpus_fingerprint([c2])
    assert fp2 != fp1


def test_corpus_is_empty(tmp_chroma):
    from app.store import corpus_is_empty, upsert_chunks

    assert corpus_is_empty() is True
    body = "v"
    c = Chunk(
        chunk_id="id1",
        scheme_id="synthetic_a",
        scheme_name="Synthetic A",
        category="Test",
        plan="Direct",
        section="F",
        source_url="https://example.com/s1",
        fetched_at="2026-10-02",
        chunk_index=0,
        is_table=False,
        content_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        embed_text=f"[Scheme: A | Cat: T | Plan: D | Section: F]\n{body}",
        display_text=body,
    )
    emb = np.zeros((1, 384), dtype=np.float32)
    upsert_chunks([c], emb, force=True)
    assert corpus_is_empty() is False
