"""Streamlit UI — Phase P9 (implementation.md §P9, PRD FR-11/FR-12).

Entry point::

    streamlit run app/ui.py

This module is deliberately thin. It owns presentation and nothing else: every
compliance guarantee (no advice, no PII echo, one whitelisted citation, the
three-sentence cap, the freshness stamp) is enforced inside
:mod:`app.generation` and :mod:`app.pipeline` before an answer reaches here.
The UI cannot widen any of them, because it only renders what it is handed.

**Imports are restricted to** ``app.config``, ``app.disclaimer``,
``app.pipeline`` and ``app.store`` (§P9). Nothing here may import
``guardrails``, ``retrieval``, ``generation`` or ``embedding`` directly — the
guard decision and the retrieved chunks are read off ``Answer.trace``, which
P8 already populated, so reaching past it would duplicate pipeline logic in the
view layer.

**No raw query is ever stored.** The user bubble renders
``trace.guard.redacted_query``, the only form of the input the pipeline is
allowed to keep (architecture §11, D4/NFR-7). A pasted PAN is displayed with
the digits removed, which is the visible proof that nothing was persisted.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any

# Streamlit puts only the *script's own* directory on sys.path — for
# `streamlit run app/ui.py` that is `<repo>/app`, never the repo root. Without
# this, `from app import ...` raises ModuleNotFoundError and the browser shows a
# traceback while `curl http://localhost:8501` still answers 200, because the
# HTTP shell is healthy even when the script is not. Prepend the repo root
# before the first project import so the documented command just works.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402  (must follow the sys.path bootstrap)

from app import disclaimer, store  # noqa: E402
from app.pipeline import SOURCES_CSV, answer_turn, new_session  # noqa: E402

# --------------------------------------------------------------------------
# Constants. Every number the UI uses is named here rather than inlined.
# --------------------------------------------------------------------------

#: Widget keys. ``st.chat_input`` has no ``value=`` argument, so prefill is done
#: by writing the widget's own session-state key before the widget is created
#: (implementation.md §P9 gotchas). The pending key carries the click across the
#: forced rerun that a button click triggers.
CHAT_INPUT_KEY = "mf_facts_chat_input"
PENDING_KEY = "mf_facts_pending_question"
CLEAR_KEY = "mf_facts_clear_input"
MESSAGES_KEY = "mf_facts_messages"
SESSION_KEY = "mf_facts_session"

#: Retrieved-chunk snippets are truncated to this many characters inside the
#: expander. A chunk is ~250 characters today, but a table-heavy chunk can be
#: several times that and would otherwise push the chat column unreadably wide
#: (§P9 gotchas).
SNIPPET_CHARS = 300

#: Decimal places for the similarity shown in the metadata badge and the
#: retrieved-chunks expander.
SIMILARITY_DP = 2

#: Decimal places for the per-turn latency badge.
LATENCY_DP = 2

#: ``store.corpus_fingerprint`` returns a 12-character hex digest (§P9 checklist
#: row 14). Rendered with the same truncation it was produced with.
FINGERPRINT_CHARS = 12

#: The setup command quoted in the empty-corpus banner.
INGEST_COMMAND = "python -m app.ingest"

# --------------------------------------------------------------------------
# Static copy
# --------------------------------------------------------------------------

#: PRD §12.3, reproduced verbatim. Kept in this module rather than
#: ``app.disclaimer`` because §P9 lists it as UI content and the disclaimer
#: module is reserved for the shared refusal/disclaimer strings (D5); copying
#: these bullets there would put project documentation next to legal copy that
#: the tests assert on exactly.
KNOWN_LIMITS = """\
**Single AMC, 5 schemes, 5 pages.** Questions about other funds or other AMCs → \
"not in sources".

**Snapshot, not a live feed.** Answers reflect the page as of `fetched_at` only.

**Aggregator pages, not official AMC/AMFI pages** (see §4.2). Values are \
cross-checkable on the linked page; for regulated facts (e.g., TER) the official \
factsheet/SID is authoritative.

**No PDF ingestion.** Factsheets/SIDs (PDF) are out of scope for v1, so some \
values may not be present in HTML.

**English only**, and no Hindi/regional-language support.

**MiniLM-L6-v2 is a small model**: good for topical retrieval, weak at very \
fine-grained numeric comparisons between schemes. Multi-scheme numeric comparison \
is deliberately refused.

**Single-turn oriented**: last 3 turns are kept for context only; no long-term \
memory, no personalization.

**No returns math**: CAGR, alpha, expense-ratio-vs-return breakeven — all out of \
scope by policy.

**Not a SEBI-registered advisor.** Educational tool for a class project."""

ABOUT_THIS_PROJECT = """\
A grounded, facts-only question-answering bot over five HDFC mutual fund scheme
pages. It retrieves published facts and cites the page each fact came from. It
does not give advice, forecast returns, or compute performance.

The pipeline, in order:

1. **Loading** — each scheme URL in `data/sources.csv` is fetched once and saved
   to `data/raw/`. A cached copy is reused until you pass `--force`.
2. **Chunking** — the page is split into section-scoped chunks (about 200
   tokens, 60-token overlap). Tables stay intact as their own chunks instead of
   being flattened into prose.
3. **Embedding** — every chunk is encoded locally with
   `sentence-transformers/all-MiniLM-L6-v2`. No chunk leaves the machine.
4. **ChromaDB** — vectors and metadata are stored in a persistent collection
   using cosine distance, with `scheme_id` filterable so a named scheme narrows
   the search to its own page.
5. **Retrieval** — the query is embedded, the nearest chunks above a similarity
   floor are kept, and Maximal Marginal Relevance (MMR) re-ranks them so the
   context is not five near-duplicates.
6. **Grounded answer** — the retrieved chunks are the only context the model
   sees. The answer is then verified in code: one citation, drawn from the
   retrieved chunks; every number in the body present in the context; at most
   three sentences; a freshness stamp. Anything that fails is served as a
   refusal rather than as a guess.

Open the **Sources** tab for the five pages and the date each was fetched. Open
**Scope & known limits** for what this bot cannot do."""

SETUP_WARNING = (
    "The corpus is empty, so there is nothing to retrieve yet.\n\n"
    f"Run `{INGEST_COMMAND}` once, then reload this page. It fetches the five "
    "scheme pages, chunks them, embeds them and fills ChromaDB — a couple of "
    "minutes the first time, and it needs network access."
)

# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------


@st.cache_resource
def _collection():
    """The Chroma collection handle, cached for the life of the server process.

    Streamlit re-executes this whole script on every click and every keystroke,
    and :func:`app.store.get_client` constructs a fresh ``PersistentClient`` on
    each call — re-opening the HNSW index every time. Caching here is what keeps
    a rerun at tens of milliseconds rather than seconds (§P9 gotchas).

    The embedding model needs the same treatment and does not need any: it is
    reached through :func:`app.embedding.get_model`, which is already
    ``@lru_cache(maxsize=1)`` at module scope, so it is loaded once per server
    process and survives reruns on its own. §P9 restricts this file's imports to
    ``config``, ``disclaimer``, ``pipeline`` and ``store``, so it cannot import
    ``app.embedding`` to wrap it — it is warmed lazily by the first answered turn
    instead, which is why the very first question takes a couple of seconds
    longer than the ones after it.
    """
    return store.get_collection()


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------


def _sources() -> list[dict[str, str]]:
    """Read ``data/sources.csv``.

    Resolved from ``__file__`` by the caller (``SOURCES_CSV``), not from the
    working directory, because Streamlit does not guarantee the repo root.
    """
    with open(SOURCES_CSV, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fetched_at_by_scheme() -> dict[str, str]:
    """Map each ``scheme_id`` to the newest ``fetched_at`` among its chunks.

    Read off the collection rather than ``corpus_stats()``, which only exposes a
    single corpus-wide ``max_fetched_at``. A per-scheme date is what tells a user
    whether *this* scheme's page is stale.
    """
    result: dict[str, str] = {}
    try:
        got = _collection().get(include=["metadatas"])
    except Exception:  # pragma: no cover - store is unreachable
        return result
    for meta in got.get("metadatas") or []:
        if not meta:
            continue
        scheme_id = str(meta.get("scheme_id") or "")
        fetched_at = str(meta.get("fetched_at") or "")
        if not scheme_id or not fetched_at:
            continue
        if fetched_at > result.get(scheme_id, ""):
            result[scheme_id] = fetched_at
    return result


def _corpus_summary() -> dict[str, Any]:
    """Fingerprint, counts and scheme coverage, read once per render."""
    stats = store.corpus_stats()
    return {
        "fingerprint": store.corpus_fingerprint(),
        "count": stats["count"],
        "schemes": stats["schemes"],
        "max_fetched_at": stats["max_fetched_at"],
    }


# --------------------------------------------------------------------------
# Message construction
# --------------------------------------------------------------------------


def _body_without_freshness(text: str) -> str:
    """Drop the trailing freshness stamp so the UI can render it once, styled.

    ``enforce`` appends ``Last updated from sources: <date>`` outside the
    sentence budget. Rendering ``answer.text`` verbatim and then rendering
    ``answer.last_updated`` as well would show the date twice; this returns the
    body only, and falls back to the full text when the stamp is absent (an
    ERROR answer carries no stamp).
    """
    body, marker, _ = text.partition(disclaimer.FRESHNESS_PREFIX)
    return body.strip() if marker else text.strip()


def _user_message(answer: Any) -> dict[str, Any]:
    """The user bubble, built from the redacted query only.

    The raw string is not passed in — ``answer_turn`` has already dropped it,
    and this reads only what survived the guard. If the guard is somehow absent
    the bubble renders nothing rather than falling back to the input, because a
    missing redacted form is a bug, not licence to show the original.
    """
    guard = answer.trace.guard
    redacted = guard.redacted_query if guard is not None else ""
    return {"role": "user", "text": redacted}


def _bot_message(answer: Any, latency_s: float) -> dict[str, Any]:
    """Flatten an :class:`~app.generation.Answer` into what the view needs."""
    retr = answer.trace.retrieval
    guard = answer.trace.guard
    return {
        "role": "bot",
        "text": _body_without_freshness(answer.text),
        "last_updated": answer.last_updated,
        "citations": [
            {"url": c.url, "scheme_name": c.scheme_name, "section": c.section}
            for c in answer.citations
        ],
        "mode": answer.generation_mode,
        "latency_s": latency_s,
        "top_similarity": retr.max_similarity if retr is not None else 0.0,
        "retrieved": [
            {
                "rank": c.rank,
                "scheme_name": c.scheme_name,
                "section": c.section,
                "similarity": c.similarity,
                "text": c.text,
            }
            for c in (retr.chunks if retr is not None else [])
        ],
        "guard": {
            "action": guard.action.value if guard is not None else "unknown",
            "matched_pattern": guard.matched_pattern if guard is not None else None,
        },
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _render_citations(citations: list[dict[str, str]]) -> None:
    """The verified source link(s). Exactly one is ever present (AC2)."""
    if not citations:
        return
    for cite in citations:
        section = cite.get("section") or ""
        label = f"{cite['scheme_name']} — {section}" if section else cite["scheme_name"]
        st.markdown(f"Source: [{label}]({cite['url']})")


def _render_badge(message: dict[str, Any]) -> None:
    """``mode · latency · top_sim`` — how the answer was produced, not just what."""
    st.caption(
        f"mode: {message['mode']} · latency: {message['latency_s']:.{LATENCY_DP}f}s"
        f" · top_sim: {message['top_similarity']:.{SIMILARITY_DP}f}"
    )


def _render_retrieved(message: dict[str, Any]) -> None:
    """What the retriever actually handed the model, and how it scored."""
    with st.expander("▸ What the bot retrieved"):
        chunks = message["retrieved"]
        if not chunks:
            st.caption("No chunks were retrieved for this turn.")
            return
        for chunk in chunks:
            st.markdown(
                f"**#{chunk['rank']}** · {chunk['scheme_name']} · "
                f"{chunk['section']} · similarity "
                f"`{chunk['similarity']:.{SIMILARITY_DP}f}`"
            )
            snippet = chunk["text"]
            if len(snippet) > SNIPPET_CHARS:
                snippet = snippet[:SNIPPET_CHARS].rstrip() + "…"
            st.caption(snippet)


def _render_guard(message: dict[str, Any]) -> None:
    """Which guardrail branch ran, and what it matched.

    Shown for every turn, not just refusals. A question that was answered
    normally still went through the classifier, and being able to see that it
    passed is the point of the guardrails being auditable.
    """
    guard = message["guard"]
    with st.expander("▸ Guardrail"):
        st.markdown(f"action: `{guard['action']}`")
        pattern = guard.get("matched_pattern")
        if pattern:
            # The matched pattern is a detector name, not user input, so it is
            # safe to show. The raw query is never rendered anywhere.
            st.markdown(f"matched pattern: `{pattern}`")
        else:
            st.caption("matched pattern: none")


def _render_bot_turn(message: dict[str, Any]) -> None:
    """One bot bubble: answer, source, freshness, badge, and the two expanders."""
    with st.chat_message("assistant"):
        st.markdown(message["text"])
        _render_citations(message["citations"])
        if message["last_updated"]:
            st.caption(f"{disclaimer.FRESHNESS_PREFIX} {message['last_updated']}")
        _render_badge(message)
        _render_retrieved(message)
        _render_guard(message)


def _render_history() -> None:
    for message in st.session_state[MESSAGES_KEY]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["text"])
        else:
            _render_bot_turn(message)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def _render_sidebar(summary: dict[str, Any]) -> None:
    with st.sidebar:
        st.subheader("Corpus")
        st.markdown(f"**Fingerprint:** `{summary['fingerprint']}`")
        st.markdown(f"**Chunks:** {summary['count']}")
        st.markdown(f"**Schemes indexed:** {len(summary['schemes'])}")
        if summary["max_fetched_at"]:
            st.markdown(f"**Newest fetch:** {summary['max_fetched_at']}")
        if summary["fingerprint"] == "":
            st.caption(
                f"The fingerprint is empty, which is what an unbuilt corpus "
                f"looks like. Run `{INGEST_COMMAND}`."
            )

        st.subheader("Chat")
        if st.button("Clear chat", width="stretch"):
            st.session_state[MESSAGES_KEY] = []
            st.session_state[SESSION_KEY] = new_session()
            # Clear the input through the deferred flag too, or the previous
            # question reappears in the box the user is about to type into.
            st.session_state[CLEAR_KEY] = True
            st.rerun()


# --------------------------------------------------------------------------
# Reference tabs
# --------------------------------------------------------------------------


def _render_sources_tab() -> None:
    rows = _sources()
    fetched = _fetched_at_by_scheme()
    if not rows:
        st.caption(f"`{SOURCES_CSV}` is missing or empty. Run `{INGEST_COMMAND}`.")
        return
    st.markdown(
        "The five pages this bot reads. Everything it can tell you comes from "
        "these and nothing else — there is no other index to fall back on."
    )
    for row in rows:
        scheme_id = row.get("scheme_id", "")
        st.markdown(f"**{row.get('scheme_name', scheme_id)}**")
        st.markdown(
            f"- [{row.get('url', '')}]({row.get('url', '')})\n"
            f"- category: {row.get('category', '')} · plan: {row.get('plan', '')}\n"
            f"- fetched_at: {fetched.get(scheme_id) or 'not indexed — run ingest'}"
        )
    missing = [r for r in rows if r.get("scheme_id") not in fetched]
    if missing:
        st.warning(
            f"{len(missing)} of {len(rows)} schemes have no chunks in the index. "
            f"Run `{INGEST_COMMAND}`."
        )


def _render_tabs() -> None:
    rows = _sources()
    sources_tab, limits_tab, disclaimer_tab, about_tab = st.tabs(
        [
            f"Sources ({len(rows)})",
            "Scope & known limits",
            "Disclaimer",
            "About this project",
        ]
    )
    with sources_tab:
        _render_sources_tab()
    with limits_tab:
        st.markdown(KNOWN_LIMITS)
    with disclaimer_tab:
        st.markdown(disclaimer.DISCLAIMER)
    with about_tab:
        st.markdown(ABOUT_THIS_PROJECT)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


st.set_page_config(page_title="MF Facts Bot", page_icon="📊", layout="centered")

st.title("MF Facts Bot")
st.caption(disclaimer.DISCLAIMER_SHORT)
# Persistent, above the chat, on every render — the disclaimer must not be
# something the user has to scroll back up to find after asking a question.
st.warning(disclaimer.DISCLAIMER)

summary = _corpus_summary()
_render_sidebar(summary)

# Session state is created explicitly rather than read with a default, so the
# keys exist before any widget tries to write them. `Session` is the pipeline's
# PII-safe audit buffer: it holds redacted turns only and is never written to
# disk, which is also why a page reload starts empty (no persistence).
if MESSAGES_KEY not in st.session_state:
    st.session_state[MESSAGES_KEY] = []
if SESSION_KEY not in st.session_state:
    st.session_state[SESSION_KEY] = new_session()
if CHAT_INPUT_KEY not in st.session_state:
    st.session_state[CHAT_INPUT_KEY] = ""

# Widget keys are read into locals at the top of the script, before any widget
# is created, so a pending prefill is in place by the time `st.chat_input` is
# constructed. Writing a widget key after that widget exists raises
# StreamlitWidgetAlreadyInstantiatedError, which is why the example buttons only
# ever set PENDING_KEY and why clearing the box is deferred through CLEAR_KEY
# rather than done inline after answering.
if st.session_state.pop(CLEAR_KEY, None):
    st.session_state[CHAT_INPUT_KEY] = ""
_pending = st.session_state.pop(PENDING_KEY, None)
if _pending:
    st.session_state[CHAT_INPUT_KEY] = _pending

if st.session_state[MESSAGES_KEY]:
    _render_history()
else:
    st.info(disclaimer.WELCOME_LINE)

for question in disclaimer.EXAMPLE_QUESTIONS:
    if st.button(question, width="stretch"):
        # Hand the text to the next run rather than answering from this one, so
        # the question appears in the input box and is submitted the same way a
        # typed one would be. Answering directly here would skip the user's
        # confirmation step and make the example buttons feel like a different
        # app.
        st.session_state[PENDING_KEY] = question
        st.rerun()

_typed = st.chat_input("Ask a factual question…", key=CHAT_INPUT_KEY)

if _typed and _typed.strip():
    # The box is emptied on the *next* run rather than here: `st.chat_input`
    # already clears what the user typed once it is submitted, but clearing the
    # key explicitly keeps the next question from arriving pre-filled with the
    # last one. It cannot be done in this run — the widget is already
    # instantiated by this point.
    st.session_state[CLEAR_KEY] = True
    started = time.perf_counter()
    answer = answer_turn(_typed, st.session_state[SESSION_KEY])
    latency_s = time.perf_counter() - started
    st.session_state[MESSAGES_KEY].append(_user_message(answer))
    st.session_state[MESSAGES_KEY].append(_bot_message(answer, latency_s))
    st.rerun()

if summary["count"] == 0:
    st.error(SETUP_WARNING)

_render_tabs()