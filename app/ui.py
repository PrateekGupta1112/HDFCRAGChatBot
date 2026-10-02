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

**Deliberate deviations from the P9 spec**, all requested by the product owner
after P9 was accepted. They are presentation-only; none touches a compliance
guarantee or the retrieval/generation path:

- The app is titled **"HDFC MF Chat Bot"** rather than "MF Facts Bot" (both the
  ``st.title`` and the browser ``page_title``).
- ``disclaimer.WELCOME_LINE`` ends with "Try one of these:", which points at the
  three example-question buttons rendered directly beneath it. The welcome line
  and those three buttons are rendered on **every** run, not only on an empty
  chat — they used to vanish after the first turn.
- Clicking an example question now **answers it directly**. It no longer prefills
  the input box and waits for Enter. Both routes share one submission path
  (``_submitted``), so a clicked example cannot behave differently from a typed
  one.
- The per-turn metadata badge (``mode · latency · top_sim``) and the two debug
  expanders ("What the bot retrieved", "Guardrail") were removed. A bot bubble is
  now the answer text plus the freshness stamp.
- The ``Source:`` citation link was removed as well, so a bubble is *just* the
  answer and its date. The pipeline still attaches and verifies exactly one
  whitelisted citation per answer (AC1/AC2) — it is simply not rendered, and
  ``_render_citations`` was deleted. Note the answer *body* can still contain the
  URL as generated text; that is governed by the citation audit in
  :mod:`app.generation`, not by this view.
- A submitted question now renders **before** the pipeline runs, with a spinner
  while it is in flight. The user bubble is first drawn with the submitted text,
  then rewritten in place with ``trace.guard.redacted_query`` once the guard has
  run. Only the redacted form is ever stored, so D4/NFR-7 still holds; the
  submitted text is on screen transiently and never persisted.
- ``disclaimer.DISCLAIMER_SHORT`` ("Facts-only. No investment advice.") is no
  longer rendered as a caption under the title. The *full*
  ``disclaimer.DISCLAIMER`` is still rendered by the ``st.warning`` directly
  beneath it on every run, so the persistent-disclaimer requirement (FR-11, PRD
  §9.6) is unaffected — only the short duplicate line is gone.
- The four reference tabs (Sources / Scope & known limits / Disclaimer / About
  this project) were removed, along with ``_render_tabs``,
  ``_render_sources_tab``, ``_sources``, ``_fetched_at_by_scheme`` and the
  ``KNOWN_LIMITS`` / ``ABOUT_THIS_PROJECT`` copy they rendered.
"""

from __future__ import annotations

import sys
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
from app.pipeline import answer_turn, new_session  # noqa: E402

# --------------------------------------------------------------------------
# Constants. Every number the UI uses is named here rather than inlined.
# --------------------------------------------------------------------------

#: Widget keys. ``st.chat_input`` has no ``value=`` argument, so the input box is
#: seeded by writing the widget's own session-state key before the widget is
#: created (implementation.md §P9 gotchas). ``PENDING_KEY`` carries an example-
#: button click across the forced rerun that the click triggers; ``CLEAR_KEY``
#: carries the "empty the box" instruction back to the next run, which is the
#: only point at which a widget key can legally be written.
CHAT_INPUT_KEY = "mf_chat_input"
PENDING_KEY = "mf_chat_pending_question"
CLEAR_KEY = "mf_chat_clear_input"
MESSAGES_KEY = "mf_chat_messages"
SESSION_KEY = "mf_chat_session"

#: ``store.corpus_fingerprint`` returns a 12-character hex digest (§P9 checklist
#: row 14). Rendered with the same truncation it was produced with.
FINGERPRINT_CHARS = 12

#: The setup command quoted in the empty-corpus banner.
INGEST_COMMAND = "python -m app.ingest"

#: Shown in the spinner while a turn is in flight, between the question bubble
#: and the answer.
SPINNER_TEXT = "Searching the five HDFC scheme pages…"

# --------------------------------------------------------------------------
# Static copy
# --------------------------------------------------------------------------

#: PRD §12.3 (known limits) and the "About this project" walkthrough used to be
#: rendered as reference tabs below the chat. They were removed on request — the
#: four-tab reference strip (Sources / Scope & known limits / Disclaimer / About
#: this project) is gone, along with the code that built it. Nothing else was
#: changed: the persistent ``st.warning`` disclaimer above the chat is a separate
#: element and is still rendered on every run, so no user-facing compliance text
#: was lost with the tabs. See README "Known defects" for the AC11 rows this
#: invalidates.

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


def _bot_message(answer: Any) -> dict[str, Any]:
    """Flatten an :class:`~app.generation.Answer` into what the view needs.

    Only the two fields the bubble renders are carried: the body and the
    freshness date. The citation list, the generation mode, latency, top
    similarity, retrieved chunks and guard decision used to be carried too, for
    the ``Source:`` link, the metadata badge and the two debug expanders. Those
    are no longer shown, so they are no longer stored — holding them would put a
    second, never-rendered copy of the trace in session state for the life of the
    conversation.
    """
    return {
        "role": "bot",
        "text": _body_without_freshness(answer.text),
        "last_updated": answer.last_updated,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _render_bot_turn(message: dict[str, Any]) -> None:
    """One bot bubble: the answer, and the freshness stamp. Nothing else.

    The ``Source:`` citation link, the ``mode · latency · top_sim`` badge and the
    "What the bot retrieved" / "Guardrail" expanders that used to follow were
    all removed on request. The bubble is now the answer text plus its date.

    This is presentation only. The pipeline still verifies the answer against the
    retrieved chunks and still attaches exactly one whitelisted citation to
    :class:`~app.generation.Answer` (P8, AC1/AC2) — that citation is what the
    three-sentence, one-number-audit checks run against, and hiding it here does
    not weaken any of them. ``Answer.citations`` is simply not rendered.

    Note the answer *body* can still contain the source URL as text: the
    extractive fallback path inlines it, and the LLM is asked to cite. That is
    generated content and is governed by the citation audit, not by this view.
    """
    with st.chat_message("assistant"):
        st.markdown(message["text"])
        if message["last_updated"]:
            st.caption(f"{disclaimer.FRESHNESS_PREFIX} {message['last_updated']}")


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
# Page
# --------------------------------------------------------------------------


st.set_page_config(page_title="HDFC MF Chat Bot", page_icon="📊", layout="centered")

st.title("HDFC MF Chat Bot")
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

# Widget keys are read into locals at the top of the script, before any widget is
# created, because writing a key after its widget exists raises
# StreamlitWidgetAlreadyInstantiatedError. That is why an example-button click
# only ever sets PENDING_KEY and why emptying the box is deferred through
# CLEAR_KEY rather than done inline after answering.
if st.session_state.pop(CLEAR_KEY, None):
    st.session_state[CHAT_INPUT_KEY] = ""
# A pending example click *is* the submitted question. It is read here and
# answered below in the same run, rather than being written into CHAT_INPUT_KEY
# and waiting for the user to press Enter.
_pending = st.session_state.pop(PENDING_KEY, None)

# The welcome line and the three example questions are chrome, not history, so
# they are rendered unconditionally and always sit above the transcript. They
# used to disappear after the first turn.
st.info(disclaimer.WELCOME_LINE)

for question in disclaimer.EXAMPLE_QUESTIONS:
    if st.button(question, width="stretch"):
        # Hand the text to the *next* run rather than answering from this one.
        # Answering inline would append to MESSAGES_KEY and then draw nothing,
        # because everything already rendered this run is discarded by the
        # rerun that must follow — the click would look like it did nothing.
        # Setting the key and rerunning is what makes the answer stick.
        st.session_state[PENDING_KEY] = question
        st.rerun()

if st.session_state[MESSAGES_KEY]:
    _render_history()

_typed = st.chat_input("Ask a factual question…", key=CHAT_INPUT_KEY)

# One submission path for both routes into a turn. A clicked example arrives as
# `_pending`; a typed question arrives as `_typed`. They are answered by exactly
# the same code, so an example click can never take a different path through the
# pipeline than typing it out would.
_submitted = (_pending or _typed or "").strip()

if _submitted:
    # The box is emptied on the *next* run rather than here: `st.chat_input`
    # already clears what the user typed once it is submitted, but clearing the
    # key explicitly keeps the next question from arriving pre-filled with the
    # last one. It cannot be done in this run — the widget is already
    # instantiated by this point.
    st.session_state[CLEAR_KEY] = True

    # Draw the question *before* the blocking pipeline call. Streamlit pushes
    # each element to the browser as it is created, so the user bubble appears
    # within a frame and then a spinner runs while the LLM answers, instead of
    # staring at an unchanged page for the whole round-trip (25 s on the first
    # turn, while all-MiniLM-L6-v2 loads). Previously the turn was appended to
    # session state and followed by `st.rerun()`, which discarded this run
    # entirely — nothing was drawn until the answer was already computed, so the
    # question and the answer always appeared together.
    question_slot = st.empty()
    with question_slot.container():
        with st.chat_message("user"):
            st.markdown(_submitted)

    with st.spinner(SPINNER_TEXT):
        answer = answer_turn(_submitted, st.session_state[SESSION_KEY])

    # Now that the guard has run, overwrite the placeholder with the redacted
    # query. `st.empty()` containers are replaced wholesale on each
    # `container()` call, so this does not append a second bubble — it rewrites
    # the one on screen. This is what keeps the D4/NFR-7 promise visible: a
    # pasted PAN appears for the duration of the call and is then shown with the
    # digits removed. Only the redacted form is ever written to session state.
    redacted = answer.trace.guard.redacted_query if answer.trace.guard else ""
    with question_slot.container():
        with st.chat_message("user"):
            st.markdown(redacted)

    st.session_state[MESSAGES_KEY].append(_user_message(answer))
    bot = _bot_message(answer)
    st.session_state[MESSAGES_KEY].append(bot)
    _render_bot_turn(bot)

    # Rerun once the turn is complete, for two reasons that both depend on it.
    #
    # 1. `CLEAR_KEY` is only consumed at the top of the script, i.e. on the run
    #    *after* the widget was created. Without this rerun there is no such
    #    run, so the box would still show the submitted question when the answer
    #    arrived — the user sees their text sitting in the input and assumes
    #    nothing was sent. Worse, the flag would survive until the *next*
    #    interaction, and that user's next Enter would be consumed by the flag
    #    wiping the widget value: their second question vanished silently.
    # 2. The transcript below is re-rendered from MESSAGES_KEY rather than from
    #    the objects drawn above, so what is on screen and what is stored cannot
    #    drift apart.
    #
    # The rerun is cheap: `_submitted` is empty by then, so no pipeline call is
    # made and only the already-built message list is redrawn.
    st.rerun()

if summary["count"] == 0:
    st.error(SETUP_WARNING)