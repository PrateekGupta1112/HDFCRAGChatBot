"""Query-path orchestration (architecture §8, implementation.md §P8).

:func:`answer_turn` is the single entry point both front ends need — ``app/chat.py``
here, ``app/ui.py`` in P9. Everything under it was built and unit-tested in P5, P6
and P7; this module's job is to call them **in the right order** and to guarantee
that no query can crash the app.

**The order is the product, not a style choice.** Three properties of §8 are
load-bearing:

* :func:`guardrails.classify` runs **first**, so a PAN is replaced before the
  string is embedded, sent to a model, or written to the session (§11).
* Refusals return **without calling** :func:`retrieval.retrieve`. That is what
  makes FR-7's "without restating or partially answering" true rather than
  aspirational — there is no scheme text in the process to leak.
* Retrieval and generation both receive ``GuardDecision.redacted_query``. The raw
  input is never passed on.

**Callables are called through their module**, not imported by name
(``guardrails.classify(...)``, ``store.corpus_is_empty()``). A direct import binds
the name at import time, so a test that patched ``app.guardrails.classify`` would
silently patch nothing and the guardrail test would pass against the *real*
classifier. Module-qualified calls are what let the P8 tests assert the
properties above instead of merely intending them.

**The result types are re-exported, not redefined.** :class:`AnswerKind`,
:class:`Citation`, :class:`Trace` and :class:`Answer` are defined in
``app.generation`` (P7) because ``enforce()`` returns an ``Answer`` and P7's file
table did not include this module. Defining them again here would create two
unrelated ``Answer`` classes and an ``isinstance`` check that lies.
"""

from __future__ import annotations

import csv
import logging
import traceback as _traceback
from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from app import disclaimer, guardrails, retrieval, store
from app.config import Settings, get_settings
from app.errors import CorpusEmptyError, LLMError
from app.generation import (
    Answer,
    AnswerKind,
    Citation,
    Trace,
    # The FR-6 stamp format belongs to P7, which appends it to every generated
    # answer. Reusing the formatter rather than re-deriving "prefix + date, or
    # the word 'unknown'" is what stops the two from drifting apart.
    _freshness_line as _freshness_stamp,
    enforce,
    generate,
)
from app.scheme_aliases import resolve_scheme_id

logger = logging.getLogger(__name__)

__all__ = [
    "Answer",
    "AnswerKind",
    "Citation",
    "MODE_STATIC",
    "Session",
    "Trace",
    "Turn",
    "answer_turn",
    "get_max_fetched_at",
    "new_session",
    "static_answer",
]


# ==========================================================================
# Constants
# ==========================================================================

#: ``Answer.generation_mode`` for every answer that no model produced. P7's
#: ``MODE_LLM`` / ``MODE_EXTRACTIVE`` cover generated text; refusals, not-found
#: responses and setup errors are not generated at all, and architecture §8.2
#: names this value "static".
MODE_STATIC = "static"

#: E6 in architecture §12. Shown when the vector store has no chunks, which means
#: ingestion has not been run — an operator problem with a known fix, not a
#: question the bot failed to answer.
SETUP_INSTRUCTIONS = "Corpus not built. Run: python -m app.ingest"

#: E14 in architecture §12. Deliberately says nothing about what went wrong:
#: a traceback is a bug report, not a user-facing answer.
APOLOGY = (
    "Sorry — something went wrong on my side while I was preparing that answer. "
    "Please try asking again."
)

#: How many stack frames the ERROR Trace records. A path through six layers is
#: enough to locate the fault; the full traceback goes to the log.
_TRACE_FRAME_LIMIT = 5

#: ``data/sources.csv`` resolved from ``__file__`` rather than the working
#: directory, matching ``scheme_aliases.ALIASES_PATH`` — the Streamlit child
#: process and the eval harness are not guaranteed to start in the repo root.
SOURCES_CSV = Path(__file__).resolve().parent.parent / "data" / "sources.csv"


# ==========================================================================
# Session (architecture §11)
# ==========================================================================


@dataclass(frozen=True)
class Turn:
    """One exchange, as it is allowed to exist in memory.

    Attributes:
        redacted_query: ``GuardDecision.redacted_query`` and nothing else. The raw
            input is never stored (D4, NFR-7).
        kind: The :class:`AnswerKind` this turn produced, so history can be
            rendered without re-deriving it.
        generation_mode: ``"llm"``, ``"extractive_fallback"`` or ``"static"``.
        citation_urls: The URLs cited, normalised-free, for the history view.
    """

    redacted_query: str
    kind: AnswerKind
    generation_mode: str
    citation_urls: tuple[str, ...]


@dataclass
class Session:
    """An in-memory conversation buffer. Ephemeral; never written to disk.

    **Intentionally mutable**, unlike every other dataclass in the project. Its
    whole purpose is to grow as the conversation does, and a ``frozen`` dataclass
    wrapping a mutable ``deque`` would still permit the append while asserting
    that it could not happen.

    Attributes:
        turns: The most recent exchanges, oldest first. The cap is enforced by
            ``deque(maxlen=...)``, so it is a property of the container rather
            than a rule somebody has to remember to apply.
        max_turns: The cap, from ``settings.max_chat_history_turns``.
    """

    turns: deque[Turn]
    max_turns: int

    def add(self, turn: Turn) -> None:
        """Append a turn, evicting the oldest if the cap is already reached."""
        self.turns.append(turn)

    def as_list(self) -> list[Turn]:
        """Return the turns oldest-first as a plain list, for rendering."""
        return list(self.turns)


def new_session(settings: Settings | None = None) -> Session:
    """Create an empty session capped at ``settings.max_chat_history_turns``.

    Args:
        settings: Injected settings, for tests. ``None`` uses the process-wide
            singleton.

    Returns:
        A :class:`Session` with no turns.
    """
    s = settings or get_settings()
    # maxlen=0 is legal and drops everything appended, which is the correct
    # reading of a cap of zero rather than an error.
    cap = max(0, s.max_chat_history_turns)
    return Session(turns=deque(maxlen=cap), max_turns=cap)


# ==========================================================================
# Corpus facts
# ==========================================================================


def get_max_fetched_at() -> str | None:
    """Return the newest ``fetched_at`` across the corpus, or ``None``.

    Drives the FR-6 freshness stamp. The *newest* source is used deliberately:
    it is the least stale claim the corpus as a whole supports (architecture §11,
    invariant 3).

    Returns:
        The ISO date, or ``None`` when the corpus is empty or unreadable.
    """
    return store.corpus_stats().get("max_fetched_at")


@lru_cache(maxsize=1)
def _source_index() -> dict[str, tuple[str, str]]:
    """Return ``{scheme_id: (scheme_name, source_url)}`` from ``data/sources.csv``.

    Needed to cite *a scheme page* without retrieving: architecture §8.2 requires
    the performance refusal to link the scheme the user asked about, and §8.3
    requires the not-found response to link the scheme page when the query names
    one. Both are refusals/off-corpus paths where retrieval must not run, so the
    URL has to come from the manifest rather than from the vector store.

    This does not reuse ``ingest.read_sources``. That function raises on a
    malformed row, which is the right behaviour for a build step and the wrong
    one here: a missing citation must never take a refusal down with it.

    Returns:
        The index, or an empty dict if the manifest is missing or unreadable.
    """
    try:
        with SOURCES_CSV.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as exc:  # noqa: BLE001 - a missing manifest must not fatal
        logger.warning("sources manifest unreadable (%s); citations fall back to AMFI",
                       type(exc).__name__)
        return {}

    index: dict[str, tuple[str, str]] = {}
    for row in rows:
        scheme_id = (row.get("scheme_id") or "").strip()
        url = (row.get("url") or "").strip()
        if scheme_id and url:
            index[scheme_id] = ((row.get("scheme_name") or "").strip(), url)
    return index


def _scheme_page_for(query: str) -> str | None:
    """Return the canonical page of the one scheme ``query`` names.

    Args:
        query: The **redacted** query. Never the raw input.

    Returns:
        The scheme's ``source_url``, or ``None`` when the query names no scheme
        or names more than one — in which case the caller falls back to the
        educational link rather than picking a winner.
    """
    scheme_id = resolve_scheme_id(query)
    if scheme_id is None:
        return None
    entry = _source_index().get(scheme_id)
    return entry[1] if entry else None


def _citation_for_url(url: str) -> Citation:
    """Build a :class:`Citation` for a URL the pipeline chose itself.

    Static answers cite a page without retrieving from it, so there is no chunk
    and therefore no section. ``scheme_name`` is filled from the manifest when the
    URL is a corpus page, so the UI has a label to show; it stays empty for the
    educational link, which is not a scheme.
    """
    name = ""
    for scheme_name, known_url in _source_index().values():
        if known_url == url:
            name = scheme_name
            break
    return Citation(url=url, section="", scheme_name=name)


# ==========================================================================
# Static answers
# ==========================================================================


def static_answer(
    kind: AnswerKind,
    text: str,
    citations: Iterable[str] = (),
    max_fetched_at: str | None = None,
    trace: Trace | None = None,
) -> Answer:
    """Build a deterministic Answer — refusals, not-found, setup errors.

    Every kind carries the FR-6 freshness stamp, **including refusals**: §8.2
    states it as invariant 2, and a refusal that omitted the date would be
    claiming to be timeless. :data:`AnswerKind.ERROR` is the single exception and
    is delegated to :func:`_error_answer`, so the rule holds for every caller
    rather than depending on each one remembering it.

    The body is **not** run through ``normalize_ws``, unlike P7's generated text.
    The copy here is ``app.disclaimer``'s, reproduced verbatim from PRD §9.3 and
    §9.4 (D5), and normalising it would fold its em dashes and curly apostrophes
    to ASCII — a silent edit of the one string the PRD pins down.

    Args:
        kind: The answer kind.
        text: Body copy, from ``app.disclaimer`` for anything user-facing.
        citations: URLs to attach. Converted to :class:`Citation` objects here so
            callers can pass the bare strings the spec names.
        max_fetched_at: From :func:`get_max_fetched_at`. Ignored for ``ERROR``.
        trace: Trace built by the caller. ``None`` produces an empty one.

    Returns:
        A frozen :class:`Answer` with ``generation_mode="static"``.
    """
    if kind is AnswerKind.ERROR:
        return _error_answer(text, trace, actions=["static_error"])
    return Answer(
        kind=kind,
        text=f"{text} {_freshness_stamp(max_fetched_at)}",
        citations=[_citation_for_url(url) for url in citations],
        last_updated=max_fetched_at,
        generation_mode=MODE_STATIC,
        trace=trace if trace is not None else Trace(),
    )


def _error_answer(text: str, trace: Trace | None = None, actions: Iterable[str] = ()) -> Answer:
    """Build the ERROR Answer. No freshness stamp, no citations, by definition.

    architecture §6.2 types ``last_updated`` as "date or None for error", and the
    exit criteria exempt ERROR from the FR-6 stamp. An error has no source, so a
    date on it would be a claim the system cannot support.

    Args:
        text: User-facing text, e.g. the setup instructions or an apology.
        trace: Trace to carry. ``None`` produces an empty one.
        actions: Postprocess actions to record. These are **appended** to whatever
            the trace already holds rather than replacing it, so a caller that
            already recorded why it was failing does not lose that context by
            reporting the failure.

    Returns:
        A frozen :class:`Answer` with ``last_updated=None``.
    """
    merged = list(trace.postprocess_actions) if trace is not None else []
    merged.extend(actions)
    return Answer(
        kind=AnswerKind.ERROR,
        text=text,
        citations=[],
        last_updated=None,
        generation_mode=MODE_STATIC,
        trace=replace(trace, postprocess_actions=merged) if trace else Trace(
            postprocess_actions=merged
        ),
    )


# ==========================================================================
# The query path
# ==========================================================================


def _traceback_actions(exc: BaseException) -> list[str]:
    """Describe where an unexpected failure came from, without quoting it.

    implementation.md §P8 asks for the traceback to be attached to the Trace. The
    exception *message* is deliberately left out: a ``ValueError`` from a cast or
    a parse carries the string that caused it, and architecture §11 forbids
    carrying raw user input anywhere — least of all into a payload a UI will
    render. The full traceback is written to the log by ``logger.exception``; what
    travels with the answer is the frame list, which is what actually locates the
    bug.

    Args:
        exc: The exception that escaped :func:`answer_turn`.

    Returns:
        Two ``postprocess_actions`` entries: the exception type and the frames.
    """
    frames = _traceback.extract_tb(exc.__traceback__ or ())
    tail = frames[-_TRACE_FRAME_LIMIT:]
    where = " -> ".join(f"{Path(f.filename).name}:{f.lineno}" for f in tail)
    return [f"pipeline_error:{type(exc).__name__}", f"pipeline_frames:{where}"]


def _record(session: Session | None, answer: Answer, guard: guardrails.GuardDecision | None) -> Answer:
    """Append this turn to ``session``, storing the redacted query only.

    The session is a PII-safe audit buffer, not conversation context: nothing
    here is fed back into the prompt, because implementation.md §P8 does not
    specify multi-turn prompting and inventing it would let an earlier turn
    change the facts in a later answer.
    """
    if session is None or guard is None:
        return answer
    session.add(
        Turn(
            redacted_query=guard.redacted_query,
            kind=answer.kind,
            generation_mode=answer.generation_mode,
            citation_urls=tuple(c.url for c in answer.citations),
        )
    )
    return answer


def answer_turn(query: str, session: Session | None = None) -> Answer:
    """Answer one query. Never raises (architecture §12 E14).

    The order below is the contract (architecture §8), and each step exists
    because reversing it breaks a stated property:

    1. ``guard = classify(query)`` — first, so PII is redacted before anything
       else can touch the string.
    2. Refusals return here, **without retrieving**. No scheme text enters the
       process, so FR-7's "without restating or partially answering" holds.
    3. An unbuilt corpus is an operator problem, not a question: ERROR with setup
       instructions.
    4. ``retrieve(guard.redacted_query)`` — the redacted form, always.
    5. Empty retrieval is a first-class NOT_FOUND, not an error (§8.3).
    6. ``generate(guard.redacted_query, retr)`` — redacted again.
    7. ``enforce(...)`` — P7 owns every compliance guarantee from here on.

    Args:
        query: The user's raw question. It is redacted at step 1 and never
            passed on.
        session: Optional :class:`Session` to append the redacted turn to.

    Returns:
        A :class:`Answer` of kind FACTUAL, REFUSAL, NOT_FOUND or ERROR.
    """
    guard: guardrails.GuardDecision | None = None
    # Hoisted so the error handlers below can still report the guard decision and
    # the corpus snapshot — an ERROR answer that cannot say which corpus it was
    # asked about is a worse bug report.
    base: Trace | None = None
    try:
        guard = guardrails.classify(query)
        max_fetched_at = get_max_fetched_at()
        base = Trace(guard=guard, corpus_fingerprint=store.corpus_fingerprint())

        # --- 2. refusals. Retrieval is NOT called on this path. -------------
        refusal = _refusal(guard, max_fetched_at, base)
        if refusal is not None:
            return _record(session, refusal, guard)

        # --- 3. corpus built? ----------------------------------------------
        if store.corpus_is_empty():
            return _error_answer(
                SETUP_INSTRUCTIONS,
                base,
                actions=["corpus_empty"],
            )

        # --- 4. retrieval on the REDACTED query -----------------------------
        retr = retrieval.retrieve(guard.redacted_query)
        if retr.is_empty:
            page = _scheme_page_for(guard.redacted_query) or disclaimer.AMFI_EDUCATION_URL
            return _record(
                session,
                static_answer(
                    AnswerKind.NOT_FOUND,
                    disclaimer.NOT_FOUND_TEXT,
                    [page],
                    max_fetched_at,
                    replace(base, retrieval=retr, postprocess_actions=["retrieval_empty"]),
                ),
                guard,
            )

        # --- 6/7. generate, then enforce -------------------------------------
        metrics: dict = {}
        raw, mode = generate(guard.redacted_query, retr, metrics=metrics)
        answer = enforce(
            raw,
            retr,
            max_fetched_at,
            mode,
            replace(
                base,
                retrieval=retr,
                prompt_chars=metrics.get("prompt_chars", 0),
                llm_latency_ms=metrics.get("llm_latency_ms"),
            ),
        )
        return _record(session, answer, guard)

    except CorpusEmptyError:
        # E6. The collection vanished between the check at step 3 and the query.
        return _error_answer(
            SETUP_INSTRUCTIONS,
            base or Trace(guard=guard),
            actions=["corpus_empty"],
        )
    except LLMError:
        # `generate` handles provider failures itself and falls back, so one
        # escaping means the fallback path itself is broken. That is a bug, and
        # saying "here is an answer" over it would be worse than saying nothing.
        logger.exception("generation failed in a way the fallback does not cover")
        return _error_answer(APOLOGY, base or Trace(guard=guard), actions=["llm_error"])
    except Exception as exc:
        # E14. Logged with the traceback, surfaced as an apology. Never a
        # fabricated answer.
        logger.exception("unexpected failure in answer_turn")
        return _error_answer(
            APOLOGY, base or Trace(guard=guard), actions=_traceback_actions(exc)
        )


#: Action -> ``(AnswerKind, body copy)``.
#:
#: The advice/PII/performance copy is ``app.disclaimer``'s single source of truth
#: (D5). ``NOT_IN_SOURCES`` reuses the not-found text because "not in the sources"
#: and "no relevant chunk retrieved" are the same statement to a user.
_REFUSAL_BODY: dict[guardrails.GuardAction, tuple[AnswerKind, str]] = {
    guardrails.GuardAction.REFUSE_PII: (AnswerKind.REFUSAL, disclaimer.REFUSAL_PII),
    guardrails.GuardAction.REFUSE_ADVICE: (AnswerKind.REFUSAL, disclaimer.REFUSAL_ADVICE),
    guardrails.GuardAction.REFUSE_PERFORMANCE: (
        AnswerKind.REFUSAL,
        disclaimer.REFUSAL_PERFORMANCE,
    ),
    guardrails.GuardAction.NOT_IN_SOURCES: (
        AnswerKind.NOT_FOUND,
        disclaimer.NOT_FOUND_TEXT,
    ),
}


def _refusal(
    guard: guardrails.GuardDecision,
    max_fetched_at: str | None,
    base: Trace,
) -> Answer | None:
    """Return the static response for ``guard``, or ``None`` to keep going.

    The citation rules differ per action and are not interchangeable (PRD §9.5,
    implementation.md §P8's note):

    * **PII** gets no citation at all. The user has just shown that they are
      pasting personal data; the refusal text already names what not to send, and
      answering it with a link is beside the point.
    * **Advice** gets AMFI's investor-education page — an educational link, never
      a scheme page, because attaching a scheme page to "I won't advise you"
      invites the reading that this one is the one to pick.
    * **Performance** gets the scheme page the user asked about, because
      architecture §8.2 promises them somewhere official to read past returns.
      When the query names no single scheme there is no page to promise, so the
      educational link stands in.
    """
    body = _REFUSAL_BODY.get(guard.action)
    if body is None:
        return None
    kind, text = body

    if guard.action is guardrails.GuardAction.REFUSE_PII:
        citations: list[str] = []
    elif guard.action is guardrails.GuardAction.REFUSE_ADVICE:
        citations = [disclaimer.AMFI_EDUCATION_URL]
    else:
        # Performance, plus NOT_IN_SOURCES — which classify() never returns
        # today (P6 left off-corpus to P8 because it needs retrieval evidence,
        # and step 5 is where that evidence arrives). Handling it here means a
        # future change to P6 cannot silently produce an unrefused off-corpus
        # answer.
        citations = [_scheme_page_for(guard.redacted_query) or disclaimer.AMFI_EDUCATION_URL]

    return static_answer(
        kind,
        text,
        citations,
        max_fetched_at,
        replace(
            base,
            postprocess_actions=[f"refused:{guard.action.value}", "retrieval_skipped"],
        ),
    )
