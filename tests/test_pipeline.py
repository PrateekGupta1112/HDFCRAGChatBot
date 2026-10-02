"""Tests for the query-path orchestration (implementation.md §P8).

Three properties are asserted here that no other test file can assert, because
they are properties of the *order* of the calls rather than of any one module:

* every refusal short-circuits **before** retrieval, so no scheme fact can reach a
  refusal (FR-7);
* retrieval and generation receive the **redacted** query, never the raw input;
* no query, however malformed, escapes as an exception.

No network. The unit tests stub the store and the vector search, so they run in
milliseconds; the two tests that need the real corpus are marked
``integration`` and skip when it has not been built.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from app import chat, config, generation, guardrails, pipeline, retrieval, store
from app.config import Settings
from app.disclaimer import AMFI_EDUCATION_URL, FRESHNESS_PREFIX
from app.generation import AnswerKind
from app.pipeline import Answer, Turn, answer_turn, new_session, static_answer

# Synthetic values only, and no real scheme fee, NAV, AUM or return appears
# anywhere in this file.
SYNTHETIC_DATE = "2020-01-01"
SYNTHETIC_FINGERPRINT = "synthetic0fp"
SYNTHETIC_PAN = "XYZZY9876A"
"""A well-formed but invented PAN. Matches the PII pattern; identifies nothing."""
SYNTHETIC_URL_A = "https://aaa.example/synthetic-page-one"
SYNTHETIC_URL_B = "https://bbb.example/synthetic-page-two"

#: Stands in for ``data/sources.csv`` so a unit test can assert the
#: "cite the scheme page, not a search result" branch without touching real
#: scheme data.
SYNTHETIC_SOURCES = {
    "synthetic_a": ("Synthetic Fund One", SYNTHETIC_URL_A),
    "synthetic_b": ("Synthetic Fund Two", SYNTHETIC_URL_B),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def offline(monkeypatch, request):
    """Force the offline path and stub the corpus for unit tests.

    ``ENV=eval`` is what the implementation guide specifies, and it disables the
    LLM even when a key is present, so the suite never touches the network and
    never depends on a developer's ``.env``.

    The store and the source manifest are stubbed too, so a unit test neither
    needs a built corpus nor reads real scheme data. Tests marked
    ``integration`` opt out and run against the real thing.
    """
    monkeypatch.setenv("ENV", "eval")
    config.get_settings.cache_clear()

    from app import generation as _gen

    def _no_client(settings):
        raise AssertionError("the offline path must never construct an LLM client")

    monkeypatch.setattr(_gen, "_build_client", _no_client)

    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setattr(
            store,
            "corpus_stats",
            lambda: {"count": 1, "max_fetched_at": SYNTHETIC_DATE, "schemes": ["synthetic_a"]},
        )
        monkeypatch.setattr(store, "corpus_fingerprint", lambda: SYNTHETIC_FINGERPRINT)
        monkeypatch.setattr(store, "corpus_is_empty", lambda: False)
        monkeypatch.setattr(pipeline, "_source_index", lambda: SYNTHETIC_SOURCES)

    yield
    config.get_settings.cache_clear()


@pytest.fixture
def empty_retrieval(monkeypatch):
    """Make retrieval return nothing, and record that it was reached.

    Returns the list of queries retrieval was called with, so a test can assert
    both *whether* it ran and *what* it was given.
    """
    seen: list[str] = []

    def _fake_retrieve(query, *args, **kwargs):
        seen.append(query)
        return retrieval.RetrievalResult(
            query=query,
            filtered_scheme_id=None,
            chunks=[],
            max_similarity=0.0,
            is_empty=True,
        )

    monkeypatch.setattr(retrieval, "retrieve", _fake_retrieve)
    return seen


@pytest.fixture
def forbidden_retrieval(monkeypatch):
    """Make any call to ``retrieve`` a hard, recorded failure."""

    def _fail(*args, **kwargs):
        raise AssertionError("retrieval must not run for a refusal")

    monkeypatch.setattr(retrieval, "retrieve", _fail)


def _decision(action: guardrails.GuardAction, redacted: str = "synthetic question"):
    """Build a GuardDecision without going through the patterns."""
    return guardrails.GuardDecision(
        action=action,
        reason=f"synthetic_{action.value}",
        matched_pattern="SYNTHETIC",
        confidence=1.0,
        redacted_query=redacted,
    )


class _FakeTty(io.StringIO):
    """A stdin that claims to be a terminal, so the REPL branch is taken."""

    def isatty(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# The result types are shared, not redefined
# ---------------------------------------------------------------------------


def test_pipeline_re_exports_the_p7_result_types() -> None:
    """``Answer`` must be the *same class* ``enforce`` returns.

    A second, structurally identical definition would satisfy every isinstance
    check in this file and still be the wrong type to the UI.
    """
    assert pipeline.Answer is generation.Answer
    assert pipeline.AnswerKind is generation.AnswerKind
    assert pipeline.Citation is generation.Citation
    assert pipeline.Trace is generation.Trace


# ---------------------------------------------------------------------------
# Refusals short-circuit before retrieval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (guardrails.GuardAction.REFUSE_PII, AnswerKind.REFUSAL),
        (guardrails.GuardAction.REFUSE_ADVICE, AnswerKind.REFUSAL),
        (guardrails.GuardAction.REFUSE_PERFORMANCE, AnswerKind.REFUSAL),
        (guardrails.GuardAction.NOT_IN_SOURCES, AnswerKind.NOT_FOUND),
    ],
)
def test_every_non_answer_action_short_circuits_to_its_kind(
    monkeypatch, forbidden_retrieval, action, expected
) -> None:
    """All four non-ANSWER actions return their own kind, with no retrieval."""
    monkeypatch.setattr(guardrails, "classify", lambda q: _decision(action))

    answer = answer_turn("anything at all")

    assert answer.kind is expected
    assert answer.generation_mode == pipeline.MODE_STATIC
    assert "retrieval_skipped" in answer.trace.postprocess_actions


def test_advice_refusal_never_calls_retrieve(monkeypatch, forbidden_retrieval) -> None:
    """The exit criterion, stated as a test.

    ``forbidden_retrieval`` raises if ``retrieval.retrieve`` is reached at all, so
    this is not "retrieval returned nothing" — it is "retrieval never happened",
    which is what FR-7's "without restating or partially answering" requires.
    """
    monkeypatch.setattr(
        guardrails, "classify", lambda q: _decision(guardrails.GuardAction.REFUSE_ADVICE)
    )

    answer = answer_turn("should I buy something")

    assert answer.kind is AnswerKind.REFUSAL
    assert [c.url for c in answer.citations] == [AMFI_EDUCATION_URL]


def test_pii_refusal_never_calls_retrieve(monkeypatch, forbidden_retrieval) -> None:
    """PII short-circuits before the store is touched, not merely before the LLM."""
    monkeypatch.setattr(
        guardrails, "classify", lambda q: _decision(guardrails.GuardAction.REFUSE_PII)
    )

    answer = answer_turn(f"my pan is {SYNTHETIC_PAN}")

    assert answer.kind is AnswerKind.REFUSAL
    assert answer.citations == []


def test_performance_refusal_cites_the_scheme_page_the_query_named(
    monkeypatch, forbidden_retrieval
) -> None:
    """architecture §8.2 promises a performance question somewhere official to read.

    The scheme page comes from the manifest, not from a search, which is what lets
    the refusal skip retrieval and still name a page.
    """
    monkeypatch.setattr(
        guardrails,
        "classify",
        lambda q: _decision(
            guardrails.GuardAction.REFUSE_PERFORMANCE, redacted="large cap synthetic question"
        ),
    )
    # The alias table is the real one; only the manifest is stubbed. "large cap"
    # is a public scheme name and carries no figures.
    monkeypatch.setattr(pipeline, "_source_index", lambda: {"hdfc_large_cap": SYNTHETIC_SOURCES["synthetic_a"]})

    answer = answer_turn("which fund performed best")

    assert answer.kind is AnswerKind.REFUSAL
    assert [c.url for c in answer.citations] == [SYNTHETIC_URL_A]


def test_performance_refusal_falls_back_to_the_educational_link(
    monkeypatch, forbidden_retrieval
) -> None:
    """No single scheme named ⇒ no scheme page to promise ⇒ AMFI."""
    monkeypatch.setattr(
        guardrails,
        "classify",
        lambda q: _decision(guardrails.GuardAction.REFUSE_PERFORMANCE, redacted="synthetic question"),
    )

    answer = answer_turn("what are the best performing funds")

    assert [c.url for c in answer.citations] == [AMFI_EDUCATION_URL]


# ---------------------------------------------------------------------------
# PII never survives the turn
# ---------------------------------------------------------------------------


def test_pii_literal_is_absent_from_the_answer_and_the_session() -> None:
    """The end-to-end property, with the real classifier.

    The refusal text is canonical copy that cannot contain the value, and the
    query itself is never rendered — so a leak would have to come from the
    redacted form being passed downstream and echoed back.
    """
    raw = f"My PAN is {SYNTHETIC_PAN}, please check my returns"
    session = new_session()

    answer = answer_turn(raw, session)

    assert answer.kind is AnswerKind.REFUSAL
    assert SYNTHETIC_PAN not in answer.text
    assert all(SYNTHETIC_PAN not in c.url for c in answer.citations)
    for turn in session.as_list():
        assert SYNTHETIC_PAN not in turn.redacted_query
        assert "[REDACTED:PAN]" in turn.redacted_query


def test_retrieval_receives_the_redacted_query_not_the_raw_one(
    monkeypatch, empty_retrieval
) -> None:
    """architecture §8 step 4: ``retrieve(guard.redacted_query)``.

    The guard is stubbed rather than the patterns, because with the real
    classifier a query containing PII is *refused* and never reaches retrieval at
    all — the two rules cannot be observed in one turn. Stubbing the decision
    isolates the step this test is about, and
    ``test_pii_refusal_never_calls_retrieve`` covers the real end-to-end case.
    """
    monkeypatch.setattr(store, "corpus_is_empty", lambda: False)
    monkeypatch.setattr(
        guardrails,
        "classify",
        lambda q: _decision(guardrails.GuardAction.ANSWER, redacted="what is the expense ratio"),
    )
    raw = "what is the expense ratio, a/c no. 123456789012"

    answer = answer_turn(raw)

    assert answer.kind is AnswerKind.NOT_FOUND
    assert empty_retrieval == ["what is the expense ratio"]
    assert "123456789012" not in empty_retrieval[0]


# ---------------------------------------------------------------------------
# Corpus state
# ---------------------------------------------------------------------------


def test_unbuilt_corpus_returns_the_setup_instructions(monkeypatch, forbidden_retrieval) -> None:
    """E6: no corpus is an operator problem with a known fix, not a bad question."""
    monkeypatch.setattr(store, "corpus_is_empty", lambda: True)

    answer = answer_turn("what is the expense ratio of the synthetic fund")

    assert answer.kind is AnswerKind.ERROR
    assert "python -m app.ingest" in answer.text
    assert answer.citations == []
    assert answer.last_updated is None


def test_corpus_empty_error_raised_by_retrieval_is_handled(monkeypatch) -> None:
    """The explicit handler, not the broad one.

    ``store.corpus_is_empty`` says the corpus is there and ``retrieve`` disagrees —
    a real race. The user must still get the setup instructions rather than an
    apology, and neither is allowed to raise.
    """
    monkeypatch.setattr(store, "corpus_is_empty", lambda: False)

    def _raise(query, *args, **kwargs):
        from app.errors import CorpusEmptyError

        raise CorpusEmptyError("synthetic empty corpus")

    monkeypatch.setattr(retrieval, "retrieve", _raise)

    answer = answer_turn("what is the synthetic expense ratio")

    assert answer.kind is AnswerKind.ERROR
    assert "python -m app.ingest" in answer.text


def test_a_refusal_is_still_answered_when_the_corpus_is_missing(monkeypatch) -> None:
    """``classify`` is pure, so a refusal does not need a corpus.

    Running the corpus check before the guard would tell a user who pasted a PAN
    into an unbuilt deployment to go run the ingestion script.
    """
    monkeypatch.setattr(store, "corpus_is_empty", lambda: True)

    answer = answer_turn(f"my pan is {SYNTHETIC_PAN}")

    assert answer.kind is AnswerKind.REFUSAL
    assert SYNTHETIC_PAN not in answer.text


# ---------------------------------------------------------------------------
# Not found
# ---------------------------------------------------------------------------


def test_empty_retrieval_returns_not_found_with_a_link(monkeypatch, empty_retrieval) -> None:
    """The brief expects a link even when there is no answer (P8 gotcha)."""
    monkeypatch.setattr(store, "corpus_is_empty", lambda: False)

    answer = answer_turn("what is the weather in mumbai")

    assert answer.kind is AnswerKind.NOT_FOUND
    assert answer.citations, "a not-found answer with no link fails the brief"
    assert "retrieval_empty" in answer.trace.postprocess_actions


def test_not_found_cites_the_scheme_page_when_the_query_names_one(
    monkeypatch, empty_retrieval
) -> None:
    """A scheme-specific question that still retrieves nothing gets that page."""
    monkeypatch.setattr(store, "corpus_is_empty", lambda: False)
    monkeypatch.setattr(
        pipeline, "_source_index", lambda: {"hdfc_elss": SYNTHETIC_SOURCES["synthetic_a"]}
    )

    answer = answer_turn("what is the ELSS tax saver minimum")

    assert answer.kind is AnswerKind.NOT_FOUND
    assert [c.url for c in answer.citations] == [SYNTHETIC_URL_A]


# ---------------------------------------------------------------------------
# The freshness stamp
# ---------------------------------------------------------------------------


def test_every_static_kind_carries_the_freshness_stamp(monkeypatch) -> None:
    """FR-6 applies to refusals too — §8.2 states it as invariant 2."""
    for action in (
        guardrails.GuardAction.REFUSE_PII,
        guardrails.GuardAction.REFUSE_ADVICE,
        guardrails.GuardAction.REFUSE_PERFORMANCE,
    ):
        monkeypatch.setattr(guardrails, "classify", lambda q, a=action: _decision(a))
        answer = answer_turn("synthetic question")
        assert FRESHNESS_PREFIX in answer.text
        assert SYNTHETIC_DATE in answer.text
        assert answer.last_updated == SYNTHETIC_DATE


def test_an_unknown_corpus_date_still_produces_a_stamp() -> None:
    """A missing date is rendered, not omitted: the line must always be there."""
    answer = static_answer(AnswerKind.NOT_FOUND, "nothing here", [], None)

    assert FRESHNESS_PREFIX in answer.text
    assert answer.text.rstrip().endswith("unknown")


def test_error_answers_carry_no_stamp_and_no_date(monkeypatch) -> None:
    """architecture §6.2 types ``last_updated`` as "date or None for error".

    An error has no source, so a date on it would assert a freshness the system
    cannot support — even when the corpus *is* built and a date was available.
    """
    monkeypatch.setattr(store, "corpus_is_empty", lambda: True)

    answer = answer_turn("what is the synthetic expense ratio")

    assert answer.kind is AnswerKind.ERROR
    assert answer.last_updated is None
    assert FRESHNESS_PREFIX not in answer.text


def test_static_answer_refuses_to_stamp_an_error() -> None:
    """The rule is enforced by ``static_answer`` itself, not by each caller.

    ``static_answer`` is exported for P9, so the "no stamp on ERROR" contract has
    to hold for a caller this phase cannot see.
    """
    answer = static_answer(AnswerKind.ERROR, "something broke", [], SYNTHETIC_DATE)

    assert answer.last_updated is None
    assert FRESHNESS_PREFIX not in answer.text
    assert answer.text == "something broke"
    assert "static_error" in answer.trace.postprocess_actions


# ---------------------------------------------------------------------------
# Static answers
# ---------------------------------------------------------------------------


def test_static_answer_carries_mode_and_converts_urls_to_citations() -> None:
    answer = static_answer(AnswerKind.REFUSAL, "no", [SYNTHETIC_URL_A], SYNTHETIC_DATE)

    assert answer.generation_mode == pipeline.MODE_STATIC
    assert [c.url for c in answer.citations] == [SYNTHETIC_URL_A]
    assert all(isinstance(c, pipeline.Citation) for c in answer.citations)
    assert answer.trace == pipeline.Trace()


def test_static_answer_resolves_a_scheme_name_from_the_manifest() -> None:
    """The UI needs a label for the link; a bare URL is a worse citation."""
    answer = static_answer(AnswerKind.REFUSAL, "no", [SYNTHETIC_URL_A], SYNTHETIC_DATE)

    assert answer.citations[0].scheme_name == "Synthetic Fund One"


def test_static_answer_does_not_alter_the_canonical_copy() -> None:
    """D5: the refusal wording is reproduced verbatim from the PRD.

    Normalising it — the way generated text is normalised — would fold the em
    dashes and curly apostrophes of ``app.disclaimer`` to ASCII, silently editing
    the one string the specification pins down.
    """
    from app import disclaimer

    answer = static_answer(AnswerKind.REFUSAL, disclaimer.REFUSAL_PII, [], SYNTHETIC_DATE)

    assert answer.text.startswith(disclaimer.REFUSAL_PII)
    assert "—" in answer.text


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def test_new_session_is_empty_and_capped() -> None:
    session = new_session(Settings(max_chat_history_turns=2, _env_file=None))

    assert session.max_turns == 2
    assert session.as_list() == []


def test_session_drops_the_oldest_turn_at_the_cap() -> None:
    """The cap is a property of the container, not a rule to remember."""
    session = new_session(Settings(max_chat_history_turns=2, _env_file=None))
    for i in range(4):
        session.add(Turn(f"synthetic query {i}", AnswerKind.NOT_FOUND, "static", ()))

    assert [t.redacted_query for t in session.as_list()] == [
        "synthetic query 2",
        "synthetic query 3",
    ]


def test_a_zero_cap_turns_stores_nothing() -> None:
    session = new_session(Settings(max_chat_history_turns=0, _env_file=None))
    session.add(Turn("synthetic query", AnswerKind.NOT_FOUND, "static", ()))

    assert session.as_list() == []


def test_session_stores_only_the_redacted_query(monkeypatch, empty_retrieval) -> None:
    monkeypatch.setattr(store, "corpus_is_empty", lambda: False)
    session = new_session()

    answer_turn(f"my a/c number is 123456789012", session)

    stored = session.as_list()
    assert len(stored) == 1
    assert "123456789012" not in stored[0].redacted_query
    assert stored[0].kind is AnswerKind.REFUSAL


def test_answer_turn_works_without_a_session() -> None:
    assert answer_turn(f"my pan is {SYNTHETIC_PAN}").kind is AnswerKind.REFUSAL


def test_session_is_optional_and_never_required() -> None:
    """``session=None`` is the default, so the UI can stay stateless per call."""
    answer = answer_turn("synthetic question")

    assert isinstance(answer, Answer)


# ---------------------------------------------------------------------------
# The turn never raises
# ---------------------------------------------------------------------------


def test_an_empty_query_does_not_raise(empty_retrieval) -> None:
    """Exit criterion, hermetic: the stubbed retrieval returns nothing, and the
    answer is a first-class NOT_FOUND rather than an error."""
    answer = answer_turn("")

    assert answer.kind is AnswerKind.NOT_FOUND


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", "x" * 5000, "?!?!", "🙂"])
def test_hostile_inputs_do_not_raise(bad, empty_retrieval) -> None:
    answer = answer_turn(bad)
    assert answer.kind is AnswerKind.NOT_FOUND


def test_an_unexpected_exception_becomes_a_polite_error(monkeypatch, caplog) -> None:
    """E14: logged with the traceback, answered with an apology, never a crash.

    The bug must be *findable*, which is why the frames travel on the Trace — but
    the exception's own message does not, because a ``ValueError`` from a cast or
    a parse quotes the string that caused it, and architecture §11 forbids
    carrying raw user input anywhere. The full message goes to the log instead.
    """
    raw = "what is the synthetic expense ratio"

    def _boom(query, *args, **kwargs):
        raise ValueError(f"synthetic failure carrying {query!r}")

    monkeypatch.setattr(retrieval, "retrieve", _boom)

    with caplog.at_level(logging.ERROR):
        answer = answer_turn(raw)

    assert answer.kind is AnswerKind.ERROR
    assert answer.citations == []
    assert answer.last_updated is None
    assert raw not in answer.text
    assert "pipeline_error:ValueError" in answer.trace.postprocess_actions
    assert any("pipeline_frames:" in a for a in answer.trace.postprocess_actions)
    assert all("synthetic failure" not in a for a in answer.trace.postprocess_actions)
    # The message reaches the log, not the answer.
    assert "synthetic failure" in caplog.text


def test_the_trace_records_the_guard_decision_and_the_corpus_fingerprint(
    monkeypatch, empty_retrieval
) -> None:
    monkeypatch.setattr(store, "corpus_is_empty", lambda: False)

    answer = answer_turn("what is the synthetic expense ratio")

    assert answer.trace.guard is not None
    assert answer.trace.guard.action is guardrails.GuardAction.ANSWER
    assert answer.trace.corpus_fingerprint == SYNTHETIC_FINGERPRINT
    assert answer.trace.retrieval is not None
    assert answer.trace.retrieval.query == "what is the synthetic expense ratio"


def test_an_error_answer_still_identifies_the_corpus_and_the_guard(
    monkeypatch, empty_retrieval
) -> None:
    """An ERROR that cannot say which corpus it was asked about is a worse bug report.

    The failure happens *after* the guard decision and the snapshot read, so both
    must survive onto the answer instead of being lost with the local scope.
    """
    def _boom(query, *args, **kwargs):
        raise RuntimeError("synthetic failure after the trace was built")

    monkeypatch.setattr(retrieval, "retrieve", _boom)

    answer = answer_turn("what is the synthetic expense ratio")

    assert answer.kind is AnswerKind.ERROR
    assert answer.trace.guard is not None
    assert answer.trace.corpus_fingerprint == SYNTHETIC_FINGERPRINT


# ---------------------------------------------------------------------------
# Integration — the real corpus
# ---------------------------------------------------------------------------

_REAL_CORPUS = Path("data/processed/chunks.jsonl")


def _real_corpus_available() -> bool:
    if not _REAL_CORPUS.exists():
        return False
    from app.config import get_settings

    if get_settings().chroma_path.startswith("tmp"):
        return False
    return not store.corpus_is_empty()


requires_real_corpus = pytest.mark.skipif(
    not _real_corpus_available(), reason="real corpus not built; run `python -m app.ingest`"
)


@pytest.mark.integration
@requires_real_corpus
def test_a_nonsense_query_is_not_found() -> None:
    """Off-corpus is a first-class answer, not an error and not a refusal."""
    answer = answer_turn("what is the weather in mumbai")

    assert answer.kind is AnswerKind.NOT_FOUND
    assert answer.citations
    assert FRESHNESS_PREFIX in answer.text


@pytest.mark.integration
@requires_real_corpus
def test_an_empty_query_does_not_raise_against_the_real_corpus() -> None:
    answer = answer_turn("")

    assert answer.kind in (AnswerKind.NOT_FOUND, AnswerKind.FACTUAL)
    assert answer.text


@pytest.mark.integration
@requires_real_corpus
def test_an_in_scope_question_is_answered_from_the_real_corpus() -> None:
    """The whole path, offline: classify → retrieve → generate → enforce.

    The body is not asserted on — which sentence wins is a P7 property — but the
    kind, the single citation and the stamp are this phase's contract.
    """
    answer = answer_turn("What is the expense ratio of HDFC Large Cap Fund?")

    assert answer.kind is AnswerKind.FACTUAL
    assert answer.generation_mode == generation.MODE_EXTRACTIVE
    assert len(answer.citations) == 1
    assert FRESHNESS_PREFIX in answer.text
    assert answer.trace.retrieval is not None
    assert answer.trace.retrieval.max_similarity > 0


# ---------------------------------------------------------------------------
# The headless CLI
# ---------------------------------------------------------------------------


def test_cli_parses_repeated_query_flags() -> None:
    args = chat.build_parser().parse_args(["-q", "one", "--query", "two", "--json"])

    assert args.queries == ["one", "two"]
    assert args.as_json is True


def test_cli_defaults_to_reading_stdin() -> None:
    args = chat.build_parser().parse_args([])

    assert args.queries is None
    assert args.as_json is False


def test_cli_piped_stdin_yields_one_query_per_line(monkeypatch) -> None:
    """A file or heredoc of questions can be replayed without any flags."""
    monkeypatch.setattr(chat.sys, "stdin", io.StringIO("first\n\nsecond\nexit\nthird\n"))

    assert list(chat._iter_queries(None)) == ["first", "second"]


def test_cli_repl_stops_on_a_blank_line(monkeypatch) -> None:
    monkeypatch.setattr(chat.sys, "stdin", _FakeTty())
    answers = iter(["a question", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    assert list(chat._iter_queries(None)) == ["a question"]


def test_cli_repl_stops_on_the_exit_word(monkeypatch) -> None:
    monkeypatch.setattr(chat.sys, "stdin", _FakeTty())
    answers = iter(["QuIt"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    assert list(chat._iter_queries(None)) == []


def test_cli_query_flags_win_over_stdin(monkeypatch) -> None:
    """An explicit query must not be appended to whatever is on stdin."""
    monkeypatch.setattr(chat.sys, "stdin", io.StringIO("ignored\n"))

    assert list(chat._iter_queries(["asked"])) == ["asked"]


def test_cli_never_echoes_the_query(capsys, empty_retrieval) -> None:
    """The PII check greps the whole output for the literal the user typed.

    A CLI that printed the question back would fail that check for the one answer
    whose purpose is not to repeat it.
    """
    rc = chat.main(["--query", f"my pan is {SYNTHETIC_PAN}"])

    out = capsys.readouterr()
    assert rc == 0
    assert SYNTHETIC_PAN not in out.out
    assert "refusal" in out.out
    assert FRESHNESS_PREFIX in out.out


def test_cli_prints_every_field_the_verify_table_reads(capsys, empty_retrieval) -> None:
    rc = chat.main(["--query", "what is the synthetic expense ratio"])

    out = capsys.readouterr()
    assert rc == 0
    for field in ("kind:", "mode:", "top_sim:", "answer:", "citation:", "last_updated:"):
        assert field in out.out


def test_cli_json_mode_emits_one_parseable_object_per_query(capsys, empty_retrieval) -> None:
    import json

    rc = chat.main(["--json", "-q", "what is the synthetic expense ratio", "-q", ""])

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert rc == 0
    assert [record["kind"] for record in lines] == ["not_found", "not_found"]
    assert lines[0]["guard_action"] == "answer"
    assert "postprocess_actions" in lines[0]
    assert "citations" in lines[0]


def test_cli_writes_the_disclaimer_to_stderr_not_stdout(capsys, empty_retrieval) -> None:
    """stdout has to stay machine-readable in both modes."""
    from app import disclaimer

    chat.main(["--json", "-q", "what is the synthetic expense ratio"])

    captured = capsys.readouterr()
    assert disclaimer.DISCLAIMER in captured.err
    assert disclaimer.DISCLAIMER not in captured.out


def test_cli_reports_a_missing_corpus_with_a_nonzero_exit(monkeypatch, capsys) -> None:
    monkeypatch.setattr(store, "corpus_is_empty", lambda: True)

    rc = chat.main(["--query", "what is the synthetic expense ratio"])

    assert rc == 1
    assert "python -m app.ingest" in capsys.readouterr().out
