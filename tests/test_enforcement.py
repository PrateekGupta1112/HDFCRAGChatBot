"""Tests for the enforcement layer (implementation.md §P7).

Pure and offline. Nothing here loads the encoder, touches Chroma, or makes a
network call: `enforce()` is the last thing between generated text and a user,
so it is tested with hand-built `RetrievalResult` objects rather than real ones,
and every fixture value is an obviously synthetic dummy — **no real mutual fund
figure, fee, NAV or return appears anywhere in this file**.
"""

from __future__ import annotations

import pytest

from app.disclaimer import FRESHNESS_PREFIX
from app.generation import AnswerKind, enforce

# Synthetic corpus. "aaa" host and "999" figures cannot collide with a real page.
URL_A = "https://aaa.example/synthetic-page-one"
URL_B = "https://bbb.example/synthetic-page-two"
URL_FAKE = "https://zzz.example/definitely-not-in-the-corpus"

# A figure that exists in the synthetic context, and one that does not.
IN_CONTEXT_NUMBER = "1.11%"
NOT_IN_CONTEXT_NUMBER = "9.99%"

SYNTHETIC_CHUNKS = [
    {
        "section": "Synthetic fees section",
        "scheme_name": "Synthetic Fund One",
        "source_url": URL_A,
        "text": f"The synthetic expense ratio is {IN_CONTEXT_NUMBER} for this example scheme.",
    },
    {
        "section": "Synthetic minimums section",
        "scheme_name": "Synthetic Fund Two",
        "source_url": URL_B,
        "text": "The synthetic minimum investment is 111 units in this example scheme.",
    },
]


def make_retr(chunks=None) -> "object":
    """Build a minimal RetrievalResult from dicts, bypassing the real encoder."""
    from app.retrieval import RetrievedChunk, RetrievalResult

    built = [
        RetrievedChunk(
            rank=i,
            chunk_id=f"synthetic{i:03d}",
            text=c["text"],
            section=c["section"],
            scheme_name=c["scheme_name"],
            source_url=c["source_url"],
            fetched_at="2020-01-01",
            similarity=0.9 - i * 0.1,
            mmr_score=None,
        )
        for i, c in enumerate(chunks if chunks is not None else SYNTHETIC_CHUNKS, start=1)
    ]
    return RetrievalResult(
        query="synthetic query",
        filtered_scheme_id=None,
        chunks=built,
        max_similarity=built[0].similarity if built else 0.0,
        is_empty=not built,
    )


@pytest.fixture
def retr():
    return make_retr()


MAXDATE = "2020-01-01"


def run(retr_obj, raw: str):
    return enforce(raw, retr_obj, MAXDATE, "test", None)


# --------------------------------------------------------------------------
# 1. sentence budget
# --------------------------------------------------------------------------


def test_six_sentences_become_three_and_record_the_action(retr) -> None:
    raw = "One. Two. Three. Four. Five. Six. " + f"Source: {URL_A}"
    a = run(retr, raw)

    body = a.text.split(FRESHNESS_PREFIX)[0].strip()
    assert body.count(".") <= 3, body
    assert "Four" not in a.text
    assert "truncated_to_3_sentences" in a.trace.postprocess_actions
    assert a.kind is AnswerKind.FACTUAL


def test_three_sentences_are_left_alone(retr) -> None:
    """Exactly at the budget: three sentences, the citation inside them."""
    raw = f"One. Two. Source: {URL_A}"
    a = run(retr, raw)
    assert "truncated_to_3_sentences" not in a.trace.postprocess_actions
    assert a.kind is AnswerKind.FACTUAL
    assert len(a.citations) == 1


# --------------------------------------------------------------------------
# 2. citation enforcement
# --------------------------------------------------------------------------


def test_two_verified_urls_collapse_to_exactly_one(retr) -> None:
    raw = f"A synthetic fact. Source: {URL_A} and also {URL_B}"
    a = run(retr, raw)

    assert len(a.citations) == 1
    assert a.citations[0].url == URL_A
    assert URL_A in a.text
    assert URL_B not in a.text
    assert any(
        act.startswith("reduced_to_one_citation") for act in a.trace.postprocess_actions
    )


def test_off_corpus_url_fails_closed_and_never_reaches_the_answer(retr) -> None:
    raw = f"A synthetic fact with a made-up link. Source: {URL_FAKE}"
    a = run(retr, raw)

    assert a.kind is AnswerKind.NOT_FOUND
    assert URL_FAKE not in a.text
    assert "fail_closed_no_citation" in a.trace.postprocess_actions
    assert a.citations == []
    # The body must not leak through just because the link was bad.
    assert "synthetic fact with a made-up link" not in a.text


def test_a_verified_link_survives_a_paraphrased_spelling(retr) -> None:
    """The model may add www., a trailing slash, or a tracking parameter.
    normalize_url is applied to BOTH sides, so these must all verify."""
    for spelling in (
        "https://aaa.example/synthetic-page-one/",
        "http://www.aaa.example/synthetic-page-one",
        "https://aaa.example/synthetic-page-one?utm_source=chatgpt",
    ):
        a = run(retr, f"A synthetic fact. Source: {spelling}")
        assert a.kind is AnswerKind.FACTUAL, spelling
        assert len(a.citations) == 1, spelling
        # The canonical corpus URL is what the citation carries.
        assert a.citations[0].url == URL_A, spelling


def test_an_answer_with_no_link_at_all_fails_closed(retr) -> None:
    a = run(retr, "A synthetic fact with no citation whatsoever.")
    assert a.kind is AnswerKind.NOT_FOUND
    assert "fail_closed_no_citation" in a.trace.postprocess_actions


def test_a_repeated_verified_url_is_rendered_only_once(retr) -> None:
    """extract_urls returns DISTINCT urls, so a model citing one page twice gives a
    one-element list and a count-based reduction misses the duplicate. Every chunk
    from one page shares a source_url, so this is the common case."""
    raw = f"Fact. Source: {URL_A} and also {URL_A}"
    a = run(retr, raw)

    assert a.kind is AnswerKind.FACTUAL
    assert a.text.count(URL_A) == 1
    assert len(a.citations) == 1
    # The dangling connective left behind must not survive either.
    assert "and also" not in a.text


def test_two_distinct_verified_urls_collapse_to_exactly_one(retr) -> None:
    raw = f"Fact. Source: {URL_A} and also {URL_B}"
    a = run(retr, raw)

    assert len(a.citations) == 1
    assert a.citations[0].url == URL_A
    assert a.text.count(URL_A) == 1
    assert URL_B not in a.text


def test_full_width_bracket_links_are_unwrapped(retr) -> None:
    """gpt-oss emits 【1](url) rather than [1](url). The URL must still verify and
    the bracket residue must not reach the user."""
    raw = f"The synthetic ratio is {IN_CONTEXT_NUMBER}. 【1]({URL_A})"
    a = run(retr, raw)

    assert a.kind is AnswerKind.FACTUAL
    assert len(a.citations) == 1
    assert "【" not in a.text
    assert "](" not in a.text
    assert URL_A in a.text


def test_a_markdown_link_is_unwrapped_not_left_broken(retr) -> None:
    raw = f"A synthetic fact. [Source]({URL_A})"
    a = run(retr, raw)
    assert a.kind is AnswerKind.FACTUAL
    assert "[Source]()" not in a.text
    assert len(a.citations) == 1


# --------------------------------------------------------------------------
# 3. numeric audit
# --------------------------------------------------------------------------


def test_unsupported_number_is_removed_and_recorded(retr) -> None:
    raw = f"The synthetic ratio is {NOT_IN_CONTEXT_NUMBER}. Source: {URL_A}"
    a = run(retr, raw)

    assert NOT_IN_CONTEXT_NUMBER not in a.text
    assert any(
        act.startswith("dropped_unsupported_number") for act in a.trace.postprocess_actions
    )
    # The figure was real and present, so it must survive untouched.
    assert IN_CONTEXT_NUMBER in run(retr, f"The ratio is {IN_CONTEXT_NUMBER}. Source: {URL_A}").text


def test_numbers_from_any_retrieved_chunk_pass_not_just_the_top_one(retr) -> None:
    """The context blob is every retrieved chunk, not only rank 1. This mirrors the
    real corpus, where the answer to 'fund size' sits in the hero block at rank 4."""
    raw = (
        f"The synthetic minimum investment is 111 units. "
        f"The synthetic expense ratio is {IN_CONTEXT_NUMBER}. Source: {URL_A}"
    )
    a = run(retr, raw)
    assert "111" in a.text
    assert IN_CONTEXT_NUMBER in a.text


def test_a_number_in_a_second_chunk_is_not_stripped(retr) -> None:
    """111 lives only in chunk 2's text, never in chunk 1's."""
    a = run(retr, f"Minimum is 111 units for the example. Source: {URL_B}")
    assert "111" in a.text
    assert not any("dropped_unsupported" in act for act in a.trace.postprocess_actions)


def test_removing_a_number_drops_the_sentence_rather_than_broken_grammar(retr) -> None:
    """Stripping the figure from 'The synthetic ratio is 9.99%' would leave
    'The synthetic ratio is.' — a whole sentence must go instead."""
    raw = f"The synthetic ratio is {NOT_IN_CONTEXT_NUMBER}. Source: {URL_A}"
    a = run(retr, raw)

    assert "The synthetic ratio is" not in a.text
    assert "dropped_mangled_sentence" in a.trace.postprocess_actions
    # The verified citation survives, because exactly one is a hard requirement.
    assert len(a.citations) == 1
    assert URL_A in a.text


def test_the_freshness_date_is_never_audited_away(retr) -> None:
    """Order matters: the numeric audit runs BEFORE the stamp is appended."""
    a = run(retr, f"The synthetic ratio is {IN_CONTEXT_NUMBER}. Source: {URL_A}")
    assert f"{FRESHNESS_PREFIX} {MAXDATE}" in a.text
    assert not any("2020" in act for act in a.trace.postprocess_actions)


def test_a_citation_placed_after_the_freshness_line_survives(retr) -> None:
    """Models sometimes emit the link *after* the timestamp, on the same line.
    Stripping "the rest of the line" would delete the only verified link and
    downgrade a correct answer for a reason unrelated to its citation."""
    raw = (
        f"The synthetic ratio is {IN_CONTEXT_NUMBER}. "
        f"{FRESHNESS_PREFIX} <fetched_at> {URL_A}"
    )
    a = run(retr, raw)

    assert a.kind is AnswerKind.FACTUAL
    assert len(a.citations) == 1
    assert URL_A in a.text
    assert a.text.count(FRESHNESS_PREFIX) == 1
    assert "<fetched_at>" not in a.text


def test_a_citation_placed_on_the_line_after_the_freshness_line_survives(retr) -> None:
    raw = (
        f"The synthetic ratio is {IN_CONTEXT_NUMBER}.\n"
        f"{FRESHNESS_PREFIX} 1999-01-01\nSource: {URL_A}"
    )
    a = run(retr, raw)
    assert a.kind is AnswerKind.FACTUAL
    assert len(a.citations) == 1
    assert "1999-01-01" not in a.text


def test_a_model_emitted_freshness_line_is_not_duplicated(retr) -> None:
    raw = (
        f"The synthetic ratio is {IN_CONTEXT_NUMBER}. "
        f"Source: {URL_A} {FRESHNESS_PREFIX} 1999-01-01"
    )
    a = run(retr, raw)
    assert a.text.count(FRESHNESS_PREFIX) == 1
    assert "1999-01-01" not in a.text


def test_digits_inside_the_url_are_never_treated_as_figures(retr) -> None:
    """The URL is masked before the audit so a slug containing digits survives."""
    url_with_digits = "https://ccc.example/synthetic/page-2020-v2"
    retr2 = make_retr(
        [
            {
                "section": "Synthetic digits section",
                "scheme_name": "Synthetic Fund Three",
                "source_url": url_with_digits,
                "text": "A synthetic sentence without figures.",
            }
        ]
    )
    a = run(retr2, f"A synthetic fact. Source: {url_with_digits}")
    assert url_with_digits in a.text
    assert not any("dropped_unsupported" in act for act in a.trace.postprocess_actions)


# --------------------------------------------------------------------------
# 4. openers
# --------------------------------------------------------------------------


def test_a_certainly_opener_is_stripped(retr) -> None:
    a = run(retr, f"Certainly! The synthetic ratio is {IN_CONTEXT_NUMBER}. Source: {URL_A}")
    assert "Certainly" not in a.text
    assert a.text.startswith("The synthetic ratio")
    assert a.kind is AnswerKind.FACTUAL


def test_chained_and_punctuated_openers_are_all_stripped(retr) -> None:
    raw = f"Sure, certainly! Of course. As an AI language model, here: the synthetic fact. Source: {URL_A}"
    a = run(retr, raw)
    assert "Sure" not in a.text
    assert "certainly" not in a.text.lower().split(FRESHNESS_PREFIX)[0].split("source")[0]
    assert a.kind is AnswerKind.FACTUAL


def test_stripping_an_opener_restores_capitalisation(retr) -> None:
    """"Of course, the synthetic ratio is ..." -> the opener takes the comma with
    it and leaves a lowercase determiner, so "the" is dropped and the sentence is
    recapped rather than emitted as "the synthetic ratio is ..."."""
    a = run(retr, f"Of course, the synthetic ratio is {IN_CONTEXT_NUMBER}. Source: {URL_A}")
    assert a.text.startswith("Synthetic ratio is")


def test_a_capitalised_first_word_is_never_eaten(retr) -> None:
    """The article repair must not fire on a sentence that already begins properly.
    Losing "The" would be silent vandalism of a perfectly good answer."""
    a = run(retr, f"Certainly! The synthetic ratio is {IN_CONTEXT_NUMBER}. Source: {URL_A}")
    assert a.text.startswith("The synthetic ratio")


def test_a_long_answer_keeps_its_citation_inside_the_budget(retr) -> None:
    """architecture §9.5 truncates before it verifies the citation. Without a slot
    reserved for the link, this fully compliant answer - three sentences plus a
    Source line - would have its citation cut and then fail closed to NOT_FOUND."""
    raw = (
        f"Fact one. Fact two. Fact three. Fact four. Source: {URL_A}"
    )
    a = run(retr, raw)

    assert a.kind is AnswerKind.FACTUAL
    assert len(a.citations) == 1
    assert URL_A in a.text
    assert "truncated_to_3_sentences" in a.trace.postprocess_actions
    # The fourth content fact is what gives way, not the citation.
    body = a.text.split(FRESHNESS_PREFIX)[0]
    assert "Fact four" not in body


# --------------------------------------------------------------------------
# 5. freshness
# --------------------------------------------------------------------------


def test_the_freshness_line_is_present_on_a_factual_answer(retr) -> None:
    a = run(retr, f"A synthetic fact. Source: {URL_A}")
    assert a.text.rstrip().endswith(f"{FRESHNESS_PREFIX} {MAXDATE}")
    assert a.last_updated == MAXDATE


def test_the_freshness_line_is_present_even_when_failing_closed(retr) -> None:
    a = run(retr, f"A synthetic fact. Source: {URL_FAKE}")
    assert a.kind is AnswerKind.NOT_FOUND
    assert FRESHNESS_PREFIX in a.text


def test_a_missing_fetch_date_still_yields_the_line(retr) -> None:
    a = enforce(f"A synthetic fact. Source: {URL_A}", retr, None, "test", None)
    assert FRESHNESS_PREFIX in a.text
    assert "unknown" in a.text


# --------------------------------------------------------------------------
# 6. enforce() must never raise
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "\n\n\t\n",
        "https://aaa.example/synthetic-page-one",
        "?",
        ".",
        "₹",
        "1",
        "a " * 10_000,
        "No citation and no punctuation whatsoever",
        "Source:",
        "]]]]]]]]]",
    ],
    ids=[
        "empty", "spaces", "newlines", "url-only", "question-mark", "period",
        "rupee", "bare-digit", "20k-a", "no-citation", "dangling-source", "brackets",
    ],
)
def test_enforce_never_raises_on_hostile_input(retr, raw: str) -> None:
    a = enforce(raw, retr, MAXDATE, "test", None)
    assert a.kind in set(AnswerKind)
    assert isinstance(a.text, str)


def test_enforce_never_raises_on_a_20kb_string(retr) -> None:
    body = " ".join(f"Sentence number {i} of synthetic text." for i in range(2_000))
    raw = f"{body} Source: {URL_A}"
    assert len(raw) > 20_000
    a = enforce(raw, retr, MAXDATE, "test", None)
    assert a.kind is AnswerKind.FACTUAL
    assert len(a.citations) == 1


def test_enforce_never_raises_on_an_empty_retrieval_result() -> None:
    from app.retrieval import RetrievalResult

    empty = RetrievalResult(
        query="synthetic", filtered_scheme_id=None, chunks=[], max_similarity=0.0, is_empty=True
    )
    a = enforce(f"A synthetic fact. Source: {URL_A}", empty, MAXDATE, "test", None)
    assert a.kind is AnswerKind.NOT_FOUND


def test_enforce_reuses_the_callers_trace_and_overwrites_actions(retr) -> None:
    from app.generation import Trace

    trace = Trace(prompt_chars=42, llm_latency_ms=7, postprocess_actions=["stale"])
    a = enforce(f"A synthetic fact. Source: {URL_A}", retr, MAXDATE, "test", trace)

    assert a.trace.prompt_chars == 42
    assert a.trace.llm_latency_ms == 7
    assert "stale" not in a.trace.postprocess_actions


# --------------------------------------------------------------------------
# 7. citation metadata
# --------------------------------------------------------------------------


def test_the_citation_url_is_the_canonical_corpus_url_not_the_emitted_spelling(retr) -> None:
    a = run(retr, f"A synthetic fact. Source: http://www.aaa.example/synthetic-page-one/")
    assert a.citations[0].url == URL_A
    assert a.citations[0].section == "Synthetic fees section"
    assert a.citations[0].scheme_name == "Synthetic Fund One"


def test_the_section_comes_from_the_highest_ranked_chunk_on_that_page() -> None:
    retr_same_url = make_retr(
        [
            {"section": "Top rank section", "scheme_name": "S1",
             "source_url": URL_A, "text": "First synthetic sentence."},
            {"section": "Second section", "scheme_name": "S1",
             "source_url": URL_A, "text": "Second synthetic sentence."},
        ]
    )
    a = run(retr_same_url, f"A synthetic fact. Source: {URL_A}")
    assert a.citations[0].section == "Top rank section"