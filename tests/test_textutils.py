"""Unit tests for app.textutils — the pure text helpers.

No network, no model, no fixtures on disk. Everything is inline so a failure
points at a specific behaviour rather than at test setup.
"""

from __future__ import annotations

import pytest

from app.textutils import (
    extract_numbers,
    extract_urls,
    normalize_url,
    normalize_ws,
    split_sentences,
    strip_number,
    truncate_sentences,
)


# --------------------------------------------------------------------------
# normalize_ws
# --------------------------------------------------------------------------


def test_normalize_ws_collapses_spaces_and_newlines() -> None:
    assert normalize_ws("a  b\n\nc\td\xa0e") == "a b c d e"


def test_normalize_ws_folds_nbsp_and_curly_quotes() -> None:
    assert normalize_ws("it\u2019s a\u00a0fee") == "it's a fee"


def test_normalize_ws_on_empty_and_whitespace_only() -> None:
    assert normalize_ws("") == ""
    assert normalize_ws("   \n\t ") == ""


# --------------------------------------------------------------------------
# split_sentences
# --------------------------------------------------------------------------


def test_split_sentences_returns_four_for_four_sentences() -> None:
    text = (
        "The expense ratio is listed on the page. "
        "The exit load applies after one year. "
        "The minimum SIP is fixed. "
        "Lock-in does not apply here."
    )
    assert len(split_sentences(text)) == 4


def test_split_sentences_does_not_split_on_decimal() -> None:
    # Regression guard: "1.23" must never become two sentences, otherwise the
    # numeric audit in P7 compares the wrong surface forms.
    assert split_sentences("The expense ratio is 1.23% today.") == [
        "The expense ratio is 1.23% today."
    ]


@pytest.mark.parametrize(
    "text",
    [
        "The minimum SIP is Rs. 500.",
        "Dr. Smith manages this fund.",
        "The fee applies, e.g. on entry.",
        "The reference is No. 12.",
    ],
)
def test_split_sentences_respects_protected_abbreviations(text: str) -> None:
    assert len(split_sentences(text)) == 1


def test_split_sentences_does_not_split_dotted_hosts() -> None:
    text = "See groww.in for details."
    assert len(split_sentences(text)) == 1


def test_split_sentences_treats_newline_as_boundary() -> None:
    text = "Expense ratio: 1.23%\nExit load: 1.00%"
    assert len(split_sentences(text)) == 2


def test_split_sentences_does_not_split_when_next_word_is_lowercase() -> None:
    assert len(split_sentences("One sentence. and then more text.")) == 1


def test_split_sentences_on_empty_input() -> None:
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_keeps_terminal_punctuation() -> None:
    assert all(s.endswith(".") for s in split_sentences("One. Two."))


# --------------------------------------------------------------------------
# truncate_sentences
# --------------------------------------------------------------------------


def test_truncate_sentences_cuts_to_three() -> None:
    text, original = truncate_sentences("A. B. C. D.", 3)
    assert original == 4
    assert len(split_sentences(text)) == 3
    assert text == "A. B. C."


def test_truncate_sentences_returns_input_when_under_budget() -> None:
    text, original = truncate_sentences("One. Two.", 3)
    assert text == "One. Two."
    assert original == 2


def test_truncate_sentences_with_n_below_one_returns_empty() -> None:
    text, original = truncate_sentences("One. Two.", 0)
    assert text == ""
    assert original == 2


# --------------------------------------------------------------------------
# normalize_url
# --------------------------------------------------------------------------


def test_normalize_url_strips_www_trailing_slash_and_tracking_params() -> None:
    assert (
        normalize_url(
            "https://www.groww.in/mutual-funds/"
            "hdfc-large-cap-fund-direct-growth/?utm_source=news&utm_medium=email"
        )
        == "groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
    )


def test_normalize_url_is_scheme_insensitive() -> None:
    assert normalize_url("https://groww.in/a/b") == normalize_url("http://groww.in/a/b")


def test_normalize_url_preserves_path_case_and_keeps_meaningful_query() -> None:
    assert normalize_url("https://groww.in/Fund-ABC") == "groww.in/Fund-ABC"
    assert normalize_url("https://example.com/x?page=2&ref=z") == "example.com/x?page=2"


def test_normalize_url_strips_trailing_sentence_punctuation() -> None:
    # A model may emit the URL with the sentence's full stop still attached.
    assert normalize_url("https://groww.in/a/b.") == "groww.in/a/b"


def test_normalize_url_on_empty_input() -> None:
    assert normalize_url("") == ""


# --------------------------------------------------------------------------
# extract_urls
# --------------------------------------------------------------------------


def test_extract_urls_finds_url_and_strips_full_stop() -> None:
    assert extract_urls("See https://groww.in/a/b for more.") == ["https://groww.in/a/b"]


def test_extract_urls_unwraps_markdown_links() -> None:
    got = extract_urls("Source: [scheme page](https://groww.in/a/b)")
    assert got == ["https://groww.in/a/b"]


def test_extract_urls_deduplicates_preserving_order() -> None:
    text = "https://a.com/1 then https://b.com/2 then https://a.com/1"
    assert extract_urls(text) == ["https://a.com/1", "https://b.com/2"]


def test_extract_urls_on_text_without_urls() -> None:
    assert extract_urls("no links here") == []


# --------------------------------------------------------------------------
# extract_numbers
# --------------------------------------------------------------------------


def test_extract_numbers_returns_surface_forms() -> None:
    # The three surface forms are exactly as written in the source text.
    assert extract_numbers("1.23% and Rs. 1,23,456 and 3 years") == [
        "1.23%",
        "Rs. 1,23,456",
        "3 years",
    ]


def test_extract_numbers_covers_currency_and_units() -> None:
    assert extract_numbers("The minimum is ₹500 or 5% of the amount.") == [
        "₹500",
        "5%",
    ]


def test_extract_numbers_deduplicates_and_keeps_order() -> None:
    assert extract_numbers("1.23% then 1.23% then 7") == ["1.23%", "7"]


def test_extract_numbers_on_text_without_numbers() -> None:
    assert extract_numbers("no figures here") == []


# --------------------------------------------------------------------------
# strip_number
# --------------------------------------------------------------------------


def test_strip_number_removes_surface_form_and_tidies_punctuation() -> None:
    assert strip_number("The expense ratio is 9.99% today.", "9.99%") == (
        "The expense ratio is today."
    )


def test_strip_number_removes_only_the_first_occurrence() -> None:
    assert strip_number("5% and again 5% here", "5%") == "and again 5% here"


def test_strip_number_returns_empty_when_nothing_legible_remains() -> None:
    assert strip_number("9.99%", "9.99%") == ""


def test_strip_number_is_a_no_op_for_unknown_surface_form() -> None:
    assert strip_number("Nothing to remove", "12.34%") == "Nothing to remove"
