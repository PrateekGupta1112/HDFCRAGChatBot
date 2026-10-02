"""Unit tests for app.ingest — the Loading stage.

Fully offline: a synthetic HTML fixture written to ``tmp_path`` stands in for a
real scheme page. No network, no real fund names, no real values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.config import Settings
from app.errors import ExtractionError, FetchError
from app.ingest import (
    MODE_HEADING_WALK,
    Document,
    Section,
    SourceSpec,
    build_document,
    extract,
    is_table_block,
    load_documents,
    read_sources,
    repair_document_aum,
    run_ingest,
    write_documents,
)
from app.textutils import normalize_ws

# Synthetic fixture. Obvious dummy values only — no real scheme data.
SYNTHETIC_HTML = """
<html><body>
<script>var junk = {"expense ratio": "9.99%", "nav": "1234.56"};</script>
<main>
  <h1>Synthetic Fund Direct Growth</h1>
  <p>Equity Large Cap Very High Risk</p>

  <h3>Minimum investments</h3>
  <div><div>Min. for 1st investment</div><div>&#8377;100</div></div>
  <div><div>Min. for 2nd investment</div><div>&#8377;100</div></div>
  <div><div>Min. for SIP</div><div>&#8377;100</div></div>

  <h2>Understand terms</h2>
  <div><p>Expense ratio: 1.23%. A fee payable to a mutual fund company for
  managing the scheme assets. It is deducted from the fund's assets and is
  therefore borne by the investor as part of the scheme's returns, so it is not
  charged separately to the investor at the time of purchase. The figure quoted
  on this page is the total expense ratio disclosed in the scheme's scheme
  information document, and it is reviewed at least once every year.</p></div>

  <h2>Home</h2>
  <div><p>Sign up for the app to start investing today with zero brokerage.</p></div>
</main>
</body></html>
"""

SPEC = SourceSpec(
    scheme_id="synthetic_a",
    scheme_name="Synthetic Fund Direct Growth",
    category="Large Cap",
    plan="Direct Growth",
    url="https://example.com/synthetic-a",
)


@pytest.fixture(autouse=True)
def isolate_ingest_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep ``run_ingest`` from writing anything outside ``tmp_path``.

    Since P4, ``run_ingest`` also chunks, embeds and upserts to Chroma. Those
    steps resolve their own destinations, and the Chroma one goes through the
    ``get_settings()`` singleton rather than the settings passed to
    ``run_ingest``. Without this fixture a unit test with a synthetic page
    overwrites the committed ``chunks.jsonl`` and — because ``upsert_chunks``
    deletes every id absent from the batch — **deletes the real vector store and
    replaces it with three synthetic chunks**. That happened; this fixture is the
    regression guard.

    The encoder is stubbed rather than loaded: a unit test must not need a 90 MB
    download, and the encoder is not what these tests are exercising.
    """
    import numpy as np

    from app import config, embedding

    monkeypatch.setattr("app.ingest.CHUNKS_JSONL", tmp_path / "chunks.jsonl")
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    # cache_clear AFTER setenv, so the singleton is rebuilt from the new value.
    config.get_settings.cache_clear()
    monkeypatch.setattr(
        embedding, "embed_texts", lambda texts: np.zeros((len(texts), 384), dtype=np.float32)
    )
    yield
    # Drop the tmp CHROMA_PATH before another module reads the singleton.
    config.get_settings.cache_clear()


def test_run_ingest_never_touches_the_real_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full run_ingest must leave every committed artifact byte-identical.

    Asserted as "unchanged", not "does not contain synthetic_a", so the test is
    independent of whether a corpus has been built yet.
    """
    real_chunks = Path("data/processed/chunks.jsonl")
    real_documents = Path("data/processed/documents.jsonl")
    before = {p: p.read_bytes() if p.exists() else None for p in (real_chunks, real_documents)}

    monkeypatch.setattr("app.ingest.DOCUMENTS_JSONL", tmp_path / "documents.jsonl")
    monkeypatch.setattr("app.ingest.INGEST_REPORT_JSON", tmp_path / "report.json")
    monkeypatch.setattr("app.ingest.RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(
        "app.ingest.fetch", lambda url, s=None: ("https://example.com/a", SYNTHETIC_HTML, 200)
    )
    run_ingest(sources=[SPEC], s=Settings(_env_file=None, fetch_delay_s=0.0))

    assert (tmp_path / "chunks.jsonl").exists(), "chunks did not land in tmp_path"
    for path, content in before.items():
        now = path.read_bytes() if path.exists() else None
        assert now == content, f"{path} was modified by a unit test"


def test_extract_returns_sections_in_document_order() -> None:
    sections, mode = extract(SYNTHETIC_HTML, min_chars=10)
    titles = [s.section_title for s in sections]
    assert mode == MODE_HEADING_WALK
    assert titles == ["Synthetic Fund Direct Growth", "Minimum investments", "Understand terms"]


def test_extract_drops_boilerplate_headings() -> None:
    sections, _ = extract(SYNTHETIC_HTML, min_chars=10)
    assert "Home" not in [s.section_title for s in sections]


def test_extract_removes_script_content() -> None:
    """__NEXT_DATA__ blobs embed return figures; they must never reach a section."""
    sections, _ = extract(SYNTHETIC_HTML, min_chars=10)
    body = " ".join(s.text for s in sections)
    assert "junk" not in body
    assert "1234.56" not in body


def test_key_value_block_is_flagged_as_a_table() -> None:
    sections, _ = extract(SYNTHETIC_HTML, min_chars=10)
    minimums = next(s for s in sections if s.section_title == "Minimum investments")
    assert minimums.is_table is True
    # Each label:value pair must survive on its own line for P3 to keep it atomic.
    assert len(minimums.text.splitlines()) >= 3


def test_prose_section_is_not_flagged_as_a_table() -> None:
    sections, _ = extract(SYNTHETIC_HTML, min_chars=10)
    terms = next(s for s in sections if s.section_title == "Understand terms")
    assert terms.is_table is False


def test_nested_deeper_heading_does_not_swallow_parent_text() -> None:
    html = """
    <html><body><main>
      <h2>Exit load, stamp duty and tax</h2>
      <p>Intro paragraph for the parent section.</p>
      <h4>Exit load</h4>
      <p>Charged as a percentage of the redemption amount.</p>
    </main></body></html>
    """
    sections, _ = extract(html, min_chars=10)
    parent = next(s for s in sections if s.section_title == "Exit load, stamp duty and tax")
    child = next(s for s in sections if s.section_title == "Exit load")
    # The child heading's body belongs to the child, and must not be copied
    # into the parent as well.
    assert parent.text == "Intro paragraph for the parent section."
    assert "Charged as a percentage" in child.text


def test_nested_section_body_is_not_duplicated_into_the_parent() -> None:
    """The nested heading owns its body; no text appears in two sections."""
    html = """
    <html><body><main>
      <h2>Understand terms</h2>
      <p>First parent paragraph.</p>
      <h4>Exit load</h4>
      <p>Child paragraph.</p>
      <p>Second paragraph of the same block.</p>
      <h3>Fund management</h3>
      <p>Manager names.</p>
    </main></body></html>
    """
    sections, _ = extract(html, min_chars=10)
    by_title = {s.section_title: s.text for s in sections}
    assert by_title["Understand terms"] == "First parent paragraph."
    assert "Child paragraph." in by_title["Exit load"]
    assert "Manager names." not in by_title["Exit load"]


def test_extract_below_the_floor_raises_rather_than_returning_a_stub() -> None:
    """Spec: raise ExtractionError if the page is too thin to be real content."""
    thin = "<html><body><main><h2>Hi</h2><p>Short.</p></main></body></html>"
    with pytest.raises(ExtractionError, match="below the 500 minimum"):
        extract(thin)


def test_table_element_forces_is_table_regardless_of_text_shape() -> None:
    """A real <table> is atomic even when its rows do not look like label:value."""
    html = (
        "<html><body><main><h2>Holdings</h2>"
        "<table><tr><td>Only</td></tr></table></main></body></html>"
    )
    # min_chars=1 keeps this on the heading-walk path so the section is real.
    sections, mode = extract(html, min_chars=1)
    assert mode == MODE_HEADING_WALK
    assert sections[0].is_table is True


def test_footer_chrome_is_pruned_by_css_module_class() -> None:
    """Groww has no <footer> tag, so chrome is only findable by class name."""
    html = """
    <html><body>
      <h2>Fund house</h2>
      <div class="fundHouse_details__aBc12"><p>HDFC Mutual Fund</p></div>
      <div class="footer_footerWrapper__Xy9Zq">
        <p>Contact Us</p><p>Careers</p><p>[email protected]</p>
      </div>
    </body></html>
    """
    sections, _ = extract(html, min_chars=1)
    text = sections[0].text
    assert "HDFC Mutual Fund" in text
    assert "Contact Us" not in text
    assert "Careers" not in text
    assert "[email protected]" not in text


def test_h1_class_named_header_is_not_mistaken_for_chrome() -> None:
    """Regression: a 'header_' prefix rule deleted the page's own <h1>."""
    html = """
    <html><body>
      <h1 class="displaySmall header_schemeName__zL6RN">Synthetic Fund Direct Growth</h1>
      <div><p>Equity Large Cap Very High Risk</p></div>
    </body></html>
    """
    sections, _ = extract(html, min_chars=1)
    assert [s.section_title for s in sections] == ["Synthetic Fund Direct Growth"]
    assert "Very High Risk" in sections[0].text


def test_pruning_chrome_does_not_empty_the_walk() -> None:
    """Decomposing mid-iteration corrupts descendants and yields 0 sections."""
    html = """
    <html><body>
      <h2>Real section</h2>
      <div><p>Real content that must survive pruning.</p></div>
      <div class="footer_footerWrapper__Xy9Zq"><p>chrome</p></div>
    </body></html>
    """
    sections, mode = extract(html, min_chars=1)
    assert mode == MODE_HEADING_WALK
    assert "Real content that must survive pruning." in sections[0].text


def test_is_table_block_requires_three_pairs() -> None:
    assert is_table_block("Min. for SIP\n\u20b9500\nMin. for STP\n\u20b9500\nMin. for SWP\n\u20b9500")
    assert is_table_block("Expense ratio: 1.23%\nExit load: 1%\nLock in: 3 years")
    assert not is_table_block("A paragraph of ordinary prose with no figures at all.")
    assert not is_table_block("Min. for SIP \u20b9500")  # a single pair is not a table
    assert not is_table_block("Min. for SIP \u20b9500\nMin. for STP \u20b9500")


def test_prose_breaks_a_pair_run() -> None:
    text = "Min. for SIP\n\u20b9500\nA sentence of prose interrupts the block.\nMin. for STP\n\u20b9500"
    assert not is_table_block(text)


def test_build_document_hash_is_stable_across_calls() -> None:
    a = build_document(SPEC, SYNTHETIC_HTML, SPEC.url, "bs4_heading_walk")
    b = build_document(SPEC, SYNTHETIC_HTML, SPEC.url, "bs4_heading_walk")
    assert a.content_sha256 == b.content_sha256
    assert len(a.content_sha256) == 64


def test_build_document_hash_changes_with_content() -> None:
    a = build_document(SPEC, SYNTHETIC_HTML, SPEC.url, "bs4_heading_walk")
    b = build_document(
        SPEC, SYNTHETIC_HTML.replace("1.23%", "9.87%"), SPEC.url, "bs4_heading_walk"
    )
    assert a.content_sha256 != b.content_sha256


def test_build_document_drops_empty_sections_and_renumbers_order() -> None:
    sections = [
        Section("Kept", 2, "real text here", False, 7),
        Section("Empty", 2, "   \n  ", False, 9),
        Section("Also kept", 3, "more text", False, 11),
    ]
    doc = build_document(SPEC, "", SPEC.url, "bs4_heading_walk", sections)
    assert [s.section_title for s in doc.sections] == ["Kept", "Also kept"]
    assert [s.order for s in doc.sections] == [0, 1]


def test_build_document_uses_the_final_url_for_citations() -> None:
    doc = build_document(
        SPEC, SYNTHETIC_HTML, "https://example.com/synthetic-a-renamed", "bs4_heading_walk"
    )
    assert doc.source_url == "https://example.com/synthetic-a-renamed"


def test_build_document_fetched_at_is_idempotent_within_a_day() -> None:
    from datetime import date

    doc = build_document(SPEC, SYNTHETIC_HTML, SPEC.url, "bs4_heading_walk")
    assert doc.fetched_at == date.today().isoformat()
    assert "T" not in doc.fetched_at  # a datetime would break same-day re-runs


def test_jsonl_round_trip(tmp_path: Path) -> None:
    doc = build_document(SPEC, SYNTHETIC_HTML, SPEC.url, "bs4_heading_walk")
    path = tmp_path / "documents.jsonl"
    write_documents([doc], path)
    assert len(path.read_text().splitlines()) == 1
    loaded = load_documents(path)
    assert loaded == [doc]


def test_load_documents_on_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_documents(tmp_path / "nope.jsonl") == []


def test_read_sources_reads_the_real_csv() -> None:
    specs = read_sources()
    assert len(specs) == 5
    assert {s.scheme_id for s in specs} == {
        "hdfc_large_cap",
        "hdfc_flexi_cap",
        "hdfc_elss",
        "hdfc_small_cap",
        "hdfc_balanced_advantage",
    }
    assert all(s.url.startswith("https://groww.in/") for s in specs)


def test_read_sources_rejects_a_csv_missing_the_url_column(tmp_path: Path) -> None:
    bad = tmp_path / "sources.csv"
    bad.write_text("scheme_id,scheme_name\nonly_a,Only A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        read_sources(bad)


def test_read_sources_rejects_an_empty_required_field(tmp_path: Path) -> None:
    bad = tmp_path / "sources.csv"
    bad.write_text(
        "scheme_id,scheme_name,category,plan,url\na,Only A,Large Cap,,https://example.com/a\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty field"):
        read_sources(bad)


def test_cached_document_picks_up_an_edited_scheme_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing the CSV must not require a re-fetch to take effect."""
    monkeypatch.setattr("app.ingest.DOCUMENTS_JSONL", tmp_path / "documents.jsonl")
    monkeypatch.setattr("app.ingest.INGEST_REPORT_JSON", tmp_path / "report.json")
    monkeypatch.setattr("app.ingest.RAW_DIR", tmp_path / "raw")
    calls = []

    def counting_fetch(url: str, s: Settings | None = None) -> tuple[str, str, int]:
        calls.append(url)
        return "https://example.com/a", SYNTHETIC_HTML, 200

    monkeypatch.setattr("app.ingest.fetch", counting_fetch)
    s = Settings(_env_file=None, fetch_delay_s=0.0)
    run_ingest(sources=[SPEC], s=s)
    run_ingest(sources=[SPEC], s=s)
    assert len(calls) == 1, "second run should reuse the cache"

    renamed = SourceSpec(
        SPEC.scheme_id, "Synthetic Fund Direct Growth (Renamed)", SPEC.category, SPEC.plan, SPEC.url
    )
    report = run_ingest(sources=[renamed], s=s)
    assert report.documents[0].scheme_name == "Synthetic Fund Direct Growth (Renamed)"
    assert len(calls) == 1


def test_run_ingest_reports_a_failed_source_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bad page must not abort the run — the failure is reported."""
    monkeypatch.setattr("app.ingest.DOCUMENTS_JSONL", tmp_path / "documents.jsonl")
    monkeypatch.setattr("app.ingest.INGEST_REPORT_JSON", tmp_path / "report.json")
    monkeypatch.setattr("app.ingest.RAW_DIR", tmp_path / "raw")

    def flaky(url: str, s: Settings | None = None) -> tuple[str, str, int]:
        if url.endswith("b"):
            raise FetchError("synthetic network failure")
        return "https://example.com/a", SYNTHETIC_HTML, 200

    monkeypatch.setattr("app.ingest.fetch", flaky)
    specs = [
        SourceSpec("a", "A", "Large Cap", "Direct Growth", "https://example.com/a"),
        SourceSpec("b", "B", "Small Cap", "Direct Growth", "https://example.com/b"),
    ]
    report = run_ingest(sources=specs, s=Settings(_env_file=None, fetch_delay_s=0.0))

    assert report.sources_ok == 1
    assert report.sources_failed == ["b"]
    assert len(report.warnings) == 1
    assert "synthetic network failure" in report.warnings[0]
    assert (tmp_path / "documents.jsonl").exists()


def test_run_ingest_persists_a_machine_readable_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.ingest.DOCUMENTS_JSONL", tmp_path / "documents.jsonl")
    monkeypatch.setattr("app.ingest.INGEST_REPORT_JSON", tmp_path / "report.json")
    monkeypatch.setattr("app.ingest.RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(
        "app.ingest.fetch", lambda url, s=None: ("https://example.com/a", SYNTHETIC_HTML, 200)
    )
    run_ingest(
        sources=[SPEC], s=Settings(_env_file=None, fetch_delay_s=0.0)
    )
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["sources_ok"] == 1
    assert payload["documents"][0]["scheme_id"] == "synthetic_a"
    assert payload["max_fetched_at"]


def test_rerun_does_not_duplicate_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR-1: ingestion must be re-runnable without duplicating documents."""
    monkeypatch.setattr("app.ingest.DOCUMENTS_JSONL", tmp_path / "documents.jsonl")
    monkeypatch.setattr("app.ingest.INGEST_REPORT_JSON", tmp_path / "report.json")
    monkeypatch.setattr("app.ingest.RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(
        "app.ingest.fetch", lambda url, s=None: ("https://example.com/a", SYNTHETIC_HTML, 200)
    )
    s = Settings(_env_file=None, fetch_delay_s=0.0)
    run_ingest(sources=[SPEC], s=s)
    run_ingest(sources=[SPEC], s=s)
    lines = (tmp_path / "documents.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert len(load_documents(tmp_path / "documents.jsonl")) == 1


def test_document_field_order_is_stable() -> None:
    """P3/P4 index these fields positionally via asdict()."""
    assert list(Document.__dataclass_fields__) == [
        "scheme_id",
        "scheme_name",
        "category",
        "plan",
        "source_url",
        "fetched_at",
        "content_sha256",
        "extraction_mode",
        "sections",
    ]


# --------------------------------------------------------------------------
# AUM mislabelling repair
#
# A source page can carry the AMC's aggregate AUM inside a sentence that calls
# it the scheme's. All five real pages did. The repair redirects that sentence to
# the figure the same page publishes under the more specific "Fund size (AUM)"
# label. Synthetic values throughout — no real scheme data.
# --------------------------------------------------------------------------

#: The scheme's own AUM, as the summary strip states it.
SCHEME_AUM_TEXT = "11,111.11"
#: The AMC's aggregate AUM, as the "Fund house" block states it.
AMC_AUM_TEXT = "99,999.99"
#: The prose figure: the AMC total rounded to 0dp, which is what the pages do.
AMC_AUM_ROUNDED_TEXT = "1,00,000"


def _aum_document(
    summary_aum: str | None = SCHEME_AUM_TEXT,
    amc_aum: str | None = AMC_AUM_TEXT,
    prose_aum: str = AMC_AUM_ROUNDED_TEXT,
) -> Document:
    """Build a document laid out like a real scheme page.

    Args:
        summary_aum: Value for ``Fund size (AUM)``; ``None`` omits the label.
        amc_aum: Value for ``Total AUM``; ``None`` omits the label.
        prose_aum: Value the About prose attributes to the scheme.
    """
    summary = f"Fund size (AUM) ₹{summary_aum} Cr" if summary_aum else "Rating 4"
    fund_house = f"Total AUM ₹{amc_aum} Cr" if amc_aum else "Custodian Example Bank"
    return Document(
        scheme_id="synthetic_a",
        scheme_name="Synthetic Fund Direct Growth",
        category="Large Cap",
        plan="Direct Growth",
        source_url="https://example.com/synthetic-a",
        fetched_at="2026-01-01",
        content_sha256="0" * 64,
        extraction_mode=MODE_HEADING_WALK,
        sections=[
            Section(
                section_title="Synthetic Fund Direct Growth",
                heading_level=1,
                text=f"Equity Large Cap {summary}",
                is_table=False,
                order=0,
            ),
            Section(
                section_title="Fund house",
                heading_level=2,
                text=f"Example Asset Management {fund_house}",
                is_table=False,
                order=1,
            ),
            Section(
                section_title="About Synthetic Fund Direct Growth",
                heading_level=2,
                text=(
                    "Synthetic Fund Direct Growth is an Equity Mutual Fund Scheme. "
                    f"The fund currently has an Asset Under Management(AUM) of ₹{prose_aum} Cr "
                    "and the Latest NAV as of 01 Jan 2026 is ₹111.11."
                ),
                is_table=False,
                order=2,
            ),
        ],
    )


def _about(doc: Document) -> str:
    return next(s.text for s in doc.sections if s.section_title.startswith("About"))


def test_prose_aum_matching_the_amc_is_redirected_to_the_scheme_figure() -> None:
    """The defect: prose repeats the AMC total, which is not the fund's size."""
    fixed = repair_document_aum(_aum_document())
    assert f"₹{SCHEME_AUM_TEXT} Cr" in _about(fixed)
    assert f"₹{AMC_AUM_ROUNDED_TEXT} Cr" not in _about(fixed)


def test_prose_aum_already_correct_is_left_alone() -> None:
    """A page that cites its own AUM must survive byte-for-byte."""
    doc = _aum_document(prose_aum=SCHEME_AUM_TEXT)
    assert repair_document_aum(doc) is doc


def test_repair_matches_across_the_pages_rounding() -> None:
    """Prose is rounded to 0dp; the AMC figure carries 2dp. Exact compare fails."""
    doc = _aum_document()
    fixed = repair_document_aum(doc)
    assert fixed is not doc, "0dp prose vs 2dp AMC value should still be recognised"


def test_repair_keeps_the_rest_of_the_sentence() -> None:
    """Only the figure changes. NAV, wording and section identity must survive."""
    doc = _aum_document()
    before = _about(doc)
    after = _about(repair_document_aum(doc))
    assert "Asset Under Management(AUM) of" in after
    assert "the Latest NAV as of 01 Jan 2026 is ₹111.11." in after
    assert after == before.replace(f"₹{AMC_AUM_ROUNDED_TEXT} Cr", f"₹{SCHEME_AUM_TEXT} Cr")


def test_repair_does_not_touch_other_sections() -> None:
    doc = _aum_document()
    fixed = repair_document_aum(doc)
    assert [s.section_title for s in fixed.sections] == [s.section_title for s in doc.sections]
    for a, b in zip(doc.sections[:2], fixed.sections[:2]):
        assert a.text == b.text


def test_repair_is_idempotent() -> None:
    """run_ingest replays cached documents, so a second pass must be a no-op."""
    once = repair_document_aum(_aum_document())
    assert repair_document_aum(once) is once


def test_repair_refreshes_the_content_hash() -> None:
    """content_sha256 covers section text; a stale hash would misattest the corpus."""
    doc = _aum_document()
    fixed = repair_document_aum(doc)
    assert fixed.content_sha256 != doc.content_sha256
    expected = hashlib.sha256(
        normalize_ws(" ".join(s.text for s in fixed.sections)).encode("utf-8")
    ).hexdigest()
    assert fixed.content_sha256 == expected


def test_no_scheme_label_means_no_repair() -> None:
    """Nothing to compare against — reporting the source as written beats guessing."""
    doc = _aum_document(summary_aum=None)
    assert repair_document_aum(doc) is doc


def test_no_amc_label_means_no_repair() -> None:
    """A scheme whose only AUM figure is its own is never mislabelled."""
    doc = _aum_document(amc_aum=None)
    assert repair_document_aum(doc) is doc


def test_unrelated_prose_figure_is_not_rewritten() -> None:
    """A figure matching neither label is left alone."""
    doc = _aum_document(prose_aum="42,424.24")
    assert repair_document_aum(doc) is doc


def test_build_document_applies_the_repair() -> None:
    """The repair must be wired into the build path, not merely available."""
    sections = [
        Section("Synthetic Fund Direct Growth", 1, f"Fund size (AUM) ₹{SCHEME_AUM_TEXT} Cr", False, 0),
        Section("Fund house", 2, f"Total AUM ₹{AMC_AUM_TEXT} Cr", False, 1),
        Section(
            "About Synthetic Fund Direct Growth",
            2,
            f"The fund currently has an Asset Under Management(AUM) of ₹{AMC_AUM_ROUNDED_TEXT} Cr.",
            False,
            2,
        ),
    ]
    doc = build_document(SPEC, html="", final_url="https://example.com/a", mode=MODE_HEADING_WALK, sections=sections)
    about = next(s.text for s in doc.sections if s.section_title.startswith("About"))
    assert f"₹{SCHEME_AUM_TEXT} Cr" in about


def test_run_ingest_repairs_a_cached_document_written_before_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached documents skip build_document, so the cached path must repair too.

    Without this the defect survives forever in any corpus that was built before
    the repair existed, with no re-fetch and no warning.
    """
    monkeypatch.setattr("app.ingest.DOCUMENTS_JSONL", tmp_path / "documents.jsonl")
    monkeypatch.setattr("app.ingest.INGEST_REPORT_JSON", tmp_path / "report.json")
    monkeypatch.setattr("app.ingest.RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr("app.ingest.CHUNKS_JSONL", tmp_path / "chunks.jsonl")
    s = Settings(_env_file=None, fetch_delay_s=0.0)
    monkeypatch.setattr("app.ingest.fetch", lambda url, s=None: ("https://example.com/a", SYNTHETIC_HTML, 200))
    monkeypatch.setattr("app.chunking.chunk_all", lambda docs: ([], []))
    monkeypatch.setattr("app.chunking.write_chunks", lambda chunks, path=None: None)
    monkeypatch.setattr("app.embedding.embed_texts", lambda texts: [])
    monkeypatch.setattr("app.store.upsert_chunks", lambda chunks, embeddings, force=False: 0)

    # Seed the cache with a document that still carries the defect.
    write_documents([_aum_document()], tmp_path / "documents.jsonl")

    # run_ingest must take the cached branch, not fetch.
    def _boom(url: str, s=None):  # pragma: no cover - must never run
        raise AssertionError("cached branch not taken; test would not cover the repair")

    monkeypatch.setattr("app.ingest.fetch", _boom)
    report = run_ingest(sources=[SPEC], s=s)

    assert report.sources_ok == 1
    about = _about(load_documents(tmp_path / "documents.jsonl")[0])
    assert f"₹{SCHEME_AUM_TEXT} Cr" in about
