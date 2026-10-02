# SPIKE_NOTES — P1 feasibility findings

**Run:** 2026-10-02 · **Script:** `spike/probe.py` (throwaway, deleted after this file)
**Raw HTML:** `data/raw/*.spike.html`

---

## Summary table (real output)

```
scheme_id                  status  main_chars  headings   kw_main   kw_html
------------------------------------------------------------------------------
hdfc_large_cap                200        4051        21        0/13        9/13
hdfc_flexi_cap                200        6443        21        0/13        8/13
hdfc_elss                     200        4503        20        0/13        9/13
hdfc_small_cap                200        6361        21        0/13        9/13
hdfc_balanced_advantage       200       27308        25        0/13        9/13
```

---

## Q1 (AR2) — Is the HTML server-rendered? **YES, resolved.**

All 5 return `200` with 450–815 KB of HTML. The target facts are present as real
DOM elements, not as a client-side shell. Example from `hdfc_large_cap`:

```html
<div class="valign-wrapper bodyLarge contentTertiary fundDetails_gap4__kM__Q">Expense ratio
  <div class="cur-po contentTertiary valign-wrapper">…</div></div>
<h5 class="bodyLargeHeavy contentPrimary exitLoadStampDutyTax_heading__QOf4f">Expense ratio</h5>
<p class="bodyBase contentSecondary">A fee payable to a mutual…</p>
```

**Decision: proceed with P2 as specified.**

---

## Q2 (AR1) — Is the heading tree recoverable? **YES, resolved.**

20–25 `h1..h4` per page, and they are real content headings:

```
HDFC Large Cap Fund Direct Growth        (h1)
  Minimum investments                    (h3)
  Understand terms                       (h2)
  Returns and rankings                   (h3)
  Exit Load                              (h3)
  Exit load, stamp duty and tax          (h3)
    Exit load                            (h4)
    Stamp duty on investment: 0.005% …   (h4)
    Tax implication                      (h4)
  Fund management                        (h3)
  Investment Objective                   (h4)
  Fund house                             (h2)
```

**Two defects to handle in P2:**
1. **Duplicate headings** — the page renders desktop and mobile DOM side by side, so
   `About HDFC Large Cap Fund Direct Growth`, `Fund house` and `Understand terms` each
   appear **twice**. Sections must be deduplicated by (title, text).
2. **Container headings have no text of their own** — `Returns and rankings`, `Exit Load`
   and `Exit load, stamp duty and tax` yield 0 lines because their content lives under
   nested `h4` children. Expected and correct; those parents drop out and the `h4`s carry
   the content.

---

## Q3 — Does trafilatura work? **NO. Decisive finding.**

`kw_main = 0/13` on **every** page — trafilatura returns 4–27 k chars of real-looking
prose containing **none** of the target facts. It drops exactly the `fundDetails_*` divs
that carry fees, minimums and exit load.

| Path | Chars (large cap) | Keywords |
| --- | --- | --- |
| `trafilatura.extract` | 4 051 | **0/13** |
| `BeautifulSoup.get_text` after stripping script/style/noscript/svg | 17 759 | **8/10** |

**Decision: the BS4 heading-tree walk is the *only* viable primary path.** This is what
P2's spec already specifies, but this run upgrades it from "preferred" to "mandatory" —
and trafilatura stays only as a last-ditch fallback per the spec.

---

## Q4 — Which facts are actually answerable?

Measured on the **visible DOM**, after decomposing `script`/`style`/`noscript`/`svg`:

| Fact | Large | Flexi | ELSS | Small | Balanced |
| --- | :-: | :-: | :-: | :-: | :-: |
| expense ratio | ✅ | ✅ | ✅ | ✅ | ✅ |
| exit load | ✅ | ✅ | ✅ | ✅ | ✅ |
| minimum SIP | ✅ | ✅ | ✅ | ✅ | ✅ |
| minimum investment | ✅ | ✅ | ✅ | ✅ | ✅ |
| benchmark | ✅ | ✅ | ✅ | ✅ | ✅ |
| capital gains | ✅ | ✅ | ✅ | ✅ | ✅ |
| fund manager | ✅ | ✅ | ✅ | ✅ | ✅ |
| NAV | ✅ | ✅ | ✅ | ✅ | ✅ |
| **lock-in** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **riskometer** | ❌ | ❌ | ❌ | ❌ | ❌ |
| **statement (download)** | ❌ | ❌ | ❌ | ❌ | ❌ |

- `lock-in` present only on the ELSS page — as expected.
- **`riskometer` is absent from every page's visible DOM.** It exists only inside the
  `__NEXT_DATA__` JSON blob: `"nfo_risk":"Moderately High Riskometer"`.
- **`statement` is absent everywhere.** No "how do I download a statement" content.
- Partially recoverable: the page header block renders `Equity | Large Cap | Very High
  Risk`, so a coarse "how risky is this fund" question may still be answerable from prose.

### 🔴 Consequence for AC8 — needs a decision before P10

Of the 15 golden factual queries, **3 target `riskometer` and 2 target `statement`**.
Those 5 facts are not in the corpus, so they cannot be answered from it.

**Best case AC8 = 12/15 = 80%, which fails the ≥90% gate in PRD §11.2.**

Options are recorded in the handoff message. This does **not** block P2 — it blocks
P3/P10 and needs a call on whether to amend the golden set or the AC8 threshold.

---

## Q5 — Do the URLs resolve? **Yes, but two schemes are renamed.**

No redirects on any URL. However the page `<h1>` disagrees with the brief for 4 of 5
schemes, so `data/sources.csv` `scheme_name` has been updated to the **page title**
(it is what appears in citations):

| `scheme_id` | Brief name | Actual page title |
| --- | --- | --- |
| `hdfc_large_cap` | HDFC Large Cap Fund | **HDFC Large Cap Fund Direct Growth** |
| `hdfc_flexi_cap` | HDFC Equity Fund | **HDFC Flexi Cap Direct Plan Growth** ⚠️ |
| `hdfc_elss` | HDFC ELSS Tax Saver Fund | HDFC ELSS Tax Saver Fund Direct Plan Growth ✓ |
| `hdfc_small_cap` | HDFC Small Cap Fund | **HDFC Small Cap Fund Direct Growth** |
| `hdfc_balanced_advantage` | HDFC Balanced Advantage Fund | **HDFC Balanced Advantage Fund Direct Growth** |

⚠️ **`hdfc_flexi_cap` is the scheme the brief calls "HDFC Equity Fund".** The AMC renamed
it to Flexi Cap; Groww serves the renamed scheme at the old URL. Aliases and display
names must use "Flexi Cap" (see P5 `data/scheme_aliases.json`, which already lists both).

---

## Structural note carried into P2

The fee and minimum-investment facts are **key:value pairs in nested `div`s, not
`<table>` elements**:

```
<h3>Minimum investments</h3>
  Min. for 1st investment | ₹100
  Min. for 2nd investment | ₹100
  Min. for SIP            | ₹100
```

So the `<table>` check in `is_table_block` will rarely fire, and the **line-based
label→value heuristic is the detector that actually matters** for P3's table-atomicity
requirement. P2 must therefore preserve per-pair line structure when building section
text — `normalize_ws` alone would collapse these into one line and destroy the signal.

Relatedly, `<script>` must be decomposed **before** extraction. It contains return
figures and the riskometer value, so leaving it in would both pollute chunks and risk
leaking performance numbers into answers (FR-8).
