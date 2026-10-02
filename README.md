# MF Facts Bot

A **facts-only** RAG chatbot that answers factual questions about 5 HDFC mutual
fund schemes, grounded in their public scheme pages, with exactly one verified
source link on every answer.

It refuses advice, performance and PII questions rather than answering them.

> **Status: Phases 0–9 complete. The bot runs end to end, in a browser and from
> the terminal.** The 5 source pages are fetched, extracted into sections, chunked
> into **131 retrieval units with 0 rule violations**, embedded and persisted to
> ChromaDB (**131 vectors, 0 dropped**); `retrieve()` answers against them with
> correct `1 − distance` similarity math, a scheme alias filter and MMR diversity;
> the refusal layer classifies all **23 gate queries** with **zero false positives**;
> and generation enforces one verified citation, a three-sentence cap and a
> freshness stamp before anything reaches the user.
> **P10 (evaluation harness) is the remaining phase.** See
> [Build status](#build-status) for exactly what does and does not work.

---

## Contents

- [Disclaimer](#disclaimer)
- [Build status](#build-status)
- [What this will do](#what-this-will-do)
- [Known defects](#known-defects)
- [Requirements](#requirements)
- [Which provider does what](#which-provider-does-what)
- [Setup](#setup)
- [Running the tests](#running-the-tests)
- [Running the app](#running-the-app)
- [Verified — the 14-point UI walkthrough (AC11)](#verified--the-14-point-ui-walkthrough-ac11)
- [Repository layout](#repository-layout)
- [Build status detail](#build-status-detail)
- [Design documents](#design-documents)
- [Disclaimer source of truth](#disclaimer-source-of-truth)

---

## Disclaimer

Facts-only. No investment advice. Answers are generated from public scheme
pages and may be incomplete or out of date. They are not a recommendation to buy,
sell or hold any security. Mutual fund investments are subject to market risks;
read all scheme-related documents carefully. Verify every fact on the linked
source page before acting.

---

## Build status

| Works today | Not built yet |
| --- | --- |
| Virtualenv + 130 pinned dependencies | Streamlit UI, evaluation |
| `app.config` — 28 centralised settings | |
| `app.disclaimer` — all user-facing copy | |
| `app.errors` — typed exceptions | |
| `app.textutils` — 8 pure text helpers | |
| `app.ingest` — fetches all 5 pages, extracts 88 clean sections | |
| `app.tokenization` — word-piece counting against the model's own tokenizer | |
| `app.chunking` — 131 chunks, 0 violations | |
| `app.embedding` — 384-dim vectors, batched, model singleton | |
| `app.store` — persistent ChromaDB collection, 131 vectors, 0 dropped | |
| `app.scheme_aliases` — all 5 schemes resolve; ambiguity ⇒ global search | |
| `app.retrieval` — alias filter → KNN → threshold → MMR → assemble | |
| `app.guardrails` — 23/23 gate queries, 0 false positives | |
| `app.generation` — prompt, Groq LLM path, extractive fallback, fail-closed enforcement | Streamlit UI, evaluation |
| `app.pipeline` — `answer_turn()` orchestration, static refusals, session | |
| `app.chat` — headless CLI: `--query`, `--json`, stdin | Streamlit UI, evaluation |
| `app.ui` — Streamlit page: chat, citation link, retrieve/guardrail expanders, sidebar | demo |
| 417 passing unit tests | |

**The bot answers questions end to end, in a browser and from the terminal.**
P10 (evaluation harness) is the remaining phase. Every section below marked
*— verified* was run; where something is still wrong it is in
[Known defects](#known-defects), not softened.

### Ingestion works — verified

```bash
.venv/bin/python -m app.ingest --fetch-only
```

| | |
| --- | --- |
| Sources fetched | **5 / 5**, HTTP 200, no failures |
| Sections extracted | **88** across 16–21 per page |
| Extraction path | BS4 heading-tree walk on all 5 (no fallback needed) |
| Content recovered | expense ratio, exit load, minimum SIP, NAV, benchmark, capital gains, fund manager, stamp duty, tax, AUM |
| Raw HTML kept | `data/raw/*.html` for re-extraction without re-fetching |
| Re-runnable | cached by `content_sha256`; a second run fetches nothing and does not duplicate lines |

### Chunking works — verified

```bash
.venv/bin/python -m app.chunking
```

| | |
| --- | --- |
| Chunks written | **131** to `data/processed/chunks.jsonl` |
| Rule violations | **0** — no chunk exceeds 240 word-pieces, header included |
| Chunk sizes | min 46 · mean 145–195 · max 239 word-pieces |
| Per scheme | large-cap 18 · flexi-cap 20 · ELSS 18 · small-cap 20 · balanced-advantage 55 |
| Table chunks | 82 of 131, never row-split, never overlapped |
| `chunk_index` | document-global and gapless on all 5 documents |
| Labels | 0 orphaned — every `Expense ratio`, `Rating`, `Min. for SIP` sits with its value |

Three P2 extraction defects were found and fixed while building this phase. They
mattered because chunking cannot repair them:

| Defect | Effect on chunks | Fix |
| --- | --- | --- |
| `build_document` ran `normalize_ws` over section text, collapsing every newline | **All 88 sections became one line.** Table splitting had no line boundaries to cut at, so `Expense ratio` ended up in an unbroken run with 7 other figures and no way to tell which number was the fee | New `normalize_lines()` in `textutils`; normalises per line and keeps the structure |
| A label div containing an info-icon div was not a "leaf", so its text was dropped | `Expense ratio` vanished while its sibling `1.03%` survived — a bare number with nothing to name it | `_leaf_line` now yields a nested block's *own* text instead of skipping it |
| `_VALUE_ONLY` did not accept a leading sign | `+7.6%` read as a label, so the whole `Returns and rankings` block failed the 3-pair test and went down the prose path, where atomicity does not apply | Sign added to the pattern; the block is now a table on all 5 pages |

### Vector store works — verified

```bash
.venv/bin/python -m app.ingest
```

| | |
| --- | --- |
| Chunks embedded | **131** in one batch |
| Vectors stored | **131** to `chroma_db/` (cosine space) — 0 dropped |
| Vector dimension | **384** — asserted against the model's own `get_sentence_embedding_dimension()` |
| Schemes indexed | all **5** |
| Persistent | yes — `chroma.sqlite3` on disk, survives process exit |
| Re-runnable | yes — incremental by `chunk_id`; a second run kept the count at 131 |

**131 chunks in, 131 vectors stored, 0 dropped.** This was *not* true until the
dedupe defect below was fixed, and the numbers are kept because the fix is easy
to mistake for a non-issue — nothing errored, the count was just quietly wrong.

The defect: `upsert_chunks` de-duplicated on `content_sha256`, and
`chunking.py` computed that hash over the chunk **body alone** (`_sha256(body)`)
— with no `scheme_id`. Eight groups of boilerplate sections are byte-identical
across different schemes and therefore collided, dropping **18 chunks** behind a
`continue  # should not happen` that fired 18 times and was never noticed. Measured
on the real corpus:

| | |
| --- | --- |
| Groups with >1 chunk sharing a hash | **8** |
| Chunks dropped | **18** |
| Collisions that are **cross-scheme** | **8 / 8** — never within one scheme |
| Collisions that are cross-section | **0 / 8** — always the same boilerplate section |
| Sections affected | `Understand terms`, `Fund house`, `Minimum investments`, `Exit load / Stamp duty`, fund-manager bios |
| Scheme the survivor is attributed to | **always `hdfc_large_cap`** (first in file order) |

Consequence, verified end to end *before* the fix: `Understand terms` existed in
all 5 schemes' `chunks.jsonl`, but only the `hdfc_large_cap` copy was stored. A
query filtered to `where={"scheme_id": "hdfc_elss"}` therefore **could not see
it**, even though the text was in the corpus — so the expense-ratio definition was
unreachable for four of the five schemes. A citation drawn from those chunks would
also have named the wrong fund.

`chunk_id` was never affected — it is correctly scoped as
`source_url|order|index|body`, and the comment there already shows awareness of
this exact bug class. Only `content_sha256` was unscoped. The fix keys
`upsert_chunks` on `chunk_id` instead, which is unique by construction (verified
131/131), and drops the fragile body-hash identity entirely rather than widening
it. `python -m app.ingest --force` rebuilds the corpus: **131 stored, 0 missing**,
fingerprint `449d9e77a656`.

### Retrieval works — verified

```bash
.venv/bin/python -m pytest tests/test_retrieval.py -q   # 49 passed
```

| | |
| --- | --- |
| Pipeline | alias filter → KNN → threshold → MMR → assemble |
| Similarity math | `1.0 − distance`, recomputed from Chroma's raw output in tests |
| Scheme filter resolves | **all 5** schemes, on the real alias file |
| Ambiguous multi-scheme query | `None` ⇒ global search, never a wrong filter |
| Word boundaries | `"elss"` does not fire inside `"else"`, `"well"`, `"itself"` |
| Nonsense query | `is_empty=True`, `max_similarity=0.0` |
| Unknown `scheme_id` in `where` | `is_empty=True`, **not** an exception |
| Empty corpus | raises `CorpusEmptyError` |
| MMR diversity | rank-1..k contiguous, no repeated section in the top 5 |
| `mmr_score` | populated when MMR is on, `None` when off |

**Threshold justification — the `0.25` floor, measured on the real corpus.**
Rank-1 similarity over all 15 golden factual queries versus best similarity over
5 deliberately off-corpus queries:

| | min | median | max |
| --- | --- | --- | --- |
| 15 golden factual queries | **0.294** | 0.730 | 0.784 |
| 5 off-corpus queries | 0.085 | 0.196 | **0.333** |

0.25 sits below the weakest real question with 0.044 of margin and rejects 4 of
the 5 off-corpus queries. The separation is **not clean** — the strongest
off-corpus query (`"recommend a laptop under fifty thousand rupees"`, 0.333)
scores *above* the weakest real one (`"How do I get my account statement?"`,
0.294). That single false positive is caught upstream by P6's `REFUSE_ADVICE`,
so it never reaches the generator; the layering, not the threshold, is what makes
this safe. Recorded plainly because a 0.044 margin is thin and a lower floor
would admit more noise.

**`riskometer` is absent from the entire corpus.** Both golden factual queries
asking for a riskometer level return five chunks at high similarity (0.744,
0.730) but **not one of those chunks contains the word** — the pages express risk
only as the string `Very High Risk`. Retrieval is working; the content is not
there. This is a P1 corpus-coverage gap that P10's AC8 will hit, since 2 of the
15 factual queries cannot be answered from the sources.

Two deviations from the guide, both deliberate:

> **MMR's relevance term references the query, not the last selected chunk.**
> The guide's pseudo-code is `lam*sim(c, selected[-1]) − (1−lam)*max_sim(c, selected)`.
> On the first greedy step only one chunk is selected, so both terms are the same
> number `s` and the score collapses to `lam*s − (1−lam)*s = (2*lam−1)*s`. At
> `lam=0.7` that is `0.4*s` — strictly increasing in `s`, so a "diversifying"
> score always picks the chunk *most* similar to the one already chosen. It is the
> exact opposite of its stated purpose and fails this phase's own exit criterion,
> "MMR returns diverse chunks (not 5 from one section)". Fixed to the standard
> form, where `sim(c, query)` is Chroma's score already on the candidate:
> `mmr(c) = lam*sim(c,query) − (1−lam)*max_sim(c, selected)`. The redundancy term
> is unchanged and `test_mmr_prefers_diversity_over_near_duplicates` pins it.

> **Ambiguity resolves to `None` on *any* two distinct schemes, not only on a tie.**
> The guide says "if two different schemes match → `None`", and separately "on tie
> across different schemes return `None`". Longest-alias-wins is implemented as the
> stricter reading of the first: no alias in this table is a substring of an alias
> belonging to a different scheme, so within one scheme the longest alias can never
> change the outcome, while across schemes longest-wins would pick a winner for
> "compare large cap and small cap" purely on alias length — precisely the wrong
> filter the design exists to prevent.

### Guardrails work — verified

```bash
.venv/bin/python -m pytest tests/test_guardrails.py -q   # 80 passed
```

| | |
| --- | --- |
| PII queries refused | **8 / 8** — PAN, Aadhaar, masked Aadhaar, account no., OTP, e-mail, 2× phone |
| Advice queries refused | **10 / 10** — intent, preference, suitability |
| Performance queries refused | **5 / 5** — forecast, comparison, ranking |
| Factual queries answered | **15 / 15** — zero false positives |
| Redaction | every PII literal absent from `redacted_query` |
| Ordering | PII short-circuits before advice; advice before performance |
| `classify()` never raises | empty, whitespace, punctuation, bare digits, 10 KB |
| Off-corpus | **not** decided here — needs retrieval evidence, so it is P8's job |

One deviation from the guide, deliberate and load-bearing:

> **Redaction order is an explicit list, not longest-pattern-first.**
> The guide says "longest first so an Aadhaar is not partially eaten by a
> shorter phone pattern", and that *intent* is honoured — but sorting by regex
> string length gets it wrong. `_AADHAAR_MASKED` (`"XXXX XXXX 4567"`) and
> `_AADHAAR` (`"2345 6789 0123"`) are near-identical in length and both also
> satisfy `_PHONE_IN`, so whichever won the sort would label the other type.
> `PII_PATTERNS` is therefore an ordered list, specific patterns first, and the
> ordering is asserted by tests (`test_redact_replaces_aadhaar_whole`,
> `test_pii_detect_suppresses_overlapping_broad_matches`).

### Generation + enforcement work — verified

```bash
.venv/bin/python -m pytest tests/test_enforcement.py tests/test_generation.py -q   # 70 passed
```

`enforce()` runs five steps in a fixed order, and the order is load-bearing:

```
strip openers → truncate to 3 sentences → verify citation → numeric audit → stamp freshness
```

| Property | Verified |
| --- | --- |
| Off-corpus URL | **`not_found`**, text does not contain it, `fail_closed_no_citation` |
| 2 URLs | 1 citation; duplicates of the *same* URL collapsed too |
| Paraphrased URL (`www.`, trailing `/`, `utm_`) | verifies — `normalize_url` on **both** sides |
| Unsupported number | removed, `dropped_unsupported_number:<num>` recorded |
| 6 sentences | 3, `truncated_to_3_sentences` |
| Mangled clause | whole sentence dropped, never "The ratio is." |
| Freshness line | present on **every** result, including `not_found` |
| `enforce()` never raises | empty, whitespace, 20 KB, URL-only, brackets, bare `₹` |
| No API key | extractive path, green, no client constructed |

Live run against the real corpus via Groq, 15 golden factual queries:

| | |
| --- | --- |
| Cited factual answers | **13 / 15** |
| Failed closed (correctly) | **2 / 15** — model paraphrased the URL, so no verified citation |
| Fabricated figures caught | `dropped_unsupported_number:₹1.25` on the ELSS redemption query |

The 2 failures are the enforcement layer working: the model shortened
`…/hdfc-equity-fund-direct-growth` and the link was rejected rather than shown.
Failing closed is the correct outcome, not a bug to tune away.

Four deviations from the guide, all deliberate:

> **The verified citation is never what truncation removes.** `architecture.md`
> §9.5 truncates *before* verifying the citation, so a model that writes three
> sentences and then a `Source:` line has its only citation cut and is failed
> closed — the "≤3 sentences" and "exactly one verified citation" invariants
> become jointly unsatisfiable. When the citation-bearing sentence falls outside
> the budget it is kept and an earlier content sentence gives up its slot
> instead. Pinned by `test_a_long_answer_keeps_its_citation_inside_the_budget`.

> **`AnswerKind`/`Citation`/`Trace`/`Answer` live in `app/generation.py`.** §P7's
> file table does not list `app/pipeline.py`, but §P8 requires those types there
> and `enforce()` already returns an `Answer`. P8 imports them rather than
> redefining them.

> **`llm_max_tokens` is 1024, not PRD §6.6's ≈220.** `openai/gpt-oss-120b` is a
> *reasoning* model: it emits ~850 characters of reasoning before the answer, so
> a 220-token budget is consumed entirely by reasoning and the completion returns
> `finish_reason="length"` with no usable content. Measured: 220 → 11 characters
> of answer, 512 → full answer. This does **not** weaken the 3-sentence limit,
> which is enforced in code, not by the token cap.

> **The provider is Groq, not OpenAI** — see the config deviation below.

### Pipeline + headless CLI work — verified

```bash
.venv/bin/python -m pytest tests/test_pipeline.py -q     # 53 passed
ENV=eval .venv/bin/python -m app.chat --query "What is the expense ratio of HDFC Large Cap Fund?" \
  --query "Should I invest in HDFC Small Cap Fund?" \
  --query "My PAN is ABCDE1234F" \
  --query "what is the weather in mumbai"
```

`answer_turn()` is the whole query path, and the **order** of the calls is the
product — each step is placed so that reversing it breaks a stated property:

```
classify() → refusals short-circuit → corpus built? → retrieve(REDACTED)
           → generate(REDACTED) → enforce()
```

| Exit criterion | Verified |
| --- | --- |
| All 6 probe queries return the expected `Answer.kind` | `factual`, `factual`, `refusal`, `refusal`, `refusal`, `not_found` |
| Advice/performance refusals skip retrieval entirely | `retrieve` patched to **raise** — never called; `retrieval_skipped` on the trace |
| PII literal absent from all output | `… \| grep -c "ABCDE1234F"` → **0** |
| `Last updated from sources:` on every answer except `ERROR` | asserted for all four static kinds and for both `not_found` paths |
| `answer_turn("")` does not raise | plus `""`, `"   "`, `"\n\t"`, 5 000 chars, `"?!?!"`, `"🙂"` |
| Works with `ENV=eval` and no API key | both verified; the CLI is the offline entry point |

Two details worth naming, because both are silent failures if you get them wrong:

> **The CLI never echoes the query.** `grep -c "ABCDE1234F"` over the output must
> print `0`, so a CLI that printed the question back would fail the check for the
> one answer whose whole purpose is not to repeat it.

> **Refusal citations come from `data/sources.csv`, not from a search.**
> `architecture.md` §8.2 requires the performance refusal to link the scheme the
> user asked about, and §8.3 requires the not-found answer to link the scheme page
> when the query names one — but both paths must **skip** retrieval, so there is no
> chunk to read a URL from. The manifest is read through a lenient module reader,
> not `ingest.read_sources`, because that one raises on a malformed row and a
> missing citation must never take a refusal down with it.

Also verified beyond the checklist: the LLM path runs end to end against real
Groq through this pipeline (`mode: llm`); a corpus that is not built returns
`kind=ERROR` with `python -m app.ingest` and a non-zero CLI exit; an unexpected
exception is logged with its traceback and answered with an apology that carries
the frame list but **not** the exception message, since a `ValueError` from a cast
quotes the string that caused it and §11 forbids carrying raw user input anywhere.

Three deviations from the guide, all deliberate:

> **The result types are re-exported, not redefined.** §P8's file table says
> `app/pipeline.py` must define `AnswerKind`/`Citation`/`Trace`/`Answer`, but P7
> already defined them in `app/generation.py` because `enforce()` returns an
> `Answer`. `pipeline.Answer is generation.Answer` is asserted by a test — two
> definitions would create two unrelated classes and an `isinstance` check that
> lies.

> **`static_answer` does not `normalize_ws` the copy.** P7 normalises generated
> text, which folds curly quotes and em dashes to ASCII. The refusal wording is
> `app/disclaimer`'s, reproduced verbatim from PRD §9.3/§9.4 (D5); normalising it
> would silently edit the one string the PRD pins down.

> **`Session` is a PII-safe audit buffer, not conversation context.** It caps at
> `max_chat_history_turns`, stores only `GuardDecision.redacted_query`, and is
> never fed back into the prompt — §P8 specifies no multi-turn prompting, and
> inventing it would let an earlier turn change the facts in a later answer.

> **One line was added to `pytest.ini`**, which is not in §P8's file table: a
> `markers = integration:` registration. §P8's own test spec requires
> `@pytest.mark.integration`, and an unregistered mark makes every run emit a
> `PytestUnknownMarkWarning`. Nothing else outside the three listed files changed.

---

## What this will do

**Scope:** 5 HDFC scheme pages, all from one AMC.

| Scheme | Category | Plan |
| --- | --- | --- |
| HDFC Large Cap Fund | Large Cap | Direct Growth |
| HDFC Flexi Cap Fund *(the brief calls this "HDFC Equity Fund" — the AMC renamed it)* | Flexi Cap | Direct Growth |
| HDFC ELSS Tax Saver Fund | ELSS (Tax) | Direct Growth |
| HDFC Small Cap Fund | Small Cap | Direct Growth |
| HDFC Balanced Advantage Fund | Balanced Advantage | Direct Growth |

**Answers** factual questions about expense ratio, exit load, minimum investment,
lock-in, riskometer, benchmark and statement download — in at most 3 sentences,
with exactly one link to a source page and a `Last updated from sources: <date>`
stamp.

**Refuses** three categories of question, in this order:

1. **PII** — PAN, Aadhaar, account numbers, OTPs, emails, phones. Refuses without
   echoing the value, and never stores or logs it.
2. **Advice** — "should I invest", "which is best", "is this right for me".
   Redirects to AMFI's investor education page.
3. **Performance** — returns, rankings, "which performed better". States that it
   makes no performance claims and points to the official factsheet.

Anything off-corpus gets a "not in my sources" answer rather than a guess.

---

## Requirements

- Python 3.10+ (developed on 3.12.6)
- Internet access on **first run only**, to download the ~90 MB embedding model
- Optional: a **Groq** API key. Without one the app uses a deterministic
  extractive fallback and still answers with citations.

### Which provider does what

| Stage | Provider | Endpoint | Notes |
| --- | --- | --- | --- |
| Retrieval — embedding | **HuggingFace**, local | none | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, L2-normalised. Downloads from the Hub once, then fully offline (PRD C4, NFR-3). **Embeddings are never sent to any API** |
| Vector store | ChromaDB, local | none | `chroma_db/`, cosine space |
| Generation — LLM | **Groq only** | `https://api.groq.com/openai/v1` | `GROQ_API_KEY`. There is no OpenAI key and no request to `api.openai.com`; the `openai` package is only the SDK for Groq's OpenAI-compatible endpoint |

Verified at runtime on 2026-10-02:

```
LLM base_url    : https://api.groq.com/openai/v1/     <- Groq, not OpenAI
Embedder repo   : sentence-transformers/all-MiniLM-L6-v2
Embedding dim   : 384 | L2 norm: 1.0                 <- local forward pass
settings fields containing 'openai': none
GET {llm_base_url}/models -> 200, 11 ids, llm_model is served: True
```

> **Provider deviation from PRD C8.** The PRD names OpenAI `gpt-4o-mini`; this
> build points the `openai` client at Groq's OpenAI-compatible endpoint
> (`https://api.groq.com/openai/v1`) instead. PRD §16 permits "an OpenAI-compatible
> LLM endpoint", so this stays inside the brief — only the C8 cell differs. **No
> new dependency:** `openai` was already required, and embedding is unaffected
> (still local `sentence-transformers/all-MiniLM-L6-v2`, per PRD C4/NFR-3).
>
> Swapping the `openai` SDK for Groq's native `groq` package would change nothing
> about which host is contacted, and would add a dependency outside
> `implementation.md` §0.3's allowed list — so it has not been done.
>
> `llm_model` must be an id Groq actually serves. Verified against
> `GET /openai/v1/models` on 2026-10-02; the general-purpose options were
> `openai/gpt-oss-120b` (default), `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`. The
> `meta-llama/llama-prompt-guard-*` entries are classifiers, not generators, and
> `llama-3.3-70b-versatile` is no longer served.

---

## Setup

```bash
git clone <this-repo>
cd MileStone4_RAG_Project

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env               # optional: add GROQ_API_KEY
```

`requirements.txt` holds 130 pins captured from a working install.

---

## Running the tests

```bash
.venv/bin/python -m pytest -q
```

Expected: **417 passed**. The suite is fully offline — no network, no ChromaDB.
Ingestion tests drive a synthetic HTML fixture and monkeypatch `fetch()`;
chunking tests use synthetic documents and assert the label/value invariant
directly. Tokenizer tests run against the real model when it is cached locally
and fall back to a character estimate when it is not. Pipeline tests force
`ENV=eval`, stub the store and the vector search, and mark the three tests that
need the real corpus as `integration` (they skip when `chroma_db/` is empty).

Quick sanity check of the config and copy:

```bash
.venv/bin/python -c "from app.config import get_settings; s=get_settings(); print(s.chunk_max_tokens, s.min_similarity, s.llm_enabled)"
# 240 0.25 False

.venv/bin/python -c "from app import disclaimer as d; print(len(d.EXAMPLE_QUESTIONS), repr(d.FRESHNESS_PREFIX))"
# 3 'Last updated from sources:'
```

---

## Running the app

**Ingestion and chunking — works now:**

```bash
.venv/bin/python -m app.ingest --fetch-only   # fetch + extract → documents.jsonl
.venv/bin/python -m app.ingest --force        # discard the cache and re-fetch
```

Chunking runs automatically as part of `app.ingest`; `app.chunking` has no
entry point of its own.

**Asking a question — works now, in a browser or from the terminal:**

### Streamlit UI

```bash
streamlit run app/ui.py
```

Then open <http://localhost:8501>. If that port is taken, add
`--server.port 8502`.

The page shows the disclaimer above the chat on every render, three one-click
example questions, and a `Sources (5)` tab carrying each scheme's `fetched_at`.
Every bot turn carries a clickable source link, the freshness date, a
`mode · latency · top_sim` badge, and two expanders — *▸ What the bot
retrieved* (rank, scheme, section, similarity, chunk text) and *▸ Guardrail*
(action and matched pattern). The sidebar shows the corpus fingerprint and
chunk count, and offers **Clear chat**.

**The first question is slow.** The encoder is a ~90 MB model loaded on the
first answered turn, not at page load, so the first click takes ~25 s and every
one after it takes ~2 s. That is the encoding step being lazy, not a hang.

`Clear chat` empties the transcript and reloading the page starts a fresh
conversation: nothing is written to disk. The user bubble is rendered from the
**redacted** query, so a pasted PAN appears as `my pan is [REDACTED:PAN]`.

### Verified — the 14-point UI walkthrough (AC11)

`implementation.md` §P9 defines AC11 as a 14-row manual walkthrough. It was
executed against `app/ui.py` with Streamlit's own `AppTest`, which runs the real
`ui.py` script rather than a mock. **14 / 14 rows pass.**

| # | Check | Result |
| --- | --- | --- |
| 1 | Disclaimer visible without scrolling | ✅ `st.warning` carries `disclaimer.DISCLAIMER` |
| 2 | Exactly 3 example questions | ✅ 3 buttons + `Clear chat` |
| 3 | Click an example → input prefilled | ✅ `chat_input` value set, not auto-submitted |
| 4 | Ask expense ratio → answer + link + date | ✅ answer body, clickable link, `Last updated from sources: 2026-10-02` |
| 5 | Click the link | ✅ HTTP 200 → `groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| 6 | Retrieved-chunks expander | ✅ 5 chunks, each with section and similarity |
| 7 | Guardrail expander | ✅ `action: answer`, `matched_pattern: none` |
| 8 | "Should I invest in…?" | ✅ `refuse_advice` + AMFI investor-education link |
| 9 | Paste a PAN | ✅ `refuse_pii`; stored user bubble is `my pan is [REDACTED:PAN]` |
| 10 | `Sources (5)` tab | ✅ 5 URLs, each with `fetched_at` |
| 11 | `Scope & known limits` tab | ✅ PRD §12.3 verbatim |
| 12 | `Clear chat` | ✅ messages `[]`, `Session` empty |
| 13 | Reload page | ✅ fresh session, 0 messages — no persistence |
| 14 | Sidebar fingerprint | ✅ 12-char hash, `449d9e77a656` |

Two notes on how this was checked, because both matter:

- **`curl` returning 200 proves nothing about the page.** Streamlit's HTTP shell
  is healthy even when the script itself throws. The first version of `ui.py`
  answered `curl` with 200 while rendering `ModuleNotFoundError` in the browser.
- `app/ui.py` prepends the repo root to `sys.path` before its first project
  import. Streamlit puts only the *script's own* directory on `sys.path`, so
  without that line `from app import …` fails under the documented command.

### Headless CLI

Also works, and is what the P10 harness will drive:

```bash
.venv/bin/python -m app.chat --query "What is the expense ratio of HDFC Large Cap Fund?"
```

Repeat `--query` for several questions, add `--json` for one JSON object per
answer (JSONL, for the P10 harness), or pipe a list of questions in:

```bash
printf 'what is the exit load on HDFC Large Cap?\nshould I buy HDFC Small Cap?\n' \
  | .venv/bin/python -m app.chat
```

With no `--query` and a terminal on stdin, `app.chat` becomes a REPL — one
question per line, ending on a blank line or `exit`. Every answer prints its
kind, generation mode, top similarity, text, citation URL and freshness date. The
disclaimer goes to stderr so stdout stays machine-readable.

**No-API-key mode:** leave `GROQ_API_KEY` blank in `.env`, or set `ENV=eval`.
Both routes use the deterministic extractive fallback, which needs no network
and returns the same answer for the same question every time.

---

## Known defects

Carried forward deliberately rather than hidden. None blocks a completed phase's
exit criteria; all of them will matter downstream.

| # | Where | Defect | Impact | Fix |
| --- | --- | --- | --- | --- |
| 1 | `app/store.py` — ✅ **fixed** | `upsert_chunks` keyed on `content_sha256`, a hash of the chunk **body alone**. Boilerplate that scheme pages share byte-for-byte collided, and 18 of 131 chunks were silently dropped by a `continue  # should not happen` — including the `Understand terms` expense-ratio definition for four of the five schemes | `Understand terms` and `Fund house` were unreachable for 4 and 3 schemes. Retrieval filters on `scheme_id`, so a question naming one of those schemes could never reach the definition that answered it | Keyed on `chunk_id` instead, which is unique by construction (verified 131/131). Corpus rebuilt with `--force`: **131 stored, 0 missing**, fingerprint `449d9e77a656` |
| 2 | corpus content | `riskometer` appears in **zero** chunks; the pages say only `Very High Risk` | 2 of the 15 golden factual queries are unanswerable from the sources. P10's AC8 targets ≥90%, so this alone puts it at 86.7% | Reword the two queries to what the pages actually say, or accept the miss honestly |
| 3 | `app/embedding.py` | `embed_texts([])` returns `np.empty((0, 256))` — it uses `embed_max_seq_length` (a *token* limit) as the row count instead of the 384 embedding dimension | Wrong-shaped empty array. Harmless today only because `retrieve` never calls it empty | Use `get_model().get_sentence_embedding_dimension()` |
| 4 | test suite | `tests/test_ingest.py` called `run_ingest`, which after P4 also writes chunks and upserts to Chroma, bypassing the redirected output paths | A plain `pytest` run **overwrote** `chunks.jsonl` and deleted all 113 real vectors, replacing them with 3 synthetic ones. This happened | Fixed — autouse fixture isolates both paths; `test_run_ingest_never_touches_the_real_corpus` is the regression guard |
| 5 | threshold | `min_similarity = 0.25` has only 0.044 of margin below the weakest real query, and one off-corpus query scores 0.333 | Thin. Caught upstream by `REFUSE_ADVICE`, but only because that query happens to be advice-shaped | Revisit with real P10 traffic |
| 6 | corpus content — ✅ **fixed** | Groww's `About <scheme>` paragraph states "The fund currently has an Asset Under Management(AUM) of ₹9,86,237 Cr" on **all five pages** — that is HDFC Mutual Fund's **AMC-wide total**, which the adjacent `Fund house` panel confirms at ₹9,86,236.84 Cr. The hero block gives the correct **scheme-level** figure (e.g. ₹39,933.36 Cr for large cap, 24.7× smaller) | The wrong figure *outranked* the right one for AUM questions, because the prose repeats the query's wording while the hero block only says "Fund size (AUM)". P7's numeric audit cannot catch it — the number **is** in the retrieved context, so it passes | `app/ingest.py:repair_document_aum` rewrites a prose AUM figure that equals the AMC total to the scheme's own `Fund size (AUM)`, using only values from the same page. Both labels must be present and the comparison tolerates the prose's rounding (1 Cr absolute / 0.1% relative). Corpus rebuilt — large cap now reports **₹39,933.36 Cr** instead of a number 24.7× too large |
| 7 | `app/generation.py` | The fail-closed `not_found` answer is built with `citations=[]` | §P8's gotcha ("not-found responses with zero citations — the brief expects a link") is satisfied on the `is_empty` path, which P8 owns and which always cites the scheme page or AMFI. The *fail-closed* path — a model that cited a URL which is not in the corpus — still ships with no link | Pass the retrieval result's top page into `_fail_closed` so the downgrade keeps a link |
| 8 | `app/retrieval.py` | `retrieval_top_k = 5` is tight for a 131-chunk corpus of long, heterogeneous sections | Two measured misses, both boundary cases rather than bugs: the ELSS **lock-in** chunk ranks 6th and is cut (it returns at `top_k=8`), and "who is the *second* fund manager of HDFC Flexi Cap" needs the chunk naming the *first* manager, which loses its slot to the chunk naming the second | Raise `retrieval_top_k` and re-measure the golden set for the cost of a noisier context. Left at 5 because the raise is unmeasured and §P9 forbids editing `retrieval.py` |

---

## Repository layout

**Current state (P0–P9 complete):**

```
MileStone4_RAG_Project/
├── PRD.md               what/why
├── architecture.md      how/why
├── implementation.md    phase-by-phase build guide
├── SPIKE_NOTES.md       P1 findings from the live feasibility probe
├── README.md            this file
├── requirements.txt     130 pinned dependencies
├── .env.example         documented settings
├── .gitignore
├── pytest.ini
├── app/
│   ├── __init__.py
│   ├── chat.py          headless CLI: --query / --json / stdin   ← new
│   ├── chunking.py      the chunker
│   ├── config.py        all tunables
│   ├── disclaimer.py    all user-facing copy
│   ├── embedding.py     384-dim encoder + model singleton
│   ├── errors.py        typed exceptions
│   ├── generation.py    prompt + Groq LLM + extractive fallback + enforcement
│   ├── guardrails.py    PII / advice / performance refusal
│   ├── ingest.py        fetch + extract
│   ├── pipeline.py      answer_turn() orchestration + session     ← new
│   ├── retrieval.py     alias filter → KNN → threshold → MMR
│   ├── scheme_aliases.py  alias → scheme_id
│   ├── store.py         persistent ChromaDB collection
│   ├── textutils.py     pure text helpers
│   ├── tokenization.py  word-piece counting
│   └── ui.py           Streamlit page: chat, citations, expanders      ← new
├── chroma_db/           persistent vector store (gitignored)
├── data/
│   ├── scheme_aliases.json   alias → scheme_id
│   ├── sources.csv      the 5 sources
│   ├── raw/             fetched HTML
│   └── processed/
│       ├── chunks.jsonl         131 retrieval units
│       ├── documents.jsonl      5 section trees
│       └── ingest_report.json   run health
├── eval/
│   ├── golden_advice.json       10 queries
│   ├── golden_factual.json      15 queries
│   ├── golden_performance.json   5 queries
│   └── golden_pii.json           8 queries
└── tests/
    ├── test_chunking.py
    ├── test_citation_fidelity.py                        ← new
    ├── test_config.py
    ├── test_disclaimer.py
    ├── test_enforcement.py                            ← new
    ├── test_generation.py                             ← new
    ├── test_guardrails.py
    ├── test_ingest.py
    ├── test_pipeline.py                               ← new
    ├── test_retrieval.py
    ├── test_store.py
    └── test_textutils.py
```

**Target state (P11 complete):** adds `eval/run_eval.py`,
`eval/golden_offcorpus.json` and `SAMPLE_QA.md`.

---

## Build status detail

| Phase | RAG stage | Status |
| --- | --- | --- |
| **P0** | scaffold | ✅ **complete** |
| **P1** | pre-Loading (feasibility spike) | ✅ **complete** — see [`SPIKE_NOTES.md`](SPIKE_NOTES.md) |
| **P2** | ① Loading | ✅ **complete** — 5/5 sources, 88 sections |
| **P3** | ② Chunking | ✅ **complete** — 131 chunks, 0 violations |
| **P4** | ③ Embedding + ④ store Vector Data | ✅ **complete** — 131 vectors in `chroma_db/` (the 18-chunk dedupe defect is fixed) |
| **P5** | ⑤ Retrieval | ✅ **complete** — 49 tests, all 7 exit criteria met |
| **P6** | cross-cutting (guardrails) | ✅ **complete** — 23/23 gate queries, 0 false positives |
| P7 | ⑥ Generation | ✅ **complete** — 70 tests, all 7 exit criteria met |
| **P8** | orchestration | ✅ **complete** — 53 tests, all 7 exit criteria met |
| **P9** | presentation (UI) | ✅ **complete** — `app/ui.py`, 14/14 of the AC11 walkthrough, 417 tests green |
| P10 | verification (evaluation) | ⬜ not started |
| P11 | deliverables (docs, video) | ⬜ not started |

---

## Design documents

| Document | Contents |
| --- | --- |
| [`PRD.md`](PRD.md) | Requirements, acceptance criteria AC1–AC12, deliverables D1–D7, milestone plan |
| [`architecture.md`](architecture.md) | Component design, data contracts, algorithms, ADRs, risks |
| [`implementation.md`](implementation.md) | Phase-by-phase build guide with copy-pasteable task specs |
| [`SPIKE_NOTES.md`](SPIKE_NOTES.md) | What the P1 probe measured on the real pages, and the decisions it forced |
| [`docs/Problemstatement.txt`](docs/Problemstatement.txt) | The original milestone brief |

---

## Disclaimer source of truth

The disclaimer and all refusal wording live in **`app/disclaimer.py`**. This
README, the Streamlit UI and `SAMPLE_QA.md` all reference that module rather than
duplicating the text by hand — except for the copy above, which a test asserts is
character-identical to the constant.
