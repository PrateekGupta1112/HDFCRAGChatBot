# PRD — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

**Milestone 4 · NextLeap**
**Type:** Academic capstone / class demo — working RAG prototype
**Status:** Draft v1.0
**Owner:** Prateek
**Last updated:** 2026-10-01

---

## 1. Document Control

| Field | Value |
| --- | --- |
| Product name | MF Facts Bot (working title) |
| Milestone | MileStone4_RAG_Project |
| Deliverable type | Working prototype (Streamlit app) + docs + sample Q&A |
| Demo format | Live app demo, ≤3-min video as fallback |
| Primary stack | Python, ChromaDB, `sentence-transformers/all-MiniLM-L6-v2` |

---

## 2. Overview

### 2.1 Problem

Retail investors and support/content teams repeatedly ask the same factual questions about
mutual fund schemes: expense ratio, exit load, minimum SIP amount, ELSS lock-in period,
riskometer category, benchmark, and how to download a capital-gains statement. These answers
already exist — openly published — on scheme pages, factsheets, KIM/SID documents and fee
guides. But the answers are scattered across many pages, change over time, and are written in
regulatory language.

Today the "answering" is done by support agents from memory or by copying from PDFs, which is
slow and error-prone.

### 2.2 Solution

A small **facts-only RAG chatbot** scoped to **one AMC (HDFC Asset Management)** and **5 schemes**.
It answers factual scheme questions **only** from a fixed corpus of 5 public URLs, and:

- cites **exactly one clear source link** in every answer,
- **refuses** opinionated / portfolio questions politely with an educational link,
- never gives investment advice and never claims or compares performance,
- never accepts or stores PII,
- keeps answers to **≤ 3 sentences** and always shows `Last updated from sources: <date>`.

### 2.3 Who this helps

| User | Need |
| --- | --- |
| Retail users comparing schemes | Quick, sourced facts (fees, lock-in, riskometer) without reading PDFs |
| Support / content teams | Automate repetitive factual MF questions with a link they can paste into a ticket |

### 2.4 Why RAG (not a fine-tune, not a hardcoded FAQ)

Facts like expense ratio and exit load **change over time**. Hardcoding them guarantees stale
answers. RAG lets us re-scrape, re-chunk and re-embed without retraining anything, and it makes
the **source link mandatory** — the answer is grounded in text we can point at.

---

## 3. Goals & Non-Goals

### 3.1 Goals

| ID | Goal | Success signal |
| --- | --- | --- |
| G1 | Answer factual scheme questions grounded in the 5-URL corpus | ≥90% retrieval hit-rate on a golden question set |
| G2 | Every factual answer carries ≥1 working source link | 100% of factual answers |
| G3 | Refuse advice/portfolio questions reliably | 100% refusal on the opinionated test set |
| G4 | Zero PII accepted or persisted | 0 PII fields stored; 100% detection on PII test set |
| G5 | Answers are short and honest about freshness | ≤3 sentences; `Last updated` always shown |
| G6 | Demonstrate the full RAG pipeline end-to-end | Ingestion → chunking → embedding → Chroma → retrieval → grounded answer, all inspectable |

### 3.2 Non-Goals

- **Not** a financial advisor, recommendation engine, or allocator.
- **Not** a returns/performance tool — no computation, ranking or comparison of returns.
- **Not** a multi-AMC platform (v1 is HDFC-only by design, to keep the corpus auditable).
- **Not** a login-gated / account-specific portal (statements are covered as *how-to* guides only).
- **Not** a production system — no auth, no rate limiting, no HA, no monitoring.

---

## 4. Scope — Corpus

### 4.1 AMC & Schemes (1 AMC, 5 schemes)

| # | Scheme (as titled on source page) | Category | Source URL |
| --- | --- | --- | --- |
| 1 | HDFC Large Cap Fund – Direct Growth | Large Cap | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| 2 | HDFC Equity Fund (Flexi Cap) – Direct Growth | Flexi Cap | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| 3 | HDFC ELSS Tax Saver Fund – Direct Plan Growth | ELSS (Tax) | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| 4 | HDFC Small Cap Fund – Direct Growth | Small Cap | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| 5 | HDFC Balanced Advantage Fund – Direct Growth | Balanced Advantage (Hybrid) | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

The category spread is intentional: it forces the retriever to handle heterogeneous page
structures and lets the demo answer **category-specific** questions (e.g. lock-in only applies
to the ELSS scheme).

### 4.2 Open decision — source authority (flagged)

The brief says *"public pages from AMC/SEBI/AMFI"* but the five URLs it lists point to
**Groww**, a broker/distributor aggregator — not an official AMC/AMFI host.

| Option | Description | Trade-off |
| --- | --- | --- |
| **A (proposed)** | Use the 5 listed URLs as the primary corpus, exactly as instructed; note the aggregator caveat in README "Known Limits" and optionally add the official HDFC AMC page as a *secondary* corroborating source | Follows the brief literally; keeps the deliverable's "5 URLs" count exact |
| B | Replace with official HDFC AMC / AMFI scheme pages | Better authority, but diverges from the explicitly given URLs |

**Decision: Option A** for v1. Rationale: the brief explicitly supplies the URLs and the
deliverable asks for "the source list of the 5 URLs you used". Flagging it is the honest move.

### 4.3 In-scope question types

| Category | Example queries |
| --- | --- |
| Fees | "What is the expense ratio of HDFC Large Cap Fund?" · "Is there an exit load?" · "What is the TER?" |
| Investments | "What is the minimum SIP amount?" · "What is the minimum lump sum?" |
| Lock-in | "What is the lock-in period in HDFC ELSS Tax Saver Fund?" |
| Risk | "What is the riskometer level of HDFC Small Cap Fund?" |
| Benchmark | "What is the benchmark of HDFC Balanced Advantage Fund?" |
| Tax / statements | "How do I download a capital-gains statement?" · "What is the tax treatment?" |
| Scheme identity | "What is the fund manager?" · "What is the scheme launch date?" · "What is the NAV?" |

### 4.4 Out-of-scope question types → refusal or "not in sources"

- "Should I buy / sell / hold X?" (advice)
- "Which is better, X or Y?" (comparison + advice)
- "How much will I make if I invest ₹X?" (performance claim)
- "Is X safe for me?" (suitability)
- "Explain the stock market in general" (off-corpus)
- Anything requiring user identity / account data (PII)

---

## 5. System Architecture

### 5.1 High-level flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  OFFLINE / INGESTION (run once, re-runnable)                            │
│                                                                         │
│  5 URLs in sources.csv                                                  │
│        │                                                                │
│        ▼                                                                │
│  ┌───────────┐   HTTP GET + retry, user-agent, politeness delay        │
│  │  LOADING  │──────────────────────────────────────────────┐         │
│  └───────────┘                                               ▼         │
│        │                                          ┌─────────────────┐  │
│        │                                          │ BeautifulSoup + │  │
│        │                                          │ trafilatura:    │  │
│        │                                          │ main content +  │  │
│        │                                          │ heading tree    │  │
│        │                                          └────────┬────────┘  │
│        ▼                                                   ▼           │
│  raw HTML ──► data/raw/<scheme_id>__<fetched_at>.html  (archival)      │
│        │                                                                │
│        ▼                                                                │
│  ┌───────────┐  heading-aware, section-preserving, overlap-aware        │
│  │ CHUNKING  │  (§6.2)                                                 │
│  └─────┬─────┘                                                          │
│        ▼                                                                │
│  ┌───────────┐  all-MiniLM-L6-v2  (384-dim, cosine)                    │
│  │ EMBEDDING │  batch=32, normalized                                   │
│  └─────┬─────┘                                                          │
│        ▼                                                                │
│  ┌───────────────────────┐                                              │
│  │ ChromaDB (Persistent) │  collection: mf_faq_v1                      │
│  │   docs + embeddings   │  metadata: scheme, category, section,       │
│  │   + metadata          │            source_url, fetched_at, hash    │
│  └───────────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ONLINE / RETRIEVAL + GENERATION (per user turn)                        │
│                                                                         │
│  user query                                                            │
│      │                                                                  │
│      ▼                                                                  │
│  ┌────────────────────┐   refuse   ┌──────────────────────────────────┐ │
│  │ 1. INPUT GUARDRAIL │──────────▶│  Facts-only refusal + education   │ │
│  │    PII scan        │           │  link (AMFI/SEBI investor        │ │
│  │    advice/opinion  │           │  education page)                  │ │
│  │    off-corpus      │           └──────────────────────────────────┘ │
│  └─────────┬──────────┘                                                │
│            │ pass                                                      │
│            ▼                                                            │
│  ┌────────────────────┐   top-k=5, cosine, min_score threshold        │
│  │ 2. VECTOR RETRIEVE │──► chunks (with metadata)                     │
│  └─────────┬──────────┘                                                │
│            ▼                                                            │
│  ┌────────────────────┐   prompt = instruction + guardrail rules +      │
│  │ 3. GROUNDED GEN    │   retrieved chunks (numbered, with URL) +        │
│  │  temp=0, ≤3 sent.  │   query + output contract                       │
│  └─────────┬──────────┘                                                │
│            ▼                                                            │
│  ┌────────────────────┐   strip extra citations, enforce 1 link,        │
│  │ 4. POST-PROCESS    │   sentence cap, drop unsupported numbers,      │
│  │                    │   attach "Last updated from sources:"           │
│  └─────────┬──────────┘                                                │
│            ▼                                                            │
│  answer text + 1 citation link + last-updated date  ──► Streamlit UI    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Component table

| # | Component | Technology | Responsibility | Stage |
| --- | --- | --- | --- | --- |
| C1 | Source registry | `sources.csv` | The 5 URLs + scheme metadata, single source of truth | Load |
| C2 | Loader | `httpx` + `beautifulsoup4`/`lxml`/`trafilatura` | Fetch pages politely, strip nav/ads/boilerplate, keep heading tree | Load |
| C3 | Chunker | Custom heading-aware splitter | Split on section semantics, attach metadata, overlap | Chunk |
| C4 | Embedder | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim sentence embeddings, L2-normalized | Embed |
| C5 | Vector store | `chromadb` (PersistentClient) | Persist docs + embeddings + metadata, cosine KNN | Store |
| C6 | Retriever | Chroma query + threshold + optional MMR | top-k=5, drop low-relevance chunks, diversify | Retrieve |
| C7 | Prompt builder | Python | Instruction, rules, numbered context, output contract | Generate |
| C8 | LLM | OpenAI `gpt-4o-mini`, `temperature=0` | Compose ≤3-sentence grounded answer | Generate |
| C9 | Extractive fallback | Deterministic Python | No-API-key mode: return best-matching sentences verbatim + link | Generate |
| C10 | Guardrails | Regex + classifier + post-processor | PII, advice-refusal, off-corpus, length, citations | Guard |
| C11 | UI | `streamlit` | Chat interface, welcome line, 3 examples, disclaimer, sources page | UI |
| C12 | Config | `.env` + `pydantic-settings` | API keys, model names, thresholds | Infra |

---

## 6. Data Pipeline Specification

### 6.1 Stage 1 — Loading

| Aspect | Spec |
| --- | --- |
| Input | 5 URLs from `data/sources.csv` |
| Method | `httpx.Client` with a descriptive `User-Agent`, `timeout=30s`, 3 retries with exponential backoff |
| Politeness | 1.5s delay between requests; sequential (5 pages only) |
| Extraction | `trafilatura` for main-content extraction (drops nav/footer/cookie banners), fallback to `BeautifulSoup` heuristics if extraction < 500 chars |
| Structure preservation | Emit a document as an **ordered list of sections**: `{section_title, heading_level, text}` derived from `h1–h4` |
| Raw archive | Save raw HTML to `data/raw/` for reproducibility/debugging (never committed if large; gitignored) |
| Metadata | `scheme_id`, `scheme_name`, `category`, `source_url`, `fetched_at` (ISO date), `content_sha256` |
| Failure handling | A failed fetch is logged, the source is skipped, and ingestion continues; the UI shows corpus health. Never silently substitutes content. |

**Output:** `data/processed/documents.jsonl` — one record per scheme with its section tree.

### 6.2 Stage 2 — Chunking *(decided based on the data)*

**Decision rationale.** We inspected the shape of the five pages before choosing. They are
**not** flowing prose — they are a stack of short, self-contained, heading-delimited fact blocks
("Expense ratio", "Exit load", "Minimum SIP", "Riskometer", "Benchmark", "Minimum investment",
"How to download statement") interleaved with **label→value tables**. Naive fixed-size character
splitting routinely breaks a label away from its value, which is exactly the failure mode that
would produce a confidently wrong numeric answer. Therefore:

**Chosen strategy: heading-aware, section-preserving, table-aware recursive splitting.**

| Parameter | Value | Why |
| --- | --- | --- |
| Primary boundary | **HTML heading hierarchy (`h1→h4`)** | Each fact lives in its own labelled section; headings give natural, semantically meaningful boundaries for free |
| Secondary boundary | Recursive character split on `"\n\n"` → `"\n"` → `" "` | Handles sections that are too long (e.g. "About the fund" prose) |
| Target chunk size | **~200 tokens** | Bounded by MiniLM-L6-v2's `max_seq_length = 256` **word-pieces**; target + context header stays safely under it. (An earlier draft used ~400 tokens — see `architecture.md` §18.1 for why that causes silent truncation.) |
| Max chunk size | **240 tokens** | Hard cap strictly below the 256 word-piece limit, so the embedder never silently truncates a chunk's tail |
| Overlap | **~60 tokens**, applied *only* to prose sections, never to table/key-value sections | Prevents facts split across a boundary from losing context, without duplicating numeric rows |
| **Table / key-value handling** | `≤ 12 rows` or `≤ 1800 chars` → kept as **one chunk, never row-split** | Separating `Expense ratio` from `1.12%` produces a hallucinated number |
| Context header | Each chunk text is prefixed: `[Scheme: HDFC Large Cap Fund – Direct Growth | Category: Large Cap | Section: Expense ratio | Plan: Direct Growth]` then embedded | MiniLM embeds bare values poorly; the header teaches the retriever *which scheme* a chunk belongs to, which is essential for cross-scheme questions like "expense ratio of X vs Y" |
| Small-section handling | Sections < 40 tokens are merged with the following section (same scheme only) | Prevents tiny fact blocks from fragmenting and retrieving poorly |
| Deduplication | Chunks whose `content_sha256` is identical are dropped | Page boilerplate repeats across sections |
| Truncation guard | Assert `len(tokenizer.encode(embed_text)) <= 256` at insert time; log a warning if exceeded | Makes silent embedder truncation visible in the ingest report instead of invisible |
| Chunk metadata | `scheme_id`, `scheme_name`, `category`, `plan`, `section`, `source_url`, `fetched_at`, `chunk_index`, `is_table` | Enables source display, per-scheme filtering, and freshness reporting |

**Expected yield — CONFIRMED in P3: 131 chunks** across 5 schemes (estimate was ~120–200).
Breakdown: `hdfc_large_cap` 18 (10 table) · `hdfc_flexi_cap` 20 (12) · `hdfc_elss` 18 (9) ·
`hdfc_small_cap` 20 (12) · `hdfc_balanced_advantage` 55 (39). 0 violations; `embed_text`
word-pieces min 46 / mean 145–195 / max 239 against the 240 ceiling. `hdfc_balanced_advantage`
dominates because its holdings table lists 326 instruments.

Two corrections to the rows above, both measured in P3 and now implemented:

* **The 1800-char cap alone does not bound a chunk.** At this corpus's ~3.55
  characters per word-piece, 1800 characters is ~507 word-pieces — more than
  twice the 240 ceiling. Splitting is therefore bounded by *both* the character
  cap and the remaining token budget.
* **`≤ 12 rows` is not enforced.** Row-count capping was implemented and removed:
  it re-cut tables that already fit the token budget, without the label/value
  check, and split a 124-token block into 12/12/1 lines. What replaced it is a
  check that the chosen split point never lands between a label and its value.
  A "row" is not a unit the splitter may cut on, so `max_table_rows` is not
  applied; the row count remains a setting because architecture §9.1 names it.

**Output:** `data/processed/chunks.jsonl`.

### 6.3 Stage 3 — Embedding

| Aspect | Spec |
| --- | --- |
| Model | `sentence-transformers/all-MiniLM-L6-v2` (as mandated) |
| Dimensions | 384 |
| Similarity | Cosine (embeddings L2-normalized on insert) |
| Batching | `batch_size=32`, `normalize_embeddings=True` |
| Caching | If `chunk.content_sha256` already exists in Chroma with an unchanged hash, skip re-embedding (**incremental re-index**) |
| Note | Model downloads once (~90 MB) and is used fully offline after the first run |

### 6.4 Stage 4 — Vector store (ChromaDB)

| Aspect | Spec |
| --- | --- |
| Client | `chromadb.PersistentClient(path="./chroma_db")` |
| Collection | `mf_faq_v1` (versioned so a schema change rebuilds cleanly) |
| Documents | The context-prefixed chunk text (this is what gets retrieved and shown in the UI debug view) |
| Embeddings | 384-dim float32, cosine space |
| Metadata | As per §6.2 (all keys are scalar/string — Chroma metadata cannot hold nested objects) |
| Retrieval | `collection.query(query_texts=[q], n_results=5, where=<optional scheme filter>)` |
| Persistence | `chroma_db/` is regenerable from `data/`; gitignored, rebuilt via one command |
| Rebuild | `python -m app.ingest --force` rebuilds collection from scratch |

### 6.5 Stage 5 — Retrieval

| Aspect | Spec |
| --- | --- |
| Top-k | 5 |
| Score threshold | Drop chunks with cosine similarity < **0.25**; if **all** chunks fall below → answer "not found in sources" + link to the source page |
| Optional MMR | λ=0.7 re-rank to diversify (avoids 5 near-duplicate chunks from one section) |
| Scheme disambiguation | If the query names a scheme we recognise (alias table in `data/scheme_aliases.json`, e.g. "large cap", "elss", "tax saver", "small cap", "balanced advantage", "flexi cap", "hdfc equity"), apply a Chroma `where` filter to that `scheme_id` before ranking. **If the query names no scheme, do not filter** — let global retrieval decide |
| Context assembly | Chunks numbered `[1] … [5]`, each shown with its `Section` and `source_url` so the LLM can attribute correctly |
| Traceability | Top chunks + scores are kept per turn and exposed in the UI ("What the bot retrieved") for demo transparency |

### 6.6 Stage 6 — Generation

**System prompt contract** (hard requirements encoded in the prompt *and* verified in post-processing):

1. Answer **only** from the provided context. If the context doesn't contain it, say so.
2. **≤ 3 sentences.** No bullets, no tables, no preamble.
3. Include **exactly one** source link, formatted as the `source_url` of the chunk used.
4. Never recommend, rank, compare suitability, or predict returns. Refuse those.
5. Never state or compute returns/performance; if asked, link to the official factsheet page instead.
6. Never request or echo PAN, Aadhaar, account numbers, OTPs, emails, phone numbers.
7. Don't invent numbers. If a figure isn't in the context, omit it.
8. Add the line `Last updated from sources: <fetched_at>`.

**LLM:** `gpt-4o-mini`, `temperature=0`, `max_tokens≈220`.

**Extractive fallback (no API key):** return the top-1/2 chunk sentences most relevant to the
query, trimmed to 3 sentences, with its section title and link. Slower to read but **never
hallucinates** — a good honest fallback for a classroom demo where Wi-Fi or keys fail.

**Post-processing / enforcement layer:**
- Truncate to 3 sentences (sentence-aware split, not character cut).
- Reduce to exactly 1 citation link (keep the most relevant).
- Verify the link is in the retrieved set's `source_url` values; drop it if not (i.e. drop hallucinated URLs).
- Append/normalize `Last updated from sources: <date>`.
- Strip any stray "Sure!", "Certainly!", "As an AI" openers.
- Final hard gate: if no verified link survives → downgrade to the "can't answer from sources" response.

---

## 7. Functional Requirements

### FR-1 — Corpus ingestion
The system **must** load the 5 URLs in `data/sources.csv` and produce structured section text with
metadata (§6.1). Ingestion **must** be re-runnable and **must** not duplicate chunks on re-run.

### FR-2 — Chunking
The system **must** chunk using the heading-aware strategy in §6.2, keeping table/key-value blocks
intact and attaching full metadata + context header to each chunk.

### FR-3 — Embedding
The system **must** embed all chunks with `sentence-transformers/all-MiniLM-L6-v2` (384-dim) and
store them in ChromaDB with metadata (§6.3, §6.4).

### FR-4 — Factual answering
The system **must** answer in-scope factual queries (§4.3) grounded in retrieved chunks, in
**≤ 3 sentences**.

### FR-5 — Mandatory single citation
**Every** factual answer **must** contain exactly one source link, and that link **must** be one of
the 5 corpus URLs. Hallucinated links **must** be removed in post-processing.

### FR-6 — Freshness stamp
**Every** answer (factual, refusal, or not-found) **must** display `Last updated from sources: <date>`
derived from the chunk's `fetched_at`.

### FR-7 — Advice / opinion refusal
The system **must** detect advice, recommendation, comparison and suitability questions and respond
politely with a facts-only message plus a relevant **educational** link (AMFI / SEBI investor
education), **without** restating or partially answering the advice question.

### FR-8 — Performance-claim handling
Asked about returns, ranking, or "which performed better", the system **must not** compute or
compare; it **must** state it doesn't provide performance claims and **must** link to the official
factsheet/source page.

### FR-9 — PII protection
The system **must** detect and **must not store** PAN, Aadhaar, account numbers, OTPs, email
addresses, and phone numbers in queries, logs, ChromaDB, or session history. On detection it
**must** refuse with a neutral message and **must not** echo the value back in full.

### FR-10 — Out-of-corpus handling
When retrieval yields no chunk above the similarity threshold, the system **must** say the
information isn't in the sources, name what it *can* answer, and link the relevant scheme page.

### FR-11 — Tiny UI
The UI **must** show: a welcome line, **3 example questions**, the persistent note
**"Facts-only. No investment advice."**, a chat input, an answer area with the citation link and
freshness date, and a sources list. It **must** additionally offer a "show retrieved chunks"
expander (demo value) and an "About / limitations" expander.

### FR-12 — Transparency/debug view
The app **must** expose, per turn: matched scheme filter, retrieved chunk sections, and similarity
scores.

### FR-13 — Configuration
All thresholds, model names, top-k and collection name **must** be configurable via `.env` /
config module, not hardcoded in logic.

---

## 8. Non-Functional Requirements

| ID | Category | Requirement |
| --- | --- | --- |
| NFR-1 | Setup | One-command setup: `pip install -r requirements.txt` then `python -m app.ingest && streamlit run app/ui.py` |
| NFR-2 | Portability | Runs on macOS/Windows/Linux, Python 3.10+, CPU-only (no GPU required) |
| NFR-3 | Offline embedding | First run downloads the embedding model (~90 MB); afterwards embedding is fully local |
| NFR-4 | Reproducibility | Pinned dependency versions; ChromaDB rebuildable from `data/` in < 2 minutes |
| NFR-5 | No key required | App must start and answer in extractive fallback mode with no OpenAI key |
| NFR-6 | Latency | Retrieval < 200 ms; end-to-end answer < 6 s (LLM mode) |
| NFR-7 | Privacy by construction | No telemetry, no external logging; user input never persisted beyond the browser session |
| NFR-8 | Maintainability | Modular packages (`ingest`, `chunk`, `embed`, `store`, `retrieve`, `generate`, `guardrails`) with clear separation |
| NFR-9 | Testability | Unit tests for chunker, guardrails, PII regex, and a golden-set retrieval test |
| NFR-10 | Cost | Free/low-cost tier; total demo cost under a small monthly budget |

---

## 9. Guardrails & Safety Specification

### 9.1 Guard order (first match wins)

```
1. PII DETECTED            → PII refusal                (§9.2)
2. ADVICE / RECOMMENDATION → Facts-only refusal + edu link (§9.3)
3. PERFORMANCE / RETURNS   → No-claims message + factsheet link (§9.4)
4. OFF-CORPUS / UNKNOWN    → Not-in-sources message + relevant page link
5. IN-SCOPE FACTUAL        → Grounded answer + 1 verified citation
```

Guardrails run **both** pre-retrieval (input classification) and **post-generation** (output
verification), so a missed pre-check is still caught by the post-check.

### 9.2 PII patterns to detect

| Type | Detection |
| --- | --- |
| PAN | Regex `[A-Z]{5}[0-9]{4}[A-Z]` (mask before any log write) |
| Aadhaar | 12-digit sequences, incl. `XXXX XXXX 1234` masked form |
| Bank account | 9–18 digit sequences near account/acc no./a/c words |
| OTP | 4–8 digit codes near OTP/one-time-password words |
| Email | Standard RFC-ish regex |
| Phone | Indian 10-digit with optional `+91`, and international `+<country><7–15 digits>` |

Behaviour: refuse, do **not** echo the value, do **not** log it, and state which types are not
accepted. Test set of ≥8 PII-bearing queries must be **100%** caught.

### 9.3 Refusal response (canonical wording)

> "I only share published facts about these schemes — I can't give investment advice, recommendations, or help pick or size a portfolio. For general guidance, [AMFI's investor education page](https://www.amfiindia.com/investor-education-centre) explains how mutual funds work and how to evaluate a scheme."

Refusals **must not** include any scheme facts, partial answers, or "hints" toward a decision.

### 9.4 Performance-claim response

> "I don't compute, compare or forecast returns — that's a performance claim I'm not allowed to make. The scheme's official factsheet on the source page lists its past performance and benchmark for reference."

### 9.5 Educational links (allowed destinations for refusals)

- AMFI Investor Education Centre — https://www.amfiindia.com/investor-education-centre
- SEBI Investor Charter / SCORES — https://www.sebi.gov.in/ (investor-grievance & redressal sections)
- HDFC Asset Management (AMC) — https://www.hdfcamc.com/

All are public, official and non-transactional.

### 9.6 Disclaimer (persistent in UI, also in README)

> **Facts-only. No investment advice.** Answers are generated from public scheme pages and may be
> incomplete or out of date. They are not a recommendation to buy, sell or hold any security.
> Mutual fund investments are subject to market risks; read all scheme-related documents carefully.
> Verify every fact on the linked source page before acting.

---

## 10. UI Specification

### 10.1 Screen layout

```
┌────────────────────────────────────────────────────────────┐
│  MF Facts Bot            Facts-only. No investment advice. │  ← disclaimer, always visible
├────────────────────────────────────────────────────────────┤
│  Welcome: Ask factual questions about 5 HDFC mutual fund    │
│  schemes. Every answer includes a source link.              │
│                                                            │
│  Try one of these:                                         │
│   1) What is the expense ratio of HDFC Large Cap Fund?      │
│   2) What is the lock-in period in HDFC ELSS Tax Saver?    │
│   3) How do I download a capital-gains statement?           │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Ask a factual question…                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                        [ Ask ]                             │
├────────────────────────────────────────────────────────────┤
│  Answer:  <≤3 sentences>                                   │
│  Source: https://…                    (one link, clickable) │
│  Last updated from sources: 2026-09-28                      │
│  ▸ What the bot retrieved (chunks + scores)                │
├────────────────────────────────────────────────────────────┤
│  ▸ Sources (5)   ▸ Scope & known limits   ▸ Disclaimer      │
└────────────────────────────────────────────────────────────┘
```

### 10.2 UI requirements

- Welcome line + exactly **3 example questions** + persistent disclaimer note (FR-11).
- Example questions are clickable → prefill the input.
- Answers render plain sentences, a clickable source link, and the freshness line.
- "What the bot retrieved" expander shows chunk sections, scores, and matched scheme filter (FR-12).
- A "Sources (5)" tab lists all URLs from `sources.csv`.
- No chat history persistence to disk (NFR-7). "Clear chat" button resets session state.
- Mobile-responsive; Streamlit default theme is fine (no custom CSS needed for the demo).

---

## 11. Success Metrics & Acceptance Criteria

### 11.1 Golden test sets

| Set | Size | Purpose |
| --- | --- | --- |
| **G-Factual** | 15 queries (3 per category × 5 categories) | Measure retrieval hit-rate + answer quality |
| **G-Advice** | 10 opinionated queries | Measure refusal accuracy (must be 100%) |
| **G-PII** | 8 PII-laden queries | Measure PII detection (must be 100%) |
| **G-Perf** | 5 return/ranking queries | Measure no-claims compliance (must be 100%) |
| **G-OffCorpus** | 5 off-topic queries | Measure graceful "not in sources" |

### 11.2 Acceptance criteria (demo is a pass only if all are met)

| # | Criterion | Threshold |
| --- | --- | --- |
| AC1 | Factual answers containing ≥1 valid corpus source link | **100%** |
| AC2 | Answers with more than 1 citation link | 0 |
| AC3 | Answers longer than 3 sentences | **0** |
| AC4 | Advice/opinion queries correctly refused (G-Advice) | **100%** |
| AC5 | PII queries refused & nothing persisted (G-PII) | **100%** |
| AC6 | Return/performance queries answered with a no-claim + link (G-Perf) | **100%** |
| AC7 | Answers showing `Last updated from sources:` | **100%** |
| AC8 | Retrieval hit-rate — key fact present in top-5 chunks (G-Factual) | **≥ 90%** |
| AC9 | Off-corpus queries gracefully declined (G-OffCorpus) | **100%** |
| AC10 | Schema/chunk/store/retrieve/generate/guard modules all present and runnable | Yes |
| AC11 | App runs end-to-end from a clean clone following README only | Yes |
| AC12 | All 5 required deliverables present (see §13) | Yes |

### 11.3 Manual quality bar

Answers must be *readable to a first-time retail investor* — no unexplained regulatory jargon
without a short gloss (e.g. "riskometer (a 1–5 scale of risk levels)").

---

## 12. Risks, Assumptions & Known Limits

### 12.1 Risks

| ID | Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- | --- |
| R1 | Groww page structure/HTML changes → scraper breaks | High | High | Store raw HTML archive; make extraction heuristics defensive; section-title-based chunking is resilient; re-ingest is one command |
| R2 | Facts go stale between ingestion and demo | Medium | Medium | `Last updated from sources:` stamp; documented re-ingest step; never claim freshness beyond the stamp |
| R3 | LLM states a wrong number despite grounding | Medium | **High** | Table-preserving chunking; post-processing drops unsupported numeric claims not present in context; always show the source so the user verifies |
| R4 | Hallucinated citation link | Medium | High | Post-processing whitelists links against retrieved `source_url`s; no valid link ⇒ no answer |
| R5 | Retriever pulls chunks from the wrong scheme | Medium | Medium | Scheme alias table + Chroma `where` filter + context header on every chunk |
| R6 | MiniLM's 256-token limit truncates long chunks | Medium | Low | Target ~400 tokens with the context header trimmed first; keep max 500 |
| R7 | No OpenAI key / no network during demo | Medium | High | Extractive fallback mode (C9); pre-run a dry demo; record a backup video |
| R8 | Advice slips through the guard and the bot advises | Low | **High** | Two-stage guard (pre + post), 100% pass requirement on G-Advice, refuse-by-default on low confidence |
| R9 | PII accidentally written to logs | Low | **High** | Mask/redact before any write; no persistence of user input at all |
| R10 | Scope creep (add schemes, add AMCs, add comparison) | Medium | Medium | Non-goals in §3.2; change control = update this PRD first |

### 12.2 Assumptions

1. The 5 URLs are reachable and render server-side or via static HTML (verified in Phase 1).
2. `trafilatura` extracts usable main content from these pages.
3. An OpenAI-compatible LLM endpoint is available for the graded demo (fallback exists regardless).
4. Facts like expense ratio and exit load are present in the page text (to be verified in Phase 1; if absent, that question type is marked "not in sources").
5. Demo environment has internet for the first model download; ChromaDB + model are cached locally afterwards.

### 12.3 Known limits (goes in README)

- **Single AMC, 5 schemes, 5 pages.** Questions about other funds or other AMCs → "not in sources".
- **Snapshot, not a live feed.** Answers reflect the page as of `fetched_at` only.
- **Aggregator pages, not official AMC/AMFI pages** (see §4.2). Values are cross-checkable on the linked page; for regulated facts (e.g., TER) the official factsheet/SID is authoritative.
- **No PDF ingestion.** Factsheets/SIDs (PDF) are out of scope for v1, so some values may not be present in HTML.
- **English only**, and no Hindi/regional-language support.
- **MiniLM-L6-v2 is a small model**: good for topical retrieval, weak at very fine-grained numeric comparisons between schemes. Multi-scheme numeric comparison is deliberately refused.
- **Single-turn oriented**: last 3 turns are kept for context only; no long-term memory, no personalization.
- **No returns math**: CAGR, alpha, expense-ratio-vs-return breakeven — all out of scope by policy.
- **Not a SEBI-registered advisor.** Educational tool for a class project.

---

## 13. Deliverables

| # | Deliverable | Format | Location | Definition of Done |
| --- | --- | --- | --- | --- |
| D1 | **Working prototype** | Streamlit app | deployed link *or* `app/` + run instructions; ≤3-min demo video as fallback | Runs end-to-end; demo passes AC1–AC12 |
| D2 | **Source list** | `sources.csv` + rendered table in README | `data/sources.csv` | Exactly the 5 URLs, with scheme, category, and `fetched_at` |
| D3 | **README** | MD | `README.md` | Setup steps, architecture, scope (AMC + schemes), chunking rationale, known limits, guardrails, test instructions |
| D4 | **Sample Q&A** | MD | `SAMPLE_QA.md` | 8–10 queries spanning all categories **plus** advice-refusal, performance-claim, PII and off-corpus cases, each with the assistant's actual answer + link + last-updated date |
| D5 | **Disclaimer snippet** | MD/constant in code | `README.md` § + `app/ui.py` + `app/disclaimer.py` | Single source of truth, identical text in UI and docs |
| D6 | **PRD** (this doc) | MD | `PRD.md` | Reviewed |
| D7 | **Tests** | pytest | `tests/` | Chunker, guardrails, PII, golden retrieval set |

**Repository layout (target):**

```
MileStone4_RAG_Project/
├── PRD.md
├── README.md
├── SAMPLE_QA.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── disclaimer.py          # D5 single source of truth
│   ├── ingest.py              # Stage 1: Loading
│   ├── chunking.py            # Stage 2: Chunking
│   ├── embedding.py           # Stage 3: Embedding
│   ├── store.py               # Stage 4: ChromaDB
│   ├── retrieval.py           # Stage 5: Retrieval
│   ├── generation.py          # Stage 6: Generation (+ extractive fallback)
│   ├── guardrails.py          # PII / advice / performance classification
│   ├── pipeline.py            # end-to-end orchestration
│   └── ui.py                  # Streamlit
├── data/
│   ├── sources.csv            # D2
│   ├── scheme_aliases.json
│   ├── raw/                   # gitignored
│   └── processed/
│       ├── documents.jsonl
│       └── chunks.jsonl
├── chroma_db/                 # gitignored, regenerable
├── eval/
│   ├── golden_factual.json
│   ├── golden_advice.json
│   ├── golden_pii.json
│   ├── golden_performance.json
│   └── report.md
└── tests/
```

---

## 14. Milestones & Plan

Assumes a ~1-week build with parallel work where possible. Durations are indicative.

| Phase | Days | Tasks | Exit criteria |
| --- | --- | --- | --- |
| **P0 — Setup & feasibility** | Day 1 | Repo scaffold, `requirements.txt`, `.env.example`, verify all 5 URLs fetchable, confirm `trafilatura` extraction quality, confirm embedding model downloads | All 5 pages fetch and extract ≥500 chars of main content; architecture confirmed |
| **P1 — Data ingestion** | Day 2 | `sources.csv`, `ingest.py` loader, section tree extraction, raw HTML archive, `documents.jsonl` | 5 documents with clean section trees, no nav/footer noise |
| **P2 — Chunking** | Day 3 | `chunking.py` per §6.2, metadata + context header, dedupe, `chunks.jsonl`, chunk-count report | Chunks inspected manually; no fact split from its value; count reported |
| **P3 — Embedding + store** | Day 3 | `embedding.py`, `store.py`, Chroma `mf_faq_v1`, incremental re-index | 100% of chunks embedded and queryable; sanity retrieval returns sensible sections |
| **P4 — Retrieval + generation** | Days 4–5 | `retrieval.py` (top-k, threshold, MMR, alias filter), `generation.py` (prompt, LLM, extractive fallback), post-processing enforcement | Grounded answers with 1 verified link; latency < 6 s |
| **P5 — Guardrails** | Day 5 | PII regexes, advice classifier, performance handler, refusal copy, post-checks | G-Advice / G-PII / G-Perf all at 100% |
| **P6 — UI** | Day 6 | Streamlit: welcome, 3 examples, disclaimer, chat, sources tab, retrieved-chunks expander, limits tab | AC11 met from clean run |
| **P7 — Eval & docs** | Day 7 | Golden sets, `eval/report.md`, README, SAMPLE_QA, demo script, backup video | AC1–AC12 all pass; D1–D7 complete |
| **Buffer** | Day 8 | Fix eval failures, polish demo narrative, contingency | Demo-ready |

### 14.1 Demo script (≤3 min) — for the video fallback

1. **0:00–0:20** Welcome screen: disclaimer + the 3 example questions.
2. **0:20–0:50** Ask "expense ratio of HDFC Large Cap Fund?" → show answer, clickable source, last-updated date; open "what the bot retrieved".
3. **0:50–1:20** Ask "lock-in period in HDFC ELSS Tax Saver?" → show the ELSS-specific answer and link.
4. **1:20–1:50** Ask "Should I buy HDFC Small Cap Fund?" → show polite facts-only refusal + education link.
5. **1:50–2:10** Ask "what will I earn if I invest ₹1 lakh for 10 years?" → show no-claims response + factsheet link.
6. **2:10–2:30** Paste a PAN in the chat → show PII refusal and nothing stored.
7. **2:30–2:50** Show the Sources tab (5 URLs) and the "Scope & known limits" tab.
8. **2:50–3:00** Close on the architecture: Loading → Chunking → Embedding → Chroma → Retrieval → Grounded answer.

---

## 15. Appendix

### 15.1 Requirement → deliverable traceability

| Requirement | Satisfied by | Verified by |
| --- | --- | --- |
| FR-1, FR-2, FR-3 (pipeline stages 1–4) | `app/ingest.py`, `chunking.py`, `embedding.py`, `store.py` | `tests/test_chunking.py`, `eval` sanity query |
| FR-4, FR-5, FR-6 (grounded answer, 1 citation, freshness) | `app/generation.py` + post-processing | AC1, AC2, AC3, AC7 |
| FR-7 (advice refusal) | `app/guardrails.py` | AC4 |
| FR-8 (no performance claims) | `app/guardrails.py` | AC6 |
| FR-9 (PII) | `app/guardrails.py` | AC5 |
| FR-10 (out-of-corpus) | `app/retrieval.py` threshold | AC9 |
| FR-11, FR-12 (UI) | `app/ui.py` | AC10, manual demo |
| FR-13 (config) | `app/config.py` | README setup walkthrough |
| D1–D7 (deliverables) | repo | §13 DoD checklist |

### 15.2 Illustrative sample Q&A format (real outputs go in `SAMPLE_QA.md`)

Values below are **placeholders for format only** — the actual file must contain answers copied
verbatim from a real run against the ingested corpus.

| # | Query | Type | Expected behaviour | Citation source |
| --- | --- | --- | --- | --- |
| 1 | Expense ratio of HDFC Large Cap Fund? | Factual | ≤3 sentences with the value | Scheme page |
| 2 | Minimum SIP for HDFC Flexi Cap? | Factual | ≤3 sentences with the amount | Scheme page |
| 3 | Exit load on HDFC Equity Fund? | Factual | ≤3 sentences; "0%" or stated period | Scheme page |
| 4 | Lock-in period in HDFC ELSS Tax Saver? | Factual | ≤3 sentences; ELSS lock-in | Scheme page |
| 5 | Riskometer level of HDFC Small Cap Fund? | Factual | ≤3 sentences with the level | Scheme page |
| 6 | Benchmark of HDFC Balanced Advantage Fund? | Factual | ≤3 sentences with the index | Scheme page |
| 7 | How do I download a capital-gains statement? | Factual | Steps, ≤3 sentences | Statement guide on scheme page |
| 8 | What is the fund manager of HDFC Large Cap Fund? | Factual | ≤3 sentences | Scheme page |
| 9 | Which of these 5 funds performed best last year? | Performance | No-claims message + factsheet link | Factsheet/scheme page |
| 10 | Should I invest in HDFC Small Cap Fund? | Advice | Facts-only refusal + education link | AMFI investor education |
| 11 | My PAN is ABCDE1234F, can you check my returns? | PII | PII refusal, value not echoed | — |
| 12 | What's the weather in Mumbai? | Off-corpus | "Not in sources" + relevant page link | Scheme page |

### 15.3 Open questions for review

1. **Source authority (§4.2)** — confirm Option A (use the 5 given Groww URLs, note the caveat) or switch to official HDFC AMC pages.
2. **LLM access** — is an OpenAI key guaranteed for the demo, or should the extractive fallback be the *default* presented path?
3. **Hosting** — deploy to Streamlit Community Cloud, or demo locally with a recorded video as the submitted artifact?
4. **PDF ingestion** — out of scope for v1, or should factsheets be added in Phase 2 if time allows? (Would materially improve coverage of TER/performance statements.)
5. **Grading emphasis** — is the *pipeline architecture* (loading → chunking → embedding → store) the primary scoring criterion, with guardrails secondary? This affects where to spend buffer time.

---

*End of PRD. Companion documents: `README.md` (setup + limits), `SAMPLE_QA.md` (captured outputs), `data/sources.csv` (source list).*