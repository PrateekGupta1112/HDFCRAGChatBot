"""User-facing copy — the single source of truth for all wording (PRD D5).

Every string a user can see lives here so the UI, the README and the disclaimer
snippet cannot drift apart. Wording for the disclaimer, the advice refusal and
the performance refusal is reproduced verbatim from PRD §9.3, §9.4 and §9.6.

Nothing in this module contains real scheme values (fees, NAVs, returns).
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Persistent disclaimer (PRD §9.6). Also reproduced in the README.
# --------------------------------------------------------------------------

# Plain text, deliberately — no markdown emphasis. This same string is rendered
# in Streamlit, copied into README.md and printed by the headless CLI, where
# literal asterisks would leak into terminal output.
DISCLAIMER = (
    "Facts-only. No investment advice. Answers are generated from public scheme "
    "pages and may be incomplete or out of date. They are not a recommendation to buy, "
    "sell or hold any security. Mutual fund investments are subject to market risks; read "
    "all scheme-related documents carefully. Verify every fact on the linked source page "
    "before acting."
)

DISCLAIMER_SHORT = "Facts-only. No investment advice."

# --------------------------------------------------------------------------
# UI chrome (PRD §10.1)
# --------------------------------------------------------------------------

WELCOME_LINE = (
    "Welcome: Ask factual questions about 5 HDFC mutual fund schemes. "
    "Every answer includes a source link."
    "Try one of these:"
)

EXAMPLE_QUESTIONS: list[str] = [
    "What is the expense ratio of HDFC Large Cap Fund?",
    "What is the lock-in period in HDFC ELSS Tax Saver?",
    "How do I download a capital-gains statement?",
]

# --------------------------------------------------------------------------
# Refusals and fallbacks
# --------------------------------------------------------------------------

# PRD §9.3 renders the AMFI link as markdown. Here the prose is kept and the URL
# is omitted, because P8 attaches AMFI_EDUCATION_URL as a proper citation next
# to this text — inlining it would render the link twice.
REFUSAL_ADVICE = (
    "I only share published facts about these schemes — I can't give investment advice, "
    "recommendations, or help pick or size a portfolio. For general guidance, AMFI's "
    "investor education page explains how mutual funds work and how to evaluate a scheme."
)

REFUSAL_PII = (
    "I don't accept or store personal identifiers, so I can't act on that request. "
    "Please don't share PAN, Aadhaar, bank account numbers, OTPs, email addresses or "
    "phone numbers here — none of them are needed to answer a factual question about "
    "these schemes, and I won't repeat back anything you've sent."
)

REFUSAL_PERFORMANCE = (
    "I don't compute, compare or forecast returns — that's a performance claim I'm not "
    "allowed to make. The scheme's official factsheet on the source page lists its past "
    "performance and benchmark for reference."
)

NOT_FOUND_TEXT = (
    "I couldn't find that in the five scheme pages I have. I only cover HDFC Large Cap, "
    "HDFC Equity (Flexi Cap), HDFC ELSS Tax Saver, HDFC Small Cap and HDFC Balanced "
    "Advantage — try rephrasing as a factual question about one of those, or check the "
    "linked scheme page directly."
)

# --------------------------------------------------------------------------
# Educational links (PRD §9.5). Public, official, non-transactional.
# --------------------------------------------------------------------------

AMFI_EDUCATION_URL = "https://www.amfiindia.com/investor-education-centre"
SEBI_URL = "https://www.sebi.gov.in/"
HDFC_AMC_URL = "https://www.hdfcamc.com/"

# --------------------------------------------------------------------------
# Freshness stamp (FR-6)
# --------------------------------------------------------------------------

FRESHNESS_PREFIX = "Last updated from sources:"
