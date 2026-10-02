"""Scheme alias resolution — mapping what a user *types* to a `scheme_id`.

A user writes "tax saver" or "hdfc equity" or nothing at all. The corpus is keyed
by `scheme_id` (`hdfc_elss`, `hdfc_flexi_cap`, …). This module is the only place
that translation happens, and it is deliberately conservative: **when it is not
sure, it returns ``None``**, which means "search the whole corpus" rather than
"filter to a scheme".

That default is the whole design. A wrong filter is invisible to the user — the
bot answers confidently from the wrong scheme's page, with a real-looking
citation, and nothing marks it as wrong. A missing filter is merely less precise.
So every ambiguity resolves toward searching everything.

Matching is on **word boundaries**. Substring matching would let "elss" fire
inside "else", "well" or "itself", silently narrowing every unrelated query to
the ELSS scheme.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

#: ``data/scheme_aliases.json`` sits next to the package, not inside it, so it is
#: resolved from ``__file__`` rather than the process working directory — the UI
#: and the eval harness are both launched from the repo root, but a test runner
#: or a Streamlit child process need not be.
ALIASES_PATH = Path(__file__).resolve().parent.parent / "data" / "scheme_aliases.json"


@lru_cache(maxsize=1)
def load_aliases() -> dict[str, list[str]]:
    """Load the alias table as ``{scheme_id: [alias, ...]}``.

    Returns:
        The mapping from ``scheme_id`` to its list of lowercase aliases.

    Raises:
        FileNotFoundError: If the alias file is missing — a packaging error, not
            a user error, so it is allowed to surface.
    """
    data = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    return {str(k): [str(a).lower() for a in v] for k, v in data.items()}


@lru_cache(maxsize=512)
def _compiled(alias: str) -> re.Pattern[str]:
    """Compile one alias into a word-boundary pattern (memoised per alias)."""
    return re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")


def resolve_scheme_id(query: str) -> str | None:
    """Resolve a free-text query to a single ``scheme_id``, or ``None``.

    ``None`` is returned in three distinct cases, all of which mean the same
    thing to the caller — search everything:

    1. no alias appears in the query ("what is the benchmark"),
    2. aliases of **two or more different** schemes appear ("compare large cap
       and small cap"),
    3. the query is empty or not a string.

    Args:
        query: The user's raw question.

    Returns:
        The matched ``scheme_id``, or ``None``.

    Note:
        The guide describes this as "longest-alias-wins, and on tie across
        different schemes return ``None``". Within this table that tie-break can
        never change the outcome: no alias is a substring of an alias belonging
        to a different scheme, so a longer match and a shorter match naming the
        same scheme always resolve alike. Across schemes, longest-wins would pick
        a winner for "compare large cap and small cap" purely on alias length —
        which is exactly the *wrong filter* the design is trying to avoid. So the
        rule applied is the stricter one: **any** two distinct schemes ⇒ ``None``.
    """
    if not isinstance(query, str) or not query.strip():
        return None

    lowered = query.lower()
    matched_schemes: set[str] = set()
    for scheme_id, aliases in load_aliases().items():
        if any(_compiled(alias).search(lowered) for alias in aliases):
            matched_schemes.add(scheme_id)

    if len(matched_schemes) == 1:
        return matched_schemes.pop()
    if len(matched_schemes) > 1:
        # Never log the query itself (architecture §11).
        logger.debug("scheme alias matched %d schemes; falling back to global search",
                     len(matched_schemes))
    return None