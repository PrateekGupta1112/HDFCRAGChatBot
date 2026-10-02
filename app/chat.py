"""Headless CLI — the query path without Streamlit (architecture §4.3, §8).

This is the entry point the evaluation harness and the demo terminal both use.
It adds no logic of its own: it calls :func:`app.pipeline.answer_turn` and prints
what comes back. The one rule it enforces for itself is **never to print the
query**, because the PII refusal would then echo back the very PAN it exists to
suppress.

Output is a labelled record per query::

    ── [1] ─────────────────────────────
    kind: factual
    mode: extractive_fallback
    top_sim: 0.71
    answer: The expense ratio ... Source: https://...
    citation: https://groww.in/...
    last_updated: 2026-10-02

``--json`` prints one JSON object per query on its own line instead (JSONL), which
is what ``eval/run_eval.py`` consumes.

The disclaimer goes to **stderr**, in both modes, so stdout stays machine
readable while a human still sees it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Iterator, Sequence

from app import disclaimer
from app.pipeline import Answer, AnswerKind, answer_turn, new_session

logger = logging.getLogger(__name__)

#: Printed when the user leaves the interactive prompt with nothing typed.
_EXIT_WORDS = frozenset({"exit", "quit", ":q", "bye"})

#: Shown when a pipeline ERROR means a turn produced no answer. The last line
#: makes the exit code self-explanatory in a shell.
_PROMPT = "> "
_NOTHING_ENTERED = "(nothing entered — exiting)"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        A parser accepting one or more ``--query`` strings and an optional
        ``--json`` switch. With no ``--query`` the CLI reads stdin instead.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.chat",
        description=(
            "Ask the MF Facts Bot factual questions from the terminal. "
            "Works with no API key (deterministic extractive fallback)."
        ),
    )
    parser.add_argument(
        "-q",
        "--query",
        dest="queries",
        action="append",
        metavar="TEXT",
        help="a question to answer; repeat for several, or omit to read stdin",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="emit one JSON object per query (JSONL) instead of the text record",
    )
    return parser


def _iter_queries(queries: Sequence[str] | None) -> Iterator[str]:
    """Yield queries from ``--query`` flags, or from stdin when there are none.

    A piped stdin is read line by line so a file or a heredoc can be replayed; an
    interactive stdin becomes a REPL. Both stop on ``exit``/``quit``, so a script
    can end a long run early, and the REPL also stops on a blank line so a
    half-typed question is not silently answered. Blank lines in a pipe are
    skipped rather than ending the run, because files of questions are routinely
    separated by them.
    """
    if queries:
        yield from queries
        return

    if sys.stdin.isatty():
        while True:
            try:
                line = input(_PROMPT).strip()
            except EOFError:  # pragma: no cover - interactive only
                print(_NOTHING_ENTERED)
                return
            if not line or line.lower() in _EXIT_WORDS:
                print(_NOTHING_ENTERED)
                return
            yield line
    else:
        for line in sys.stdin:
            text = line.strip()
            if text.lower() in _EXIT_WORDS:
                return
            if text:
                yield text


def _answer_payload(answer: Answer) -> dict:
    """Flatten an :class:`Answer` into the JSON record.

    The ``trace`` is flattened rather than dumped: the retrieval result and the
    guard decision hold dataclasses, and an evaluation harness wants the
    decisions (``refused:refuse_advice``, ``retrieval_skipped``) rather than the
    raw objects behind them.
    """
    guard = answer.trace.guard
    retr = answer.trace.retrieval
    return {
        "kind": answer.kind.value,
        "generation_mode": answer.generation_mode,
        "top_similarity": round(retr.max_similarity, 4) if retr else None,
        "text": answer.text,
        "citations": [
            {"url": c.url, "section": c.section, "scheme_name": c.scheme_name}
            for c in answer.citations
        ],
        "last_updated": answer.last_updated,
        "guard_action": guard.action.value if guard else None,
        "guard_reason": guard.reason if guard else None,
        "postprocess_actions": list(answer.trace.postprocess_actions),
        "llm_latency_ms": answer.trace.llm_latency_ms,
        "corpus_fingerprint": answer.trace.corpus_fingerprint,
    }


def _print_record(answer: Answer, index: int) -> None:
    """Print one human-readable answer record.

    The query is deliberately absent. implementation.md §P8's PII check greps the
    whole output for the literal the user typed, and echoing the question would
    make that check fail for the one answer whose entire purpose is not to repeat
    it.
    """
    retr = answer.trace.retrieval
    top_sim = f"{retr.max_similarity:.2f}" if retr else "n/a"
    citation = answer.citations[0].url if answer.citations else "-"

    print(f"── [{index}] ─────────────────────────────")
    print(f"kind: {answer.kind.value}")
    print(f"mode: {answer.generation_mode}")
    print(f"top_sim: {top_sim}")
    print(f"answer: {answer.text}")
    print(f"citation: {citation}")
    print(f"last_updated: {answer.last_updated or 'n/a'}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list, excluding the program name. ``None`` uses
            ``sys.argv[1:]``.

    Returns:
        ``0`` if every turn produced an answer, ``1`` if any turn was an ERROR
        (usually a corpus that was never built).
    """
    args = build_parser().parse_args(argv)
    print(disclaimer.DISCLAIMER, file=sys.stderr)

    session = new_session()
    failures = 0
    for index, query in enumerate(_iter_queries(args.queries), start=1):
        answer = answer_turn(query, session)
        if args.as_json:
            print(json.dumps(_answer_payload(answer), ensure_ascii=False))
        else:
            _print_record(answer, index)
        if answer.kind is AnswerKind.ERROR:
            failures += 1

    if failures:
        print(f"{failures} turn(s) returned an error.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
