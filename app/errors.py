"""Typed exceptions for the pipeline (architecture §12).

Every failure mode the system can recover from has its own class so callers can
catch precisely instead of swallowing broad `Exception`.
"""

from __future__ import annotations


class MFFactsError(Exception):
    """Base class for every error raised by this application."""


class FetchError(MFFactsError):
    """A source page could not be fetched after exhausting retries."""


class ExtractionError(MFFactsError):
    """A fetched page yielded no usable main content."""


class CorpusEmptyError(MFFactsError):
    """The vector store has no chunks; the ingestion pipeline must run first."""


class ModelUnavailableError(MFFactsError):
    """The embedding model could not be loaded (usually: not downloaded yet)."""


class LLMError(MFFactsError):
    """The generation call failed in a way not handled by the fallback path."""
