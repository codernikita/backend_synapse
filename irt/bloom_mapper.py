"""
bloom_mapper.py — CHANGE 1: question difficulty (b) from Bloom's Taxonomy.

Responsibility
--------------
Standard IRT estimates item difficulty statistically from thousands of
responses. We don't have that data yet, so the supervisor's fix is to read
difficulty directly off each question's Bloom level using a configurable
mapping (see config.BLOOM_DIFFICULTY_MAP).

This module is pure: no DB, no I/O, no pandas dependency even. Given a
Bloom label, it returns a difficulty float. Given a bad/unknown label, it
fails loudly rather than silently defaulting — a mis-tagged question
should be caught at data-loading time, not silently treated as "Apply".

Why it exists as its own module (not inlined into feature_builder.py or
theta.py)
---------------------------------------------------------------------------
Both feature_builder.py (bucketing into easy/medium/hard) and theta.py
(needing a known b per question to plug into the logistic equation) need
the same Bloom -> difficulty logic. Centralizing it means the two never
drift out of sync, and the mapping can be retuned in config.py without
touching either consumer.

How it interacts with the rest of the architecture
----------------------------------------------------------------------
    questions.csv / questions table (bloom_level column)
        -> bloom_mapper.difficulty_for(bloom_level)   => b, used by theta.py
        -> bloom_mapper.bucket_for(bloom_level)        => "easy"/"medium"/"hard",
                                                           used by feature_builder.py
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import BLOOM_DIFFICULTY_BUCKETS, BLOOM_DIFFICULTY_MAP


class UnknownBloomLevelError(ValueError):
    """Raised when a question/concept's bloom_level isn't in the configured
    mapping. Deliberately not silently defaulted — a typo like 'anaylze'
    should surface immediately, not become an invisible 'Apply'."""


def _normalize(bloom_level: str) -> str:
    if bloom_level is None:
        raise UnknownBloomLevelError("bloom_level is None/missing.")
    key = bloom_level.strip().lower()
    if not key:
        raise UnknownBloomLevelError("bloom_level is an empty string.")
    return key


def difficulty_for(bloom_level: str) -> float:
    """Return the configured difficulty parameter (b) for a Bloom level.

    >>> difficulty_for("Understand")
    -1.0
    >>> difficulty_for("Create")
    2.5
    """
    key = _normalize(bloom_level)
    try:
        return BLOOM_DIFFICULTY_MAP[key]
    except KeyError:
        raise UnknownBloomLevelError(
            f"Unknown bloom_level {bloom_level!r}. Configured levels: "
            f"{sorted(BLOOM_DIFFICULTY_MAP)}. Add it to config.BLOOM_DIFFICULTY_MAP "
            "(and config.BLOOM_DIFFICULTY_BUCKETS) if this is a legitimate new level."
        ) from None


def bucket_for(bloom_level: str) -> str:
    """Return the easy/medium/hard accuracy bucket for a Bloom level.
    Used by feature_builder.py so difficulty and bucketing share one source
    of truth instead of drifting apart.

    >>> bucket_for("Remember")
    'easy'
    >>> bucket_for("Analyze")
    'hard'
    """
    key = _normalize(bloom_level)
    try:
        return BLOOM_DIFFICULTY_BUCKETS[key]
    except KeyError:
        raise UnknownBloomLevelError(
            f"Unknown bloom_level {bloom_level!r}. Configured levels: "
            f"{sorted(BLOOM_DIFFICULTY_BUCKETS)}."
        ) from None


@dataclass(frozen=True)
class BloomInfo:
    """Convenience bundle when a caller wants both b and the bucket for one
    question in a single lookup (feature_builder.py does this per response)."""

    bloom_level: str
    difficulty: float
    bucket: str


def describe(bloom_level: str) -> BloomInfo:
    """Return difficulty + bucket together. Raises UnknownBloomLevelError
    once (via difficulty_for) rather than duplicating the lookup+validation
    twice per call site."""
    key = _normalize(bloom_level)
    return BloomInfo(
        bloom_level=key,
        difficulty=difficulty_for(key),
        bucket=bucket_for(key),
    )
