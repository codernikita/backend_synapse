"""
mastery_initializer.py — CHANGE 5: turns a student's single overall theta
into a per-CONCEPT initial mastery probability, ready to seed the Student
Knowledge Graph before Bayesian Knowledge Tracing takes over.

Responsibility
--------------
This module does NOT update mastery over time — that is BKT's job,
already implemented in the quiz portal (`knowledge.service.ts`). It only
computes the very first mastery value for each concept, immediately after
the diagnostic quiz, by combining:

  1. The student's theta (from theta.py's ThetaResult — reused directly,
     not duplicated, per the requirement).
  2. Concept performance: correct/attempted per concept, derived from
     concept-tagged responses.
  3. Bloom-derived difficulty of the items that probed each concept.

Why theta.py's own probability_correct() is reused here (design rationale)
---------------------------------------------------------------------
The algorithm needs a way to fold "how hard was this concept's diagnostic
item" and "how able is this student" into a single number that behaves
correctly at the extremes (higher theta -> higher mastery; harder Bloom
level -> lower mastery, for the same observed accuracy). Re-deriving a new
ad hoc formula for that would duplicate logic that already exists and is
already validated in theta.py: the 2PL curve itself,
`P(correct) = 1 / (1 + exp(-a(theta - b)))`, is already exactly a
"theta + difficulty -> expected accuracy" function. This module calls
`theta.probability_correct()` directly, using
`config.MASTERY_REFERENCE_DISCRIMINATION` (a=1.0) as a fixed reference
slope — because per-item, segregation-derived discrimination is
deliberately NOT part of this module's declared input set (theta,
concept accuracy, Bloom difficulty only) — to get a "theta-implied
accuracy" for each concept. That already has every monotonicity property
the algorithm requires, for free, from math already proven correct.

The mathematical reasoning (Change 5 algorithm)
---------------------------------------------------------------------
For a concept c, let:
  - observed_accuracy(c)     = (# correct) / (# attempted) among c's responses
  - theta_implied_accuracy(c) = mean over c's responses of
        probability_correct(a=MASTERY_REFERENCE_DISCRIMINATION,
                             b=bloom_mapper.difficulty_for(bloom_level),
                             theta=student's theta)
  - n(c) = number of attempted responses for concept c

Initial mastery is a precision-weighted (Bayesian shrinkage) blend:

    weight_observed = n(c) / (n(c) + K)
    initial_mastery(c) = weight_observed * observed_accuracy(c)
                        + (1 - weight_observed) * theta_implied_accuracy(c)

where K = config.MASTERY_PRIOR_STRENGTH is the "effective sample size" of
the theta-implied prior. This is the same structure as a Beta-Bernoulli
posterior mean with an informative prior: a concept probed by only 1-2
diagnostic items is not trusted to speak for itself and is pulled toward
what the student's overall ability + that concept's difficulty would
predict; a concept probed by many items is dominated by what was actually
observed. This directly satisfies the stated requirement — higher theta,
higher concept accuracy, and easier Bloom levels all independently push
initial_mastery up; lower theta, poor accuracy, and harder Bloom levels
all independently push it down — because each of those three inputs
enters the formula through a term whose monotonicity is already
established (theta_implied_accuracy is monotonically increasing in theta
and decreasing in difficulty, by the 2PL curve's own well-known shape;
observed_accuracy is monotonic by construction).

The result is clamped to (SEED_PRIOR_MIN, SEED_PRIOR_MAX) — see config.py
— rather than allowed to reach exact 0 or 1, since Bayesian Knowledge
Tracing's update equations cannot recover from a hard 0/1 prior.

How this fits into the Hybrid IRT architecture
---------------------------------------------------------------------
    theta.estimate_theta()          -> ThetaResult
    (question_id -> concept_id, bloom_level) mapping, from the question
        dataset/repository (not built yet — repository.py)
        + per-question correctness for this student
            -> mastery_initializer.build_concept_attempts()  (helper)
                -> ConceptAttempt list
                    -> mastery_initializer.initialize_mastery(theta_result, concept_attempts)
                        -> MasteryInitializationResult
                            -> consumed by the quiz portal to seed
                               student_masteries (no schema change needed —
                               see Compatibility section below)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .bloom_mapper import UnknownBloomLevelError, difficulty_for
from .config import (
    MASTERY_PRIOR_STRENGTH,
    MASTERY_REFERENCE_DISCRIMINATION,
    SEED_PRIOR_MAX,
    SEED_PRIOR_MIN,
)
from .theta import ThetaResult, probability_correct


class MissingThetaError(ValueError):
    """Raised when initialize_mastery() is given no ThetaResult (None).
    Mastery cannot be initialized without a student ability estimate —
    this must be caught upstream (e.g. skip students whose diagnostic
    couldn't be scored), not silently defaulted to some neutral mastery."""


class EmptyConceptDataError(ValueError):
    """Raised when concept_attempts is empty. There is nothing to
    initialize mastery for — this is a caller error (e.g. the diagnostic
    quiz had no concept-tagged items), not something to silently no-op on."""


class DuplicateConceptAttemptError(ValueError):
    """Raised when the same question_id appears more than once across
    concept_attempts (whether under the same or different concept_id) —
    ambiguous (which attempt counts?), mirrors theta.py's
    DuplicateResponseError for the same reason: never silently continue
    with an arbitrary pick."""


class InvalidBloomLevelError(ValueError):
    """Raised when a ConceptAttempt's bloom_level isn't recognized by
    bloom_mapper (config.BLOOM_DIFFICULTY_MAP). Wraps
    bloom_mapper.UnknownBloomLevelError with mastery-initializer-specific
    context so callers can catch one exception type from this module."""


@dataclass(frozen=True)
class ConceptAttempt:
    """One (concept, question, correctness) record for a single student.
    Deliberately does NOT reuse feature_builder.ResponseRow (which has no
    concept_id field, and carries student_id which is redundant here since
    initialize_mastery() is called once per student) or theta.AnswerRecord
    (which has no concept_id or bloom_level). This IS the new, minimal
    data model this module needs — not a duplicate of an existing one."""

    concept_id: str
    question_id: str
    is_correct: bool
    bloom_level: str


@dataclass(frozen=True)
class ConceptMastery:
    """Initial mastery for one concept, with the intermediate values that
    produced it kept visible for debugging/research (per Change 5's
    'initialization summary' requirement) rather than only exposing the
    final blended number."""

    concept_id: str
    initial_mastery: float
    observed_accuracy: float
    theta_implied_accuracy: float
    n_attempted: int
    n_correct: int
    weight_observed: float  # how much observed_accuracy vs theta contributed


@dataclass(frozen=True)
class MasteryInitializationSummary:
    """Cohort-of-one-student roll-up, useful for a quick sanity read
    without iterating every concept."""

    n_concepts: int
    average_initial_mastery: float
    lowest_mastery_concept_id: str
    lowest_mastery_value: float
    highest_mastery_concept_id: str
    highest_mastery_value: float


@dataclass
class MasteryInitializationResult:
    """Output of initialize_mastery(). student_id and theta are carried
    alongside the per-concept breakdown so this single object is
    everything a caller needs to persist student_masteries rows without
    re-threading theta_result separately."""

    student_id: str
    theta: float
    theta_converged: bool
    concept_masteries: Dict[str, ConceptMastery]
    summary: MasteryInitializationSummary

    def mastery_for(self, concept_id: str) -> float:
        return self.concept_masteries[concept_id].initial_mastery


def _theta_implied_accuracy(theta: float, bloom_levels: Sequence[str]) -> float:
    """Mean 2PL P(correct) at `theta`, over each item's Bloom-derived
    difficulty, using theta.py's exact probability_correct() (reused, not
    duplicated) with a fixed reference discrimination — see config.py's
    MASTERY_REFERENCE_DISCRIMINATION for why per-item segregation-based a
    is intentionally not used here."""
    probs = [
        probability_correct(MASTERY_REFERENCE_DISCRIMINATION, difficulty_for(level), theta)
        for level in bloom_levels
    ]
    return sum(probs) / len(probs)


def _clamp(value: float) -> float:
    return max(SEED_PRIOR_MIN, min(SEED_PRIOR_MAX, value))


def initialize_mastery(
    student_id: str,
    theta_result: Optional[ThetaResult],
    concept_attempts: Sequence[ConceptAttempt],
) -> MasteryInitializationResult:
    """Compute initial per-concept mastery for one student.

    Raises
    ------
    MissingThetaError            if theta_result is None.
    EmptyConceptDataError        if concept_attempts is empty.
    DuplicateConceptAttemptError if a question_id appears more than once.
    InvalidBloomLevelError       if any ConceptAttempt's bloom_level is unrecognized.
    """
    if theta_result is None:
        raise MissingThetaError(
            f"Cannot initialize mastery for student {student_id!r} without a ThetaResult."
        )
    if not concept_attempts:
        raise EmptyConceptDataError(
            f"Cannot initialize mastery for student {student_id!r}: concept_attempts is empty."
        )

    seen_question_ids: set[str] = set()
    for attempt in concept_attempts:
        if attempt.question_id in seen_question_ids:
            raise DuplicateConceptAttemptError(
                f"question_id {attempt.question_id!r} appears more than once in "
                f"concept_attempts for student {student_id!r}."
            )
        seen_question_ids.add(attempt.question_id)
        try:
            difficulty_for(attempt.bloom_level)
        except UnknownBloomLevelError as exc:
            raise InvalidBloomLevelError(
                f"question_id {attempt.question_id!r} (concept {attempt.concept_id!r}) "
                f"has an unrecognized bloom_level {attempt.bloom_level!r}: {exc}"
            ) from exc

    attempts_by_concept: Dict[str, List[ConceptAttempt]] = {}
    for attempt in concept_attempts:
        attempts_by_concept.setdefault(attempt.concept_id, []).append(attempt)

    theta = theta_result.theta
    concept_masteries: Dict[str, ConceptMastery] = {}
    for concept_id, attempts in attempts_by_concept.items():
        n_attempted = len(attempts)
        n_correct = sum(1 for a in attempts if a.is_correct)
        observed_accuracy = n_correct / n_attempted

        bloom_levels = [a.bloom_level for a in attempts]
        theta_implied = _theta_implied_accuracy(theta, bloom_levels)

        weight_observed = n_attempted / (n_attempted + MASTERY_PRIOR_STRENGTH)
        blended = weight_observed * observed_accuracy + (1 - weight_observed) * theta_implied
        initial_mastery = _clamp(blended)

        concept_masteries[concept_id] = ConceptMastery(
            concept_id=concept_id,
            initial_mastery=initial_mastery,
            observed_accuracy=observed_accuracy,
            theta_implied_accuracy=theta_implied,
            n_attempted=n_attempted,
            n_correct=n_correct,
            weight_observed=weight_observed,
        )

    values = [(cid, cm.initial_mastery) for cid, cm in concept_masteries.items()]
    lowest_id, lowest_val = min(values, key=lambda t: t[1])
    highest_id, highest_val = max(values, key=lambda t: t[1])
    summary = MasteryInitializationSummary(
        n_concepts=len(concept_masteries),
        average_initial_mastery=sum(v for _, v in values) / len(values),
        lowest_mastery_concept_id=lowest_id,
        lowest_mastery_value=lowest_val,
        highest_mastery_concept_id=highest_id,
        highest_mastery_value=highest_val,
    )

    return MasteryInitializationResult(
        student_id=student_id,
        theta=theta,
        theta_converged=theta_result.converged,
        concept_masteries=concept_masteries,
        summary=summary,
    )
