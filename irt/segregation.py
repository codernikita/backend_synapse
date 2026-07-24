"""
segregation.py — CHANGE 3: per-question discrimination parameter (a),
computed as strong-cluster accuracy minus weak-cluster accuracy, instead
of statistically estimated from thousands of responses.

Responsibility
--------------
Given a ClusterResult (from clustering.py — who is strong, who is weak)
and the raw responses, compute for every question:

    strong_accuracy - weak_accuracy = segregation_score

and classify it into a discrimination-quality label so poor items can be
flagged for review. This module does NOT touch difficulty (bloom_mapper.py)
or theta (theta.py, not built yet) — it only answers "does this question
separate strong students from weak ones?".

Why segregation approximates the IRT discrimination parameter
---------------------------------------------------------------------
Classical 2PL discrimination (a) describes how steeply a question's
probability-of-correct curve rises with ability — a high-a question is
answered correctly by almost everyone above a certain ability and almost
no one below it; a low-a question is answered right/wrong roughly
independent of ability. Segregation score measures the same underlying
idea directly rather than fitting a logistic curve to estimate it: if a
question is a good discriminator, students already identified as strong
(by the independent, multi-feature KMeans split) should answer it
correctly far more often than students already identified as weak. This
is exactly the logic behind Ebel's classical item-discrimination index
(upper-group accuracy minus lower-group accuracy), which has been used in
classroom psychometrics for decades specifically because it doesn't
require large-sample MLE — it only requires being able to split students
into "does better overall" vs "does worse overall", which is exactly what
clustering.py already gives us.

Why negative segregation indicates a problematic question
-------------------------------------------------------------------
A negative score means weak-cluster students answered the question
correctly MORE often than strong-cluster students. That's the opposite of
what a valid assessment item should do, and it usually signals one of a
few concrete problems: an ambiguous question stem, a mis-keyed correct
answer, a question answerable by test-taking tricks/guessing rather than
the targeted skill, or a question testing something uncorrelated with
(or actively penalized by) the ability the rest of the quiz measures.
It should never be treated as "a slightly weak but usable item" — it
should be flagged for review before it's trusted as a Bloom-difficulty (b)
anchor or fed into theta estimation.

How this module fits into the Hybrid IRT architecture
---------------------------------------------------------------------
    clustering.cluster_students()  ->  ClusterResult (strong/weak split)
    responses (ResponseRow, from feature_builder.py — reused, not duplicated)
        -> segregation.compute_segregation_scores(cluster_result, responses)
            -> SegregationBatchResult
                -> .results: list[SegregationResult], one per scoreable question
                -> .as_dict_by_question(): {question_id: SegregationResult}
                   consumed directly by theta.py (next module) to look up
                   each question's discrimination parameter (a) by id,
                   alongside bloom_mapper.difficulty_for() for (b).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .clustering import ClusterResult
from .config import DISCRIMINATION_QUALITY_THRESHOLDS, FLAGGED_DISCRIMINATOR_QUALITIES
from .feature_builder import ResponseRow


class InsufficientAttemptsError(ValueError):
    """Raised by the strict single-question API when a question can't be
    scored because one or both cluster groups have zero attempts. The
    batch API (compute_segregation_scores) does NOT raise this — it
    records the question as skipped (with a reason) so one bad question
    doesn't abort scoring for the rest of the item bank. Never silently
    continue: skipped questions are always visible in the returned
    SegregationBatchResult.skipped list, not dropped."""


def classify_discrimination(segregation_score: float) -> str:
    """Map a segregation score to a quality label using
    config.DISCRIMINATION_QUALITY_THRESHOLDS (Ebel-aligned, see config.py
    for the literature reasoning)."""
    for label, min_inclusive in DISCRIMINATION_QUALITY_THRESHOLDS:
        if segregation_score >= min_inclusive:
            return label
    # Unreachable: the last threshold is -inf, so something always matches.
    raise AssertionError("DISCRIMINATION_QUALITY_THRESHOLDS must end with a -inf floor")


@dataclass(frozen=True)
class SegregationResult:
    """Discrimination result for one question. This IS the "discrimination
    parameter (a)" referenced throughout the architecture — theta.py reads
    `segregation_score` directly as a in the 2PL logistic equation."""

    question_id: str
    strong_accuracy: float
    weak_accuracy: float
    n_strong_attempted: int
    n_weak_attempted: int
    segregation_score: float  # strong_accuracy - weak_accuracy; this IS "a"
    discriminator_quality: str  # excellent / good / moderate / poor / negative
    is_flagged: bool  # True for poor/negative — surfaced for review

    @property
    def discrimination(self) -> float:
        """Alias for segregation_score, named to match the IRT vocabulary
        (a) that theta.py's docstrings/equation will use. Both names stay
        available so callers can use whichever reads better in context."""
        return self.segregation_score


@dataclass(frozen=True)
class SkippedQuestion:
    """A question that could not be scored, with why. Reasons:
    'no_attempts', 'only_strong_attempted', 'only_weak_attempted'."""

    question_id: str
    reason: str


@dataclass
class SegregationBatchResult:
    """Output of compute_segregation_scores(). Designed so theta.py can
    consume it without any changes to earlier modules: as_dict_by_question()
    gives O(1) lookup of a question's (a) by id, exactly how theta.py will
    need it alongside bloom_mapper.difficulty_for() for (b)."""

    results: List[SegregationResult]
    skipped: List[SkippedQuestion] = field(default_factory=list)
    unknown_student_response_count: int = 0
    unknown_student_ids: List[str] = field(default_factory=list)

    def sorted_by_segregation_score(self, descending: bool = True) -> List[SegregationResult]:
        return sorted(self.results, key=lambda r: r.segregation_score, reverse=descending)

    def flagged(self) -> List[SegregationResult]:
        """Questions marked poor/negative discriminators — surfaced per
        Change 3 ('questions with poor segregation should be flagged')."""
        return [r for r in self.results if r.is_flagged]

    def as_dict_by_question(self) -> Dict[str, SegregationResult]:
        return {r.question_id: r for r in self.results}

    def warnings(self) -> List[str]:
        msgs = []
        if self.unknown_student_response_count:
            msgs.append(
                f"{self.unknown_student_response_count} response(s) referenced "
                f"student(s) not present in the ClusterResult and were excluded "
                f"from scoring: {self.unknown_student_ids}"
            )
        for s in self.skipped:
            msgs.append(f"Question {s.question_id} skipped: {s.reason}")
        return msgs


def _accuracy_for_group(
    responses: List[ResponseRow], member_ids: frozenset
) -> tuple[Optional[float], int]:
    """Returns (accuracy, n_attempted) for the subset of `responses`
    whose student_id is in `member_ids`. accuracy is None (not 0.0!) when
    n_attempted is 0 — 0.0 would be indistinguishable from 'everyone got
    it wrong', which is a real and different outcome from 'nobody
    attempted it'. Callers must handle None explicitly."""
    attempted = [r for r in responses if r.student_id in member_ids]
    n = len(attempted)
    if n == 0:
        return None, 0
    correct = sum(1 for r in attempted if r.is_correct)
    return correct / n, n


def compute_segregation_score(
    question_id: str,
    responses_for_question: Iterable[ResponseRow],
    cluster_result: ClusterResult,
) -> SegregationResult:
    """Strict single-question API: computes and returns a SegregationResult,
    or raises InsufficientAttemptsError if either cluster group has zero
    attempts for this question. Use compute_segregation_scores() for
    whole-item-bank scoring — it degrades gracefully per-question instead
    of raising and aborting the batch.
    """
    responses_for_question = [r for r in responses_for_question if r.question_id == question_id]
    strong_ids = frozenset(cluster_result.strong_student_ids())
    weak_ids = frozenset(cluster_result.weak_student_ids())

    strong_acc, n_strong = _accuracy_for_group(responses_for_question, strong_ids)
    weak_acc, n_weak = _accuracy_for_group(responses_for_question, weak_ids)

    if strong_acc is None and weak_acc is None:
        raise InsufficientAttemptsError(
            f"Question {question_id!r} has no attempts from either cluster."
        )
    if strong_acc is None:
        raise InsufficientAttemptsError(
            f"Question {question_id!r} has no attempts from the strong cluster "
            f"(only {n_weak} weak-cluster attempt(s)); cannot compute segregation."
        )
    if weak_acc is None:
        raise InsufficientAttemptsError(
            f"Question {question_id!r} has no attempts from the weak cluster "
            f"(only {n_strong} strong-cluster attempt(s)); cannot compute segregation."
        )

    score = strong_acc - weak_acc
    quality = classify_discrimination(score)
    return SegregationResult(
        question_id=question_id,
        strong_accuracy=strong_acc,
        weak_accuracy=weak_acc,
        n_strong_attempted=n_strong,
        n_weak_attempted=n_weak,
        segregation_score=score,
        discriminator_quality=quality,
        is_flagged=quality in FLAGGED_DISCRIMINATOR_QUALITIES,
    )


def compute_segregation_scores(
    cluster_result: ClusterResult,
    responses: Iterable[ResponseRow],
    question_ids: Optional[Iterable[str]] = None,
) -> SegregationBatchResult:
    """Batch scoring across the whole item bank.

    Parameters
    ----------
    cluster_result:
        Output of clustering.cluster_students() — defines strong/weak
        membership.
    responses:
        All students' responses (reuses feature_builder.ResponseRow —
        no duplicate data model, per the requirement).
    question_ids:
        Optional explicit list of question ids to score. If omitted,
        every question_id that appears anywhere in `responses` is scored.
        Pass this explicitly when you want questions with ZERO responses
        at all (e.g. a newly added item nobody has attempted yet) to show
        up as a recorded 'no_attempts' skip rather than being invisibly
        absent from the output — this is the 'missing responses' case.

    Never silently continues: every question ends up in either
    `.results` (scored) or `.skipped` (with a reason). Responses
    referencing a student_id absent from cluster_result are excluded from
    scoring and counted in `.unknown_student_response_count` /
    `.unknown_student_ids` rather than crashing the whole batch — mirrors
    feature_builder.build_feature_matrix()'s handling of unknown students.
    """
    responses = list(responses)
    strong_ids = frozenset(cluster_result.strong_student_ids())
    weak_ids = frozenset(cluster_result.weak_student_ids())
    known_ids = strong_ids | weak_ids

    unknown_responses = [r for r in responses if r.student_id not in known_ids]
    known_responses = [r for r in responses if r.student_id in known_ids]

    if question_ids is None:
        ordered_question_ids = list(dict.fromkeys(r.question_id for r in known_responses))
    else:
        ordered_question_ids = list(dict.fromkeys(question_ids))

    responses_by_question: Dict[str, List[ResponseRow]] = {qid: [] for qid in ordered_question_ids}
    for r in known_responses:
        if r.question_id in responses_by_question:
            responses_by_question[r.question_id].append(r)

    results: List[SegregationResult] = []
    skipped: List[SkippedQuestion] = []
    for qid in ordered_question_ids:
        qresponses = responses_by_question[qid]
        strong_acc, n_strong = _accuracy_for_group(qresponses, strong_ids)
        weak_acc, n_weak = _accuracy_for_group(qresponses, weak_ids)

        if strong_acc is None and weak_acc is None:
            skipped.append(SkippedQuestion(question_id=qid, reason="no_attempts"))
            continue
        if strong_acc is None:
            skipped.append(SkippedQuestion(question_id=qid, reason="only_weak_attempted"))
            continue
        if weak_acc is None:
            skipped.append(SkippedQuestion(question_id=qid, reason="only_strong_attempted"))
            continue

        score = strong_acc - weak_acc
        quality = classify_discrimination(score)
        results.append(
            SegregationResult(
                question_id=qid,
                strong_accuracy=strong_acc,
                weak_accuracy=weak_acc,
                n_strong_attempted=n_strong,
                n_weak_attempted=n_weak,
                segregation_score=score,
                discriminator_quality=quality,
                is_flagged=quality in FLAGGED_DISCRIMINATOR_QUALITIES,
            )
        )

    return SegregationBatchResult(
        results=results,
        skipped=skipped,
        unknown_student_response_count=len(unknown_responses),
        unknown_student_ids=sorted({r.student_id for r in unknown_responses}),
    )
