"""
service.py — orchestrates the complete Hybrid IRT pipeline end-to-end:

    Feature Builder -> Clustering -> Segregation -> Question Parameters
    -> Theta -> Mastery Initializer

Responsibility
--------------
This module contains NO mathematical or statistical logic of its own —
every number anywhere in its output was computed by one of the modules
under irt/ (feature_builder.py, clustering.py, segregation.py,
item_parameters.py, theta.py, mastery_initializer.py). service.py's only
job is to:

  1. Pull data from an IRTRepository (repository.py — the only module
     allowed to know where that data actually comes from).
  2. Call the six pipeline stages in the documented order, wiring each
     stage's output into the next stage's input.
  3. Assemble the results into plain dataclasses a caller can consume directly.
  4. Decide what to do when one student's data can't be scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .clustering import (
    ClusteringFailedError,
    ClusterResult,
    EmptyFeatureMatrixError,
    InsufficientStudentsError,
    cluster_students,
)
from .feature_builder import (
    FeatureMatrix,
    build_feature_matrix,
    normalize_feature_matrix,
)
from .item_parameters import (
    QuestionIRTParameters,
    SkippedQuestionParameters,
    build_question_parameters,
)
from .mastery_initializer import (
    DuplicateConceptAttemptError,
    EmptyConceptDataError,
    InvalidBloomLevelError,
    MasteryInitializationResult,
    MissingThetaError,
    initialize_mastery,
)
from .repository import IRTRepository, RecordNotFoundError
from .segregation import SegregationBatchResult, compute_segregation_scores
from .theta import (
    DuplicateResponseError,
    EmptyResponsesError,
    MissingParameterError,
    ThetaResult,
    estimate_theta,
)


# ── Exceptions ───────────────────────────────────────────────────────────


class PipelineError(Exception):
    """Base class for every exception raised by this module."""


class ItemBankBuildError(PipelineError):
    """Raised when build_item_bank() cannot produce a usable item bank."""


class StudentScoringError(PipelineError):
    """Raised by score_student() when one student's data can't be scored."""


# ── Cohort-level result ──────────────────────────────────────────────────


@dataclass
class ItemBankResult:
    """Output of build_item_bank(): cohort-level parameters."""

    feature_matrix: FeatureMatrix
    cluster_result: ClusterResult
    segregation_batch: SegregationBatchResult
    parameters: List[QuestionIRTParameters]
    skipped_parameters: List[SkippedQuestionParameters]

    def warnings(self) -> List[str]:
        msgs = list(self.feature_matrix.warnings())
        msgs.extend(self.segregation_batch.warnings())
        for s in self.skipped_parameters:
            msgs.append(f"Question {s.question_id} has no IRT parameters: {s.reason}")
        return msgs


# ── Student-level result ─────────────────────────────────────────────────


@dataclass(frozen=True)
class StudentPipelineResult:
    """Output of score_student(): one student's full pipeline result."""

    student_id: str
    cluster_label: str
    theta_result: ThetaResult
    mastery_result: MasteryInitializationResult


# ── Batch (cohort) result ────────────────────────────────────────────────


@dataclass(frozen=True)
class SkippedStudent:
    """A student who could not be scored, with reason."""

    student_id: str
    reason: str


@dataclass
class CohortPipelineResult:
    """Output of run_pipeline(): complete results for a cohort."""

    item_bank: ItemBankResult
    student_results: Dict[str, StudentPipelineResult] = field(default_factory=dict)
    skipped_students: List[SkippedStudent] = field(default_factory=list)

    def result_for(self, student_id: str) -> StudentPipelineResult:
        return self.student_results[student_id]

    def scored_student_ids(self) -> List[str]:
        return list(self.student_results.keys())

    def warnings(self) -> List[str]:
        msgs = list(self.item_bank.warnings())
        for s in self.skipped_students:
            msgs.append(f"Student {s.student_id} skipped: {s.reason}")
        return msgs


# ── Orchestration ─────────────────────────────────────────────────────────


def build_item_bank(repo: IRTRepository) -> ItemBankResult:
    """Run cohort-level pipeline stages:
        Feature Builder -> Clustering -> Segregation -> Question Parameters
    """
    profiles = repo.get_student_profiles()
    responses = repo.get_responses()

    raw = build_feature_matrix(profiles, responses)
    normalized = normalize_feature_matrix(raw)

    try:
        cluster_result = cluster_students(normalized, raw)
    except (EmptyFeatureMatrixError, InsufficientStudentsError, ClusteringFailedError) as exc:
        raise ItemBankBuildError(
            f"Could not build the item bank: clustering failed ({exc})"
        ) from exc

    segregation_batch = compute_segregation_scores(cluster_result, responses)

    bloom_levels = repo.get_question_bloom_levels()
    parameters, skipped_parameters = build_question_parameters(bloom_levels, segregation_batch)

    return ItemBankResult(
        feature_matrix=raw,
        cluster_result=cluster_result,
        segregation_batch=segregation_batch,
        parameters=parameters,
        skipped_parameters=skipped_parameters,
    )


def score_student(
    repo: IRTRepository,
    student_id: str,
    item_bank: ItemBankResult,
) -> StudentPipelineResult:
    """Run student-level stages for one student:
        Theta -> Mastery Initializer
    """
    try:
        answers = repo.get_answer_records(student_id)
    except RecordNotFoundError as exc:
        raise StudentScoringError(
            f"Cannot score student {student_id!r}: {exc}"
        ) from exc

    try:
        theta_result = estimate_theta(answers, item_bank.parameters)
    except (EmptyResponsesError, DuplicateResponseError, MissingParameterError) as exc:
        raise StudentScoringError(
            f"Cannot estimate theta for student {student_id!r}: {exc}"
        ) from exc

    try:
        concept_attempts = repo.get_concept_attempts(student_id)
    except RecordNotFoundError as exc:
        raise StudentScoringError(
            f"Cannot score student {student_id!r}: {exc}"
        ) from exc

    try:
        mastery_result = initialize_mastery(student_id, theta_result, concept_attempts)
    except (
        MissingThetaError,
        EmptyConceptDataError,
        DuplicateConceptAttemptError,
        InvalidBloomLevelError,
    ) as exc:
        raise StudentScoringError(
            f"Cannot initialize mastery for student {student_id!r}: {exc}"
        ) from exc

    try:
        cluster_label = item_bank.cluster_result.label_for(student_id)
    except ValueError as exc:
        raise StudentScoringError(
            f"Student {student_id!r} was not part of the cohort used to "
            f"build this item bank's cluster split: {exc}"
        ) from exc

    return StudentPipelineResult(
        student_id=student_id,
        cluster_label=cluster_label,
        theta_result=theta_result,
        mastery_result=mastery_result,
    )


def run_pipeline(
    repo: IRTRepository,
    student_ids: Optional[Iterable[str]] = None,
) -> CohortPipelineResult:
    """Run the complete Hybrid IRT pipeline for a cohort of students."""
    item_bank = build_item_bank(repo)

    ids = list(student_ids) if student_ids is not None else repo.get_all_student_ids()

    student_results: Dict[str, StudentPipelineResult] = {}
    skipped_students: List[SkippedStudent] = []
    for sid in ids:
        try:
            student_results[sid] = score_student(repo, sid, item_bank)
        except StudentScoringError as exc:
            skipped_students.append(SkippedStudent(student_id=sid, reason=str(exc)))

    return CohortPipelineResult(
        item_bank=item_bank,
        student_results=student_results,
        skipped_students=skipped_students,
    )


__all__ = [
    "PipelineError",
    "ItemBankBuildError",
    "StudentScoringError",
    "ItemBankResult",
    "StudentPipelineResult",
    "SkippedStudent",
    "CohortPipelineResult",
    "build_item_bank",
    "score_student",
    "run_pipeline",
]