"""
feature_builder.py — CHANGE 2 (part 1): builds the per-student feature
vector that clustering.py will feed into KMeans(k=2).

Responsibility
--------------
For every student, produce:

    [previous_class_percentage, iq_score, total_correct,
     easy_accuracy, medium_accuracy, hard_accuracy]

in that exact order (irt.config.FEATURE_VECTOR_FIELDS), normalized, ready
for clustering. Nothing here decides who is "strong" or "weak" — that's
clustering.py's job. This module only turns raw rows into numbers.

Why it exists as its own module
--------------------------------
Change 2 replaces statistical discrimination estimation with
KMeans-on-features. The feature vector is the single most decision-heavy
part of that change (three real ambiguities were resolved with the
supervisor before writing this: class9_marks is already a 0-100
percentage; iq_score is optional and imputed with the cohort mean when
missing; easy/medium/hard buckets come from Bloom level, the same source
as the b parameter, not a separate raw difficulty column). Isolating all
of that here means clustering.py and every future consumer just call
`build_feature_matrix(...)` and get back clean, already-decided numbers.

How it interacts with the rest of the architecture
----------------------------------------------------------------------
    repository.fetch_student_profiles()  -\
    repository.fetch_responses()          }-> feature_builder.build_feature_matrix()
    repository.fetch_questions()         -/         |
                                                      v
                                          clustering.assign_clusters()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .bloom_mapper import bucket_for
from .config import ACCURACY_BUCKETS, FEATURE_VECTOR_FIELDS


# ── Input shapes (plain, DB-agnostic — repository.py maps rows into these) ──


@dataclass(frozen=True)
class StudentProfileRow:
    """One student's profile fields relevant to feature-building.
    iq_score is Optional by design (Change 2 decision: psychometric test
    is a separate workstream; missing values are imputed, not required)."""

    student_id: str
    previous_class_percentage: Optional[float]  # StudentProfile.class9_marks, already 0-100
    iq_score: Optional[float] = None


@dataclass(frozen=True)
class ResponseRow:
    """One (student, question) response. bloom_level comes from the
    question, joined in by the repository layer — feature_builder never
    touches a question table/CSV directly, only this flattened row."""

    student_id: str
    question_id: str
    is_correct: bool
    bloom_level: str


@dataclass
class ImputationReport:
    """Tracks which students had a value imputed, and with what, so the
    caller can log/audit it instead of it happening silently. Required by
    the Change 2 decision: iq_score missing -> impute with cohort mean AND
    log a warning."""

    field_name: str
    fill_value: float
    imputed_student_ids: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.imputed_student_ids)

    def as_warning(self) -> Optional[str]:
        if not self.imputed_student_ids:
            return None
        return (
            f"{len(self.imputed_student_ids)} student(s) missing '{self.field_name}'; "
            f"imputed with cohort mean ({self.fill_value:.3f}): "
            f"{self.imputed_student_ids}"
        )


@dataclass
class FeatureMatrix:
    """Output of build_feature_matrix(). `matrix` rows are aligned 1:1 with
    `student_ids`, columns aligned with config.FEATURE_VECTOR_FIELDS."""

    student_ids: list[str]
    matrix: np.ndarray  # shape (n_students, len(FEATURE_VECTOR_FIELDS))
    field_names: tuple[str, ...] = FEATURE_VECTOR_FIELDS
    imputations: list[ImputationReport] = field(default_factory=list)

    def warnings(self) -> list[str]:
        return [w for r in self.imputations if (w := r.as_warning())]

    def as_dict_rows(self) -> list[dict[str, float]]:
        """Convenience for tests/CLI printing: list of {field: value} per
        student, useful before anything is normalized/clustered."""
        return [
            {"student_id": sid, **dict(zip(self.field_names, row))}
            for sid, row in zip(self.student_ids, self.matrix)
        ]


def _accuracy_by_bucket(
    responses: Sequence[ResponseRow],
) -> tuple[dict[str, float], int]:
    """For one student's responses, return {bucket: accuracy} for every
    bucket in ACCURACY_BUCKETS (0.0 if the student saw no questions in that
    bucket — a student with no 'hard' questions isn't good or bad at hard
    ones, they're simply unmeasured; 0.0 is the documented, safe default
    for feeding a distance-based clustering algorithm) and total_correct."""
    counts = {b: [0, 0] for b in ACCURACY_BUCKETS}  # bucket -> [correct, seen]
    total_correct = 0
    for r in responses:
        bucket = bucket_for(r.bloom_level)
        counts[bucket][1] += 1
        if r.is_correct:
            counts[bucket][0] += 1
            total_correct += 1

    accuracy = {
        b: (correct / seen if seen > 0 else 0.0) for b, (correct, seen) in counts.items()
    }
    return accuracy, total_correct


def _impute_iq_scores(
    profiles: Sequence[StudentProfileRow],
) -> tuple[dict[str, float], ImputationReport]:
    known = [p.iq_score for p in profiles if p.iq_score is not None]
    cohort_mean = float(np.mean(known)) if known else 100.0  # neutral IQ default if NO data at all
    report = ImputationReport(field_name="iq_score", fill_value=cohort_mean)

    resolved: dict[str, float] = {}
    for p in profiles:
        if p.iq_score is None:
            resolved[p.student_id] = cohort_mean
            report.imputed_student_ids.append(p.student_id)
        else:
            resolved[p.student_id] = float(p.iq_score)
    return resolved, report


def _impute_previous_class_percentage(
    profiles: Sequence[StudentProfileRow],
) -> tuple[dict[str, float], ImputationReport]:
    """class9_marks can also be NULL for a given student (new transfer,
    missing record, etc.) even though it isn't Optional-by-design like
    iq_score. Same cohort-mean-imputation strategy, kept as a separate
    report so callers can tell the two apart."""
    known = [p.previous_class_percentage for p in profiles if p.previous_class_percentage is not None]
    cohort_mean = float(np.mean(known)) if known else 50.0
    report = ImputationReport(field_name="previous_class_percentage", fill_value=cohort_mean)

    resolved: dict[str, float] = {}
    for p in profiles:
        if p.previous_class_percentage is None:
            resolved[p.student_id] = cohort_mean
            report.imputed_student_ids.append(p.student_id)
        else:
            resolved[p.student_id] = float(p.previous_class_percentage)
    return resolved, report


def build_feature_matrix(
    profiles: Iterable[StudentProfileRow],
    responses: Iterable[ResponseRow],
) -> FeatureMatrix:
    """Build the Change-2 feature matrix for every student in `profiles`.

    - previous_class_percentage: from StudentProfileRow, treated as already
      0-100 (decision), cohort-mean-imputed if NULL.
    - iq_score: optional (decision), cohort-mean-imputed if missing.
    - total_correct: count of correct responses.
    - easy/medium/hard_accuracy: fraction correct within each Bloom-derived
      bucket (decision: bucketed from bloom_level, not a raw difficulty
      column), 0.0 if the student had no responses in that bucket.

    Students with zero responses still get a row (all accuracy fields 0.0,
    total_correct 0) — clustering should see everyone, not silently drop
    students who haven't taken the diagnostic yet as an error case; that's
    for the caller to filter if desired.
    """
    profiles = list(profiles)
    if not profiles:
        return FeatureMatrix(student_ids=[], matrix=np.zeros((0, len(FEATURE_VECTOR_FIELDS))))

    responses_by_student: dict[str, list[ResponseRow]] = {p.student_id: [] for p in profiles}
    for r in responses:
        if r.student_id not in responses_by_student:
            # Response for a student with no profile row — skip rather than
            # silently fabricate a profile; repository.py should keep these
            # in sync, but feature_builder stays defensive.
            continue
        responses_by_student[r.student_id].append(r)

    iq_by_student, iq_report = _impute_iq_scores(profiles)
    pct_by_student, pct_report = _impute_previous_class_percentage(profiles)

    student_ids: list[str] = []
    rows: list[list[float]] = []
    for p in profiles:
        accuracy, total_correct = _accuracy_by_bucket(responses_by_student[p.student_id])
        row_values = {
            "previous_class_percentage": pct_by_student[p.student_id],
            "iq_score": iq_by_student[p.student_id],
            "total_correct": float(total_correct),
            "easy_accuracy": accuracy["easy"],
            "medium_accuracy": accuracy["medium"],
            "hard_accuracy": accuracy["hard"],
        }
        student_ids.append(p.student_id)
        rows.append([row_values[f] for f in FEATURE_VECTOR_FIELDS])

    matrix = np.array(rows, dtype=float)
    imputations = [r for r in (iq_report, pct_report) if r]
    return FeatureMatrix(student_ids=student_ids, matrix=matrix, imputations=imputations)


def normalize_feature_matrix(fm: FeatureMatrix) -> FeatureMatrix:
    """Z-score normalize each column (mean 0, std 1) so KMeans in
    clustering.py doesn't let total_correct or iq_score dominate purely
    because of scale. Columns with zero variance (e.g. every student
    scored identically) are left as all-zeros rather than dividing by
    zero.

    Kept as a separate function (not baked into build_feature_matrix) so
    callers/tests can inspect raw feature values before normalization.
    """
    if fm.matrix.shape[0] == 0:
        return fm
    mean = fm.matrix.mean(axis=0)
    std = fm.matrix.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    normalized = (fm.matrix - mean) / std_safe
    normalized = np.where(std == 0, 0.0, normalized)
    return FeatureMatrix(
        student_ids=fm.student_ids,
        matrix=normalized,
        field_names=fm.field_names,
        imputations=fm.imputations,
    )
