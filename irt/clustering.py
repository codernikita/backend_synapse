"""
clustering.py — CHANGE 2 (part 2): splits students into a strong and a
weak cluster using KMeans(k=2) on the feature vectors feature_builder.py
already built and normalized.

Responsibility
--------------
Take a FeatureMatrix (already normalized — this module normalizes
nothing) and produce a ClusterResult: which raw sklearn cluster id each
student landed in, which of the two cluster ids is "strong" vs "weak",
the fitted centroids, and per-cluster descriptive statistics.

Why it exists as its own module
--------------------------------
This is the discrimination-parameter groundwork from Change 2/3: instead
of estimating item discrimination mathematically, the supervisor's plan
needs a strong/weak split of the student population first — segregation.py
(not built yet) will later compute each *question's* discrimination as
strong-cluster-accuracy minus weak-cluster-accuracy, using the split
produced here. Keeping clustering fully separate from segregation means
this module has exactly one job (label students) and can be tested,
swapped, or re-tuned (e.g. a different k, a different algorithm) without
touching how questions get scored.

How it interacts with the rest of the architecture
----------------------------------------------------------------------
    feature_builder.build_feature_matrix()
        -> feature_builder.normalize_feature_matrix()
            -> clustering.cluster_students(normalized, raw)
                -> ClusterResult  (consumed by segregation.py, next module)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans

from .config import N_CLUSTERS, RANDOM_STATE
from .feature_builder import FeatureMatrix


class EmptyFeatureMatrixError(ValueError):
    """Raised when cluster_students() is given a FeatureMatrix with zero
    students. There is nothing to cluster — this is a caller error
    (e.g. querying an empty cohort), not something to silently no-op on."""


class InsufficientStudentsError(ValueError):
    """Raised when there are fewer students than N_CLUSTERS (currently 2).
    KMeans(k=2) is mathematically undefined with 0 or 1 samples, and we'd
    rather fail loudly than let sklearn's own (less specific) error leak
    through, or worse, silently produce a meaningless one-cluster result."""


class ClusteringFailedError(RuntimeError):
    """Raised when scikit-learn itself fails to produce N_CLUSTERS distinct
    labels for a request that passed the size checks above (e.g. some
    unexpected sklearn internal error). Wraps the underlying exception so
    callers get one exception type to catch regardless of cause."""


@dataclass(frozen=True)
class ClusterStatistics:
    """Human-interpretable, real-unit summary of one cluster — computed
    from RAW (non-normalized) feature values, since "average IQ 0.42" is
    meaningless to a person debugging the pipeline but "average IQ 104.3"
    is not. See cluster_students()'s raw_feature_matrix parameter."""

    cluster_id: int
    n_students: int
    avg_previous_class_percentage: float
    avg_iq_score: float
    avg_total_correct: float
    avg_easy_accuracy: float
    avg_medium_accuracy: float
    avg_hard_accuracy: float


@dataclass
class ClusterResult:
    """Output of cluster_students().

    cluster_labels: raw sklearn cluster id (0 or 1) per student, aligned
        1:1 with student_ids. These are NOT "strong"/"weak" directly —
        sklearn's cluster ids are arbitrary; use strong_cluster_id /
        weak_cluster_id (or the helper methods below) to interpret them.
    cluster_centroids: KMeans' fitted centroids, in whatever feature space
        was passed in for clustering (normalized units, per this module's
        contract) — useful for debugging/plotting, not for direct
        human-readable reporting (use cluster_statistics for that).
    """

    student_ids: List[str]
    cluster_labels: np.ndarray  # shape (n_students,), values in {0, 1}
    strong_cluster_id: int
    weak_cluster_id: int
    cluster_centroids: np.ndarray  # shape (N_CLUSTERS, n_features)
    cluster_statistics: Dict[int, ClusterStatistics]

    def label_for(self, student_id: str) -> str:
        """Return 'strong' or 'weak' for a given student_id."""
        idx = self.student_ids.index(student_id)
        raw_label = int(self.cluster_labels[idx])
        return "strong" if raw_label == self.strong_cluster_id else "weak"

    def strong_student_ids(self) -> List[str]:
        return [
            sid
            for sid, label in zip(self.student_ids, self.cluster_labels)
            if int(label) == self.strong_cluster_id
        ]

    def weak_student_ids(self) -> List[str]:
        return [
            sid
            for sid, label in zip(self.student_ids, self.cluster_labels)
            if int(label) == self.weak_cluster_id
        ]

    def statistics_for(self, cluster_name: str) -> ClusterStatistics:
        """Look up statistics by 'strong'/'weak' instead of raw cluster id."""
        cluster_id = self.strong_cluster_id if cluster_name == "strong" else self.weak_cluster_id
        return self.cluster_statistics[cluster_id]


def _compute_cluster_statistics(
    cluster_id: int,
    student_ids_in_cluster: List[str],
    raw_rows_by_student: Dict[str, np.ndarray],
    field_index: Dict[str, int],
) -> ClusterStatistics:
    rows = np.array([raw_rows_by_student[sid] for sid in student_ids_in_cluster])
    return ClusterStatistics(
        cluster_id=cluster_id,
        n_students=len(student_ids_in_cluster),
        avg_previous_class_percentage=float(rows[:, field_index["previous_class_percentage"]].mean()),
        avg_iq_score=float(rows[:, field_index["iq_score"]].mean()),
        avg_total_correct=float(rows[:, field_index["total_correct"]].mean()),
        avg_easy_accuracy=float(rows[:, field_index["easy_accuracy"]].mean()),
        avg_medium_accuracy=float(rows[:, field_index["medium_accuracy"]].mean()),
        avg_hard_accuracy=float(rows[:, field_index["hard_accuracy"]].mean()),
    )


def cluster_students(
    feature_matrix: FeatureMatrix,
    raw_feature_matrix: Optional[FeatureMatrix] = None,
) -> ClusterResult:
    """Cluster students into a strong and a weak group.

    Parameters
    ----------
    feature_matrix:
        The matrix KMeans actually clusters on. Per the module contract,
        this must already be normalized (feature_builder.normalize_feature_matrix) —
        clustering.py performs no normalization of its own.
    raw_feature_matrix:
        The same students' RAW (non-normalized) feature values, used only
        to compute human-readable ClusterStatistics (avg IQ, avg
        percentage, etc. in real units). If omitted, statistics are
        computed from `feature_matrix` itself — meaning they'll be in
        normalized (z-score) units, which is still internally consistent
        but much less useful for a person reading a debug report. Callers
        (e.g. the demo script, and later segregation.py) should pass the
        raw matrix whenever it's available.
        Must contain exactly the same student_ids as feature_matrix (order
        may differ; they are matched by id, not position).

    Raises
    ------
    EmptyFeatureMatrixError
        if feature_matrix has zero students.
    InsufficientStudentsError
        if feature_matrix has fewer than config.N_CLUSTERS (2) students.
    ClusteringFailedError
        if scikit-learn fails to produce N_CLUSTERS distinct cluster ids
        for a request that otherwise passed validation.
    """
    n_students = len(feature_matrix.student_ids)

    if n_students == 0:
        raise EmptyFeatureMatrixError(
            "Cannot cluster an empty FeatureMatrix (0 students). "
            "Check the upstream query/CSV that produced it."
        )
    if n_students < N_CLUSTERS:
        raise InsufficientStudentsError(
            f"Need at least {N_CLUSTERS} students to form {N_CLUSTERS} clusters, "
            f"got {n_students}."
        )

    if raw_feature_matrix is not None:
        if set(raw_feature_matrix.student_ids) != set(feature_matrix.student_ids):
            raise ValueError(
                "raw_feature_matrix and feature_matrix must contain the same "
                "student_ids. Got mismatched sets: "
                f"{set(feature_matrix.student_ids) ^ set(raw_feature_matrix.student_ids)}"
            )
        raw_source = raw_feature_matrix
    else:
        raw_source = feature_matrix  # falls back to normalized units, documented above

    try:
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(feature_matrix.matrix)
    except Exception as exc:  # sklearn's own errors vary by version; normalize to one type
        raise ClusteringFailedError(f"scikit-learn KMeans failed: {exc}") from exc

    distinct_labels = set(int(l) for l in labels)
    if len(distinct_labels) < N_CLUSTERS:
        raise ClusteringFailedError(
            f"KMeans produced only {len(distinct_labels)} distinct cluster(s) "
            f"instead of {N_CLUSTERS}. This can happen with degenerate input "
            "(e.g. duplicate feature vectors driving centroids together); "
            "inspect the feature matrix before retrying."
        )

    # Build raw-value lookup for statistics, keyed by student_id.
    raw_field_index = {name: i for i, name in enumerate(raw_source.field_names)}
    raw_rows_by_student = dict(zip(raw_source.student_ids, raw_source.matrix))

    student_ids = feature_matrix.student_ids
    cluster_statistics: Dict[int, ClusterStatistics] = {}
    avg_total_correct_by_cluster: Dict[int, float] = {}
    for cluster_id in sorted(distinct_labels):
        members = [sid for sid, label in zip(student_ids, labels) if int(label) == cluster_id]
        stats = _compute_cluster_statistics(cluster_id, members, raw_rows_by_student, raw_field_index)
        cluster_statistics[cluster_id] = stats
        avg_total_correct_by_cluster[cluster_id] = stats.avg_total_correct

    # CHANGE 2 requirement: strong/weak is decided by average total_correct,
    # never assumed from the raw cluster id. Ties are broken deterministically
    # (higher cluster id wins strong) so results stay reproducible even when
    # two clusters perform identically (e.g. the "identical students" test case).
    strong_cluster_id = max(
        avg_total_correct_by_cluster,
        key=lambda cid: (avg_total_correct_by_cluster[cid], cid),
    )
    weak_cluster_id = next(cid for cid in distinct_labels if cid != strong_cluster_id)

    return ClusterResult(
        student_ids=student_ids,
        cluster_labels=labels,
        strong_cluster_id=strong_cluster_id,
        weak_cluster_id=weak_cluster_id,
        cluster_centroids=kmeans.cluster_centers_,
        cluster_statistics=cluster_statistics,
    )
