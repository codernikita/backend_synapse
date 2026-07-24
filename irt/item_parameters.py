"""
item_parameters.py — the seam between "how a and b were produced" and
"how theta is estimated from them".

Responsibility
--------------
Defines QuestionIRTParameters: a plain (question_id, discrimination,
difficulty) triple. That's the entire public contract theta.py depends
on. theta.py never imports bloom_mapper or segregation — it only ever
sees this dataclass.

Why this module exists (an architectural change, not just a dataclass)
-------------------------------------------------------------------------
Change 1 sources difficulty (b) from Bloom's Taxonomy; Change 3 sources
discrimination (a) from KMeans segregation. Both are implementation
details of *how the parameters get produced*. theta.py's job — estimating
θ from known a/b via 2PL — is mathematically identical no matter how a
and b were derived (Bloom mapping today, statistical calibration once
enough response data exists later, manual override by a subject-matter
expert, etc.). If theta.py imported bloom_mapper.difficulty_for() and
segregation.SegregationResult directly, swapping out either source later
would mean touching theta.py's code, and every test of theta.py would
need to know about Bloom levels and clustering even though neither is
relevant to Newton-Raphson MLE. This module is the boundary that makes
that swap free.

build_question_parameters() is the one place that DOES know about both
sources — it's the assembly step, called by service.py (not built yet),
never by theta.py itself.

How it interacts with the rest of the architecture
----------------------------------------------------------------------
    bloom_mapper.difficulty_for(bloom_level)      -\
    segregation.SegregationBatchResult             }-> item_parameters.build_question_parameters()
                                                    -/         |
                                                                v
                                                   list[QuestionIRTParameters]
                                                                |
                                                                v
                                                          theta.estimate_theta()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

from .bloom_mapper import UnknownBloomLevelError, difficulty_for
from .segregation import SegregationBatchResult


@dataclass(frozen=True)
class QuestionIRTParameters:
    """The entire contract theta.py depends on. Deliberately minimal —
    no bloom_level, no cluster info, no accuracy stats. Just what the 2PL
    equation needs."""

    question_id: str
    discrimination: float  # a
    difficulty: float  # b


@dataclass(frozen=True)
class SkippedQuestionParameters:
    """A question that could not get a QuestionIRTParameters built for it,
    with why — mirrors segregation.SkippedQuestion's never-silently-drop
    pattern. Reasons: 'unknown_bloom_level', 'no_segregation_score'."""

    question_id: str
    reason: str


def build_question_parameters(
    question_bloom_levels: Mapping[str, str],
    segregation_result: SegregationBatchResult,
) -> Tuple[List[QuestionIRTParameters], List[SkippedQuestionParameters]]:
    """Assemble QuestionIRTParameters for every question that has BOTH a
    Bloom level (-> difficulty) and a scored segregation result (-> discrimination).

    Parameters
    ----------
    question_bloom_levels:
        {question_id: bloom_level}, e.g. sourced from the question
        dataset/repository.
    segregation_result:
        Output of segregation.compute_segregation_scores() for the same
        item bank.

    Returns
    -------
    (parameters, skipped) — every question_id present in EITHER input ends
    up in exactly one of the two lists, never silently absent from both.
    A question skipped by segregation.py (e.g. 'only_strong_attempted')
    stays skipped here too, with reason 'no_segregation_score'; a question
    with a segregation score but an unrecognized/missing bloom_level is
    skipped here with reason 'unknown_bloom_level' or 'missing_bloom_level'.
    """
    scored_by_question = segregation_result.as_dict_by_question()
    all_question_ids = set(question_bloom_levels) | set(scored_by_question)

    parameters: List[QuestionIRTParameters] = []
    skipped: List[SkippedQuestionParameters] = []

    for qid in sorted(all_question_ids):
        segregation_entry = scored_by_question.get(qid)
        bloom_level = question_bloom_levels.get(qid)

        if segregation_entry is None:
            skipped.append(SkippedQuestionParameters(qid, reason="no_segregation_score"))
            continue
        if bloom_level is None:
            skipped.append(SkippedQuestionParameters(qid, reason="missing_bloom_level"))
            continue
        try:
            b = difficulty_for(bloom_level)
        except UnknownBloomLevelError:
            skipped.append(SkippedQuestionParameters(qid, reason="unknown_bloom_level"))
            continue

        parameters.append(
            QuestionIRTParameters(
                question_id=qid,
                discrimination=segregation_entry.discrimination,
                difficulty=b,
            )
        )

    return parameters, skipped
