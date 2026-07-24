"""
theta.py — CHANGE 4: estimate the student ability parameter (theta) via
the 2PL logistic model, given ALREADY KNOWN discrimination (a) and
difficulty (b) for every question.

Responsibility
--------------
This module does exactly one thing: given one student's responses and
the QuestionIRTParameters for the questions they answered, find the
theta that maximizes the 2PL log-likelihood of that response pattern.
It knows nothing about Bloom levels, clustering, or where a/b came from
— see item_parameters.py for that decoupling.

Why theta represents latent student ability
---------------------------------------------------------------------
Theta is a single real number placed on the same scale as difficulty (b):
positive theta means "more likely than average to answer hard questions
correctly", negative means the opposite. It's "latent" because nobody
observes it directly — it's inferred purely from the pattern of which
questions a student got right and wrong, weighted by how hard (b) and how
discriminating (a) each of those questions is. Two students with the same
raw score (e.g. 6/10) can get different thetas if one got the harder
questions right and the other got the easier ones right — theta accounts
for that in a way a raw percentage can't.

Why only theta needs to be estimated in our Hybrid IRT model
---------------------------------------------------------------------
Standard 2PL calibration jointly estimates a, b, AND theta for every
student/item from a large response matrix — that's the "thousands of
responses" the supervisor said we don't have yet. Here, a and b are fixed
inputs (Change 1 and Change 3 already resolved them independently of any
particular student's responses), so the only unknown left in the 2PL
equation P(correct) = 1 / (1 + exp(-a(theta - b))) is theta. That
collapses a hard joint-estimation problem into a much easier one-
dimensional numerical optimization per student, which is exactly why this
is tractable with the amount of data actually available.

Why Newton-Raphson (not e.g. gradient descent or grid search)
---------------------------------------------------------------------
The 2PL log-likelihood, as a function of theta alone with a/b fixed, is
globally concave (its second derivative -sum(a_i^2 * P_i * (1-P_i)) is
always <= 0, regardless of the sign of a_i). A concave, twice-
differentiable, one-dimensional function is exactly the case Newton-
Raphson is designed for: it converges quadratically (typically 3-6
iterations to 1e-5 precision here) instead of gradient descent's linear
convergence, and unlike grid search its precision isn't limited by a
step-size choice. The only failure mode Newton-Raphson has on a concave
function is divergence on pathological/extreme inputs — handled below by
detecting those cases up front (see _extreme_pattern_theta) rather than
letting the optimizer run into them.

How theta will later initialize the Student Knowledge Graph
---------------------------------------------------------------------
mastery_initializer.py (not built yet) will combine this ThetaResult with
per-concept correctness to seed each concept's initial mastery probability
before BKT starts updating it from live quiz activity. ThetaResult is
designed (see Future Compatibility below) to be consumed there without
any change to this module.

Future Compatibility
---------------------------------------------------------------------
ThetaResult carries everything mastery_initializer.py should need
(theta, standard_error, converged, n_responses) as plain fields — no
private/internal state, no dependency on numpy types leaking out (theta,
standard_error, log_likelihood are all plain Python floats).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .config import (
    THETA_CONVERGENCE_TOLERANCE,
    THETA_EXPONENT_CLAMP,
    THETA_EXTREME_PATTERN_CLAMP,
    THETA_INITIAL,
    THETA_MAX,
    THETA_MAX_ITERATIONS,
    THETA_MAX_STEP,
    THETA_MIN,
)
from .item_parameters import QuestionIRTParameters


class EmptyResponsesError(ValueError):
    """Raised when estimate_theta() is given zero responses. There is no
    likelihood function to maximize with no data — this must be caught
    upstream (e.g. skip students who haven't taken the diagnostic yet),
    not silently defaulted to theta=0."""


class DuplicateResponseError(ValueError):
    """Raised when the same question_id appears more than once in a single
    student's responses. This is ambiguous (which attempt counts?) and
    should be resolved by the caller (e.g. keep-latest-attempt policy),
    not guessed at here."""


class DuplicateParameterError(ValueError):
    """Raised when the same question_id appears more than once in the
    supplied QuestionIRTParameters list, even if the values are identical
    — it signals an upstream data assembly bug (item_parameters.py should
    never produce this) and should be caught immediately rather than
    silently picking one."""


class MissingParameterError(ValueError):
    """Raised when a student's response references a question_id that has
    no corresponding QuestionIRTParameters. Cannot score a response
    without knowing that question's a and b."""


@dataclass(frozen=True)
class AnswerRecord:
    """One (question, correctness) pair for a single student. Deliberately
    does NOT carry student_id or bloom_level — estimate_theta() is called
    once per student, and doesn't need to know about Bloom levels at all
    (that's item_parameters.py's job)."""

    question_id: str
    is_correct: bool


@dataclass(frozen=True)
class ThetaResult:
    """Output of estimate_theta(). All fields are plain Python
    types (no numpy leakage) so downstream consumers (mastery_initializer.py,
    eventually a JSON API response) don't need to know this was computed
    with numpy internally."""

    theta: float
    iterations: int
    converged: bool
    log_likelihood: float
    standard_error: Optional[float]
    n_responses: int


def probability_correct(a: float, b: float, theta: float) -> float:
    """The 2PL logistic curve itself: P(correct) = 1 / (1 + exp(-a(theta - b))).
    Public and reused by BOTH estimate_theta()'s internal Newton-Raphson
    loop and the demo script's per-question P(correct) table, so the
    formula lives in exactly one place (no duplicated logic).

    Numerically stable: the exponent is clamped to +-THETA_EXPONENT_CLAMP
    before calling exp(), and the two mathematically equivalent forms of
    the sigmoid are used depending on the sign of the exponent, to avoid
    ever computing exp() of a large positive number (which is where
    overflow actually happens; exp() of a very negative number safely
    underflows to 0.0, which is the numerically correct answer).
    """
    z = -a * (theta - b)
    z = max(-THETA_EXPONENT_CLAMP, min(THETA_EXPONENT_CLAMP, z))
    if z >= 0:
        ez = math.exp(-z)
        return ez / (1.0 + ez)
    else:
        ez = math.exp(z)
        return 1.0 / (1.0 + ez)


def _score_and_information(
    theta: float, joined: Sequence[tuple]
) -> tuple[float, float, float]:
    """Returns (score, information, log_likelihood) at a given theta.
    score = dL/dtheta = sum(a_i * (y_i - P_i))
    information = -d2L/dtheta2 = sum(a_i^2 * P_i * (1 - P_i))  (always >= 0)
    log_likelihood = sum(y_i*log(P_i) + (1-y_i)*log(1-P_i))
    `joined` is a sequence of (a, b, is_correct) tuples, precomputed once
    per estimate_theta() call rather than re-zipping every iteration.
    """
    score = 0.0
    information = 0.0
    log_likelihood = 0.0
    # Probabilities are clamped away from exact 0/1 before taking log(), so
    # a perfectly-fit response (P essentially 1.0 or 0.0) never produces
    # log(0) = -inf or a NaN log-likelihood.
    eps = 1e-9
    for a, b, is_correct in joined:
        p = probability_correct(a, b, theta)
        p_clamped = min(max(p, eps), 1.0 - eps)
        y = 1.0 if is_correct else 0.0
        score += a * (y - p)
        information += (a * a) * p_clamped * (1.0 - p_clamped)
        log_likelihood += y * math.log(p_clamped) + (1.0 - y) * math.log(1.0 - p_clamped)
    return score, information, log_likelihood


def _extreme_pattern_theta(joined: Sequence[tuple]) -> Optional[float]:
    """If every response is correct or every response is incorrect, the
    2PL log-likelihood has no finite maximum (it's monotonic in theta
    forever) — this is a real mathematical property of MLE for extreme
    response patterns, not a bug to iterate around. Returns the clamp
    boundary theta to report in that case, or None if the pattern is mixed
    (has both a correct and an incorrect response) and Newton-Raphson
    should proceed normally.

    Note: this also correctly covers the single-response case, since one
    response is trivially "all correct" or "all incorrect" — there is no
    special-cased n==1 branch anywhere in this module; the math handles it
    naturally.
    """
    outcomes = {is_correct for _, _, is_correct in joined}
    if len(outcomes) > 1:
        return None  # mixed pattern, has an interior maximum
    all_correct = True in outcomes
    return THETA_EXTREME_PATTERN_CLAMP if all_correct else -THETA_EXTREME_PATTERN_CLAMP


def estimate_theta(
    responses: Sequence[AnswerRecord],
    parameters: Sequence[QuestionIRTParameters],
) -> ThetaResult:
    """Estimate one student's ability (theta) via Newton-Raphson MLE on
    the 2PL log-likelihood.

    Raises
    ------
    EmptyResponsesError       if responses is empty.
    DuplicateResponseError    if a question_id appears twice in responses.
    DuplicateParameterError   if a question_id appears twice in parameters.
    MissingParameterError     if a response's question_id has no matching
                               QuestionIRTParameters.
    """
    if not responses:
        raise EmptyResponsesError("Cannot estimate theta from zero responses.")

    seen_response_qids: set[str] = set()
    for r in responses:
        if r.question_id in seen_response_qids:
            raise DuplicateResponseError(
                f"question_id {r.question_id!r} appears more than once in responses."
            )
        seen_response_qids.add(r.question_id)

    param_by_qid: Dict[str, QuestionIRTParameters] = {}
    for p in parameters:
        if p.question_id in param_by_qid:
            raise DuplicateParameterError(
                f"question_id {p.question_id!r} appears more than once in parameters."
            )
        param_by_qid[p.question_id] = p

    missing = [r.question_id for r in responses if r.question_id not in param_by_qid]
    if missing:
        raise MissingParameterError(
            f"No QuestionIRTParameters found for question_id(s): {missing}"
        )

    joined = [
        (param_by_qid[r.question_id].discrimination, param_by_qid[r.question_id].difficulty, r.is_correct)
        for r in responses
    ]

    extreme_theta = _extreme_pattern_theta(joined)
    if extreme_theta is not None:
        _, information, log_likelihood = _score_and_information(extreme_theta, joined)
        se = _standard_error(information)
        return ThetaResult(
            theta=extreme_theta,
            iterations=0,
            converged=False,  # no finite MLE exists for this pattern; boundary reported, not a true optimum
            log_likelihood=log_likelihood,
            standard_error=se,
            n_responses=len(responses),
        )

    theta = THETA_INITIAL
    converged = False
    iterations = 0
    log_likelihood = 0.0
    information = 0.0

    for iterations in range(1, THETA_MAX_ITERATIONS + 1):
        score, information, log_likelihood = _score_and_information(theta, joined)

        if information <= 0.0:
            # All items have effectively zero discrimination at this theta
            # (e.g. every a == 0) — no information to update theta with.
            # Stop rather than divide by zero.
            break

        step = score / information
        step = max(-THETA_MAX_STEP, min(THETA_MAX_STEP, step))  # cap runaway steps
        theta_new = theta + step
        theta_new = max(THETA_MIN, min(THETA_MAX, theta_new))  # keep within sane bounds

        if abs(score) < THETA_CONVERGENCE_TOLERANCE:
            theta = theta_new
            converged = True
            break

        theta = theta_new

    # Recompute final log-likelihood/information at the theta we're returning,
    # in case the loop exited on the information<=0 break (where the last
    # computed log_likelihood/information are already correct for `theta`,
    # since no update happened after that computation).
    _, information, log_likelihood = _score_and_information(theta, joined)
    se = _standard_error(information)

    return ThetaResult(
        theta=theta,
        iterations=iterations,
        converged=converged,
        log_likelihood=log_likelihood,
        standard_error=se,
        n_responses=len(responses),
    )


def _standard_error(information: float) -> Optional[float]:
    """SE(theta) = 1 / sqrt(information), the standard asymptotic MLE
    standard error under IRT. None (not NaN, not infinity) when
    information is zero or negative — that state means "no information
    to estimate a standard error from", which is a fact about the data,
    not a numeric edge case to hide behind inf."""
    if information <= 0.0:
        return None
    return 1.0 / math.sqrt(information)
