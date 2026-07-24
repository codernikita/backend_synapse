"""
mastery.py — mastery thresholding and IRT cold-start P(L0) helpers.

Responsibility
--------------
This module contains two distinct but related responsibilities:

1. Mastery adjudication: given a P(L) and a threshold, decide whether the
   skill is mastered. This is extracted from engine.py so that threshold
   logic is testable independently of the full update cycle.

2. IRT cold-start: a standalone replication of the IRT engine's
   mastery_initializer.py shrinkage formula, callable without importing
   the full IRT pipeline (useful for lightweight orchestrators that receive
   IRT results as plain dicts rather than typed dataclasses).

Why this module is not just a helper in engine.py
--------------------------------------------------
Mastery thresholding is a *policy* decision (what P(L) is high enough to
call "mastered?"), separate from the *mathematical* BKT update (how does
each observation change P(L)?). Keeping them in separate modules means:
  - The threshold can be changed without touching the update math.
  - Tests can verify mastery decisions in isolation.
  - Future multi-threshold policies (e.g. "practice mastery" at 0.80 vs
    "assessment mastery" at 0.95) can be added here without touching engine.py.

Why guess-detection lives here (not in engine.py)
--------------------------------------------------
Guess detection (`is_probable_guess`) is also a policy decision: it classifies
an observation before the update runs. It depends on IRT parameters and
response-time heuristics — neither of which are core BKT math. Placing it
here keeps engine.py's concern purely "given these BKT parameters and this
is_correct flag, compute the update."
"""

from __future__ import annotations

import math
from typing import Optional

from .config import (
    IRT_SURPRISE_THRESHOLD,
    MASTERY_REFERENCE_DISCRIMINATION,
    MASTERY_THRESHOLD,
    RT_GUESS_THRESHOLD_MS,
)
from .exceptions import MasteryThresholdError
from .parameters import irt_p_correct


# ── Mastery adjudication ─────────────────────────────────────────────────────


def check_mastery(p_l: float, threshold: float = MASTERY_THRESHOLD) -> bool:
    """Return True if p_l meets or exceeds the mastery threshold.

    Parameters
    ----------
    p_l : float
        Current posterior P(Learned) from a BKTState.
    threshold : float
        Mastery threshold. Defaults to MASTERY_THRESHOLD from config.
        Must be in (0, 1) exclusive.

    Returns
    -------
    bool
        True when p_l >= threshold.

    Raises
    ------
    MasteryThresholdError
        If threshold is outside (0, 1).
    """
    if not (0.0 < threshold < 1.0):
        raise MasteryThresholdError(
            f"Mastery threshold must be in (0, 1), got {threshold}. "
            f"A threshold of 0 trivially masters everything; a threshold of 1 "
            f"is unattainable given the P_L_CLAMP_MAX invariant."
        )
    return p_l >= threshold


def mastery_gap(p_l: float, threshold: float = MASTERY_THRESHOLD) -> float:
    """Return how far P(L) is from the mastery threshold.

    Positive value = still below threshold (gap to close).
    Negative value = already above threshold (margin above mastery).

    Useful for ranking skills by urgency: larger positive gap = further from mastery.
    """
    return threshold - p_l


# ── Guess detection ──────────────────────────────────────────────────────────


def is_probable_guess(
    *,
    is_correct: bool,
    response_time_ms: Optional[int] = None,
    theta: Optional[float] = None,
    difficulty: Optional[float] = None,
    discrimination: Optional[float] = None,
    rt_threshold_ms: int = RT_GUESS_THRESHOLD_MS,
    surprise_threshold: float = IRT_SURPRISE_THRESHOLD,
    a_ref: float = MASTERY_REFERENCE_DISCRIMINATION,
) -> bool:
    """Classify a correct answer as a probable guess.

    Mirrors the TypeScript `isProbableGuess()` in knowledge.service.ts
    exactly — same two-condition logic, same config constants (imported from
    config.py so they are always in sync). A correct answer is a probable
    guess when it is BOTH:
      (a) answered faster than rt_threshold_ms, AND
      (b) "surprising" under the 2PL model: IRT P(correct) < surprise_threshold.

    If no IRT parameters are available, criterion (b) is skipped and the
    classification falls back to time-only.

    Parameters
    ----------
    is_correct : bool
        Whether the student answered correctly. Incorrect answers are never
        guesses (guessing only inflates correct answers).
    response_time_ms : int | None
        Time spent on this question in milliseconds. None = unknown.
    theta : float | None
        Student IRT ability. None = IRT params not yet available.
    difficulty : float | None
        Item IRT difficulty (b). None = IRT params not yet available.
    discrimination : float | None
        Item IRT discrimination (a). None = use a_ref.
    rt_threshold_ms : int
        Speed threshold. Defaults to RT_GUESS_THRESHOLD_MS from config.
    surprise_threshold : float
        IRT P(correct) below this → answer is "surprising". Defaults to
        IRT_SURPRISE_THRESHOLD from config.
    a_ref : float
        Reference discrimination when per-item a is not available.

    Returns
    -------
    bool
        True if and only if the answer is classified as a probable guess.
    """
    # Incorrect answers are never guesses
    if not is_correct:
        return False

    # Criterion (a): was the answer too fast?
    too_fast = (
        response_time_ms is not None
        and response_time_ms > 0
        and response_time_ms < rt_threshold_ms
    )
    if not too_fast:
        return False

    # Criterion (b): was the correct answer surprising under the IRT model?
    if theta is not None and difficulty is not None:
        a = discrimination if discrimination is not None else a_ref
        try:
            p_irt = irt_p_correct(theta, difficulty, a)
            return p_irt < surprise_threshold
        except Exception:
            # If IRT computation fails for any reason, fall back to time-only
            return True

    # No IRT params available: time-only classification
    return True


# ── Mastery trajectory analysis ───────────────────────────────────────────────


def expected_attempts_to_mastery(
    p_l: float,
    p_t: float,
    p_g: float,
    p_s: float,
    threshold: float = MASTERY_THRESHOLD,
    max_steps: int = 200,
) -> Optional[int]:
    """Simulate the BKT update to estimate how many more correct answers
    a student at `p_l` would need (assuming all-correct future responses)
    before reaching the mastery threshold.

    This is a forward simulation, NOT a closed-form solution. It is
    provided for dashboard/reporting use (e.g. "~5 more correct answers to
    mastery") and is intentionally separate from engine.py to avoid coupling
    reporting logic to the core update path.

    Parameters
    ----------
    p_l : float
        Current P(Learned).
    p_t, p_g, p_s : float
        BKT parameters for this skill.
    threshold : float
        Mastery threshold to simulate toward.
    max_steps : int
        Maximum simulation steps before returning None (skill requires
        very many attempts, possibly unreachable under these parameters).

    Returns
    -------
    int | None
        Number of all-correct steps needed, or None if not reached within
        max_steps.
    """
    if check_mastery(p_l, threshold):
        return 0  # already mastered

    current_p_l = p_l
    for step in range(1, max_steps + 1):
        # Correct answer conditional update
        numerator = current_p_l * (1.0 - p_s)
        denominator = numerator + (1.0 - current_p_l) * p_g
        if denominator <= 0:
            return None
        p_conditional = numerator / denominator
        # Learning transition
        current_p_l = p_conditional + (1.0 - p_conditional) * p_t
        if check_mastery(current_p_l, threshold):
            return step

    return None  # did not reach mastery within max_steps


__all__ = [
    "check_mastery",
    "mastery_gap",
    "is_probable_guess",
    "expected_attempts_to_mastery",
]
