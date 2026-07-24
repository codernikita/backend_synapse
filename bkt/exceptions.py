"""
exceptions.py — BKT engine domain-specific exception hierarchy.

Every failure mode gets its own named exception type. Callers can
catch the base `BKTError` for generic handling, or a specific subclass
when the distinction matters (e.g. an invalid parameter vs. a corrupted
state object). No downstream module raises plain ValueError or TypeError
for conditions that are semantically BKT errors.
"""

from __future__ import annotations


class BKTError(Exception):
    """Base class for every exception raised by the BKT engine.
    Catching this catches any BKT-specific error regardless of which
    sub-module raised it."""


class InvalidParameterError(BKTError):
    """Raised when a BKTSkillParameters instance violates a mathematical
    constraint:
      - Any probability outside [0, 1]
      - P(G) + (1 - P(S)) >= 1  →  correct answer always implies mastery
        regardless of current state (degenerate BKT)
      - P(S) + (1 - P(G)) >= 1  →  incorrect always implies non-mastery
        regardless of current state (degenerate BKT)
      - P(T) <= 0  →  absorbing non-mastery state (no learning possible)
    The message always names the offending parameter and its value.
    """


class InvalidStateError(BKTError):
    """Raised when a BKTState's `p_l` field is outside the open interval
    (0, 1) exclusive. A hard 0 or 1 makes the Bayes conditional update
    undefined (0/0 or division by zero in the denominator) — the engine
    refuses to operate on such a state rather than producing NaN silently.
    """


class MasteryThresholdError(BKTError):
    """Raised when a mastery threshold value is outside (0, 1) exclusive.
    A threshold of 0 means every state is mastered (trivial / useless);
    a threshold of 1 is unattainable given the P_L_CLAMP_MAX < 1 invariant.
    """


class IRTParameterRangeError(BKTError):
    """Raised when an IRT parameter passed to the BKT bridge (theta, a, b)
    is outside a numerically safe range:
      - discrimination `a` must be > 0  (non-positive a reverses the ICC)
      - theta must be finite
      - difficulty `b` must be finite
    Does NOT enforce specific numeric bounds (the IRT engine's own clamps
    handle that); only rejects clearly nonsensical values.
    """


__all__ = [
    "BKTError",
    "InvalidParameterError",
    "InvalidStateError",
    "MasteryThresholdError",
    "IRTParameterRangeError",
]
