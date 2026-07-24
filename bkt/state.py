"""
state.py — BKTState: the complete HMM hidden state for one student × skill pair.

Responsibility
--------------
Defines the immutable snapshot of everything the BKT engine needs to know
about a student's knowledge state for a particular skill. Every field is a
plain Python type — no numpy types, no database handles — so this dataclass
can be serialized to JSON, stored in a database, or passed across service
boundaries without any import dependency beyond the standard library.

BKTState is ALWAYS the input and output of BKTEngine.update(). The engine
never mutates a state in place; it always produces a new BKTState. This
makes the update function a pure function: same inputs → same output, no
hidden side effects, trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .config import (
    DEFAULT_P_L0,
    MASTERY_THRESHOLD,
    P_L_CLAMP_MAX,
    P_L_CLAMP_MIN,
)
from .exceptions import InvalidStateError, MasteryThresholdError


@dataclass(frozen=True)
class BKTState:
    """Immutable snapshot of a student's BKT knowledge state for one skill.

    Fields
    ------
    student_id : str
        Unique student identifier (maps to User.id in the Prisma schema).
    skill_id : str
        Skill/concept identifier (maps to Concept.concept_id).
    p_l : float
        Current P(Learned) — the HMM's posterior probability that the
        student knows this skill. Strictly in (P_L_CLAMP_MIN, P_L_CLAMP_MAX).
    attempt_count : int
        Total number of observations (questions answered) for this skill.
    correct_count : int
        Number of correct responses among those observations.
    is_mastered : bool
        True if p_l >= mastery_threshold at the time of last update.
        This is a derived flag — always consistent with p_l and the
        threshold used; never set independently.
    last_updated : datetime
        UTC timestamp of the most recent update. Uses timezone-aware UTC
        by default to avoid ambiguous local-time comparisons.
    mastery_threshold_used : float
        The mastery threshold that was in effect when is_mastered was last
        evaluated. Stored so audit logs can show which threshold produced
        a mastery decision, even if the threshold is later reconfigured.
    """

    student_id: str
    skill_id: str
    p_l: float
    attempt_count: int = 0
    correct_count: int = 0
    is_mastered: bool = False
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mastery_threshold_used: float = MASTERY_THRESHOLD

    def __post_init__(self) -> None:
        """Enforce invariants on construction."""
        if not (P_L_CLAMP_MIN <= self.p_l <= P_L_CLAMP_MAX):
            raise InvalidStateError(
                f"BKTState.p_l must be in [{P_L_CLAMP_MIN}, {P_L_CLAMP_MAX}], "
                f"got {self.p_l:.6f} for student={self.student_id!r}, "
                f"skill={self.skill_id!r}. Use BKTEngine.initial_state() to "
                "construct a valid starting state."
            )
        if self.attempt_count < 0:
            raise InvalidStateError(
                f"attempt_count must be >= 0, got {self.attempt_count}"
            )
        if self.correct_count < 0 or self.correct_count > self.attempt_count:
            raise InvalidStateError(
                f"correct_count must be in [0, attempt_count], got "
                f"correct_count={self.correct_count}, attempt_count={self.attempt_count}"
            )
        if not (0.0 < self.mastery_threshold_used < 1.0):
            raise MasteryThresholdError(
                f"mastery_threshold_used must be in (0, 1), got {self.mastery_threshold_used}"
            )

    @property
    def accuracy(self) -> Optional[float]:
        """Observed accuracy so far: correct_count / attempt_count.
        Returns None if no attempts have been made yet (avoids 0/0)."""
        if self.attempt_count == 0:
            return None
        return self.correct_count / self.attempt_count

    def __repr__(self) -> str:
        return (
            f"BKTState(student={self.student_id!r}, skill={self.skill_id!r}, "
            f"p_l={self.p_l:.4f}, attempts={self.attempt_count}, "
            f"correct={self.correct_count}, mastered={self.is_mastered})"
        )


def initial_state(
    student_id: str,
    skill_id: str,
    p_l0: float = DEFAULT_P_L0,
    mastery_threshold: float = MASTERY_THRESHOLD,
) -> BKTState:
    """Construct the canonical starting BKTState for a student × skill pair.

    Parameters
    ----------
    student_id : str
        Unique student identifier.
    skill_id : str
        Skill/concept identifier.
    p_l0 : float
        Initial knowledge probability. Defaults to DEFAULT_P_L0 (from config).
        Pass the output of `parameters.seed_p_l0_from_irt()` to use an
        IRT-derived prior instead.
    mastery_threshold : float
        Threshold that will be used to evaluate is_mastered on this state.
        Defaults to MASTERY_THRESHOLD (from config). Stored on the state so
        mastery decisions are auditable even if the global default changes.

    Returns
    -------
    BKTState
        A freshly initialized state with zero attempts and is_mastered=False.
    """
    if not (0.0 < mastery_threshold < 1.0):
        raise MasteryThresholdError(
            f"mastery_threshold must be in (0, 1), got {mastery_threshold}"
        )
    # p_l0 is already validated by BKTState.__post_init__; clamp first for safety
    p_l_clamped = max(P_L_CLAMP_MIN, min(P_L_CLAMP_MAX, p_l0))
    return BKTState(
        student_id=student_id,
        skill_id=skill_id,
        p_l=p_l_clamped,
        attempt_count=0,
        correct_count=0,
        is_mastered=p_l_clamped >= mastery_threshold,  # edge case: very high p_l0
        last_updated=datetime.now(timezone.utc),
        mastery_threshold_used=mastery_threshold,
    )


__all__ = [
    "BKTState",
    "initial_state",
]
