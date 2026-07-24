"""
engine.py — BKTEngine: the pure, stateless BKT update computation.

Responsibility
--------------
This module contains exactly one thing: the mathematics of one BKT
update step. It reads parameters from BKTSkillParameters, reads the
current hidden state from BKTState, executes the two-stage HMM update,
and returns a new BKTState together with a BKTUpdateResult audit record.

Zero side effects. No database calls. No IRT imports. No config references
that could vary between invocations — all constants enter through the
BKTSkillParameters argument or the explicitly passed override arguments.
If this module has a bug, the bug is in the mathematics, not in any
plumbing or policy logic (which belong in parameters.py and mastery.py).

Two-stage BKT update (Corbett & Anderson, 1994)
-----------------------------------------------
Given:
  - P(L_{t-1}) = current posterior P(Learned) from BKTState.p_l
  - P(G) = guess rate from BKTSkillParameters.p_g  (or override_p_g)
  - P(S) = slip rate from BKTSkillParameters.p_s  (or override_p_s)
  - P(T) = learn rate from BKTSkillParameters.p_t

Stage 1 — Conditional update (Bayes' theorem):
  If correct:
    P(L_t | correct) = P(L_{t-1}) × (1 - P(S))
                       ─────────────────────────────────────────────────────
                       P(L_{t-1}) × (1 - P(S)) + (1 - P(L_{t-1})) × P(G)

  If incorrect:
    P(L_t | incorrect) = P(L_{t-1}) × P(S)
                         ─────────────────────────────────────────────────────
                         P(L_{t-1}) × P(S) + (1 - P(L_{t-1})) × (1 - P(G))

Stage 2 — Learning transition:
  P(L_t) = P(L_t | obs) + (1 - P(L_t | obs)) × P(T)

Result clamped to (P_L_CLAMP_MIN, P_L_CLAMP_MAX).
Mastery evaluated by mastery.check_mastery(p_l_new, threshold).

The IRT hook
------------
The update() method accepts optional override_p_g and override_p_s
arguments. When provided, these replace the skill-level P(G) and P(S)
from BKTSkillParameters for this single invocation. This is the
"IRT hook" that allows the orchestrator to inject per-question dynamic
values derived from the student's theta and the item's (a, b) parameters
(via parameters.derive_p_g_from_irt / derive_p_s_from_irt), while the
BKT engine itself remains agnostic to IRT.

Guess-detection integration
---------------------------
When response_time_ms and IRT parameters (theta, difficulty, discrimination)
are provided, the engine calls mastery.is_probable_guess() to determine
whether a correct answer should be treated with an inflated P(G) (making
the upward update weaker). The effective P(G) used is always recorded in
BKTUpdateResult.p_g_used so the decision is fully auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .config import (
    MASTERY_THRESHOLD,
    P_G_CLAMP_MAX,
    P_G_CLAMP_MIN,
    P_L_CLAMP_MAX,
    P_L_CLAMP_MIN,
    P_S_CLAMP_MAX,
    P_S_CLAMP_MIN,
)
from .exceptions import InvalidParameterError, InvalidStateError
from .mastery import check_mastery, is_probable_guess
from .parameters import BKTSkillParameters, derive_p_g_inflated
from .state import BKTState, initial_state


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BKTUpdateResult:
    """Output of BKTEngine.update().

    Carries both the new state (what the caller persists) and a full
    audit trail of the intermediate values that produced it. Every number
    in the update is recorded here — nothing is computed then discarded —
    so that tests, logging, and human review can all trace the exact
    calculation.

    Fields
    ------
    new_state : BKTState
        The updated HMM state after incorporating this observation.
    previous_p_l : float
        P(L_{t-1}) before this update.
    p_conditional : float
        P(L_t | observation) after the Bayesian conditional update (Stage 1).
    p_g_used : float
        The effective P(G) used in Stage 1. May differ from the skill's
        baseline P(G) if a guess was detected or an override was passed.
    p_s_used : float
        The effective P(S) used in Stage 1.
    p_t_used : float
        The P(T) used in Stage 2.
    was_flagged_guess : bool
        True if the correct answer was classified as a probable guess and
        p_g_used was inflated accordingly.
    override_p_g_applied : bool
        True if the caller supplied an override_p_g that replaced the
        skill-level baseline.
    override_p_s_applied : bool
        True if the caller supplied an override_p_s that replaced the
        skill-level baseline.
    is_correct : bool
        The observation that triggered this update.
    timestamp : datetime
        UTC timestamp when the update was computed.
    """

    new_state: BKTState
    previous_p_l: float
    p_conditional: float
    p_g_used: float
    p_s_used: float
    p_t_used: float
    was_flagged_guess: bool
    override_p_g_applied: bool
    override_p_s_applied: bool
    is_correct: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def delta_p_l(self) -> float:
        """Change in P(L): positive for correct, negative for incorrect."""
        return self.new_state.p_l - self.previous_p_l


# ── BKTEngine ─────────────────────────────────────────────────────────────────


class BKTEngine:
    """Pure, stateless BKT update service.

    All methods are static — BKTEngine carries no instance state and has
    no constructor arguments. Instantiation is allowed (for dependency-
    injection patterns) but not required; callers may call methods on the
    class directly.

    The engine has no knowledge of databases, HTTP, or IRT internals.
    Its only job is to compute the two-stage BKT update correctly.
    """

    @staticmethod
    def initial_state(
        student_id: str,
        skill_id: str,
        params: BKTSkillParameters,
        mastery_threshold: float = MASTERY_THRESHOLD,
    ) -> BKTState:
        """Create the starting BKTState for a student × skill pair.

        Delegates to state.initial_state() using params.p_l0 as the
        prior. Provided here so callers can use BKTEngine as a single
        entry point without importing state.py directly.

        Parameters
        ----------
        student_id : str
            Unique student identifier.
        skill_id : str
            Skill/concept identifier.
        params : BKTSkillParameters
            Must include a valid p_l0 (from defaults() or from_irt()).
        mastery_threshold : float
            Threshold stored on the state for audit purposes.

        Returns
        -------
        BKTState
        """
        return initial_state(
            student_id=student_id,
            skill_id=skill_id,
            p_l0=params.p_l0,
            mastery_threshold=mastery_threshold,
        )

    @staticmethod
    def update(
        state: BKTState,
        is_correct: bool,
        params: BKTSkillParameters,
        *,
        override_p_g: Optional[float] = None,
        override_p_s: Optional[float] = None,
        response_time_ms: Optional[int] = None,
        theta: Optional[float] = None,
        difficulty: Optional[float] = None,
        discrimination: Optional[float] = None,
        mastery_threshold: Optional[float] = None,
    ) -> BKTUpdateResult:
        """Execute one BKT update step.

        Stage 1: Conditional update (Bayes' theorem)
        Stage 2: Learning transition
        Then: clamp, mastery check, build new BKTState.

        Parameters
        ----------
        state : BKTState
            Current knowledge state. Must have p_l in (P_L_CLAMP_MIN, P_L_CLAMP_MAX).
        is_correct : bool
            True if the student answered correctly.
        params : BKTSkillParameters
            Skill-level BKT parameters. Provides the baseline P(G), P(S), P(T).
        override_p_g : float | None
            **IRT Hook.** If provided, replaces params.p_g for this single
            invocation. Must be in (P_G_CLAMP_MIN, P_G_CLAMP_MAX). Pass the
            output of parameters.derive_p_g_from_irt(theta, b, a) here.
        override_p_s : float | None
            **IRT Hook.** If provided, replaces params.p_s for this single
            invocation. Must be in (P_S_CLAMP_MIN, P_S_CLAMP_MAX). Pass the
            output of parameters.derive_p_s_from_irt(theta, b, a) here.
        response_time_ms : int | None
            Time the student spent on this question (milliseconds). Used by
            the guess-detection heuristic when combined with theta/difficulty.
        theta : float | None
            Student IRT ability. Required for IRT-aware guess detection.
        difficulty : float | None
            Item IRT difficulty (b). Required for IRT-aware guess detection.
        discrimination : float | None
            Item IRT discrimination (a). Optional; falls back to reference a.
        mastery_threshold : float | None
            Override the mastery threshold for this update. If None, uses the
            threshold stored on the incoming `state` (state.mastery_threshold_used).

        Returns
        -------
        BKTUpdateResult
            Immutable audit record containing new_state and all intermediates.

        Raises
        ------
        InvalidStateError
            If state.p_l is outside the valid open interval.
        InvalidParameterError
            If override_p_g or override_p_s are outside their clamp ranges,
            or if P(G)+P(S) would become degenerate after overrides.
        """
        # ── Validate state ────────────────────────────────────────────────────
        if not (P_L_CLAMP_MIN <= state.p_l <= P_L_CLAMP_MAX):
            raise InvalidStateError(
                f"state.p_l={state.p_l:.6f} is outside [{P_L_CLAMP_MIN}, {P_L_CLAMP_MAX}]. "
                "The state may have been constructed outside BKTEngine."
            )

        # ── Resolve effective threshold ────────────────────────────────────────
        effective_threshold = (
            mastery_threshold
            if mastery_threshold is not None
            else state.mastery_threshold_used
        )

        # ── Resolve effective P(G) ─────────────────────────────────────────────
        override_p_g_applied = False
        if override_p_g is not None:
            if not (P_G_CLAMP_MIN <= override_p_g <= P_G_CLAMP_MAX):
                raise InvalidParameterError(
                    f"override_p_g={override_p_g:.4f} is outside the valid range "
                    f"[{P_G_CLAMP_MIN}, {P_G_CLAMP_MAX}]."
                )
            p_g = override_p_g
            override_p_g_applied = True
        else:
            p_g = params.p_g

        # ── Resolve effective P(S) ─────────────────────────────────────────────
        override_p_s_applied = False
        if override_p_s is not None:
            if not (P_S_CLAMP_MIN <= override_p_s <= P_S_CLAMP_MAX):
                raise InvalidParameterError(
                    f"override_p_s={override_p_s:.4f} is outside the valid range "
                    f"[{P_S_CLAMP_MIN}, {P_S_CLAMP_MAX}]."
                )
            p_s = override_p_s
            override_p_s_applied = True
        else:
            p_s = params.p_s

        # ── Guess detection ────────────────────────────────────────────────────
        # Must run BEFORE the update so the inflated P(G) is used in Stage 1.
        flagged_guess = is_probable_guess(
            is_correct=is_correct,
            response_time_ms=response_time_ms,
            theta=theta,
            difficulty=difficulty,
            discrimination=discrimination,
        )
        if flagged_guess:
            # Inflate P(G) to dampen the upward update for this suspicious answer.
            # The inflated value comes from config (P_G_INFLATED = DEFAULT_P_G × multiplier),
            # clamped so it can never produce a degenerate model.
            p_g = min(derive_p_g_inflated(), P_G_CLAMP_MAX)

        # ── Degeneracy guard after all overrides are resolved ─────────────────
        if p_g + p_s >= 1.0:
            raise InvalidParameterError(
                f"Resolved P(G) + P(S) >= 1.0 after overrides: "
                f"P(G)={p_g:.4f}, P(S)={p_s:.4f}. "
                "This makes the BKT model uninformative. Adjust overrides or config."
            )

        p_l_prev = state.p_l
        p_t = params.p_t

        # ── Stage 1: Conditional update (Bayes) ───────────────────────────────
        if is_correct:
            # P(correct | L) = 1 − P(S)   /   P(correct | ¬L) = P(G)
            numerator = p_l_prev * (1.0 - p_s)
            denominator = numerator + (1.0 - p_l_prev) * p_g
        else:
            # P(incorrect | L) = P(S)   /   P(incorrect | ¬L) = 1 − P(G)
            numerator = p_l_prev * p_s
            denominator = numerator + (1.0 - p_l_prev) * (1.0 - p_g)

        # denominator can only be 0 if p_l_prev is exactly 0 or 1 AND the
        # parameters are degenerate — both already caught above, but guard anyway.
        if denominator <= 0.0:
            raise InvalidStateError(
                "BKT conditional update denominator is zero or negative — "
                f"p_l={p_l_prev:.6f}, p_g={p_g:.4f}, p_s={p_s:.4f}. "
                "This should have been caught by parameter validation."
            )

        p_conditional = numerator / denominator

        # ── Stage 2: Learning transition ──────────────────────────────────────
        p_l_new_raw = p_conditional + (1.0 - p_conditional) * p_t

        # ── Clamp ─────────────────────────────────────────────────────────────
        p_l_new = max(P_L_CLAMP_MIN, min(P_L_CLAMP_MAX, p_l_new_raw))

        # ── Mastery adjudication ──────────────────────────────────────────────
        mastered = check_mastery(p_l_new, effective_threshold)

        # ── Build new state ───────────────────────────────────────────────────
        now = datetime.now(timezone.utc)
        new_state = BKTState(
            student_id=state.student_id,
            skill_id=state.skill_id,
            p_l=p_l_new,
            attempt_count=state.attempt_count + 1,
            correct_count=state.correct_count + (1 if is_correct else 0),
            is_mastered=mastered,
            last_updated=now,
            mastery_threshold_used=effective_threshold,
        )

        return BKTUpdateResult(
            new_state=new_state,
            previous_p_l=p_l_prev,
            p_conditional=p_conditional,
            p_g_used=p_g,
            p_s_used=p_s,
            p_t_used=p_t,
            was_flagged_guess=flagged_guess,
            override_p_g_applied=override_p_g_applied,
            override_p_s_applied=override_p_s_applied,
            is_correct=is_correct,
            timestamp=now,
        )

    @staticmethod
    def batch_update(
        state: BKTState,
        observations: list[tuple[bool, Optional[int]]],
        params: BKTSkillParameters,
        *,
        irt_overrides: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
        mastery_threshold: Optional[float] = None,
    ) -> tuple[BKTState, list[BKTUpdateResult]]:
        """Apply a sequence of observations in order, threading the state.

        Useful for replaying a student's full session history or seeding
        a new student's state from historical quiz data.

        Parameters
        ----------
        state : BKTState
            Starting state.
        observations : list of (is_correct, response_time_ms | None)
            Ordered sequence of observations to apply.
        params : BKTSkillParameters
            Skill parameters applied to every observation. Per-observation
            IRT overrides go in `irt_overrides`.
        irt_overrides : list of (override_p_g | None, override_p_s | None) | None
            Per-observation IRT-derived P(G) and P(S) overrides. If provided,
            must have the same length as `observations`. Use None for either
            element if no override for that observation.
        mastery_threshold : float | None
            Applied uniformly across all updates.

        Returns
        -------
        (final_state, list_of_results)
        """
        if irt_overrides is not None and len(irt_overrides) != len(observations):
            raise InvalidParameterError(
                f"irt_overrides length ({len(irt_overrides)}) must match "
                f"observations length ({len(observations)})."
            )

        current_state = state
        results: list[BKTUpdateResult] = []

        for i, (is_correct, rt_ms) in enumerate(observations):
            pg_ov, ps_ov = (None, None) if irt_overrides is None else irt_overrides[i]
            result = BKTEngine.update(
                current_state,
                is_correct,
                params,
                override_p_g=pg_ov,
                override_p_s=ps_ov,
                response_time_ms=rt_ms,
                mastery_threshold=mastery_threshold,
            )
            current_state = result.new_state
            results.append(result)

        return current_state, results


__all__ = [
    "BKTUpdateResult",
    "BKTEngine",
]
