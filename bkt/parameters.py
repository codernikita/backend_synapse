"""
parameters.py — BKTSkillParameters dataclass and the IRT → BKT parameter
derivation bridge.

Responsibility
--------------
This module does two things:

1. Defines BKTSkillParameters: the four canonical BKT parameters (P_L0,
   P_T, P_G, P_S) for a single skill. This is the only object the BKT
   engine (engine.py) ever reads parameters from. engine.py never imports
   IRT types or calls any IRT function — the bridge lives here, not there.

2. Provides factory functions that DERIVE those four parameters from IRT
   data (theta, a, b) using documented mathematical formulas. These are
   optional: callers without IRT data can construct BKTSkillParameters
   directly using the config defaults.

Why IRT parameters change P(G) and P(S) — mathematical rationale
-----------------------------------------------------------------
The classical BKT model treats P(G) and P(S) as fixed skill-level
constants, which means the model assumes a student's chance of guessing
correctly is identical whether the question is a trivially easy recall
item or a hard synthesis problem. That assumption is fine when all items
in a skill have similar difficulty, but it breaks down across our
heterogeneous item bank, where the same skill is probed by items spanning
multiple Bloom levels.

The 2PL model already gives us a principled, per-item estimate of how
likely a student at ability θ is to answer item (a, b) correctly:

    P_irt(correct | θ, a, b) = σ(a(θ - b))   where σ is the logistic sigmoid.

We use this as a *moderating signal*:

  P(G)_effective = P_G_IRT_BASE × (1 − P_irt) + P_G_IRT_FLOOR
    ↑ When P_irt → 1 (easy item for this student), effective P(G) → floor.
      A student who would almost certainly get this item right without BKT
      knowledge has essentially no "guess" component for this observation.
    ↑ When P_irt → 0 (hard item), effective P(G) → base default.
      A student with very low expected P(correct) is essentially guessing
      even if they do get it right.

  P(S)_effective = P_S_IRT_BASE × (1 − P_irt) + P_S_IRT_FLOOR
    ↑ Same structure: a high-ability student on an easy item has a lower
      slip rate (less likely to make a careless error on something routine).
    ↑ A low-ability student on a hard item has a higher effective slip rate
      (even if they "know" it they're more likely to err under difficulty).

Both formulas are linear interpolations in (1 − P_irt), which:
  - Is monotonically decreasing in P_irt (higher IRT prediction → lower param)
  - Maps [0, 1] → [floor, base] (bounded by construction)
  - Degrades gracefully to the base default when P_irt = 0
  - Is analytically simple — no extra hyperparameters beyond the IRT params

P(T) difficulty scaling:
  P(T) = DEFAULT_P_T × sigmoid(−mean_b)
where mean_b is the mean IRT difficulty of items probing this concept.
  - mean_b < 0 (easy concept): sigmoid(−mean_b) > 0.5 → faster learning
  - mean_b = 0 (neutral): sigmoid(0) = 0.5 → P(T) = DEFAULT_P_T × 0.5
    NOTE: We normalize by dividing by sigmoid(0)=0.5 to keep neutral=default:
    P(T)_scaled = DEFAULT_P_T × sigmoid(-mean_b) / sigmoid(0.0)
               = DEFAULT_P_T × 2 × sigmoid(-mean_b)
  - mean_b > 0 (hard concept): sigmoid(−mean_b) < 0.5 → slower learning

P(L0) from IRT:
  Uses the same Beta-Bernoulli shrinkage blend as the IRT engine's own
  mastery_initializer.py — the initial mastery output from that module is
  the direct P(L0). If calling without the full IRT pipeline, this module
  provides `seed_p_l0_from_irt()` which replicates the identical formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .config import (
    DEFAULT_P_G,
    DEFAULT_P_L0,
    DEFAULT_P_S,
    DEFAULT_P_T,
    LOGISTIC_EXPONENT_CLAMP,
    MASTERY_PRIOR_STRENGTH,
    MASTERY_REFERENCE_DISCRIMINATION,
    P_G_CLAMP_MAX,
    P_G_CLAMP_MIN,
    P_G_INFLATED,
    P_G_IRT_BASE,
    P_G_IRT_FLOOR,
    P_S_CLAMP_MAX,
    P_S_CLAMP_MIN,
    P_S_IRT_BASE,
    P_S_IRT_FLOOR,
    P_T_CLAMP_MAX,
    P_T_CLAMP_MIN,
    P_T_DIFFICULTY_SCALING_ENABLED,
    SEED_PRIOR_MAX,
    SEED_PRIOR_MIN,
    _sigmoid,
)
from .exceptions import InvalidParameterError, IRTParameterRangeError


# ── Core 2PL ICC ─────────────────────────────────────────────────────────────


def irt_p_correct(theta: float, b: float, a: float = MASTERY_REFERENCE_DISCRIMINATION) -> float:
    """2PL Item Characteristic Curve: P(correct | θ, a, b) = 1 / (1 + exp(−a(θ − b))).

    This is the ONLY place in the BKT engine where the 2PL formula is
    defined — engine.py and mastery.py reuse this function rather than
    re-implementing the sigmoid. Mirrors theta.py's `probability_correct()`
    in the IRT engine so both modules are always mathematically identical.

    The formula matches the IRT engine's theta.py exactly:
        z = -a * (theta - b)
        P = 1 / (1 + exp(z))   [computed in numerically stable form]
    So when theta >> b: z → -inf → P → 1 (high ability, easy item → correct).
    When theta << b: z → +inf → P → 0 (low ability, hard item → incorrect).

    Parameters
    ----------
    theta : float
        Student ability on the logit scale (from IRT engine ThetaResult.theta).
    b : float
        Item difficulty on the same logit scale (QuestionIRTParameters.difficulty).
    a : float
        Item discrimination (QuestionIRTParameters.discrimination). Defaults to
        MASTERY_REFERENCE_DISCRIMINATION (= 1.0) when no per-item value exists.

    Raises
    ------
    IRTParameterRangeError
        if `a` <= 0 (non-positive discrimination reverses the ICC and produces
        a decreasing curve, which is psychometrically nonsensical for a scored item),
        or if theta or b are non-finite.
    """
    if not math.isfinite(theta):
        raise IRTParameterRangeError(f"theta must be finite, got {theta!r}")
    if not math.isfinite(b):
        raise IRTParameterRangeError(f"difficulty b must be finite, got {b!r}")
    if a <= 0:
        raise IRTParameterRangeError(
            f"discrimination a must be > 0, got {a!r}. "
            "A non-positive discrimination reverses the item characteristic curve."
        )
    # z = -a*(theta - b); numerically stable: avoid exp(large positive)
    z = -a * (theta - b)
    z_clamped = max(-LOGISTIC_EXPONENT_CLAMP, min(LOGISTIC_EXPONENT_CLAMP, z))
    if z_clamped >= 0:
        # z >= 0 means theta <= b (student below difficulty): P(correct) < 0.5
        # Use: 1/(1+exp(z)) = exp(-z)/(exp(-z)+1) — but simpler: direct formula
        ez = math.exp(-z_clamped)
        return ez / (1.0 + ez)
    else:
        # z < 0 means theta > b (student above difficulty): P(correct) > 0.5
        ez = math.exp(z_clamped)
        return 1.0 / (1.0 + ez)


# ── BKTSkillParameters ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BKTSkillParameters:
    """The four canonical BKT parameters for a single skill.

    This dataclass is the ONLY source of parameter information that
    engine.py reads. It does not know whether the values were derived from
    IRT data or set directly — that distinction belongs to the factory
    functions below.

    Fields
    ------
    p_l0 : float
        P(L_0) — prior probability the student knows the skill before any
        observation. In [SEED_PRIOR_MIN, SEED_PRIOR_MAX].
    p_t : float
        P(T) — probability of transitioning from "not known" to "known"
        after a single practice opportunity. In (P_T_CLAMP_MIN, P_T_CLAMP_MAX).
    p_g : float
        P(G) — probability a student who does NOT know the skill answers
        correctly (guess rate). In [P_G_CLAMP_MIN, P_G_CLAMP_MAX].
    p_s : float
        P(S) — probability a student who DOES know the skill answers
        incorrectly (slip rate). In [P_S_CLAMP_MIN, P_S_CLAMP_MAX].
    skill_id : str | None
        Optional skill/concept identifier for logging purposes.
    irt_derived : bool
        True when values were computed from IRT parameters; False when
        config defaults were used. Kept for audit/logging.
    """

    p_l0: float
    p_t: float
    p_g: float
    p_s: float
    skill_id: Optional[str] = None
    irt_derived: bool = False

    def __post_init__(self) -> None:
        """Validate all parameter constraints on construction."""
        for name, val in [("p_l0", self.p_l0), ("p_t", self.p_t),
                          ("p_g", self.p_g), ("p_s", self.p_s)]:
            if not (0.0 < val < 1.0):
                raise InvalidParameterError(
                    f"BKTSkillParameters.{name} must be in the open interval (0, 1), "
                    f"got {val:.6f}. Values at exactly 0 or 1 make the BKT update "
                    "equations undefined."
                )

        # Degeneracy check: if P(G) + P(S) >= 1, then for a student who KNOWS
        # the skill the probability of a correct answer (1 - P_S) is no greater
        # than for a student who does NOT know it (P_G). The model becomes
        # uninformative or inverted.
        if self.p_g + self.p_s >= 1.0:
            raise InvalidParameterError(
                f"P(G) + P(S) must be < 1.0 to keep the BKT model identifiable. "
                f"Got P(G)={self.p_g:.4f} + P(S)={self.p_s:.4f} = {self.p_g + self.p_s:.4f}. "
                "When P(G) + P(S) >= 1, a correct answer provides no evidence of mastery."
            )

        if self.p_t <= 0:
            raise InvalidParameterError(
                f"P(T) must be > 0 (got {self.p_t:.6f}); a zero learn-rate creates "
                "an absorbing non-mastery state that can never be escaped by any observation."
            )


def defaults(skill_id: Optional[str] = None) -> BKTSkillParameters:
    """Construct a BKTSkillParameters from pure config defaults — no IRT data.

    Use this only when no IRT information is available at all (e.g. before
    the diagnostic quiz has been taken). All four parameters come directly
    from config.py with their documented derivations.
    """
    return BKTSkillParameters(
        p_l0=DEFAULT_P_L0,
        p_t=DEFAULT_P_T,
        p_g=DEFAULT_P_G,
        p_s=DEFAULT_P_S,
        skill_id=skill_id,
        irt_derived=False,
    )


# ── IRT → P(G) and P(S) per-invocation derivation ───────────────────────────


def derive_p_g_from_irt(
    theta: float,
    b: float,
    a: float = MASTERY_REFERENCE_DISCRIMINATION,
) -> float:
    """Derive an IRT-informed effective guess probability for a single item.

    Formula:
        P(G)_eff = P_G_IRT_BASE × (1 − P_irt) + P_G_IRT_FLOOR

    where P_irt = irt_p_correct(θ, a, b).

    When P_irt → 1 (student is very likely to answer correctly under IRT),
    effective P(G) → P_G_IRT_FLOOR (minimal guessing component).
    When P_irt → 0 (very hard for this student), effective P(G) → base default.

    Result is clamped to [P_G_CLAMP_MIN, P_G_CLAMP_MAX].
    """
    p_irt = irt_p_correct(theta, b, a)
    p_g_eff = P_G_IRT_BASE * (1.0 - p_irt) + P_G_IRT_FLOOR
    return max(P_G_CLAMP_MIN, min(P_G_CLAMP_MAX, p_g_eff))


def derive_p_s_from_irt(
    theta: float,
    b: float,
    a: float = MASTERY_REFERENCE_DISCRIMINATION,
) -> float:
    """Derive an IRT-informed effective slip probability for a single item.

    Formula:
        P(S)_eff = P_S_IRT_BASE × (1 − P_irt) + P_S_IRT_FLOOR

    When P_irt → 1 (easy item for this student), effective P(S) → floor.
    When P_irt → 0 (hard item), effective P(S) → base default.

    Result is clamped to [P_S_CLAMP_MIN, P_S_CLAMP_MAX].
    """
    p_irt = irt_p_correct(theta, b, a)
    p_s_eff = P_S_IRT_BASE * (1.0 - p_irt) + P_S_IRT_FLOOR
    return max(P_S_CLAMP_MIN, min(P_S_CLAMP_MAX, p_s_eff))


def derive_p_g_inflated() -> float:
    """Return the inflated guess probability for guess-flagged observations.

    The value is read from config (P_G_INFLATED = DEFAULT_P_G × P_G_INFLATED_MULTIPLIER)
    and clamped. This function exists so callsites never reference config
    directly — all parameter access is through parameters.py.
    """
    return max(P_G_CLAMP_MIN, min(P_G_CLAMP_MAX, P_G_INFLATED))


# ── IRT → P(T) difficulty scaling ────────────────────────────────────────────


def derive_p_t_from_difficulty(mean_b: float) -> float:
    """Derive a difficulty-scaled learn rate for a skill.

    Formula:
        P(T) = DEFAULT_P_T × sigmoid(−mean_b) / sigmoid(0.0)

    Here sigmoid(x) = 1/(1+exp(-x)), so:
        sigmoid(−mean_b) = 1/(1+exp(mean_b))

    Division by sigmoid(0.0) = 0.5 normalizes so that mean_b = 0 (neutral
    difficulty) preserves the exact DEFAULT_P_T value:
        mean_b = 0  → sigmoid(0)/sigmoid(0) = 1  → P(T) = DEFAULT_P_T  ✓
        mean_b < 0  → sigmoid(|mean_b|) > 0.5   → ratio > 1 → P(T) > DEFAULT_P_T  ✓ (easier → faster)
        mean_b > 0  → sigmoid(-mean_b) < 0.5    → ratio < 1 → P(T) < DEFAULT_P_T  ✓ (harder → slower)

    The _sigmoid in config computes 1/(1+exp(-x)), so _sigmoid(-mean_b)
    = 1/(1+exp(mean_b)), which is correctly > 0.5 when mean_b < 0.

    If P_T_DIFFICULTY_SCALING_ENABLED is False in config, returns DEFAULT_P_T.

    Result is clamped to [P_T_CLAMP_MIN, P_T_CLAMP_MAX].
    """
    if not P_T_DIFFICULTY_SCALING_ENABLED:
        return DEFAULT_P_T
    sigmoid_neutral = _sigmoid(0.0)  # = 0.5 exactly, but computed, not hard-coded
    # _sigmoid(-mean_b): for mean_b < 0, arg is positive → result > 0.5 → P(T) scales up ✓
    scaled = DEFAULT_P_T * _sigmoid(-mean_b) / sigmoid_neutral
    return max(P_T_CLAMP_MIN, min(P_T_CLAMP_MAX, scaled))


# ── IRT → P(L0) cold-start seeding ──────────────────────────────────────────


def seed_p_l0_from_irt(
    theta: float,
    bloom_difficulties: Sequence[float],
    n_correct: int,
    n_attempted: int,
    a_ref: float = MASTERY_REFERENCE_DISCRIMINATION,
    prior_strength: float = MASTERY_PRIOR_STRENGTH,
) -> float:
    """Compute IRT-seeded P(L0) using the same Beta-Bernoulli shrinkage blend
    as the IRT engine's mastery_initializer.py.

    This function is a standalone replica of that module's core formula,
    callable without the full IRT pipeline. Use it when the orchestrator has
    already run the IRT engine and passes the outputs directly.

    Formula (mirrors mastery_initializer.initialize_mastery):
        theta_implied = mean_over_items[ irt_p_correct(theta, b_i, a_ref) ]
        observed      = n_correct / n_attempted   (or theta_implied if n=0)
        weight        = n_attempted / (n_attempted + prior_strength)
        p_l0          = clamp( weight × observed + (1−weight) × theta_implied )

    Parameters
    ----------
    theta : float
        Student IRT ability from ThetaResult.theta.
    bloom_difficulties : Sequence[float]
        IRT difficulty values (b) for each item that probed this concept.
        These come from QuestionIRTParameters.difficulty, already mapped from
        Bloom level by bloom_mapper.py in the IRT engine.
    n_correct : int
        Number of correct responses for this concept in the diagnostic.
    n_attempted : int
        Total responses for this concept. Must be >= n_correct.
    a_ref : float
        Reference discrimination for the theta-implied accuracy projection.
        Defaults to MASTERY_REFERENCE_DISCRIMINATION (= 1.0).
    prior_strength : float
        Effective sample size K of the theta-implied prior.
        Defaults to MASTERY_PRIOR_STRENGTH (= 3.0).

    Returns
    -------
    float
        P(L0) clamped to [SEED_PRIOR_MIN, SEED_PRIOR_MAX].
    """
    if not bloom_difficulties:
        raise IRTParameterRangeError(
            "bloom_difficulties must be non-empty to seed P(L0) from IRT. "
            "Pass at least one item difficulty value for this concept."
        )
    if n_attempted < 0:
        raise IRTParameterRangeError(f"n_attempted must be >= 0, got {n_attempted}")
    if n_correct < 0 or n_correct > n_attempted:
        raise IRTParameterRangeError(
            f"n_correct must be in [0, n_attempted], got n_correct={n_correct}, "
            f"n_attempted={n_attempted}"
        )

    # Theta-implied accuracy: mean 2PL P(correct) at this student's theta over
    # all items that probed this concept, using the reference discrimination.
    theta_implied = sum(
        irt_p_correct(theta, b, a_ref) for b in bloom_difficulties
    ) / len(bloom_difficulties)

    # Observed accuracy (falls back to theta-implied when no attempts recorded)
    observed = n_correct / n_attempted if n_attempted > 0 else theta_implied

    # Bayesian shrinkage weight
    weight = n_attempted / (n_attempted + prior_strength)

    blended = weight * observed + (1.0 - weight) * theta_implied
    return max(SEED_PRIOR_MIN, min(SEED_PRIOR_MAX, blended))


# ── Full IRT-derived parameter set ────────────────────────────────────────────


def from_irt(
    theta: float,
    b: float,
    a: float = MASTERY_REFERENCE_DISCRIMINATION,
    mean_b: Optional[float] = None,
    p_l0: Optional[float] = None,
    skill_id: Optional[str] = None,
) -> BKTSkillParameters:
    """Construct a fully IRT-derived BKTSkillParameters for a skill/item.

    This is the primary factory function the orchestrator calls when IRT
    data is available. All four parameters are computed from IRT inputs:

    - P(L0): use `p_l0` if provided (from mastery_initializer output or
      `seed_p_l0_from_irt()`); otherwise falls back to DEFAULT_P_L0.
    - P(T): difficulty-scaled from `mean_b` if provided; else from `b`.
    - P(G): derived from irt_p_correct(theta, b, a) via derive_p_g_from_irt().
    - P(S): derived from irt_p_correct(theta, b, a) via derive_p_s_from_irt().

    Parameters
    ----------
    theta : float
        Student IRT ability.
    b : float
        Item difficulty (used for P(G) and P(S) derivation).
    a : float
        Item discrimination.
    mean_b : float | None
        Mean difficulty of ALL items for this skill (used for P(T) scaling).
        If None, the single item's `b` is used as an approximation.
    p_l0 : float | None
        Pre-computed P(L0) (e.g. from mastery_initializer). If None,
        falls back to DEFAULT_P_L0.
    skill_id : str | None
        Optional identifier for logging.

    Returns
    -------
    BKTSkillParameters
        Fully IRT-derived, validated parameter set.
    """
    effective_p_l0 = max(SEED_PRIOR_MIN, min(SEED_PRIOR_MAX, p_l0)) \
        if p_l0 is not None else DEFAULT_P_L0
    effective_mean_b = mean_b if mean_b is not None else b

    return BKTSkillParameters(
        p_l0=effective_p_l0,
        p_t=derive_p_t_from_difficulty(effective_mean_b),
        p_g=derive_p_g_from_irt(theta, b, a),
        p_s=derive_p_s_from_irt(theta, b, a),
        skill_id=skill_id,
        irt_derived=True,
    )


__all__ = [
    "irt_p_correct",
    "BKTSkillParameters",
    "defaults",
    "derive_p_g_from_irt",
    "derive_p_s_from_irt",
    "derive_p_g_inflated",
    "derive_p_t_from_difficulty",
    "seed_p_l0_from_irt",
    "from_irt",
]
