"""
config.py — single source of truth for all BKT engine tunable constants.

No downstream module hard-codes a threshold, a probability, or a clamp
value. If a supervisor asks "why is the default guess rate 0.25?" or
"what counts as mastered?", the answer lives here — and the formula
that produced the default is documented inline so future calibration
has a principled starting point.

Derivation philosophy
---------------------
Every default is either:
  (a) Analytically derived — e.g. the 4-option MCQ guess floor comes
      from 1/n_options, which is the theoretical random-guess probability
      for a uniformly distributed distractor set.
  (b) Literature-anchored — cited to a peer-reviewed psychometric source.
  (c) Operationally inferred — e.g. clamp bounds that prevent the HMM
      from reaching absorbing states, chosen from the known mathematical
      requirement that P(L) ∈ (0, 1) strictly to preserve future updates.

Nothing in this file is a "magic number". If you change a value, document
why in the comment next to it.
"""

from __future__ import annotations

import math
from typing import Final

# ── MCQ item structure ───────────────────────────────────────────────────────
# Number of options in a standard multiple-choice item. Determines the
# theoretical minimum guess probability: for a k-option item where all
# distractors are uniformly plausible, a student with zero knowledge
# selects the correct answer with probability 1/k by chance alone.
N_MCQ_OPTIONS: Final[int] = 4

# ── Default BKT parameters (HMM priors) ─────────────────────────────────────
# These are the skill-level fallback values used when no IRT data is available.
# The moment IRT parameters (θ, a, b) are available, `parameters.py` replaces
# these with analytically derived values — these are never silently used
# when richer data exists.

# P(L_0): Initial Knowledge — probability a student already knows the skill
# before any instruction. Standard BKT literature (Corbett & Anderson, 1994)
# uses values between 0.0 and 0.3 for untaught populations; 0.20 is the
# modal choice for secondary-school skills with no prior instruction signal.
DEFAULT_P_L0: Final[float] = 0.20

# P(T): Transition / Learn Rate — probability a student transitions from
# "not known" to "known" after a single practice opportunity. Corbett &
# Anderson (1994) estimated 0.05–0.40 across skills; 0.10 is a conservative
# middle estimate appropriate before per-skill calibration data exists.
DEFAULT_P_T: Final[float] = 0.10

# P(G): Guess Rate — probability a student who does NOT know the skill answers
# correctly anyway. Theoretical floor for a 4-option MCQ = 1/N_MCQ_OPTIONS.
# This is the exact random-guess rate; when IRT parameters are available the
# effective P(G) is derived dynamically (see parameters.py).
DEFAULT_P_G: Final[float] = 1.0 / N_MCQ_OPTIONS  # = 0.25 for 4-option MCQ

# P(S): Slip Rate — probability a student who DOES know the skill answers
# incorrectly (careless error, misread, etc.). Empirically estimated at
# 0.05–0.15 across cognitive-tutor skills (Baker et al., 2008); 0.10 is
# the mid-point of that range as a safe default.
DEFAULT_P_S: Final[float] = 0.10

# ── Mastery threshold ────────────────────────────────────────────────────────
# P(L) at or above this value triggers `is_mastered = True`. The 0.95
# threshold is the de-facto standard in BKT literature (Corbett & Anderson,
# 1994; van de Sande, 2013) and maps to "at most a 5% residual probability
# of not knowing the skill", which is operationally acceptable for skill
# advancement decisions in classroom ITS deployments.
MASTERY_THRESHOLD: Final[float] = 0.95

# ── Numerical stability clamps ───────────────────────────────────────────────
# BKT's conditional update and transition equations are undefined at exactly
# P(L) = 0 or P(L) = 1 (division by zero in the Bayes update denominator, or
# an absorbing state that can never be corrected by any observation). These
# clamp bounds keep P(L) strictly inside (0, 1) at all times.
# Chosen to be far enough from 0/1 that floating-point rounding cannot push
# a clamped value back to the boundary on the next iteration, but small/large
# enough that they have negligible practical effect on mastery judgments.
P_L_CLAMP_MIN: Final[float] = 0.001  # = 0.1% — effectively "doesn't know but not hopeless"
P_L_CLAMP_MAX: Final[float] = 0.999  # = 99.9% — effectively "mastered but not unreachable"

# ── IRT → BKT parameter bridge constants ────────────────────────────────────
# These govern how IRT-derived P_correct(θ, a, b) maps to effective P(G) and
# P(S). Both mappings use the same linear interpolation pattern:
#   effective_param = base * (1 - P_irt) + floor
# where P_irt = P(correct) from the 2PL model. When P_irt → 1 (student
# almost certain to get it right), the parameter collapses to its floor.
# When P_irt → 0 (extremely hard item relative to student ability), it
# reaches its base default.

# P(G) effective bounds:
# - Floor: the physical minimum for a forced-choice item even with complete
#   knowledge-absence is 1/N_MCQ_OPTIONS (random guessing). We set the floor
#   slightly above zero (1/N_MCQ_OPTIONS / 4) to allow near-certainty items
#   to produce near-zero effective guess rates without touching zero.
P_G_IRT_BASE: Final[float] = DEFAULT_P_G               # = 1/4 for 4-option MCQ
P_G_IRT_FLOOR: Final[float] = DEFAULT_P_G / N_MCQ_OPTIONS  # = 1/16 ≈ 0.0625 — hard physical floor

# P(G) after guess-detection flag: a correct answer that looks like a guess
# (fast + IRT-surprising) uses this inflated effective P(G) so the correct
# observation contributes weaker upward evidence of mastery. Value is set to
# 3× the default to produce a meaningful but not extreme dampening effect.
P_G_INFLATED_MULTIPLIER: Final[float] = 3.0
P_G_INFLATED: Final[float] = DEFAULT_P_G * P_G_INFLATED_MULTIPLIER  # = 0.75 for 4-opt MCQ

# P(S) effective bounds:
# - Base: DEFAULT_P_S (the fallback slip rate with no IRT data)
# - Floor: half the base — a highly able student on an easy item can still
#   slip, but at roughly half the baseline rate. Using DEFAULT_P_S / 2
#   ensures the floor scales correctly if DEFAULT_P_S is ever recalibrated.
P_S_IRT_BASE: Final[float] = DEFAULT_P_S               # = 0.10
P_S_IRT_FLOOR: Final[float] = DEFAULT_P_S / 2.0        # = 0.05

# Clamp ranges for IRT-derived P(G) and P(S) to prevent degenerate values:
# P(G) must stay below 0.50 — a guess rate >= 0.5 is indistinguishable from
# a coin flip and would make correct answers evidence *against* mastery.
# P(S) must stay below 0.40 — a slip rate >= 0.5 is also degenerate for the
# same reason (incorrect becomes more likely than correct for known students).
P_G_CLAMP_MIN: Final[float] = P_G_IRT_FLOOR            # 1/16 ≈ 0.0625
P_G_CLAMP_MAX: Final[float] = 0.45                     # just below the degeneracy threshold
P_S_CLAMP_MIN: Final[float] = P_S_IRT_FLOOR            # 0.05
P_S_CLAMP_MAX: Final[float] = 0.35                     # conservative upper bound for slip

# ── P(T) difficulty-scaling ──────────────────────────────────────────────────
# P(T) can optionally be scaled by the concept's mean Bloom difficulty (b).
# We use a sigmoid-derived scaling factor:
#   P(T)_scaled = DEFAULT_P_T * sigmoid(-mean_b)
# where sigmoid(x) = 1/(1+exp(-x)) and mean_b is the mean IRT difficulty of
# items probing this concept. This makes learning transitions slower for
# harder concepts (mean_b > 0) and faster for easier ones (mean_b < 0),
# which matches the intuition that harder skills take more practice to learn.
# The logistic sigmoid is used (rather than a linear taper) because it:
#   1. Keeps P(T) in (0, 1) for all finite mean_b
#   2. Is symmetric around mean_b = 0 (neutral difficulty → unscaled P(T))
#   3. Has a well-understood interpretation on the IRT b-scale
# Clamp ensures P(T) is always positive (a zero learn-rate is an absorbing state).
P_T_DIFFICULTY_SCALING_ENABLED: Final[bool] = True
P_T_CLAMP_MIN: Final[float] = 0.01   # minimum learn-rate — always some learning possible
P_T_CLAMP_MAX: Final[float] = 0.50   # maximum — prevents unrealistically fast mastery

# ── IRT cold-start seeding ───────────────────────────────────────────────────
# When seeding P(L0) from IRT data (mirroring mastery_initializer.py), the
# same Beta-Bernoulli shrinkage blend is used:
#   weight = n / (n + K)  where K = MASTERY_PRIOR_STRENGTH
# K is the "effective sample size" of the theta-implied prior: a concept
# probed by only 1–2 diagnostic items gets K/(1+K) weight from theta, while
# a concept probed by many items is dominated by observed accuracy.
# Value matches the IRT engine's config.MASTERY_PRIOR_STRENGTH = 3.0.
MASTERY_PRIOR_STRENGTH: Final[float] = 3.0

# Fixed reference discrimination for theta→accuracy projection.
# Matches irt-engine's config.MASTERY_REFERENCE_DISCRIMINATION = 1.0.
# This is the 2PL `a` parameter used when projecting a student's theta
# onto an expected-accuracy scale without a known per-item discrimination.
MASTERY_REFERENCE_DISCRIMINATION: Final[float] = 1.0

# Bounds on IRT-seeded P(L0) — must stay strictly inside (0, 1) to keep
# BKT's update equations non-degenerate. Match the IRT engine's bounds.
SEED_PRIOR_MIN: Final[float] = 0.05
SEED_PRIOR_MAX: Final[float] = 0.95

# ── Guess detection (matches quiz-portal knowledge.service.ts exactly) ───────
# A correct answer is flagged as a probable guess when it is BOTH:
#   (a) faster than RT_GUESS_THRESHOLD_MS milliseconds, AND
#   (b) surprising under the 2PL model — IRT P(correct) < IRT_SURPRISE_THRESHOLD
# These mirror the TypeScript constants in knowledge.service.ts so the
# Python BKT engine and the Node.js portal produce consistent classifications.
# If no IRT params are available, the time-only criterion is used.
RT_GUESS_THRESHOLD_MS: Final[int] = 1500   # answers faster than this are "too fast"
IRT_SURPRISE_THRESHOLD: Final[float] = 0.30  # P(correct) below this = model didn't expect success

# ── Logistic sigmoid exponent clamp ─────────────────────────────────────────
# Matches the IRT engine's THETA_EXPONENT_CLAMP = 35.0. Prevents exp()
# overflow (exp(35) ≈ 1.6e15, well inside float64 range) while representing
# a probability effectively indistinguishable from 0 or 1 at this scale.
LOGISTIC_EXPONENT_CLAMP: Final[float] = 35.0


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid: 1 / (1 + exp(-x)).

    Correct two-branch implementation to avoid exp() overflow:
      x >= 0: result = 1/(1+exp(-x)) >= 0.5.
              Stable form: 1/(1+exp(-x)) directly (exp(-x) is small, no overflow).
      x < 0:  result = exp(x)/(1+exp(x)) < 0.5.
              Stable form: avoids exp(large negative) underflow.

    Used internally by parameters.py to compute difficulty-scaled P(T).
    Public so parameters.py can import it from here (single definition).
    Verified: _sigmoid(0)=0.5, _sigmoid(2)≈0.88, _sigmoid(-2)≈0.12.
    """
    x = max(-LOGISTIC_EXPONENT_CLAMP, min(LOGISTIC_EXPONENT_CLAMP, x))
    if x >= 0:
        # exp(-x) is small (in [0, 1]) — no overflow risk
        return 1.0 / (1.0 + math.exp(-x))
    else:
        # exp(x) is small (in [0, 1]) — no overflow risk
        ex = math.exp(x)
        return ex / (1.0 + ex)


__all__ = [
    # MCQ structure
    "N_MCQ_OPTIONS",
    # Default BKT parameters
    "DEFAULT_P_L0",
    "DEFAULT_P_T",
    "DEFAULT_P_G",
    "DEFAULT_P_S",
    # Mastery
    "MASTERY_THRESHOLD",
    # Stability clamps
    "P_L_CLAMP_MIN",
    "P_L_CLAMP_MAX",
    # IRT bridge
    "P_G_IRT_BASE",
    "P_G_IRT_FLOOR",
    "P_G_INFLATED_MULTIPLIER",
    "P_G_INFLATED",
    "P_S_IRT_BASE",
    "P_S_IRT_FLOOR",
    "P_G_CLAMP_MIN",
    "P_G_CLAMP_MAX",
    "P_S_CLAMP_MIN",
    "P_S_CLAMP_MAX",
    # P(T) scaling
    "P_T_DIFFICULTY_SCALING_ENABLED",
    "P_T_CLAMP_MIN",
    "P_T_CLAMP_MAX",
    # Cold-start seeding
    "MASTERY_PRIOR_STRENGTH",
    "MASTERY_REFERENCE_DISCRIMINATION",
    "SEED_PRIOR_MIN",
    "SEED_PRIOR_MAX",
    # Guess detection
    "RT_GUESS_THRESHOLD_MS",
    "IRT_SURPRISE_THRESHOLD",
    # Math utils
    "LOGISTIC_EXPONENT_CLAMP",
    "_sigmoid",
]
