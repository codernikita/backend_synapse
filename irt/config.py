"""
config.py — single source of truth for the hybrid IRT engine's tunable
constants and database connection.

Nothing downstream hard-codes a threshold, a Bloom weight, or a column
default. If a supervisor asks "what does 'Apply' map to?" or "what counts
as a poor discriminator?", the answer lives here and nowhere else.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# ── CHANGE 1: Bloom -> difficulty (b) mapping ───────────────────────────────
# Configurable per your supervisor's instruction. Keys are matched
# case-insensitively against the question/concept dataset's `bloom_level`
# column (e.g. "remember", "Understand", "APPLY").
BLOOM_DIFFICULTY_MAP: dict[str, float] = {
    "remember": -2.0,
    "understand": -1.0,
    "apply": 0.0,
    "analyze": 1.0,
    "evaluate": 2.0,
    "create": 2.5,
}

# ── CHANGE 2: easy/medium/hard bucketing ────────────────────────────────────
# Per-decision: buckets are derived from the SAME Bloom signal as difficulty
# (b), not from a separate raw difficulty column, so there's exactly one
# source of truth for "how hard is this question" in the whole system.
# Every Bloom level named in BLOOM_DIFFICULTY_MAP must appear in exactly one
# bucket below — this is validated at import time (see _validate_buckets).
BLOOM_DIFFICULTY_BUCKETS: dict[str, str] = {
    "remember": "easy",
    "understand": "easy",
    "apply": "medium",
    "analyze": "hard",
    "evaluate": "hard",
    "create": "hard",
}
ACCURACY_BUCKETS = ("easy", "medium", "hard")

# ── CHANGE 2: feature vector construction ───────────────────────────────────
# Decision: class9_marks is already a 0-100 percentage; used directly, no
# max-marks normalization step.
PREVIOUS_CLASS_PERCENTAGE_IS_NORMALIZED = True

# Decision: iq_score is optional (psychometric test ships separately).
# Missing values are imputed with the cohort mean at feature-build time,
# and every imputation is logged/recorded so callers can see how much of
# the clustering input was real vs imputed.
IQ_SCORE_COLUMN_OPTIONAL = True

# The exact feature vector order from the spec. Every module that builds or
# consumes a feature matrix must agree on this order — defined once here.
FEATURE_VECTOR_FIELDS: tuple[str, ...] = (
    "previous_class_percentage",
    "iq_score",
    "total_correct",
    "easy_accuracy",
    "medium_accuracy",
    "hard_accuracy",
)

# ── CHANGE 2/3: clustering + segregation ────────────────────────────────────
N_CLUSTERS = 2
RANDOM_STATE = 42  # KMeans seed, for reproducible strong/weak assignment

# CHANGE 3: discrimination-quality thresholds for the segregation score
# (strong_accuracy - weak_accuracy). The task brief suggested
# 0.50/0.30/0.10/0.00 boundaries; these are adjusted to match Ebel's (1979)
# classic item-discrimination classification, the most widely cited
# standard in classroom-test psychometrics (Ebel & Frisbie, "Essentials of
# Educational Measurement"). Ebel's D-index uses the SAME formula this
# module computes (upper-group accuracy - lower-group accuracy), just on a
# fixed 27%/27% split rather than a KMeans-derived one, which is exactly
# what makes it the right literature anchor here:
#   D >= 0.40            -> excellent
#   0.30 <= D < 0.40      -> good
#   0.20 <= D < 0.30      -> moderate ("marginal" in Ebel's terms; usable but flagged for review)
#   0.00 <= D < 0.20      -> poor (should be revised)
#   D < 0.00              -> negative (discriminates backwards; should be pulled/rewritten)
# Ordered highest-threshold-first; classify_discrimination() picks the first
# (label, min_inclusive) where score >= min_inclusive.
DISCRIMINATION_QUALITY_THRESHOLDS: list[tuple[str, float]] = [
    ("excellent", 0.40),
    ("good", 0.30),
    ("moderate", 0.20),
    ("poor", 0.00),
    ("negative", float("-inf")),
]
# ── CHANGE 4: theta (ability) estimation ────────────────────────────────────
# Newton-Raphson MLE on the 2PL log-likelihood, with a and b already known
# (from bloom_mapper.py and segregation.py respectively) — see theta.py for
# why Newton-Raphson is the right choice here.
THETA_INITIAL = 0.0            # starting guess; 0.0 = "average ability", a neutral prior
THETA_MIN = -4.0                # clamp bounds. +-4 logits covers ~99.97% of a logistic
THETA_MAX = 4.0                 # ability distribution — matching common IRT software defaults
THETA_MAX_STEP = 1.0            # per-iteration Newton step cap, prevents a single noisy
                                  # iteration from launching theta out of a sane range
THETA_MAX_ITERATIONS = 50
THETA_CONVERGENCE_TOLERANCE = 1e-5  # |score (dL/dtheta)| below this => converged
# Response patterns that are all-correct or all-incorrect have NO finite
# maximum-likelihood theta (the log-likelihood is strictly increasing or
# decreasing forever) — this is a known, expected property of MLE for
# extreme response patterns, not a numerical bug. Those cases are detected
# up front and theta is reported at this clamp boundary with converged=False,
# rather than burning iterations approaching it asymptotically.
THETA_EXTREME_PATTERN_CLAMP = 4.0

# Bound on the logistic exponent |a * (theta - b)| before calling exp(), to
# avoid float overflow (exp of a large number) or underflow (exp of a very
# negative number silently becoming exactly 0.0, which would later divide
# cleanly but incorrectly). 35 is comfortably inside float64's range
# (exp(35) ~ 1.6e15) while already representing a probability of
# effectively 0 or 1 to more precision than any real answer key needs.
THETA_EXPONENT_CLAMP = 35.0

# Per Change 3 ("questions with poor segregation should be flagged"),
# these quality labels mark a question for review.
FLAGGED_DISCRIMINATOR_QUALITIES = frozenset({"poor", "negative"})

# ── CHANGE 5: mastery initialization ─────────────────────────────────────
# mastery_initializer.py's inputs are explicitly theta + concept accuracy +
# Bloom-derived difficulty (per-question segregation-based discrimination
# is NOT part of that module's declared input set). To still reuse
# theta.py's exact 2PL probability_correct() formula (no duplicated math)
# when folding difficulty into the theta-implied accuracy term, a fixed
# reference discrimination is used in place of a per-item segregation
# score. This is a deliberate engineering simplification, documented here
# and in mastery_initializer.py: it treats every concept's items as
# "moderately discriminating" for the purpose of the theta -> accuracy
# projection, which is a reasonable default when the mastery-initializer's
# job is a first-pass estimate that BKT will immediately start correcting
# with live data anyway.
MASTERY_REFERENCE_DISCRIMINATION = 1.0

# Blend weight between OBSERVED concept accuracy (correct/attempted) and
# THETA-IMPLIED accuracy (mean 2PL P(correct) at the student's theta over
# that concept's items), as a function of how many items were attempted
# for that concept. This is the effective sample size of a Beta-Bernoulli
# shrinkage prior centered at the theta-implied accuracy: with N attempts,
# observed accuracy gets weight N/(N+K) and the theta-implied prior gets
# K/(N+K). A concept probed by only 1-2 diagnostic items should not be
# initialized purely from that noisy handful of observations; a concept
# probed by 10+ items should be dominated by what was actually observed.
MASTERY_PRIOR_STRENGTH = 3.0

# Initial mastery is clamped into (SEED_PRIOR_MIN, SEED_PRIOR_MAX) — reusing
# the bounds already defined above for exactly this purpose — rather than
# allowing exact 0.0/1.0, since Bayesian Knowledge Tracing's update
# equations (already implemented in the quiz portal) can behave poorly
# starting from a hard 0 or 1 prior (a single slip/guess can never be
# recovered from mathematically once mastery is exactly 0 or 1).

# ── Guess-detection parameters (unchanged from the prior IRT module; mirror
# the constants in the TypeScript submit path — see docs/ for the sync note) ─
RT_GUESS_THRESHOLD_MS = 1500
SURPRISE_THRESHOLD = 0.30
MIN_STUDENTS_WARN = 50

# Bounds used when seeding cold-start mastery from theta.
SEED_PRIOR_MIN = 0.05
SEED_PRIOR_MAX = 0.95


def _validate_buckets() -> None:
    missing = set(BLOOM_DIFFICULTY_MAP) - set(BLOOM_DIFFICULTY_BUCKETS)
    if missing:
        raise ValueError(
            f"BLOOM_DIFFICULTY_BUCKETS is missing an entry for: {sorted(missing)}. "
            "Every Bloom level in BLOOM_DIFFICULTY_MAP must map to an accuracy bucket."
        )
    bad = set(BLOOM_DIFFICULTY_BUCKETS.values()) - set(ACCURACY_BUCKETS)
    if bad:
        raise ValueError(f"BLOOM_DIFFICULTY_BUCKETS has unknown bucket name(s): {sorted(bad)}")


_validate_buckets()


def load_database_url(override: Optional[str] = None) -> str:
    """Resolve DATABASE_URL: explicit override > backend/.env > environment.
    Only used by the Postgres repository path; CSV-mode callers never need
    this."""
    if override:
        return override
    try:
        from dotenv import dotenv_values

        here = os.path.dirname(os.path.abspath(__file__))
        env = dotenv_values(os.path.join(here, "..", ".env"))
        if env.get("DATABASE_URL"):
            return env["DATABASE_URL"]
    except ImportError:
        pass
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("No DATABASE_URL found. Pass --database-url or set it in .env")
    return url
