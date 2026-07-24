"""
__init__.py — BKT engine public API.

Everything a caller (orchestrator, test, or future API layer) needs is
re-exported here. Internal implementation details (config constants,
_sigmoid, etc.) are not re-exported — callers who need them must import
from the specific submodule explicitly.
"""

from .config import MASTERY_THRESHOLD, N_MCQ_OPTIONS
from .engine import BKTEngine, BKTUpdateResult
from .exceptions import (
    BKTError,
    InvalidParameterError,
    InvalidStateError,
    IRTParameterRangeError,
    MasteryThresholdError,
)
from .mastery import (
    check_mastery,
    expected_attempts_to_mastery,
    is_probable_guess,
    mastery_gap,
)
from .parameters import (
    BKTSkillParameters,
    defaults,
    derive_p_g_from_irt,
    derive_p_s_from_irt,
    derive_p_t_from_difficulty,
    from_irt,
    irt_p_correct,
    seed_p_l0_from_irt,
)
from .state import BKTState, initial_state

__all__ = [
    # Core classes
    "BKTEngine",
    "BKTUpdateResult",
    "BKTState",
    "BKTSkillParameters",
    # State factory
    "initial_state",
    # Parameter factories
    "defaults",
    "from_irt",
    # IRT bridge functions
    "irt_p_correct",
    "derive_p_g_from_irt",
    "derive_p_s_from_irt",
    "derive_p_t_from_difficulty",
    "seed_p_l0_from_irt",
    # Mastery utilities
    "check_mastery",
    "mastery_gap",
    "is_probable_guess",
    "expected_attempts_to_mastery",
    # Exceptions
    "BKTError",
    "InvalidParameterError",
    "InvalidStateError",
    "MasteryThresholdError",
    "IRTParameterRangeError",
    # Config re-exports (the ones callers commonly need)
    "MASTERY_THRESHOLD",
    "N_MCQ_OPTIONS",
]
