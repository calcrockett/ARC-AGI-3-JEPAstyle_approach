"""The per-game world-model contract, plus a sandboxed loader for
LLM-authored source code. See architecture.md's "Per-game artifact" section.

The interface an LLM-authored module must implement:

    class WorldModel:
        def __init__(self):
            ...  # may set up self.counters / any hidden state

        def predict(self, state, action_name, x=None, y=None):
            '''state: list of layers, each a 64x64 list of ints 0-15.
            action_name: one of ACTION1..ACTION5, ACTION6, ACTION7.
            x, y: only set when action_name == "ACTION6".
            Returns (next_state, levels_delta, done) where:
              - next_state has the same shape as state
              - levels_delta is an int (usually 0 or 1)
              - done is a bool (True on predicted WIN or GAME_OVER)
            Must be a pure-ish function of (self, state, action, x, y) --
            self may only be used to store small hidden counters inferred
            from repeated probes, not anything encoding a specific state.'''

        def goal_hint(self, state):
            '''Returns a float: higher = closer to a win condition, used
            only as a search heuristic, never as a hard oracle. Only the
            ORDERING of its values is used (see `hints_tied`); its scale
            is arbitrary and never compared against a fixed threshold.'''
"""

from __future__ import annotations

import builtins
import copy
import math
import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .types import Action, Grid

logger = logging.getLogger(__name__)

# Names the LLM's module is allowed to see at exec() time. Deliberately
# small -- this is a lightweight safety net (LLM-authored code still runs
# in-process, not a real sandbox), not a security boundary. It exists to
# catch the common failure mode of generated code reaching for `os`/`sys`/
# `open` out of habit, not to defend against a hostile model.
_SAFE_BUILTIN_NAMES = [
    # `__build_class__` is required by the interpreter for any `class`
    # statement to work under exec() with a restricted __builtins__ dict --
    # without it, defining WorldModel itself raises NameError. `__name__`
    # is read implicitly by class bodies (for __module__ / __qualname__).
    "__build_class__", "__name__",
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "isinstance", "len", "list", "max", "min", "range", "round", "set",
    "sorted", "str", "sum", "tuple", "zip", "print", "getattr", "setattr",
    "hasattr", "ValueError", "TypeError", "IndexError", "KeyError",
    "Exception", "StopIteration",
    # --- added 2026-09-07 -------------------------------------------
    # Everything below was missing, and each omission silently converted
    # ordinary, idiomatic generated Python into a load or predict
    # failure that the model then had to guess its way out of:
    #   super()            -> NameError in __init__, so the whole class
    #                         fails to instantiate (load_world_model
    #                         reports "constructor failed")
    #   map/filter/reversed-> NameError at predict() time, i.e. a replay
    #                         failure indistinguishable from a wrong rule
    #   AttributeError &c. -> a model's own defensive `try/except` block
    #                         raises NameError while handling an error
    # These are all pure builtins with no I/O, no import machinery and no
    # filesystem/network reach, so adding them does not widen what this
    # namespace can do -- see the note above about this being a
    # foot-gun guard, not a security boundary. `__import__`, `open`,
    # `eval`, `exec` and `compile` remain deliberately absent.
    "map", "filter", "reversed", "type", "object", "super",
    "staticmethod", "classmethod", "property", "frozenset", "divmod",
    "pow", "iter", "next", "repr", "format", "slice", "bytes", "complex",
    "id", "hash", "chr", "ord", "bin", "hex", "oct",
    "AttributeError", "NameError", "RuntimeError", "ZeroDivisionError",
    "NotImplementedError", "AssertionError", "ArithmeticError",
    "LookupError", "OverflowError", "RecursionError",
]
_SAFE_BUILTINS = {
    name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(builtins, name)
}
_SAFE_BUILTINS.setdefault("__name__", "world_model_sandbox")


class WorldModelProtocol(Protocol):
    def predict(
        self, state: Grid, action_name: str, x: Optional[int] = None, y: Optional[int] = None
    ) -> tuple[Grid, int, bool]: ...

    def goal_hint(self, state: Grid) -> float: ...


@dataclass
class LoadResult:
    ok: bool
    world_model: Optional[WorldModelProtocol] = None
    error: Optional[str] = None


def load_world_model(source: str) -> LoadResult:
    """exec() the LLM-authored source in a restricted namespace and
    instantiate its WorldModel class. Any exception here (syntax error,
    missing class, constructor error) is treated the same as a replay
    failure -- fed back to the LLM as the thing to fix, not raised."""
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    try:
        compiled = compile(source, "<world_model>", "exec")
        exec(compiled, namespace)  # noqa: S102 -- see module docstring
    except Exception as e:  # noqa: BLE001 -- intentionally broad, see below
        return LoadResult(ok=False, error=f"exec failed: {type(e).__name__}: {e}")

    cls = namespace.get("WorldModel")
    if cls is None:
        return LoadResult(ok=False, error="no `WorldModel` class defined")
    try:
        instance = cls()
    except Exception as e:  # noqa: BLE001
        return LoadResult(ok=False, error=f"WorldModel() constructor failed: {type(e).__name__}: {e}")

    if not hasattr(instance, "predict") or not hasattr(instance, "goal_hint"):
        return LoadResult(ok=False, error="WorldModel must define predict() and goal_hint()")

    return LoadResult(ok=True, world_model=instance)


def safe_predict(
    model: WorldModelProtocol, state: Grid, action: Action
) -> tuple[Optional[Grid], Optional[int], Optional[bool], Optional[str]]:
    """Call model.predict, catching any exception so a broken model
    degrades to 'always wrong' (a replay failure) instead of crashing the
    agent process.

    The return value is *also* type-checked here, not just the call. This
    is load-bearing: before 2026-09-07 a `predict()` that returned a
    structurally-malformed grid (a plausible LLM slip -- e.g. returning
    one layer instead of the list of layers) sailed through this function
    and only blew up later, inside the diagnostic formatter that tried to
    describe the mismatch, as an unhandled TypeError that killed the game
    thread. Catching it here turns that class of near-miss model into
    ordinary, actionable repair feedback: 'your predict() returned the
    wrong shape, here is how'.
    """
    try:
        state_copy = copy.deepcopy(state)
        result = model.predict(state_copy, action.name, x=action.x, y=action.y)
    except Exception as e:  # noqa: BLE001
        return None, None, None, f"predict() raised {type(e).__name__}: {e}"

    error = _describe_bad_prediction(result)
    if error is not None:
        return None, None, None, error

    next_state, levels_delta, done = result
    return next_state, int(levels_delta), bool(done), None


def _describe_bad_prediction(result: Any) -> Optional[str]:
    """None if `result` is a usable (next_state, levels_delta, done)
    triple, else a description of what is wrong with it, phrased for the
    LLM that has to fix it."""
    from .diff import describe_malformed  # local import: diff imports types only

    if not isinstance(result, (tuple, list)):
        return (
            f"predict() returned {type(result).__name__}, expected a "
            "(next_state, levels_delta, done) tuple"
        )
    if len(result) != 3:
        return (
            f"predict() returned {len(result)} value(s), expected exactly 3: "
            "(next_state, levels_delta, done)"
        )

    next_state, levels_delta, done = result
    malformed = describe_malformed(next_state, "predict()'s next_state")
    if malformed is not None:
        return malformed
    if isinstance(levels_delta, bool) or not isinstance(levels_delta, int):
        return (
            f"predict()'s levels_delta is {type(levels_delta).__name__}, "
            "expected an int (usually 0 or 1)"
        )
    if not isinstance(done, bool):
        return f"predict()'s done is {type(done).__name__}, expected a bool"
    return None


def isolated(model: WorldModelProtocol) -> WorldModelProtocol:
    """A private copy of `model` for HYPOTHETICAL calls.

    LLM-written models keep instance state -- 70 of the 83 installed or
    candidate models saved from kernels v3-v7 do -- and `predict()` updates
    it (ls20: `self.counter_x += 1`). The planner used to search on the
    installed instance itself: one plan() call moved ls20's counter from 13
    to 34, writing 21 imagined moves into the real simulator. It then
    mispredicted real moves, and each false "divergence" triggered a repair
    that a freshly loaded copy passed with no LLM call at all -- 7 on ls20
    in kernel v7, each charged against the coder budget.

    Only REAL transitions may advance the installed instance. Everything
    imagined -- search, gate probes, peak tests -- runs on a copy. All 83
    saved models deep-copy cleanly (worst 0.45 ms); if one ever does not,
    the original is returned, which is the old behaviour.
    """
    try:
        return copy.deepcopy(model)
    except Exception:  # noqa: BLE001 -- LLM-authored object
        return model


def safe_goal_hint(model: WorldModelProtocol, state: Grid) -> float:
    """goal_hint for use inside search: never raises, never returns a
    non-finite value. A NaN in a beam's sort key makes the ordering
    undefined (NaN compares false against everything), so it is mapped to
    0.0 exactly like an exception."""
    try:
        value = float(model.goal_hint(copy.deepcopy(state)))
    except Exception:  # noqa: BLE001
        return 0.0
    return value if math.isfinite(value) else 0.0


#: Relative tolerance under which two goal_hint values count as equal.
#: goal_hint is LLM-authored and has no defined scale -- the two real
#: examples on record are cell-count RATIOS over 4096 cells, where one
#: action moving five cells changes the value by ~0.0012. The original
#: stall test compared the spread against an ABSOLUTE 0.05, which such a
#: hint can only clear if ~205 cells differ between the best and worst
#: candidate -- so a perfectly sensible objective stalled the planner on
#: every step and the world model was never consulted. Deterministic code
#: evaluated on different boards gives exactly equal floats when it does
#: not distinguish them, so only genuine float noise needs absorbing here.
HINT_REL_TOL = 1e-9


def hints_tied(values: "list[float]", rel_tol: float = HINT_REL_TOL) -> bool:
    """True iff `values` do not distinguish anything: fewer than two, or
    max and min equal up to `rel_tol` of their own magnitude. Scale-free:
    multiplying every value by any positive constant never changes the
    answer, which an absolute epsilon cannot promise."""
    if len(values) < 2:
        return True
    lo, hi = min(values), max(values)
    return (hi - lo) <= rel_tol * max(abs(lo), abs(hi))


WORLD_MODEL_SKELETON = '''\
class WorldModel:
    """Fill in predict() and goal_hint() based on the transcript above.
    Keep any inferred hidden state (e.g. a suspected counter) on self,
    initialized in __init__ -- do not hardcode specific grids/positions
    from the examples, generalize the *rule* you infer from them."""

    def __init__(self):
        pass  # e.g. self.counters = {}

    def predict(self, state, action_name, x=None, y=None):
        # state: list of layers, each layer a 64x64 list of ints (0-15).
        # Return (next_state, levels_delta, done).
        # Default: assume nothing changes (replace with your inferred rule).
        return state, 0, False

    def goal_hint(self, state):
        # Higher = closer to a win. Only the ORDER of values matters, any
        # scale is fine. Must NOT be constant: the search picks actions by
        # comparing it, so a constant gives it nothing to choose between.
        return 0.0
'''
