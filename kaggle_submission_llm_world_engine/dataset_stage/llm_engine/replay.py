"""Validate a candidate WorldModel by replaying the observed transcript
through it. This is the hard gate described in architecture.md's "Validate
by replay, not by trust" -- a code revision is only ever accepted if it
reproduces every transition seen so far, not just the one that motivated
the revision.

The gate has two halves (2026-09-25). `predict()` must reproduce every
transition, and `goal_hint()` must be *usable* as the planner's objective.
Action selection is driven by goal_hint alone -- the planner ranks
rollouts by (levels predicted, goal_hint), and a transcript with no
level-up gives predict() no basis to ever predict one.

"Usable" means (2026-09-26) that goal_hint **separates the boards predict()
says different actions lead to**, checked from observed boards. The first
version asked only that goal_hint vary across the transcript's history,
and a live run showed why that is the wrong property: the installed
objectives counted cells in the bottom strip (dc22 row 63, ls20 rows
61-62) -- which changes on every step, so it passed -- and every action
advances that strip identically, so the planner stalled on 299 of 320
calls. predict() distinguished the actions on the playfield on 31/31
sampled frames; goal_hint threw it away.

"Separates the successors" turned out to be gameable by the same strip: an
action that only ticks the counter scores differently from one that moves
something, so dc22's HUD objective still "separated" actions on 13 of 31
boards -- and on 13 of 13 of those the action it preferred left the
playfield unchanged. So the check is counterfactual: two actions'
predicted boards are given the SAME edge band and differ only in the
interior, and goal_hint must respond to that difference somewhere. On the
live run's boards this rejects both installed objectives (0/31) and
accepts a position-sensitive whole-board control (31/31).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .diff import format_diff
from .types import GameTranscript, Transition
from .world_model import WorldModelProtocol, hints_tied, safe_predict


@dataclass
class TransitionCheck:
    index: int
    passed: bool
    reason: Optional[str] = None  # populated on failure


@dataclass
class GoalHintCheck:
    """Is goal_hint usable as a search objective on the boards we have
    actually seen? Deliberately a *minimum* bar -- it establishes that the
    function runs and distinguishes something, not that what it
    distinguishes is progress."""

    ok: bool
    states_checked: int = 0
    distinct_values: int = 0
    problem: Optional[str] = None
    #: Observed boards from which predict() sends different actions to
    #: different boards -- the only boards where goal_hint can matter.
    informative_boards: int = 0
    #: False when predict() never distinguishes actions from any observed
    #: board: then no objective could plan, and goal_hint is not blamed.
    predict_distinguishes_actions: bool = True


@dataclass
class ReplayResult:
    #: Accepted: predict() reproduced every transition AND goal_hint is
    #: usable. This is what callers gate installation on.
    passed: bool
    checks: list[TransitionCheck]
    first_failure: Optional[TransitionCheck] = None
    goal_hint: GoalHintCheck = field(default_factory=lambda: GoalHintCheck(ok=True))

    @property
    def predict_passed(self) -> bool:
        """The predict half alone -- the quantity every pre-2026-09-25
        write-up calls a "replay pass"."""
        return self.first_failure is None

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total(self) -> int:
        return len(self.checks)


def replay(transcript: GameTranscript, model: WorldModelProtocol) -> ReplayResult:
    """Run every transition in order through model.predict and compare
    against what actually happened. Stops recording new checks after the
    first failure is enough context for a repair prompt, but still counts
    every transition so pass_count/total is meaningful."""
    checks: list[TransitionCheck] = []
    first_failure: Optional[TransitionCheck] = None

    for i, t in enumerate(transcript.transitions):
        check = _check_one(i, t, model)
        checks.append(check)
        if not check.passed and first_failure is None:
            first_failure = check

    goal = check_goal_hint(transcript, model)
    return ReplayResult(
        passed=first_failure is None and goal.ok,
        checks=checks,
        first_failure=first_failure,
        goal_hint=goal,
    )


def _observed_states(transcript: GameTranscript) -> list[Any]:
    """Every distinct board the transcript shows, in order of first
    appearance: each transition's before-board plus the final after-board.
    (A discontinuity between transitions is harmless here -- both boards
    are real observations either way.)"""
    seen: set = set()
    out: list[Any] = []
    boards = [t.frame_before for t in transcript.transitions]
    if transcript.transitions:
        boards.append(transcript.transitions[-1].frame_after)
    for b in boards:
        key = repr(b)
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out


observed_states = _observed_states


#: Observed boards probed, spread evenly over the transcript. Each costs one
#: predict() per candidate action (13) -- the planner spends ~5x that on
#: every single step.
MAX_SEPARATION_BOARDS = 12
#: Distinct playfield outcomes compared per board (pairs = n * (n - 1)).
MAX_OUTCOMES_PER_BOARD = 6
#: Cells within this distance of the border are treated as the "edge band"
#: where step counters and status bars live. The repo's status-bar detector
#: (graph_explorer_agent.identify_status_bars_with_rule, ported from
#: arXiv:2512.24156), run on real frames, found bars only 0-2 cells from the
#: edge on the 22 of 25 games where it fired; dc22's objective read row 63,
#: ls20's rows 61-62. Applied only to boards at least MIN_BAND_BOARD on a side
#: -- on a tiny board there is no room for a HUD and no interior without it.
EDGE_BAND = 3
MIN_BAND_BOARD = 16


def _spread(items: list, k: int) -> list[tuple[int, Any]]:
    if len(items) <= k:
        return list(enumerate(items))
    step = (len(items) - 1) / (k - 1)
    idx = sorted({round(i * step) for i in range(k)})
    return [(i, items[i]) for i in idx]


def _band(board: Any) -> int:
    try:
        h, w = len(board[0]), len(board[0][0])
    except (IndexError, TypeError):
        return 0
    return EDGE_BAND if min(h, w) >= MIN_BAND_BOARD else 0


def _interior(board: Any, band: int) -> Any:
    if band == 0:
        return board
    return [[row[band:len(row) - band] for row in layer[band:len(layer) - band]] for layer in board]


def _transplant(interior_from: Any, edge_from: Any, band: int) -> Any:
    """`edge_from`'s edge band around `interior_from`'s interior."""
    if band == 0:
        return copy.deepcopy(interior_from)
    out = copy.deepcopy(edge_from)
    for layer_out, layer_in in zip(out, interior_from):
        h = len(layer_out)
        for y in range(band, h - band):
            w = len(layer_out[y])
            layer_out[y][band:w - band] = layer_in[y][band:w - band]
    return out


def _where_they_differ(a: Any, b: Any) -> str:
    cells = [
        (y, x)
        for layer_a, layer_b in zip(a, b)
        for y, (row_a, row_b) in enumerate(zip(layer_a, layer_b))
        for x, (va, vb) in enumerate(zip(row_a, row_b))
        if va != vb
    ]
    if not cells:
        return "in no cell of layer 0 (shape or extra layers only)"
    ys = [c[0] for c in cells]
    xs = [c[1] for c in cells]
    return (
        f"in {len(cells)} cell(s), rows {min(ys)}-{max(ys)}, "
        f"columns {min(xs)}-{max(xs)}"
    )


def _score(model: WorldModelProtocol, board: Any) -> tuple[Optional[float], Optional[str]]:
    """goal_hint on one board, or the reason it is unusable there."""
    try:
        raw = model.goal_hint(copy.deepcopy(board))
    except Exception as e:  # noqa: BLE001 -- LLM-authored code
        return None, f"goal_hint() raised {type(e).__name__}: {e}"
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None, f"goal_hint() returned {type(raw).__name__}, expected a float"
    value = float(raw)
    if not math.isfinite(value):
        return None, f"goal_hint() returned {value} (not finite)"
    return value, None


def check_goal_hint(transcript: GameTranscript, model: WorldModelProtocol) -> GoalHintCheck:
    """Is goal_hint usable as the planner's objective?

    1. It must run and return a finite number on every observed board.
    2. It must respond to what the actions do on the playfield. From
       observed boards, predict every candidate action's successor and keep
       those whose interior (everything outside the edge band) differs from
       the current board and from each other. For a pair of such outcomes
       A and B, build a counterfactual: A's edge band around B's interior.
       If goal_hint(A) differs from goal_hint(counterfactual) on at least
       one pair, it responds to the playfield. A score that reads only a
       step counter or edge bar gives the two the same value, always.

    If predict() never sends two actions to different playfields from any
    observed board, there is nothing to respond to and goal_hint is not
    blamed (`predict_distinguishes_actions=False`): no objective could plan
    with that simulator, and rejecting it would only spend retries.
    """
    from .planner import candidate_actions  # planner does not import replay

    states = _observed_states(transcript)
    values: list[float] = []
    for i, board in enumerate(states):
        value, problem = _score(model, board)
        if problem is not None:
            return GoalHintCheck(
                ok=False, states_checked=i, problem=f"{problem} on observed board #{i}",
            )
        values.append(value)  # type: ignore[arg-type]
    distinct = len(set(values))

    informative = 0
    example = None
    for board_idx, board in _spread(states, MAX_SEPARATION_BOARDS):
        band = _band(board)
        here = repr(_interior(board, band))
        outcomes: dict[str, tuple[Any, Any]] = {}
        for action in candidate_actions():
            nxt, _delta, _done, error = safe_predict(model, board, action)
            if error is not None or nxt is None:
                continue
            key = repr(_interior(nxt, band))
            if key != here:
                outcomes.setdefault(key, (action, nxt))
        if len(outcomes) < 2:
            continue
        informative += 1
        picked = list(outcomes.values())[:MAX_OUTCOMES_PER_BOARD]
        for i, (a_i, s_i) in enumerate(picked):
            v_i, problem = _score(model, s_i)
            if problem is not None:
                return GoalHintCheck(
                    ok=False, states_checked=len(states), distinct_values=distinct,
                    informative_boards=informative,
                    problem=(f"{problem} on the board predict() says {a_i} leads to "
                             f"from observed board #{board_idx}"),
                )
            for j, (a_j, s_j) in enumerate(picked):
                if i == j:
                    continue
                v_c, problem = _score(model, _transplant(s_j, s_i, band))
                if problem is not None:
                    return GoalHintCheck(
                        ok=False, states_checked=len(states), distinct_values=distinct,
                        informative_boards=informative,
                        problem=(f"{problem} on a board combining the results of {a_i} "
                                 f"and {a_j} from observed board #{board_idx}"),
                    )
                if not hints_tied([v_i, v_c]):  # type: ignore[list-item]
                    return GoalHintCheck(
                        ok=True, states_checked=len(states), distinct_values=distinct,
                        informative_boards=informative,
                    )
                if example is None:
                    example = (board_idx, a_i, a_j, s_i, s_j, v_i, band)

    if informative == 0:
        return GoalHintCheck(
            ok=True, states_checked=len(states), distinct_values=distinct,
            informative_boards=0, predict_distinguishes_actions=False,
        )

    board_idx, a1, a2, s1, s2, v, band = example  # type: ignore[misc]
    where = _where_they_differ(_transplant(s2, s1, band), s1)
    edge_note = (
        f" It ignores every change your actions make away from the outer {band} "
        "cells of the board -- the edge, where step counters, timers and status "
        "bars live, which change the same way whatever action is taken."
        if band else ""
    )
    return GoalHintCheck(
        ok=False, states_checked=len(states), distinct_values=distinct,
        informative_boards=informative,
        problem=(
            f"goal_hint() cannot tell actions apart. From observed board #{board_idx}, "
            f"your predict() says {a1} and {a2} change the playfield differently -- "
            f"their results differ {where} -- yet goal_hint gives the same score "
            f"({v!r}) whichever of the two that region looks like. This happened on "
            f"all {informative} observed board(s) checked where actions change the "
            f"playfield differently.{edge_note} The agent picks each action by "
            "comparing these scores, so it cannot choose. Make goal_hint depend on "
            f"the part of the board your actions change (here: {where.split(', ', 1)[-1]}). "
            "If your actions MOVE things, counting colours will not work -- a moved "
            "object has the same colour counts -- so measure positions instead, e.g. "
            "minus the distance from the object you move to where you think it should go."
        ),
    )


def _check_one(index: int, t: Transition, model: WorldModelProtocol) -> TransitionCheck:
    predicted_state, predicted_levels_delta, predicted_done, error = safe_predict(
        model, t.frame_before, t.action
    )
    if error is not None:
        return TransitionCheck(index=index, passed=False, reason=error)

    reasons = []
    if predicted_state != t.frame_after:
        reasons.append(f"grid mismatch: {format_diff(predicted_state, t.frame_after)}")
    if predicted_levels_delta != t.levels_delta:
        reasons.append(f"levels_delta mismatch: predicted {predicted_levels_delta}, actual {t.levels_delta}")
    if predicted_done != t.done:
        reasons.append(f"done mismatch: predicted {predicted_done}, actual {t.done}")

    if reasons:
        return TransitionCheck(index=index, passed=False, reason="; ".join(reasons))
    return TransitionCheck(index=index, passed=True)


def describe_rejection(transcript: GameTranscript, result: ReplayResult) -> str:
    """Why a candidate was not accepted, for a retry/repair prompt.

    A predict() failure comes first -- it is the harder half and the one
    the transcript speaks to directly -- with any goal_hint problem noted
    after it, so neither half is fixed at the expense of the other."""
    parts = []
    if result.first_failure is not None:
        parts.append(describe_failure(transcript, result.first_failure))
    if not result.goal_hint.ok and result.goal_hint.problem:
        parts.append(
            ("Separately: " if parts else "predict() reproduces the whole transcript -- keep it. ")
            + result.goal_hint.problem
        )
    return "\n".join(parts) or "the candidate was rejected"


def describe_failure(transcript: GameTranscript, check: TransitionCheck) -> str:
    """Human/LLM-readable description of one failing transition, for use
    in a repair prompt."""
    t = transcript.transitions[check.index]
    return (
        f"Transition #{check.index}: action={t.action}, "
        f"levels_completed {t.levels_completed_before}->{t.levels_completed_after}, "
        f"state_after={t.state_after}\n"
        f"Your model's predict() was wrong: {check.reason}"
    )
