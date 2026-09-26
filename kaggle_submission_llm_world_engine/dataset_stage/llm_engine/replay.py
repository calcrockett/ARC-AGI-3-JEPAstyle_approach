"""Validate a candidate WorldModel by replaying the observed transcript
through it. This is the hard gate described in architecture.md's "Validate
by replay, not by trust" -- a code revision is only ever accepted if it
reproduces every transition seen so far, not just the one that motivated
the revision.

The gate has two halves (2026-09-25). `predict()` must reproduce every
transition, and `goal_hint()` must be *usable*: it must run on every
board the transcript shows, return a finite number, and not return the
same value for all of them. The second half exists because action
selection is driven by goal_hint alone -- the planner ranks rollouts by
(levels predicted, goal_hint), and a transcript with no level-up gives
predict() no basis to ever predict one. Until this gate existed nothing,
anywhere, ever executed goal_hint before a model was installed: models
passed 28/28 on predict and were installed with an objective no one had
looked at.
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


def check_goal_hint(transcript: GameTranscript, model: WorldModelProtocol) -> GoalHintCheck:
    """Run goal_hint on every distinct observed board.

    Fails if it raises, returns a non-number or non-finite value, or gives
    the same value (up to `hints_tied`'s relative tolerance) to every one
    of two or more distinct boards. With fewer than two distinct boards
    there is nothing to distinguish and the check passes vacuously.
    """
    states = _observed_states(transcript)
    values: list[float] = []
    for i, board in enumerate(states):
        try:
            raw = model.goal_hint(copy.deepcopy(board))
        except Exception as e:  # noqa: BLE001 -- LLM-authored code
            return GoalHintCheck(
                ok=False, states_checked=i,
                problem=f"goal_hint() raised {type(e).__name__}: {e} on observed board #{i}",
            )
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return GoalHintCheck(
                ok=False, states_checked=i,
                problem=f"goal_hint() returned {type(raw).__name__}, expected a float",
            )
        value = float(raw)
        if not math.isfinite(value):
            return GoalHintCheck(
                ok=False, states_checked=i,
                problem=f"goal_hint() returned {value} (not finite) on observed board #{i}",
            )
        values.append(value)

    distinct = len(set(values))
    if len(states) >= 2 and hints_tied(values):
        return GoalHintCheck(
            ok=False, states_checked=len(states), distinct_values=distinct,
            problem=(
                f"goal_hint() returned the same value ({values[0]!r}) for all "
                f"{len(states)} different boards in the transcript. The agent picks "
                "actions ONLY by comparing goal_hint across the boards predict() says "
                "each action leads to, so a constant goal_hint means it cannot choose "
                "between actions at all. Make goal_hint measure progress toward what "
                "you think the win condition is, so that it differs between at least "
                "some of these boards. Only the order of its values matters, not "
                "their scale."
            ),
        )
    return GoalHintCheck(ok=True, states_checked=len(states), distinct_values=distinct)


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
