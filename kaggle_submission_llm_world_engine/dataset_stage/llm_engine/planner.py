"""Search over a validated WorldModel to choose actions, without calling
the LLM. See architecture.md step 4: "Plan without the LLM" -- this is the
efficiency lever that keeps LLM calls rare while still making a real
decision every step.

Uses beam search: cheap, bounded, and degrades gracefully (fewer/shorter
candidate sequences) rather than blowing up when goal_hint is uninformative
-- a real risk early in a game, before the model has learned much. When
that degradation is total -- no candidate action distinguishes itself at
all -- `plan()` reports `stalled=True` so the caller can fall back to the
sparse action-head LLM (see llm_engine/action_head.py) instead of guessing
via search alone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .types import ALL_ACTIONS, Action, Grid, allowed_action_names
from .world_model import HINT_REL_TOL, WorldModelProtocol, hints_tied, isolated, safe_goal_hint, safe_predict

# Coordinates to try for ACTION6 -- a coarse grid rather than all 4096
# (x, y) pairs, matching the opening-probe's spread-sample approach. Kept
# small because each candidate multiplies the branching factor, and because
# the competition's own Swarm runs many games concurrently via Python
# threads (GIL-bound) -- a wide beam search here contends for the same
# core across every concurrent game, so this stays intentionally cheap
# rather than exhaustive.
_ACTION6_SAMPLE_POINTS = [(16, 16), (48, 16), (16, 48), (48, 48)]

# The search stalls when no rollout predicts a level AND the first-ply
# goal_hints are tied -- i.e. it is not distinguishing between actions at
# all. "Tied" is judged RELATIVE to the hints' own magnitude
# (world_model.hints_tied), never against an absolute threshold: goal_hint
# is LLM-authored and has no defined scale.
#
# This replaced `STALL_GOAL_HINT_EPSILON = 0.05` (2026-09-25). Both
# goal_hints on record from the live prototype were cell-count ratios over
# the 4096-cell board, which move by ~0.0012 per five cells changed; an
# absolute 0.05 needs ~205 cells to differ between the best and worst
# candidate. Such a model stalled on every step and was never consulted
# -- a mechanistic account of "16 replay passes, 0 levels".
#
# Tie-breaking is randomised (candidates are shuffled before a stable
# sort). With a fixed candidate order, any tie at the top of the beam went
# to whichever action happened to be listed first -- ACTION1 -- every
# step, a deterministic loop no worse-informed than random but far less
# exploratory.
_RNG = random.Random()


def candidate_actions(allowed: "Optional[set[str]]" = None) -> list[Action]:
    """The actions the search considers from any state, restricted to
    `allowed` names when given. Public because the replay gate checks
    goal_hint against exactly this set.

    The restriction was missing until 2026-09-26: the planner searched all
    seven actions in every game, and on the live run 164 of 324 of its
    decisions were moves the game does not accept (ACTION5/ACTION7 in games
    without them; non-click actions in click-only ft09, 96 of 96) -- each
    discarded by the agent and replaced with a random action.
    """
    names = [a for a in ALL_ACTIONS if allowed is None or a in allowed]
    actions = [Action(name=a) for a in names if a != "ACTION6"]
    if "ACTION6" in names:
        actions += [Action(name="ACTION6", x=x, y=y) for x, y in _ACTION6_SAMPLE_POINTS]
    return actions


_candidate_actions = candidate_actions  # old name, kept for callers


@dataclass
class PlanResult:
    actions: list[Action]
    predicted_levels_gained: int
    predicted_final_goal_hint: float
    stalled: bool
    goal_hint_spread: float


def plan(
    model: WorldModelProtocol,
    state: Grid,
    depth: int = 2,
    beam_width: int = 4,
    stall_rel_tol: float = HINT_REL_TOL,
    rng: Optional[random.Random] = None,
    available_actions: Optional[list[int]] = None,
) -> PlanResult:
    """Beam search over predicted futures. Score = (levels gained so far
    in this rollout, goal_hint of the resulting state) -- levels gained
    dominates (it's the real signal), goal_hint only breaks ties among
    rollouts that haven't won anything yet.
    """
    candidates = candidate_actions(allowed_action_names(available_actions))
    (rng or _RNG).shuffle(candidates)

    # Each beam entry: (score_tuple, action_sequence, resulting_state,
    # cumulative_levels, terminal, model_copy). Every branch runs on its own
    # copy of the model: the installed instance is never touched (only real
    # transitions may advance it), and sibling branches cannot see each
    # other's imagined moves through shared hidden state.
    root = isolated(model)
    beam: list = [((0, safe_goal_hint(root, state)), [], state, 0, False, root)]

    first_ply_hints: list[float] = []
    first_ply_any_levels = False

    for depth_idx in range(depth):
        expanded: list = []
        for entry in beam:
            _score, seq, cur_state, cum_levels, terminal, branch_model = entry
            if cum_levels > 0 or terminal:
                # Already predicts a level, or reached a predicted terminal
                # state -- carry it forward without extending it.
                expanded.append(entry)
                continue
            for action in candidates:
                child = isolated(branch_model)
                next_state, levels_delta, done, error = safe_predict(child, cur_state, action)
                if error is not None or next_state is None:
                    continue
                new_cum = cum_levels + (levels_delta or 0)
                hint = safe_goal_hint(child, next_state)
                if depth_idx == 0:
                    first_ply_hints.append(hint)
                    if new_cum > 0:
                        first_ply_any_levels = True
                # A predicted terminal state ends THIS rollout only. This
                # used to `break` out of the candidate loop, which silently
                # skipped every candidate listed after the first action
                # predicting `done` -- and thinned the stall test's sample.
                expanded.append(((new_cum, hint), seq + [action], next_state, new_cum, bool(done), child))
        if not expanded:
            break
        expanded.sort(key=lambda e: e[0], reverse=True)  # score only: never compare models
        beam = expanded[:beam_width]

    goal_hint_spread = (max(first_ply_hints) - min(first_ply_hints)) if first_ply_hints else 0.0

    if not beam:
        # Total planning failure (model errors on everything) -- caller
        # should fall back to the action-head LLM or a scripted/random
        # action.
        return PlanResult(
            actions=[], predicted_levels_gained=0, predicted_final_goal_hint=0.0,
            stalled=True, goal_hint_spread=goal_hint_spread,
        )

    best_score, best_seq, _, best_levels, _terminal, _m = beam[0]
    stalled = (
        best_levels == 0
        and not first_ply_any_levels
        and hints_tied(first_ply_hints, rel_tol=stall_rel_tol)
    )

    if not best_seq:
        return PlanResult(
            actions=[], predicted_levels_gained=0, predicted_final_goal_hint=best_score[1],
            stalled=stalled, goal_hint_spread=goal_hint_spread,
        )
    return PlanResult(
        actions=best_seq, predicted_levels_gained=best_levels, predicted_final_goal_hint=best_score[1],
        stalled=stalled, goal_hint_spread=goal_hint_spread,
    )


def next_action(
    model: WorldModelProtocol, state: Grid, depth: int = 2, beam_width: int = 4,
    available_actions: Optional[list[int]] = None,
) -> tuple[Optional[Action], PlanResult]:
    """Convenience wrapper: plan, then return the first action of the best
    sequence (re-planning every step against the latest real state is
    cheap since this never calls the LLM) alongside the full PlanResult so
    the caller can inspect `stalled` and decide whether to consult the
    action-head fallback."""
    result = plan(model, state, depth=depth, beam_width=beam_width, available_actions=available_actions)
    action = result.actions[0] if result.actions else None
    return action, result
