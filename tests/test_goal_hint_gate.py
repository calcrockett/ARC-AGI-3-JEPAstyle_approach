"""goal_hint is what CodeWorldAgent plays with -- these tests pin the three
fixes made on 2026-09-25 so that it is actually consulted, and validated.

Background: `planner.plan()` ranks rollouts by (levels predicted,
goal_hint). A transcript with no level-up gives predict() no basis to
ever predict one, so in practice goal_hint is the entire objective. Yet:

  1. the stall test compared goal_hint's spread against an ABSOLUTE 0.05,
     and both goal_hints on record from the live prototype were cell-count
     ratios over 4096 cells (~0.0012 per five cells) -- so a sensible
     objective stalled the planner on every step;
  2. on a stall with no action-head budget the agent played the planner's
     arbitrary tie-winner anyway, which with a fixed candidate order was
     ACTION1 every step;
  3. nothing executed goal_hint before a model was installed;
  4. the prompts said "return 0.0 if you have no idea".

A predicted terminal state also used to `break` out of the candidate loop,
hiding every later candidate from the search and from the stall test.
"""

from __future__ import annotations

import math
import random

import pytest

from conftest import (  # type: ignore[import-not-found]
    CORRECT_WORLD_MODEL_SOURCE,
    FakeLLMClient,
    build_transcript,
    with_agent_at,
)

from llm_engine import drafting
from llm_engine.planner import _candidate_actions, next_action, plan
from llm_engine.replay import check_goal_hint, describe_rejection, replay
from llm_engine.types import Action, GameTranscript, Transition
from llm_engine.world_model import hints_tied, load_world_model, safe_goal_hint


def fenced(source: str) -> str:
    return f"```python\n{source}\n```\n"


def with_goal_hint(body: str) -> str:
    """CORRECT_WORLD_MODEL_SOURCE -- whose predict() is exactly right for the
    toy game -- with its goal_hint body replaced."""
    head, _sep, _tail = CORRECT_WORLD_MODEL_SOURCE.partition("    def goal_hint(self, state):\n")
    return head + "    def goal_hint(self, state):\n" + body


CONSTANT = with_goal_hint("        return 0.0\n")

# The shape of both real goal_hints on record (ar25, bp35): a cell-count
# ratio over the whole board. On the 6x6 toy it is scaled to what the same
# ratio does on 64x64: one cell is 1/4096.
RATIO_4096 = with_goal_hint(
    "        found = self._find(state)\n"
    "        if found is None:\n"
    "            return 0.0\n"
    "        return (found[0] + found[1]) / 4096.0\n"
)


def load(source: str):
    res = load_world_model(source)
    assert res.ok, res.error
    return res.world_model


# =====================================================================
# 1. the stall test is scale-free
# =====================================================================


def test_hints_tied_is_scale_free():
    base = [0.0, 1.0, 2.0]
    for k in (1e-9, 1e-3, 1.0, 1e6):
        assert not hints_tied([v * k for v in base]), k
    assert hints_tied([5.0, 5.0, 5.0])
    assert hints_tied([0.0, 0.0])
    assert hints_tied([7.0])
    assert hints_tied([])


def test_ratio_scaled_goal_hint_no_longer_stalls():
    """Regression for the defect: a one-cell move is worth 1/4096 ~ 0.00024
    here, far below the old absolute 0.05 -- which stalled every step."""
    model = load(RATIO_4096)
    result = plan(model, with_agent_at(2, 2), rng=random.Random(0))
    assert result.goal_hint_spread < 0.05, "fixture must reproduce the old stall condition"
    assert not result.stalled
    assert result.actions, "a non-stalled plan must carry a decision"


def test_constant_goal_hint_still_stalls():
    result = plan(load(CONSTANT), with_agent_at(2, 2), rng=random.Random(0))
    assert result.stalled


def test_stall_decision_is_invariant_to_rescaling_the_hint():
    for scale in ("1e-9", "1.0", "1e9"):
        src = with_goal_hint(
            "        found = self._find(state)\n"
            "        if found is None:\n"
            "            return 0.0\n"
            f"        return (found[0] + found[1]) * {scale}\n"
        )
        assert not plan(load(src), with_agent_at(2, 2), rng=random.Random(0)).stalled, scale


def test_informative_hint_picks_the_improving_action():
    """goal = x + y, agent at (2,2) on a 6x6 board: the first move of the
    chosen plan must increase x + y (ACTION2 down, ACTION4 right, or a
    click that lands further along)."""
    model = load(CORRECT_WORLD_MODEL_SOURCE)
    for seed in range(10):
        action, result = next_action(model, with_agent_at(2, 2))
        assert action is not None and not result.stalled
        nxt, _, _ = model.predict(with_agent_at(2, 2), action.name, action.x, action.y)
        assert model.goal_hint(nxt) >= model.goal_hint(with_agent_at(2, 2))


def test_ties_are_broken_randomly_not_by_candidate_order():
    """With a fixed candidate order every tie went to ACTION1."""
    model = load(CONSTANT)
    firsts = {plan(model, with_agent_at(2, 2), rng=random.Random(seed)).actions[0].name
              for seed in range(40)}
    assert len(firsts) > 1


def test_a_terminal_prediction_does_not_hide_later_candidates():
    """ACTION1 predicts done=True. It used to `break` the candidate loop, so
    if ACTION1 came first nothing else was ever evaluated."""
    src = with_goal_hint(
        "        found = self._find(state)\n"
        "        return 0.0 if found is None else float(found[0])\n"
    ).replace(
        "        return ns, 0, False\n",
        "        return ns, 0, action_name == \"ACTION1\"\n",
    )
    model = load(src)

    class FirstIsAction1(random.Random):
        def shuffle(self, x):  # keep ACTION1 first, as the old fixed order did
            x.sort(key=lambda a: a.name != "ACTION1")

    result = plan(model, with_agent_at(2, 2), depth=1, beam_width=50, rng=FirstIsAction1())
    assert not result.stalled
    # ACTION4 (x+1) must have been seen, and wins on goal = x.
    assert result.actions[0].name in {"ACTION4", "ACTION6"}


def test_safe_goal_hint_maps_non_finite_to_zero():
    for bad in ("float('nan')", "float('inf')", "-float('inf')"):
        m = load(with_goal_hint(f"        return {bad}\n"))
        assert safe_goal_hint(m, with_agent_at(0, 0)) == 0.0


# =====================================================================
# 2. the replay gate validates goal_hint
# =====================================================================


def test_constant_goal_hint_is_rejected_even_with_a_perfect_predict():
    t = build_transcript()
    result = replay(t, load(CONSTANT))
    assert result.predict_passed, "predict() is the correct one"
    assert not result.passed
    assert not result.goal_hint.ok
    assert "same value" in result.goal_hint.problem
    assert result.goal_hint.distinct_values == 1


def test_correct_model_passes_both_halves():
    result = replay(build_transcript(), load(CORRECT_WORLD_MODEL_SOURCE))
    assert result.passed and result.predict_passed and result.goal_hint.ok
    assert result.goal_hint.distinct_values > 1


def test_ratio_scaled_goal_hint_passes_the_gate():
    """The gate is scale-free too -- the prototype's objectives must pass."""
    assert replay(build_transcript(), load(RATIO_4096)).passed


@pytest.mark.parametrize("body,needle", [
    ("        raise ValueError('boom')\n", "raised ValueError"),
    ("        return float('nan')\n", "not finite"),
    ("        return 'high'\n", "returned str"),
    ("        return True\n", "returned bool"),
    ("        return None\n", "returned NoneType"),
])
def test_broken_goal_hints_are_rejected(body, needle):
    result = replay(build_transcript(), load(with_goal_hint(body)))
    assert result.predict_passed
    assert not result.passed
    assert needle in result.goal_hint.problem


def test_integer_goal_hint_is_accepted():
    src = with_goal_hint(
        "        found = self._find(state)\n"
        "        return 0 if found is None else found[0] + found[1]\n"
    )
    assert replay(build_transcript(), load(src)).passed


def test_one_distinct_board_passes_vacuously():
    """Nothing to distinguish -> no evidence either way -> do not reject."""
    g = with_agent_at(0, 0)
    t = GameTranscript(game_id="t")
    t.append(Transition(g, Action(name="ACTION1"), g, 0, 0, "NOT_FINISHED"))
    check = check_goal_hint(t, load(CONSTANT))
    assert check.ok and check.states_checked == 1


def test_goal_hint_gets_a_copy_it_may_mutate():
    """A goal_hint that mutates its input must not corrupt the transcript."""
    src = with_goal_hint(
        "        state[0][0][0] = 9\n"
        "        found = self._find(state)\n"
        "        return 0.0 if found is None else float(found[0] + found[1])\n"
    )
    t = build_transcript()
    before = repr([tr.frame_before for tr in t.transitions])
    replay(t, load(src))
    assert repr([tr.frame_before for tr in t.transitions]) == before


def test_rejection_text_puts_predict_first_then_goal_hint():
    t = build_transcript()
    wrong_predict = CONSTANT.replace("ny = max(0, py - 1)", "ny = py")
    result = replay(t, load(wrong_predict))
    text = describe_rejection(t, result)
    assert text.index("predict() was wrong") < text.index("Separately:")
    assert "same value" in text


def test_goal_only_rejection_says_keep_predict():
    t = build_transcript()
    text = describe_rejection(t, replay(t, load(CONSTANT)))
    assert "keep it" in text and "same value" in text
    assert "predict() was wrong" not in text


# =====================================================================
# the draft loop acts on the gate
# =====================================================================


def test_draft_rejects_a_constant_goal_hint_then_accepts_the_fix():
    client = FakeLLMClient([fenced(CONSTANT), fenced(CORRECT_WORLD_MODEL_SOURCE)])
    outcome = drafting.draft_world_model(client, build_transcript(), max_attempts=3)
    assert outcome.ok and outcome.attempts == 2
    assert outcome.source.strip() == CORRECT_WORLD_MODEL_SOURCE.strip()
    retry_prompt = client.prompts[1][1]
    assert "same value" in retry_prompt and "keep it" in retry_prompt


def test_draft_never_installs_a_constant_goal_hint():
    client = FakeLLMClient([fenced(CONSTANT)] * 3)
    outcome = drafting.draft_world_model(client, build_transcript(), max_attempts=3)
    assert not outcome.ok and outcome.world_model is None
    assert outcome.replay_result.predict_passed


# =====================================================================
# 3. the prompts ask for a usable goal_hint
# =====================================================================


def test_drafting_prompt_demands_a_non_constant_ordering_objective():
    p = drafting._SYSTEM_PROMPT
    assert "must NOT be constant" in p
    assert "ORDER" in p and "scale" in p
    assert "Return 0.0 if you have no idea" not in p


def test_skeleton_no_longer_invites_a_constant():
    from llm_engine.world_model import WORLD_MODEL_SKELETON
    assert "Return 0.0 if you have no idea" not in WORLD_MODEL_SKELETON
    assert "NOT be constant" in WORLD_MODEL_SKELETON


# =====================================================================
# the agent does not play a stalled planner's pick
# =====================================================================


def _stall_agent(code_world_agent_module, source):
    from test_code_world_agent import make_agent  # type: ignore[import-not-found]

    agent = make_agent(code_world_agent_module)
    agent._probe_index = len(agent._probe_plan)  # probes done
    agent.model = load(source)
    agent.model_source = source
    agent.action_budget.max_calls_per_game = 0  # the prototype's config
    return agent


def test_stalled_plan_falls_back_instead_of_playing_the_tie_winner(code_world_agent_module):
    agent = _stall_agent(code_world_agent_module, CONSTANT)
    sentinel = Action(name="ACTION7")
    agent._fallback_action = lambda available: sentinel
    chosen = agent._choose_engine_action(with_agent_at(2, 2), [])
    assert chosen is sentinel
    assert agent.plan_stats == {"calls": 1, "stalled": 1, "planned": 0}


def test_informative_plan_is_played_and_counted(code_world_agent_module):
    agent = _stall_agent(code_world_agent_module, CORRECT_WORLD_MODEL_SOURCE)
    agent._fallback_action = lambda available: pytest.fail("planner decision was discarded")
    chosen = agent._choose_engine_action(with_agent_at(2, 2), [])
    assert chosen is not None
    assert agent.plan_stats == {"calls": 1, "stalled": 0, "planned": 1}
