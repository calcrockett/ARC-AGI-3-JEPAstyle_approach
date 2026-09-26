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
from llm_engine.planner import candidate_actions, next_action, plan
from llm_engine.replay import check_goal_hint, describe_rejection, observed_states, replay
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
    assert "cannot tell actions apart" in result.goal_hint.problem
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


def test_one_distinct_board_is_no_longer_a_vacuous_pass():
    """The first gate compared goal_hint across the transcript's history,
    so a transcript with one distinct board passed automatically -- which is
    how lp85's model got in. The separation check predicts successors from
    that board, so a constant objective is caught even here."""
    g = with_agent_at(2, 2)
    t = GameTranscript(game_id="t")
    t.append(Transition(g, Action(name="ACTION5"), g, 0, 0, "NOT_FINISHED"))
    check = check_goal_hint(t, load(CONSTANT))
    assert not check.ok and check.informative_boards == 1


IDENTITY_PREDICT = with_goal_hint("        return 0.0\n").replace(
    "    def predict(self, state, action_name, x=None, y=None):\n",
    "    def predict(self, state, action_name, x=None, y=None):\n"
    "        return [[row[:] for row in state[0]]], 0, False\n",
)


def test_simulator_that_never_distinguishes_actions_is_not_blamed_on_goal_hint():
    """predict() says every action does the same thing from every board:
    nothing to separate, so no goal_hint could help and no retry is spent."""
    g = with_agent_at(2, 2)
    t = GameTranscript(game_id="t")
    t.append(Transition(g, Action(name="ACTION5"), g, 0, 0, "NOT_FINISHED"))
    check = check_goal_hint(t, load(IDENTITY_PREDICT))
    assert check.ok
    assert check.predict_distinguishes_actions is False
    assert check.informative_boards == 0


# ---------------------------------------------------------------------------
# The live failure, 2026-09-26: a goal_hint reading an edge strip that every
# action advances identically. It varies across the transcript's history --
# so the first gate passed it -- and ties across every action.
# ---------------------------------------------------------------------------

HUD_N = 20  # >= replay.MIN_BAND_BOARD, so the edge band applies
HUD_ROW = HUD_N - 1
HUD_COLOR = 9


def _hud_board(ax: int, ay: int, bar: int):
    g = [[0] * HUD_N for _ in range(HUD_N)]
    g[ay][ax] = 3
    for x in range(min(bar, HUD_N)):
        g[HUD_ROW][x] = HUD_COLOR
    return [g]


LO, HI = 3, HUD_N - 4  # the agent lives in the interior, outside the edge band


def _hud_true_step(ax, ay, bar, action):
    dx, dy = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}.get(action, (0, 0))
    nx = min(max(ax + dx, LO), HI)
    ny = min(max(ay + dy, LO), HI)
    return nx, ny, bar + 1


HUD_MODEL_TEMPLATE = """\
class WorldModel:
    def _find(self, state):
        for y in range(%(lo)d, %(hi)d + 1):
            for x in range(%(lo)d, %(hi)d + 1):
                if state[0][y][x] == 3:
                    return x, y
        return None

    def predict(self, state, action_name, x=None, y=None):
        n = %(n)d
        layer = [row[:] for row in state[0]]
        bar = sum(1 for c in layer[n - 1] if c == %(hud)d)
        found = self._find(state)
        if found is not None:
            px, py = found
            d = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}
            dx, dy = d.get(action_name, (0, 0))
            nx = min(max(px + dx, %(lo)d), %(hi)d)
            ny = min(max(py + dy, %(lo)d), %(hi)d)
            layer[py][px] = 0
            layer[ny][nx] = 3
        if bar < n:
            layer[n - 1][bar] = %(hud)d
        return [layer], 0, False

    def goal_hint(self, state):
%(goal)s
"""


def _hud_model(goal_body: str):
    return load(HUD_MODEL_TEMPLATE % {"n": HUD_N, "hud": HUD_COLOR, "goal": goal_body, "lo": LO, "hi": HI})


def _hud_transcript():
    t = GameTranscript(game_id="hud")
    ax, ay, bar = 8, 8, 0
    for action in ["ACTION4", "ACTION2", "ACTION4", "ACTION1", "ACTION3"]:
        nx, ny, nbar = _hud_true_step(ax, ay, bar, action)
        t.append(Transition(_hud_board(ax, ay, bar), Action(name=action),
                            _hud_board(nx, ny, nbar), 0, 0, "NOT_FINISHED"))
        ax, ay, bar = nx, ny, nbar
    return t


HUD_COUNTER_GOAL = "        return float(sum(1 for c in state[0][%d] if c == %d))" % (HUD_ROW, HUD_COLOR)
PLAYFIELD_GOAL = (
    "        f = self._find(state)\n"
    "        return 0.0 if f is None else float(f[0])"
)


def test_hud_fixture_predict_is_exact():
    """The model under test must be a perfect simulator, or the goal_hint
    tests below would be testing predict() instead."""
    result = replay(_hud_transcript(), _hud_model(PLAYFIELD_GOAL))
    assert result.predict_passed


def test_hud_counter_varies_over_history_so_the_first_gate_would_pass_it():
    t = _hud_transcript()
    m = _hud_model(HUD_COUNTER_GOAL)
    history = {m.goal_hint(b) for b in observed_states(t)}
    assert len(history) == len(observed_states(t)) > 1


def test_hud_counter_goal_is_rejected_because_it_ties_across_actions():
    result = replay(_hud_transcript(), _hud_model(HUD_COUNTER_GOAL))
    assert result.predict_passed
    assert not result.passed
    problem = result.goal_hint.problem
    assert "cannot tell actions apart" in problem
    assert "edge, where step counters" in problem
    # It says WHERE the actions' outcomes differ -- the playfield, never the
    # HUD row -- so the retry is pointed at the part of the board that matters.
    import re
    lo, hi = map(int, re.search(r"rows (\d+)-(\d+)", problem).groups())
    assert 0 <= lo <= hi < HUD_ROW


COLOUR_COUNT_GOAL = "        return float(sum(1 for r in state[0] for c in r if c == 3))"


def test_colour_count_is_blind_to_movement_and_is_rejected():
    """Counting colours -- what the live models kept writing -- gives every
    move the same score: a moved object has the same colour counts."""
    result = replay(_hud_transcript(), _hud_model(COLOUR_COUNT_GOAL))
    assert result.predict_passed and not result.passed
    assert "counting colours will not work" in result.goal_hint.problem


def test_objective_reading_both_hud_and_playfield_is_accepted():
    """Reading the HUD is not banned -- ignoring the playfield is."""
    both = HUD_COUNTER_GOAL.replace("        return ", "        f = self._find(state)\n        return (0.0 if f is None else f[0]) + ")
    assert replay(_hud_transcript(), _hud_model(both)).passed


def test_playfield_goal_passes():
    result = replay(_hud_transcript(), _hud_model(PLAYFIELD_GOAL))
    assert result.passed and result.goal_hint.informative_boards >= 1


def test_the_planner_stalls_on_the_hud_goal_and_plans_on_the_playfield_goal():
    """The gate's verdict matches what the planner actually does."""
    board = _hud_board(8, 8, 3)
    assert plan(_hud_model(HUD_COUNTER_GOAL), board, rng=random.Random(0)).stalled
    assert not plan(_hud_model(PLAYFIELD_GOAL), board, rng=random.Random(0)).stalled


def test_separation_probe_is_bounded():
    from llm_engine import replay as replay_mod

    calls = {"n": 0}
    m = _hud_model(PLAYFIELD_GOAL)
    orig = m.predict

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    m.predict = counting
    t = GameTranscript(game_id="long")
    ax, ay, bar = 8, 8, 0
    for i in range(60):
        action = ["ACTION4", "ACTION2"][i % 2]
        nx, ny, nbar = _hud_true_step(ax, ay, bar % HUD_N, action)
        t.append(Transition(_hud_board(ax, ay, bar % HUD_N), Action(name=action),
                            _hud_board(nx, ny, nbar % HUD_N), 0, 0, "NOT_FINISHED"))
        ax, ay, bar = nx, ny, nbar
    check = replay_mod.check_goal_hint(t, m)
    assert check.ok
    n_cands = len(candidate_actions())
    assert calls["n"] <= replay_mod.MAX_SEPARATION_BOARDS * n_cands


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
    assert "cannot tell actions apart" in text


def test_goal_only_rejection_says_keep_predict():
    t = build_transcript()
    text = describe_rejection(t, replay(t, load(CONSTANT)))
    assert "keep it" in text and "cannot tell actions apart" in text
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
    assert "cannot tell actions apart" in retry_prompt and "keep it" in retry_prompt


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
    assert "tell ACTIONS apart" in p
    assert "edge of the board" in p
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


def test_agent_does_not_draft_before_the_board_has_changed(code_world_agent_module):
    """One distinct board: no rule to infer, no way to check goal_hint."""
    from test_code_world_agent import make_agent  # type: ignore[import-not-found]

    agent = make_agent(code_world_agent_module, coder_responses=[fenced(CORRECT_WORLD_MODEL_SOURCE)] * 3)
    g = with_agent_at(2, 2)
    for _ in range(code_world_agent_module.CodeWorldAgent.REDRAFT_AFTER_NEW_TRANSITIONS + 2):
        agent.transcript.append(Transition(g, Action(name="ACTION5"), g, 0, 0, "NOT_FINISHED"))
    agent._maybe_draft_model()
    assert agent.model is None
    assert agent.coder_budget.calls_used == 0, "no coder call may be spent on zero evidence"
    assert agent._last_draft_attempt_len == -1, "the first change must trigger a draft at once"


def test_agent_drafts_once_the_board_has_changed(code_world_agent_module):
    from test_code_world_agent import make_agent  # type: ignore[import-not-found]

    agent = make_agent(code_world_agent_module, coder_responses=[fenced(CORRECT_WORLD_MODEL_SOURCE)] * 3)
    agent.transcript = build_transcript()
    agent._maybe_draft_model()
    assert agent.model is not None


# =====================================================================
# legal moves only (2026-09-26): on kernel v4, 164 of 324 planner decisions
# were actions the game does not accept -- ft09, click-only, 96 of 96
# =====================================================================

from llm_engine.types import allowed_action_names  # noqa: E402


def test_allowed_action_names():
    assert allowed_action_names(None) is None
    assert allowed_action_names([]) is None, "empty = not reported = no restriction"
    assert allowed_action_names([6]) == {"ACTION6"}
    assert allowed_action_names([0, 1, 2, 99]) == {"ACTION1", "ACTION2"}, "RESET/unknown ids ignored"


def test_candidates_respect_the_allowed_set():
    assert {a.name for a in candidate_actions({"ACTION6"})} == {"ACTION6"}
    assert len(candidate_actions({"ACTION6"})) == 4, "the four click sample points"
    names = {a.name for a in candidate_actions({"ACTION1", "ACTION2", "ACTION3", "ACTION4"})}
    assert names == {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
    assert len(candidate_actions()) == len(candidate_actions(None))


def test_planner_never_picks_a_move_the_game_does_not_accept():
    model = load(CORRECT_WORLD_MODEL_SOURCE)
    for seed in range(30):
        for avail, legal in (([1, 2, 3, 4], {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}),
                             ([6], {"ACTION6"})):
            r = plan(model, with_agent_at(2, 2), rng=random.Random(seed), available_actions=avail)
            assert all(a.name in legal for a in r.actions), (avail, [a.name for a in r.actions])


def test_gate_checks_goal_hint_only_against_accepted_moves():
    """HUD fixture: the playfield goal separates the arrow keys. If the game
    only accepts ACTION5 -- which moves nothing -- there is nothing the
    planner could choose between, so the gate must see no informative board."""
    m = _hud_model(PLAYFIELD_GOAL)
    t = _hud_transcript()
    t.available_actions = [1, 2, 3, 4]
    assert check_goal_hint(t, m).informative_boards >= 1
    t.available_actions = [5]
    c = check_goal_hint(t, m)
    assert c.ok and c.informative_boards == 0 and c.predict_distinguishes_actions is False


def test_agent_skips_probes_the_game_does_not_accept(code_world_agent_module):
    from test_code_world_agent import make_agent  # type: ignore[import-not-found]

    agent = make_agent(code_world_agent_module)
    first = agent._choose_engine_action(with_agent_at(2, 2), [6])
    assert first.name == "ACTION6", "simple-action probes must be skipped in a click-only game"
    assert agent.transcript.available_actions == [6]


def test_agent_passes_available_actions_to_the_planner(code_world_agent_module):
    agent = _stall_agent(code_world_agent_module, CORRECT_WORLD_MODEL_SOURCE)
    agent._fallback_action = lambda available: pytest.fail("a legal planner decision was discarded")
    for _ in range(10):
        chosen = agent._choose_engine_action(with_agent_at(2, 2), [6])
        assert chosen.name == "ACTION6"
    assert agent.plan_stats["planned"] == 10
