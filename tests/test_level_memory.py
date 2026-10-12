"""Solved-level memory for the milestone-2 harness.

Unit tests run on a fake agent. The integration tests run the REAL patched
ToolAgent (base bundle + dfranzen's harness patch): ARC3_M2_SRC, or the src that
tests/m2_harness.py rebuilds from a local base bundle checkout; skipped when neither exists.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle_submission_milestone2_fork" / "level_memory"))
import level_memory as lm  # noqa: E402


def entry(action, level):
    return SimpleNamespace(action=action, frame=SimpleNamespace(level=level))


def game(level1_actions, level2_actions=()):
    hist = [entry("", 1)]
    hist += [entry(a, 1) for a in level1_actions[:-1]]
    hist.append(entry(level1_actions[-1], 2))          # the winning action lands on level 2's frame
    hist += [entry(a, 2) for a in level2_actions]
    return hist


class FakeAgent:
    def __init__(self):
        self._system_prompt = "SYSTEM"
        self._history_messages = []
        self._context_was_trimmed = False
        self._resume_after_yield = False


TRANSITION = {"level_transition": True, "level": 2}
RULE = "Rule for level 1: push the red block onto the blue target; UP/DOWN move the player, ACTION5 grabs."


def test_level_actions_attributes_each_action_to_the_level_it_was_taken_on():
    hist = game(["UP", "UP", "LEFT"], ["DOWN"])
    assert lm.level_actions(hist, 1) == ["UP", "UP", "LEFT"]
    assert lm.level_actions(hist, 2) == ["DOWN"]


def test_find_rule_reads_one_line_or_a_short_bulleted_block():
    assert lm.find_rule("blah\n" + RULE + "\nmore", 1).startswith("push the red block")
    bulleted = "Rule for level 3:\n- goal: fill the frame\n- ACTION6 paints a cell\n\nnext paragraph"
    assert lm.find_rule(bulleted, 3) == "- goal: fill the frame - ACTION6 paints a cell"
    assert lm.find_rule("Rule for level 1: short", 1) is None
    assert lm.find_rule(RULE, 2) is None, "the level number must match"


def test_transition_records_facts_and_asks_for_the_rule_once_per_turn():
    a = FakeAgent()
    a._history_messages = [{"role": "assistant", "content": "I cleared it by pushing left.",
                            "reasoning_content": "thinking about the block"}]
    out = lm.after_user_prompt(a, "PROMPT", previous_step_summary=TRANSITION,
                               current_frame=SimpleNamespace(level=2), history_entries=game(["UP"] * 40))
    assert "Rule for level 1:" in out and out.startswith("PROMPT")
    rec = a._lm_state["levels"][0]
    assert rec["level"] == 1 and rec["n_actions"] == 40 and len(rec["actions"]) == lm.ACTIONS_KEPT
    assert "pushing left" in rec["reasoning"] and "thinking about the block" in rec["reasoning"]
    # the same transition summary seen again does not record a duplicate
    lm.after_user_prompt(a, "P", previous_step_summary=TRANSITION, current_frame=SimpleNamespace(level=2),
                         history_entries=game(["UP"] * 40))
    assert len(a._lm_state["levels"]) == 1


def test_rule_is_captured_from_the_reply_and_asking_stops():
    a = FakeAgent()
    lm.after_user_prompt(a, "P", previous_step_summary=TRANSITION, current_frame=SimpleNamespace(level=2),
                         history_entries=game(["UP"]))
    a._history_messages.append({"role": "assistant", "content": "ok", "reasoning_content": RULE})
    out = lm.after_user_prompt(a, "P2", previous_step_summary={}, current_frame=SimpleNamespace(level=2),
                               history_entries=game(["UP"]))
    assert out == "P2", "no ask once the rule is captured"
    assert a._lm_state["levels"][0]["rule"].startswith("push the red block")


def test_asking_gives_up_after_ask_turns():
    a = FakeAgent()
    outs = [lm.after_user_prompt(a, "P", previous_step_summary=TRANSITION if i == 0 else {},
                                 current_frame=SimpleNamespace(level=2), history_entries=game(["UP"]))
            for i in range(5)]
    assert sum("Rule for level 1:" in o for o in outs) == lm.ASK_TURNS


def test_no_ask_on_a_resumed_turn_whose_opener_is_replaced():
    a = FakeAgent()
    a._resume_after_yield = True
    out = lm.after_user_prompt(a, "P", previous_step_summary=TRANSITION, current_frame=SimpleNamespace(level=2),
                               history_entries=game(["UP"]))
    assert out == "P" and a._lm_state["levels"][0]["asks"] == 0
    a._resume_after_yield = False
    assert "Rule for level 1:" in lm.after_user_prompt(a, "P", previous_step_summary={},
                                                       current_frame=SimpleNamespace(level=2),
                                                       history_entries=game(["UP"]))


def test_block_is_pinned_only_when_history_was_evicted():
    a = FakeAgent()
    lm.after_user_prompt(a, "P", previous_step_summary=TRANSITION, current_frame=SimpleNamespace(level=2),
                         history_entries=game(["UP", "LEFT"]))
    msgs = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "u"}]
    assert lm.apply_on_evict(a, msgs) is msgs and a._system_prompt == "SYSTEM", "no eviction -> cache kept"
    a._context_was_trimmed = True
    out = lm.apply_on_evict(a, msgs)
    assert out[0]["content"].startswith("SYSTEM\n\n" + lm.HEADER) and a._system_prompt == out[0]["content"]
    assert "UP, LEFT" in out[0]["content"] and out[1] is msgs[1]
    again = lm.apply_on_evict(a, out)
    assert again is out, "an unchanged block does not rewrite the prompt"


def test_older_levels_keep_only_their_rule():
    levels = [dict(level=i, n_actions=5, actions=["UP"], reasoning="r", rule=f"rule {i} " * 5, asks=1, checks=1)
              for i in range(1, 6)]
    block = lm.render_block(levels)
    assert block.count("Last actions before it cleared") == lm.FULL_LEVELS
    assert block.count("Your rule:") == 5


def test_reset_restores_the_unpinned_prompt_for_a_new_game():
    a = FakeAgent()
    lm.after_user_prompt(a, "P", previous_step_summary=TRANSITION, current_frame=SimpleNamespace(level=2),
                         history_entries=game(["UP"]))
    a._context_was_trimmed = True
    lm.apply_on_evict(a, [{"role": "system", "content": "SYSTEM"}])
    assert a._system_prompt != "SYSTEM"
    lm.reset(a)
    assert a._system_prompt == "SYSTEM" and a._lm_state is None


# ----------------------------------------------------------------------------- real harness
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_harness  # noqa: E402

M2_SRC = m2_harness.m2_src()
real = pytest.mark.skipif(not M2_SRC, reason="milestone-2 src unavailable (ARC3_M2_SRC / ARC3_M2_BASE)")
_LM_WRAPPED = ("_build_user_prompt", "_trim_messages_for_context", "_ensure_session")


@pytest.fixture()
def real_agent():
    saved_env = dict(os.environ)
    if "inference.agent.tool_agent" not in sys.modules:
        os.environ.update(m2_harness.notebook_env())       # module constants read them at import
    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")
    os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "flashnext")
    ta, rs, _, _ = m2_harness.import_harness(M2_SRC)
    cls = ta.ToolAgent
    saved = {n: cls.__dict__[n] for n in _LM_WRAPPED} if not getattr(cls, "_lm_installed", False) else None
    lm.install(cls)
    try:
        yield ta, cls(model="local"), rs.Frame, rs.HistoryEntry
    finally:
        if saved is not None:          # leave the class as found, for the other real-harness tests
            for n, f in saved.items():
                setattr(cls, n, f)
            del cls._lm_installed
        os.environ.clear()
        os.environ.update(saved_env)


def _frame(Frame, level, step):
    grid = [[0] * 8 for _ in range(8)]
    try:
        return Frame(grid=grid, step=step, level=level)
    except TypeError:
        import inspect
        params = inspect.signature(Frame).parameters
        kw = {k: (grid if k == "grid" else step if k == "step" else level if k == "level" else None) for k in params}
        return Frame(**kw)


@real
def test_real_harness_methods_are_wrapped(real_agent):
    ta, agent, _, _ = real_agent
    assert ta.ToolAgent._lm_installed
    for name in ("_build_user_prompt", "_trim_messages_for_context", "_ensure_session"):
        assert getattr(ta.ToolAgent, name).__module__ == "level_memory"


@real
def test_real_user_prompt_carries_the_rule_request(real_agent, tmp_path):
    ta, agent, Frame, HistoryEntry = real_agent
    agent._ensure_session(tmp_path / "state.json")
    hist = [HistoryEntry(action="", frame=_frame(Frame, 1, 0)),
            HistoryEntry(action="UP", frame=_frame(Frame, 1, 1)),
            HistoryEntry(action="LEFT", frame=_frame(Frame, 2, 2))]
    out = agent._build_user_prompt(2, valid_actions=["UP", "LEFT"], current_frame=_frame(Frame, 2, 2),
                                   history_entries=hist,
                                   previous_step_summary={"level_transition": True, "level": 2,
                                                          "executed_actions": ["LEFT"]})
    assert "Rule for level 1:" in out
    assert agent._lm_state["levels"][0]["actions"] == ["UP", "LEFT"]


@real
def test_real_trim_pins_the_block_exactly_when_it_evicts(real_agent, tmp_path):
    ta, agent, Frame, HistoryEntry = real_agent
    agent._ensure_session(tmp_path / "state.json")
    base = agent._system_prompt
    agent._lm_state = None
    lm._state(agent)["levels"].append(dict(level=1, n_actions=2, actions=["UP", "LEFT"], reasoning="r",
                                           rule="push the block onto the target " * 2, asks=1, checks=1))
    small = [{"role": "system", "content": base}, {"role": "user", "content": "hi"}]
    out = agent._trim_messages_for_context(small)
    assert out[0]["content"] == base, "nothing evicted -> system prompt untouched"
    filler = "x " * 40000
    big = [{"role": "system", "content": base}]
    for i in range(40):
        big += [{"role": "user", "content": f"turn {i} {filler}"}, {"role": "assistant", "content": f"ok {i}"}]
    agent._context_was_trimmed = False
    out = agent._trim_messages_for_context(big)
    assert len(out) < len(big), "the real trimmer evicted history"
    assert lm.HEADER in out[0]["content"] and agent._system_prompt == out[0]["content"]
    assert out[0]["content"].startswith(base)
