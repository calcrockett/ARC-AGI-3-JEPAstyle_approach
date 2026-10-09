"""Tried-facts extension of solved-level memory (LEVEL_MEMORY_TRIED_FACTS, default off)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle_submission_milestone2_fork" / "level_memory"))
import level_memory as lm  # noqa: E402


def entry(action, level, game_over=False, automatic=False):
    return SimpleNamespace(action=action, frame=SimpleNamespace(level=level),
                           result={"game_over": game_over, "automatic": automatic} if action else {})


def lives(level, lifespans, tail=()):
    """History entering `level` and dying after each len(lifespan) actions, then `tail` alive actions."""
    hist = [entry("", level)]
    for i, span in enumerate(lifespans):
        for j in range(span):
            hist.append(entry(f"A{i}_{j}", level, game_over=(j == span - 1)))
        hist.append(entry("RESET", level, automatic=True))
    hist += [entry(a, level) for a in tail]
    return hist


class FakeAgent:
    def __init__(self):
        self._system_prompt = "SYSTEM"
        self._history_messages = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "visible", "reasoning_content": "early. " + "r" * 800 + " FINAL THOUGHT"},
        ]
        self._context_was_trimmed = False
        self._resume_after_yield = False


@pytest.fixture()
def on(monkeypatch):
    monkeypatch.setenv(lm.TRIED_FACTS_ENV, "1")
    for k in lm.TF_STATS:
        lm.TF_STATS[k] = 0


@pytest.fixture()
def off(monkeypatch):
    monkeypatch.delenv(lm.TRIED_FACTS_ENV, raising=False)


def prompt(agent, hist, level=3):
    return lm.after_user_prompt(agent, "PROMPT", previous_step_summary={},
                                current_frame=SimpleNamespace(level=level), history_entries=hist)


def evict(agent, msgs):
    agent._context_was_trimmed = True
    return lm.apply_on_evict(agent, msgs)


MSGS = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]


def test_level_life_facts_counts_lives_and_excludes_resets_from_them():
    f = lm.level_life_facts(lives(3, [42, 42, 7], tail=["X", "Y"]), 3)
    assert [len(g) for g in f["game_overs"]] == [42, 42, 7]
    assert f["game_overs"][2][-1] == "A2_6"
    assert f["actions"] == 42 + 42 + 7 + 3 + 2          # three auto RESETs count as actions spent
    assert lm.level_life_facts(lives(3, [5]), 4) == {"level": 4, "actions": 0, "game_overs": []}


def test_other_levels_game_overs_are_not_counted():
    hist = lives(2, [4, 4])
    hist.append(entry("WIN", 3))                         # the action that cleared level 2 lands on level 3
    hist += lives(3, [9])[1:]
    f = lm.level_life_facts(hist, 3)
    assert [len(g) for g in f["game_overs"]] == [9]


def test_block_content_equal_game_overs_recent_runs_and_reasoning(on):
    a = FakeAgent()
    prompt(a, lives(3, [42, 42, 42, 42], tail=["UP"]))
    out = evict(a, MSGS)
    sp = out[0]["content"]
    assert sp.startswith("SYSTEM\n\n" + lm.TF_HEADER) and sp.endswith(lm.TF_FOOTER)
    assert "Level 3:" in sp and "4 game overs on it" in sp
    assert "All 4 game overs occurred after exactly 42 actions." in sp
    assert sp.count("Game over #") == lm.TRIED_RUNS and "Game over #2" in sp and "Game over #4" in sp
    assert "Game over #1" not in sp
    assert "A3_41" in sp and "A3_32" in sp and "A3_31" not in sp, "exactly the last 10 actions of a fatal run"
    assert "FINAL THOUGHT" in sp and "early." not in sp, "only the ~600 char tail of the reasoning"
    assert out[1:] == MSGS[1:]


def test_unequal_counts_are_listed(on):
    a = FakeAgent()
    prompt(a, lives(3, [10, 20, 30]))
    sp = evict(a, MSGS)[0]["content"]
    assert "Actions taken before each game over, in order: 10, 20, 30." in sp
    assert "exactly" not in sp


def test_facts_are_not_advice(on):
    a = FakeAgent()
    prompt(a, lives(3, [10, 10]))
    sp = evict(a, MSGS)[0]["content"].lower()
    body = sp.split(lm.TF_HEADER.lower())[1].split("your reasoning")[0]
    for word in ("should", "try ", "avoid", "do not", "don't", "must", "recommend", "instead"):
        assert word not in body, word


def test_hard_cap_three_kb_with_hostile_input(on):
    a = FakeAgent()
    long_action = "ACTION6(" + "9" * 40 + ")"
    hist = [entry("", 3)]
    for _ in range(60):
        for j in range(50):
            hist.append(entry(long_action, 3, game_over=(j == 49)))
        hist.append(entry("RESET", 3, automatic=True))
    a._history_messages = [{"role": "assistant", "content": "é" * 5000}]
    prompt(a, hist)
    text, truncated = lm.render_tried_facts(a._lm_state["facts"])
    assert truncated and len(text.encode("utf-8")) <= lm.TRIED_FACTS_MAX_BYTES
    assert text.startswith(lm.TF_HEADER) and "60 game overs" in text
    sp = evict(a, MSGS)[0]["content"]
    assert len(sp.split("\n\n", 1)[1].encode("utf-8")) <= lm.TRIED_FACTS_MAX_BYTES
    s = lm.summary()
    assert s["tried_facts_blocks"] == 1 and s["tried_facts_truncations"] == 1
    assert s["tried_facts_bytes_max"] <= lm.TRIED_FACTS_MAX_BYTES and s["tried_facts_avg_bytes"] > 0


def test_hard_cap_when_nothing_can_be_degraded_further(on):
    snap = {"level": 3, "actions": 5, "game_overs": [["x" * 4000]], "reasoning": "r" * 600}
    text, truncated = lm.render_tried_facts(snap)
    assert truncated and len(text.encode("utf-8")) <= lm.TRIED_FACTS_MAX_BYTES


def test_small_block_is_not_marked_truncated(on):
    a = FakeAgent()
    prompt(a, lives(3, [5]))
    evict(a, MSGS)
    s = lm.summary()
    assert s["tried_facts_blocks"] == 1 and s["tried_facts_truncations"] == 0


def test_composes_with_the_solved_level_block(on):
    a = FakeAgent()
    lm.after_user_prompt(a, "P", previous_step_summary={"level_transition": True, "level": 2},
                         current_frame=SimpleNamespace(level=2),
                         history_entries=[entry("", 1), entry("UP", 2)])
    prompt(a, lives(2, [6]), level=2)
    sp = evict(a, MSGS)[0]["content"]
    assert sp.index(lm.HEADER) < sp.index(lm.TF_HEADER)


def test_no_actions_on_level_means_no_block(on):
    a = FakeAgent()
    prompt(a, [entry("", 3)])
    assert evict(a, MSGS) is MSGS
    assert a._system_prompt == "SYSTEM"


def test_same_snapshot_does_not_rewrite_the_prompt_twice(on):
    a = FakeAgent()
    prompt(a, lives(3, [8]))
    first = evict(a, MSGS)
    second = evict(a, first)
    assert second is first


def test_no_mutation_of_pre_eviction_messages(on):
    a = FakeAgent()
    prompt(a, lives(3, [8, 8]))
    msgs = copy.deepcopy(MSGS)
    before = copy.deepcopy(msgs)
    a._context_was_trimmed = False
    assert lm.apply_on_evict(a, msgs) is msgs, "no eviction -> same object, nothing pinned"
    assert a._system_prompt == "SYSTEM"
    out = evict(a, msgs)
    assert msgs == before, "input list and its dicts are untouched"
    assert out[1:] == before[1:] and out[1] is msgs[1] and out[0]["content"] != before[0]["content"]


def test_flag_off_is_byte_identical_to_the_solved_level_only_behavior(off):
    base, flagged = FakeAgent(), FakeAgent()
    trans = {"level_transition": True, "level": 2}
    hist = [entry("", 1), entry("UP", 1), entry("LEFT", 2)]
    for ag in (base, flagged):
        lm.after_user_prompt(ag, "P", previous_step_summary=trans, current_frame=SimpleNamespace(level=2),
                             history_entries=hist)
    out_a = evict(base, MSGS)
    # a rich history of game overs: with the flag off it must be ignored entirely
    out_prompt = lm.after_user_prompt(flagged, "P", previous_step_summary={},
                                      current_frame=SimpleNamespace(level=2),
                                      history_entries=hist + lives(2, [9, 9])[1:])
    out_b = evict(flagged, MSGS)
    assert out_a == out_b and lm.TF_HEADER not in out_b[0]["content"]
    assert out_b[0]["content"] == "SYSTEM\n\n" + lm.render_block(flagged._lm_state["levels"])
    assert flagged._lm_state["facts"] is None
    assert out_prompt.startswith("P") and "tried_facts_blocks" not in lm.summary()


def test_flag_off_without_levels_pins_nothing(off):
    a = FakeAgent()
    prompt(a, lives(3, [8, 8]))
    assert evict(a, MSGS) is MSGS and a._system_prompt == "SYSTEM"


def test_reset_clears_facts_for_a_new_game(on):
    a = FakeAgent()
    prompt(a, lives(3, [8]))
    evict(a, MSGS)
    assert lm.TF_HEADER in a._system_prompt
    lm.reset(a)
    assert a._system_prompt == "SYSTEM" and a._lm_state is None


def test_reasoning_tail_falls_back_to_visible_reply():
    assert lm.reasoning_tail([{"role": "assistant", "content": "just a reply"}]) == "just a reply"
    assert lm.reasoning_tail([]) == ""
    assert len(lm.reasoning_tail([{"role": "assistant", "reasoning": "x" * 2000}])) == lm.TRIED_REASONING_CHARS


# ----------------------------------------------------------------------------- build script wiring
def test_build_script_variant_naming_and_install_lines():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import _build_m2_level_memory_kernel as b
    assert b.kernel_id(()) == "calamitychasm/arc3-m2-level-memory"
    assert b.kernel_id(["triedfacts"]) == "calamitychasm/arc3-m2-lm-triedfacts"
    plain = "".join(b.install_cell(())["source"])
    flagged = "".join(b.install_cell(["triedfacts"])["source"])
    assert "TRIED_FACTS installed" not in plain and "_os.environ" not in plain
    assert "TRIED_FACTS installed" in flagged and flagged.index("TRIED_FACTS installed") < flagged.index("LEVEL_MEMORY installed")
    assert b.VARIANT_MARKERS["triedfacts"] == "TRIED_FACTS installed"
