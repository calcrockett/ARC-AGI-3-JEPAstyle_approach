"""Momentum time allocation for the anim solver: each rule, the time the
model is shown, and the install hook -- against a fake clock and session."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle_submission_duck_nvfp4_anim_momentum"))
from momentum_time import BASE_S, MomentumTimePolicy, install  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(levels=0, gid="g1", start=1000.0):
    return SimpleNamespace(
        started_at=start,
        game=SimpleNamespace(current_state=SimpleNamespace(levels_completed=levels),
                             game_run=SimpleNamespace(game_id=gid)),
    )


@pytest.fixture()
def pc():
    c = Clock()
    return MomentumTimePolicy(clock=c), c


def at(c, s, elapsed):
    c.t = s.started_at + elapsed


def started(p, c, **kw):
    """A session the policy has seen at elapsed 0 -- as in the real solver,
    whose play() loop calls should_stop() (and so the policy) before the
    first turn."""
    s = make(**kw)
    at(c, s, 0)
    assert not p.limit_reached(s)
    return s


def test_before_base_nothing_changes_while_leveling(pc):
    p, c = pc
    s = started(p, c)
    for minute in range(0, 130, 10):
        at(c, s, minute * 60)
        if minute % 60 == 0:
            s.game.current_state.levels_completed += 1
        assert not p.limit_reached(s)


def test_stuck_game_is_stopped_at_the_stall_threshold(pc):
    p, c = pc
    s = started(p, c)
    at(c, s, p.stall_s - 1)
    assert not p.limit_reached(s)
    at(c, s, p.stall_s)
    assert p.limit_reached(s)
    assert p.decisions[-1][1] == "stall"


def test_a_level_up_resets_the_stall_clock(pc):
    p, c = pc
    s = started(p, c)
    at(c, s, p.stall_s - 60)
    s.game.current_state.levels_completed = 1
    assert not p.limit_reached(s)
    at(c, s, p.stall_s + 60)
    assert not p.limit_reached(s), "stall is measured from the last level-up"


def test_no_momentum_at_base_stops_exactly_like_the_baseline(pc):
    p, c = pc
    s = started(p, c)
    at(c, s, BASE_S - 60 * 60)
    s.game.current_state.levels_completed = 1   # 60 min before BASE: no momentum, not stalled
    p.limit_reached(s)
    at(c, s, BASE_S - 1)
    assert not p.limit_reached(s)
    at(c, s, BASE_S)
    assert p.limit_reached(s)
    assert p.decisions[-1][1] == "base_cap"


def test_momentum_at_base_extends_the_game(pc):
    p, c = pc
    s = started(p, c)
    at(c, s, BASE_S - 10 * 60)
    s.game.current_state.levels_completed = 3
    p.limit_reached(s)
    at(c, s, BASE_S + 1)
    assert not p.limit_reached(s)
    assert [d[1] for d in p.decisions] == ["extended"]


def test_extended_game_stops_when_momentum_runs_out(pc):
    p, c = pc
    s = started(p, c)
    at(c, s, BASE_S - 10 * 60)
    s.game.current_state.levels_completed = 1
    p.limit_reached(s)
    at(c, s, BASE_S + 1)
    assert not p.limit_reached(s)
    at(c, s, BASE_S - 10 * 60 + p.momentum_s)   # 45 min after the last level-up
    assert p.limit_reached(s)
    assert p.decisions[-1][1] == "base_cap"


def test_extension_keeps_going_while_leveling_but_never_past_the_hard_cap(pc):
    p, c = pc
    s = started(p, c)
    t = BASE_S - 60
    while t < p.hard_s + 600:
        at(c, s, t)
        s.game.current_state.levels_completed += 1   # a level every 10 minutes
        stopped = p.limit_reached(s)
        assert stopped == (t >= p.hard_s), t
        if stopped:
            break
        t += 600
    assert p.decisions[-1][1] == "hard_cap"


def test_a_stopped_game_stays_stopped(pc):
    p, c = pc
    s = started(p, c)
    at(c, s, p.stall_s)
    assert p.limit_reached(s)
    s.game.current_state.levels_completed = 5
    assert p.limit_reached(s)


def test_model_sees_the_baseline_time_until_extended(pc):
    p, c = pc
    s = started(p, c)
    at(c, s, 1000)
    assert p.timing_payload(s)["time_remaining_seconds"] == pytest.approx(BASE_S - 1000)
    at(c, s, BASE_S - 300)
    s.game.current_state.levels_completed = 2
    p.limit_reached(s)
    assert p.timing_payload(s)["time_remaining_seconds"] == pytest.approx(300)
    at(c, s, BASE_S + 100)
    assert not p.limit_reached(s)
    remaining = p.timing_payload(s)["time_remaining_seconds"]
    assert remaining == pytest.approx(p.hard_s - BASE_S - 100)
    assert remaining > 0, "an extended game must not get 0 s request timeouts"


def test_games_are_tracked_independently(pc):
    p, c = pc
    a, b = started(p, c, gid="a"), started(p, c, gid="b")
    at(c, b, BASE_S - 60 * 60)
    b.game.current_state.levels_completed = 1   # no momentum by BASE
    p.limit_reached(b)
    at(c, a, BASE_S - 60)
    a.game.current_state.levels_completed = 1
    p.limit_reached(a)
    at(c, a, BASE_S + 1)
    assert not p.limit_reached(a)
    assert p.limit_reached(b)
    s = p.summary()
    assert s["games_seen"] == 2 and s["decisions"] == {"extended": 1, "base_cap": 1}


def test_broken_level_readout_never_stops_a_game_early(pc):
    p, c = pc
    s = SimpleNamespace(started_at=1000.0, game=None)
    at(c, s, 60)
    assert not p.limit_reached(s)


def test_install_replaces_both_methods():
    class Session:
        started_at = 0.0
        game = SimpleNamespace(current_state=SimpleNamespace(levels_completed=0),
                               game_run=SimpleNamespace(game_id="x"))

        def runtime_limit_reached(self):
            raise AssertionError("original must be replaced")

        def timing_payload(self):
            raise AssertionError("original must be replaced")

    c = Clock()
    c.t = 5.0
    policy = MomentumTimePolicy(clock=c)
    install(Session, policy)
    s = Session()
    assert s.runtime_limit_reached() is False
    assert s.timing_payload()["time_remaining_seconds"] == pytest.approx(BASE_S - 5.0)
    assert Session._momentum_policy is policy
