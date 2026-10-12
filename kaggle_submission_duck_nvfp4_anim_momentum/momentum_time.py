"""Momentum time allocation for the anim (Duck/TAAF) solver.

The run is time-bound: every game plays until its fixed 7,920 s clock runs
out. Measured on 225 game-runs from 9 saved public-25 runs of this solver
family (experiments/stage7_momentum_time.md):

  * a turn in a game that leveled up within the last ~9 turns earns 0.423
    completion points; a turn in a game stuck 10+ turns earns 0.155 -- 2.7x
    less -- and stuck turns are 35% of all turns;
  * the level-up rate of a game WITH momentum does not fall with depth
    (0.065 / 0.080 / 0.082 per turn at levels 0-1 / 2-3 / 4+);
  * 48% of games were still progressing when the clock cut them off.

Simulated on a 110-game hidden run (28 slots), merely stopping stuck games
HURTS -- every game gets started anyway, so the freed time has nowhere to go.
What helps is the opposite emphasis: do not cut a game that is still
progressing at 7,920 s, and pay for it by stopping games stuck a long time.

Policy (all times wall-clock seconds, per game, from the moment it starts):
  * stop a game whose last level-up (or start) was >= STALL_S ago;
  * at BASE_S, stop -- unless it leveled up within the last MOMENTUM_S, in
    which case it continues;
  * never beyond HARD_MULT * BASE_S.
The framework's own soft deadline still cancels everything at the notebook
budget, so an extension can never overrun it.

The model is shown `time_remaining_seconds`. Until BASE_S it is shown exactly
what the unmodified solver would show; only an extended game sees the
extension. Request timeouts derive from the same number.

Installed by replacing two methods on `_HarnessGameSession` -- no bundle file
is edited.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

BASE_S = 7920.0
MOMENTUM_S = 45 * 60.0
STALL_S = 100 * 60.0
HARD_MULT = 1.5


@dataclass
class _GameClock:
    started_at: float
    levels: int
    last_progress_at: float
    extended: bool = False
    stopped_reason: Optional[str] = None


@dataclass
class MomentumTimePolicy:
    base_s: float = BASE_S
    momentum_s: float = MOMENTUM_S
    stall_s: float = STALL_S
    hard_mult: float = HARD_MULT
    clock: Callable[[], float] = time.monotonic
    _games: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    #: One record per game whose clock was changed from the baseline's:
    #: (game_id, reason, elapsed_s, levels). Reasons: "stall", "base_cap",
    #: "extended", "hard_cap".
    decisions: list = field(default_factory=list)

    @property
    def hard_s(self) -> float:
        return self.hard_mult * self.base_s

    # -- bookkeeping -----------------------------------------------------

    def _state(self, session: Any) -> _GameClock:
        key = id(session)
        with self._lock:
            st = self._games.get(key)
            if st is None:
                started = float(getattr(session, "started_at", self.clock()))
                st = _GameClock(started, _levels(session), started)
                self._games[key] = st
            return st

    def _observe(self, session: Any, now: float) -> _GameClock:
        st = self._state(session)
        levels = _levels(session)
        if levels > st.levels:
            st.levels = levels
            st.last_progress_at = now
        return st

    def _record(self, session: Any, st: _GameClock, reason: str, now: float) -> None:
        with self._lock:
            self.decisions.append((_game_id(session), reason, round(now - st.started_at, 1), st.levels))

    # -- the two replaced methods ----------------------------------------

    def limit_reached(self, session: Any) -> bool:
        now = self.clock()
        st = self._observe(session, now)
        if st.stopped_reason is not None:
            return True
        elapsed = now - st.started_at
        since_progress = now - st.last_progress_at
        reason = None
        if since_progress >= self.stall_s:
            reason = "stall"
        elif elapsed >= self.hard_s:
            reason = "hard_cap"
        elif elapsed >= self.base_s:
            if since_progress < self.momentum_s:
                if not st.extended:
                    st.extended = True
                    self._record(session, st, "extended", now)
                return False
            reason = "base_cap"
        if reason is None:
            return False
        st.stopped_reason = reason
        # base_cap at BASE_S on a non-extended game is exactly the baseline's
        # own stop; recorded anyway so the run reports every decision.
        self._record(session, st, reason, now)
        return True

    def effective_limit_s(self, session: Any) -> float:
        """The horizon the game is told about: the baseline's BASE_S, or the
        hard cap once the game has been extended past it."""
        st = self._state(session)
        return self.hard_s if st.extended else self.base_s

    def timing_payload(self, session: Any) -> dict:
        elapsed = max(0.0, self.clock() - self._state(session).started_at)
        remaining = max(0.0, self.effective_limit_s(session) - elapsed)
        return {"run_elapsed_seconds": elapsed, "time_remaining_seconds": remaining}

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict:
        with self._lock:
            counts: dict = {}
            for _gid, reason, _el, _lv in self.decisions:
                counts[reason] = counts.get(reason, 0) + 1
            return {
                "params": {"base_s": self.base_s, "momentum_s": self.momentum_s,
                           "stall_s": self.stall_s, "hard_mult": self.hard_mult},
                "games_seen": len(self._games),
                "decisions": counts,
                "log": list(self.decisions),
            }


def _levels(session: Any) -> int:
    try:
        return int(session.game.current_state.levels_completed)
    except Exception:  # noqa: BLE001 -- never let bookkeeping stop a game
        return 0


def _game_id(session: Any) -> str:
    run = getattr(getattr(session, "game", None), "game_run", None)
    return str(getattr(run, "game_id", "?"))


def install(session_cls: Any, policy: MomentumTimePolicy) -> None:
    """Replace `runtime_limit_reached` and `timing_payload` on the solver's
    session class. `request_timeout_seconds` reads `timing_payload`, so it
    follows automatically."""
    session_cls.runtime_limit_reached = lambda self: policy.limit_reached(self)
    session_cls.timing_payload = lambda self: policy.timing_payload(self)
    session_cls._momentum_policy = policy
