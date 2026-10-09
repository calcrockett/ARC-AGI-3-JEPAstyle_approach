"""Run the REAL patched milestone-2 ToolAgent locally (no model, no game engine).

`m2_src()` finds the reconstructed `src/` (base bundle + dfranzen's harness patch): ARC3_M2_SRC
if set, else it rebuilds one by `git apply`-ing the patch embedded in the upstream notebook onto
a local copy of the base bundle (ARC3_M2_BASE, or a known checkout path) -- exactly what the
notebook's setup cell does on Kaggle. None when neither is available (tests then skip).

`FakeGame` plays the solver's role: it appends HistoryEntry objects and persists them through
the solver's own module-level `write_runtime_state` (so the history cache's writer is the one
exercised once installed), one write per executed action, exactly like
`_HarnessGameSession._execute_action`; `step_env` executes a batch.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM_NB = REPO / "kaggle_submission_milestone2_fork" / "upstream" / "arc-agi-3-milestone-2-solution.ipynb"
BASE_CANDIDATES = [
    "/home/user/tonghuikang/daniel-franzen-arc-agi-3/kaggle/dfranzen/dataset-taaf-kaggle-source-bundle-copy/src",
]


def _cells():
    nb = json.loads(UPSTREAM_NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def notebook_env() -> dict[str, str]:
    """The analyzer/harness env knobs the notebook sets (its `setup_env` dict + priority block)."""
    cell = next(s for s in _cells() if "setup_env = {" in s)
    body = cell[cell.index("setup_env = {") + len("setup_env = "):]
    body = body[: body.index("\n}\n") + 2]
    env = eval(body, {"__builtins__": {}}, {"SERVER_BASE_URL": "http://127.0.0.1:9/v1",
                                            "SERVED_MODEL_NAME": "flashnext"})
    start = cell.index("setup_env.update(") + len("setup_env.update(")
    prio = cell[start: cell.index("})", start) + 1]
    env.update(eval(prio, {"__builtins__": {}}, {}))
    return {k: str(v) for k, v in env.items()}


def m2_src() -> Path | None:
    if os.environ.get("ARC3_M2_SRC"):
        return Path(os.environ["ARC3_M2_SRC"])
    base = os.environ.get("ARC3_M2_BASE") or next((b for b in BASE_CANDIDATES if Path(b).is_dir()), None)
    if not base or not Path(base).is_dir() or shutil.which("git") is None:
        return None
    patch = next(s for s in _cells() if s.startswith("%%writefile /kaggle/harness-changes.patch\n"))
    patch = patch.split("\n", 1)[1] + "\n"          # %%writefile ends the file with a newline
    tag = hashlib.sha256((patch + str(base)).encode()).hexdigest()[:12]
    out = Path(tempfile.gettempdir()) / f"arc3_m2src_{tag}"
    if (out / ".ok").exists():
        return out
    tmp = Path(tempfile.mkdtemp(prefix="arc3_m2src_build_"))
    shutil.copytree(base, tmp / "src", symlinks=True)
    pf = tmp / "harness-changes.patch"
    pf.write_text(patch, encoding="utf-8")
    r = subprocess.run(["git", "apply", "--include=ARC3-Inference/*", str(pf)], cwd=tmp / "src",
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    (tmp / "src" / ".ok").write_text("ok")
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    try:
        (tmp / "src").rename(out)
    except OSError:
        return tmp / "src"
    return out


def import_harness(src: Path):
    """(tool_agent, runtime_state, python_tool_sandbox, solver-or-None) from the reconstructed src.
    The notebook's env knobs must be set by the caller first (module constants read them)."""
    for p in (str(src / "ARC3-Inference"), str(src / "tufa-arc-agi-framework" / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import inference.agent.python_tool_sandbox as SB
    import inference.agent.runtime_state as RS
    import inference.agent.tool_agent as TA
    try:
        import inference.framework.solver as S
    except Exception:  # noqa: BLE001 -- solver needs taaf's deps (matplotlib, arc_agi, ...)
        S = None
    return TA, RS, SB, S


class FakeGame:
    """A deterministic 64x64 game: each action recolors a few cells; every `level_every` actions
    the level advances (new board); a cell-walk eventually 'dies' and RESET restores the level's
    start board -- so histories contain RESETs and level changes. ACTION5 is a pure no-op."""

    def __init__(self, RS, writer_module, state_path: Path, *, seed=0, size=64, level_every=40,
                 die_every=17):
        self.RS, self.W = RS, writer_module
        self.state_path = state_path
        self.history_entries = []
        self.rng = random.Random(seed)
        self.size, self.level_every, self.die_every = size, level_every, die_every
        self.level, self.n, self.since_reset = 1, 0, 0
        self.grid = self._board()
        self.level_start = [r[:] for r in self.grid]
        self.calls = []                 # perf_counter() at each step_env entry (timing tests)

    def _board(self):
        # ARC-like: a background with a few solid rectangles. (Uniform-random cells would make the
        # sandbox's segmentation take ~9 s per frame -- every snippet would time out and the
        # equivalence checks would compare two empty stdouts.)
        n = self.size
        bg = self.rng.randrange(16)
        grid = [[bg] * n for _ in range(n)]
        for _ in range(8):
            c = self.rng.randrange(16)
            y, x = self.rng.randrange(n - 8), self.rng.randrange(n - 8)
            for i in range(y, y + self.rng.randrange(2, 8)):
                for j in range(x, x + self.rng.randrange(2, 8)):
                    grid[i][j] = c
        return grid

    def current_frame(self):
        return self.RS.Frame(grid=tuple(tuple(r) for r in self.grid), step=self.n, level=self.level)

    _quiet = False

    def write_runtime_state(self):      # body of _HarnessGameSession.write_runtime_state
        if self._quiet:
            return
        self.W.write_runtime_state(self.state_path, current_frame=self.current_frame(),
                                   history=self.history_entries)

    def seed(self):
        self.history_entries.append(self.RS.HistoryEntry(action="", frame=self.current_frame()))
        self.write_runtime_state()

    def _apply(self, name):
        level_completed = game_over = False
        if name == "RESET":
            self.grid = [r[:] for r in self.level_start]
            self.since_reset = 0
            changed = True
        elif name == "ACTION5":     # a pure no-op: never ends a level or a life
            changed = False
        else:
            for _ in range(3):
                self.grid[self.rng.randrange(self.size)][self.rng.randrange(self.size)] = self.rng.randrange(16)
            changed = True
            self.since_reset += 1
            if (self.n + 1) % self.level_every == 0:
                self.level += 1
                self.grid = self._board()
                self.level_start = [r[:] for r in self.grid]
                level_completed = True
            elif self.since_reset >= self.die_every:
                game_over = True
        self.n += 1
        return changed, level_completed, game_over

    def advance(self, k, names=("ACTION1", "ACTION2", "ACTION3", "ACTION4"), *, write_each=True):
        """Grow the history by k actions without a sandbox (setup for long histories). With
        write_each=False the state is written once at the end (setup is O(N), not O(N^2))."""
        self._quiet = not write_each
        try:
            for i in range(k):
                self._execute(names[i % len(names)])
        finally:
            self._quiet = False
        if k and not write_each:
            self.write_runtime_state()

    def _execute(self, name):
        changed, done_level, over = self._apply(name)
        result = {"executed": True, "action_num": self.n, "level": self.level, "score": self.level - 1,
                  "state": "GAME_OVER" if over else "NOT_FINISHED",
                  "valid_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "RESET"],
                  "board_changed": changed, "gameplay_changed": changed, "no_op": not changed,
                  "done": False, "level_completed": done_level, "game_over": over,
                  "run_complete": False, "action_name": name, "action_data": {}, "action_display": name,
                  "animation": {"frames": 1, "note": "é"}, "automatic": False}
        self.history_entries.append(self.RS.HistoryEntry(action=name, frame=self.current_frame(),
                                                         result=dict(result)))
        self.write_runtime_state()
        if over:   # the solver's auto-reset after a death
            self._apply("RESET")
            self.history_entries.append(self.RS.HistoryEntry(action="RESET", frame=self.current_frame(),
                                                             result={"executed": True, "automatic": True}))
            self.write_runtime_state()
        return result, changed

    def step_env(self, arguments):
        import time
        self.calls.append(time.perf_counter())
        if str(arguments.get("query") or "") == "animation":
            return {"executed": False, "query": "animation", "record": None}
        names = [str(a.get("action")) for a in arguments.get("actions") or []]
        per, last = [], {}
        for name in names:
            last, changed = self._execute(name)
            per.append(changed)
            if last["level_completed"] or last["game_over"]:
                break
        return {**last, "executed_actions": names[: len(per)], "requested_count": len(names),
                "executed_count": len(per), "gameplay_changed_per_action": per,
                "stopped_early": len(per) < len(names)}


def make_agent(TA, state_path: Path, game: FakeGame):
    agent = TA.ToolAgent(model="local")
    agent._ensure_session(state_path)
    agent._step_env_callback = game.step_env
    agent._current_valid_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "RESET"]
    return agent
