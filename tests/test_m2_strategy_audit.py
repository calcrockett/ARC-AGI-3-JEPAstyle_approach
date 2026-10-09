"""Strategy-audit variant `audit` (arc3-m2-turbo-lossless-tail-audit): lordhansolo's Milestone-2 strategy-audit
prompt appended to the turn opener once a game has generated ARC3_STRATEGY_AUDIT_TOKENS on one level. Unit tests
on a fake agent; integration tests on the REAL patched ToolAgent (skipped when the milestone-2 src is unavailable);
builder tests on the committed kernel."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "kaggle_submission_milestone2_fork" / "strategy_audit"))
sys.path.insert(0, str(ROOT / "kaggle_submission_milestone2_fork" / "level_memory"))

import _build_m2_level_memory_kernel as B  # noqa: E402
import level_memory as lm  # noqa: E402
import strategy_audit as sa  # noqa: E402
from _check_notebook_cell import check_notebook  # noqa: E402

VARIANTS = B.TURBO_LOSSLESS + ("tail", "audit")
SLUG, DIRNAME = "arc3-m2-turbo-lossless-tail-audit", "kaggle_submission_m2_turbo_lossless_tail_audit"
NB = ROOT / DIRNAME / "notebook" / f"{SLUG}.ipynb"
BASE_NB = ROOT / "kaggle_submission_m2_turbo_lossless_tail" / "notebook" / "arc3-m2-turbo-lossless-tail.ipynb"
T = sa.DEFAULT_AUDIT_TOKENS


def _cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv(sa.AUDIT_TOKENS_ENV, raising=False)
    monkeypatch.delenv("ARC3_YIELD_RESUME_PROMPT", raising=False)
    for k in sa.STATS:
        sa.STATS[k] = "" if k == "last_error" else 0
    yield


class Agent:
    def __init__(self, session="g1"):
        self._session_runtime_dir = session
        self._session_generated_tokens = 0
        self._resume_after_yield = False


def entry(action, level, automatic=False):
    return SimpleNamespace(action=action, frame=SimpleNamespace(level=level), result={"automatic": automatic})


def turn(agent, level, hist=()):
    return sa.after_user_prompt(agent, "OPENER", current_frame=SimpleNamespace(level=level), history_entries=list(hist))


# ------------------------------------------------------------------------------------------------- unit

def test_prompt_is_lordhansolos_text_with_only_the_documented_adaptations():
    p = sa.PROMPT
    assert p.startswith("\nStrategy audit for this turn\nYou may be stuck on this level.")
    assert "Do not reset the game or abandon a plan solely because of this audit." in p
    assert "Confidence, passing tests of your own model or a growing search count alone are not" in p
    assert "plan and execute" not in p and "python call's plan" not in p and "add the observed counterexample" not in p
    assert "comments at the top of the python snippet" in p and "action(...)" in p
    assert 2000 < len(p) < 3200    # ~600 tokens, once per period


def test_not_due_before_the_period_then_appended_once():
    a = Agent()
    assert turn(a, 1) == "OPENER"
    a._session_generated_tokens = T - 1
    assert turn(a, 1) == "OPENER"
    a._session_generated_tokens = T
    out = turn(a, 1)
    assert out.startswith("OPENER\n") and out.endswith(sa.PROMPT) and sa.STATS["audits_sent"] == 1
    a._session_generated_tokens = T + 3000          # the model answered: confirmed, clock restarts at T
    assert turn(a, 1) == "OPENER"
    assert sa.STATS["audits_confirmed"] == 1 and sa.STATS["levels_audited"] == 1
    a._session_generated_tokens = 2 * T - 1
    assert turn(a, 1) == "OPENER"
    a._session_generated_tokens = 2 * T
    assert turn(a, 1).endswith(sa.PROMPT)


def test_a_turn_that_generated_nothing_is_retracted_and_resent():
    a = Agent()
    turn(a, 1)
    a._session_generated_tokens = T
    assert turn(a, 1).endswith(sa.PROMPT)
    assert turn(a, 1).endswith(sa.PROMPT), "rolled-back turn: no tokens since delivery -> re-sent"
    assert sa.STATS["audits_retracted"] == 1 and sa.STATS["audits_confirmed"] == 0


def test_level_change_restarts_the_clock_and_counts_audited_levels_cleared():
    a = Agent()
    turn(a, 1)
    a._session_generated_tokens = T
    turn(a, 1)
    a._session_generated_tokens = T + 500
    assert turn(a, 2) == "OPENER"          # the audit turn cleared level 1
    assert sa.STATS["audited_levels_cleared"] == 1 and sa.STATS["max_audits_one_level"] == 1
    a._session_generated_tokens = 2 * T + 499
    assert turn(a, 2) == "OPENER"
    a._session_generated_tokens = 2 * T + 500
    assert turn(a, 2).endswith(sa.PROMPT)
    # a level cleared without an audit is not counted
    b = Agent("g2")
    turn(b, 1)
    b._session_generated_tokens = 100
    turn(b, 2)
    assert sa.STATS["audited_levels_cleared"] == 1


def test_new_game_resets_state():
    a = Agent()
    turn(a, 3)
    a._session_generated_tokens = T - 10
    a._session_runtime_dir, a._session_generated_tokens = "g2", 0
    turn(a, 1)
    a._session_generated_tokens = T - 10
    assert turn(a, 1) == "OPENER"
    assert sa.STATS["games"] == 2


def test_voluntary_resets_counted_overall_and_after_an_audit():
    a = Agent()
    hist = [entry("", 1), entry("UP", 1), entry("RESET", 1), entry("RESET", 1, automatic=True)]
    turn(a, 1, hist)
    assert sa.STATS["voluntary_resets_total"] == 1 and sa.STATS["voluntary_resets_after_audit"] == 0
    a._session_generated_tokens = T
    turn(a, 1, hist)
    hist += [entry("LEFT", 1), entry("RESET", 1)]
    a._session_generated_tokens = T + 10
    turn(a, 1, hist)
    assert sa.STATS["voluntary_resets_total"] == 2 and sa.STATS["voluntary_resets_after_audit"] == 1


def test_disabled_and_short_resume(monkeypatch):
    a = Agent()
    monkeypatch.setenv(sa.AUDIT_TOKENS_ENV, "0")
    turn(a, 1)
    a._session_generated_tokens = 10 * T
    assert turn(a, 1) == "OPENER"
    monkeypatch.setenv(sa.AUDIT_TOKENS_ENV, "bogus")
    assert sa.audit_tokens() == 0
    monkeypatch.delenv(sa.AUDIT_TOKENS_ENV)
    monkeypatch.setenv("ARC3_YIELD_RESUME_PROMPT", "short")
    a._resume_after_yield = True
    assert turn(a, 1) == "OPENER" and sa.STATS["audits_skipped_short_resume"] == 1
    a._resume_after_yield = False
    assert turn(a, 1).endswith(sa.PROMPT)


def test_install_wraps_and_never_breaks_a_turn():
    class TA:
        def _build_user_prompt(self, action_num, **kw):
            return "BASE"
    assert sa.install(TA) and sa.install(TA)
    agent = TA()
    agent._session_runtime_dir, agent._session_generated_tokens = "g", 0
    assert agent._build_user_prompt(1, current_frame=SimpleNamespace(level=1), history_entries=[]) == "BASE"
    agent._session_generated_tokens = "not a number"        # broken state -> error counted, prompt unchanged
    assert agent._build_user_prompt(1, current_frame=SimpleNamespace(level=1), history_entries=[]) == "BASE"
    assert sa.STATS["errors"] == 1 and sa.summary()["threshold_tokens"] == T


# ------------------------------------------------------------------------------------------ real harness
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_harness  # noqa: E402

M2_SRC = m2_harness.m2_src()
real = pytest.mark.skipif(not M2_SRC, reason="milestone-2 src unavailable (ARC3_M2_SRC / ARC3_M2_BASE)")


@pytest.fixture()
def real_agent():
    saved_env = dict(os.environ)
    if "inference.agent.tool_agent" not in sys.modules:
        os.environ.update(m2_harness.notebook_env())
    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")
    os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "flashnext")
    ta, rs, _, _ = m2_harness.import_harness(M2_SRC)
    cls = ta.ToolAgent
    names = ("_build_user_prompt", "_trim_messages_for_context", "_ensure_session")
    saved = {n: cls.__dict__[n] for n in names}
    flags = {f: cls.__dict__.get(f) for f in ("_lm_installed", "_sa_installed")}
    lm.install(cls)
    sa.install(cls)
    try:
        yield ta, cls(model="local"), rs.Frame, rs.HistoryEntry
    finally:
        for n, f in saved.items():
            setattr(cls, n, f)
        for f, v in flags.items():
            if v is None:
                if f in cls.__dict__:
                    delattr(cls, f)
            else:
                setattr(cls, f, v)
        os.environ.clear()
        os.environ.update(saved_env)


@real
def test_real_opener_carries_the_audit_after_the_period(real_agent, tmp_path):
    ta, agent, Frame, HistoryEntry = real_agent
    assert ta.ToolAgent._build_user_prompt.__module__ == "strategy_audit"
    agent._ensure_session(tmp_path / "state.json")
    grid = [[0] * 8 for _ in range(8)]
    frame = Frame(grid=grid, step=1, level=1)
    hist = [HistoryEntry(action="", frame=Frame(grid=grid, step=0, level=1)), HistoryEntry(action="UP", frame=frame)]
    kw = dict(valid_actions=["UP"], current_frame=frame, history_entries=hist,
              previous_step_summary={"executed_count": 1, "executed_actions": ["UP"], "level": 1})
    first = agent._build_user_prompt(1, **kw)
    assert sa.MARK not in first
    agent._session_generated_tokens = T
    out = agent._build_user_prompt(1, **kw)
    assert out.startswith(first.rstrip("\n")) and out.endswith(sa.PROMPT)
    # a new game (new runtime dir) starts a fresh clock
    agent._ensure_session(tmp_path / "g2" / "state.json")
    assert sa.MARK not in agent._build_user_prompt(1, **kw)


@real
def test_real_harness_resume_mode_is_full():
    """The audit relies on a resumed turn re-sending the full opener (the incumbent sets no ARC3_YIELD_RESUME_PROMPT)."""
    assert "ARC3_YIELD_RESUME_PROMPT" not in m2_harness.notebook_env()


# ---------------------------------------------------------------------------------------------- builder

def test_slug_dir_markers_counters():
    assert B.kernel_slug(VARIANTS) == SLUG and B.kernel_slug(tuple(reversed(VARIANTS))) == SLUG
    assert B.kernel_dir(VARIANTS).relative_to(B.ROOT).as_posix() == f"{DIRNAME}/notebook"
    m = B.kernel_markers(VARIANTS)
    assert "STRATEGY_AUDIT installed" in m and "SPEC_ACCEPT 0.5" not in " ".join(m)
    assert set(B.kernel_markers(B.TURBO_LOSSLESS + ("tail",))) | {"STRATEGY_AUDIT installed"} == set(m)
    assert B.kernel_counters(VARIANTS) == B.kernel_counters(B.TURBO_LOSSLESS) [:2] + [
        "strategy_audit_summary.json", "timeout_fix_summary.json"]
    # existing slugs unchanged
    assert B.kernel_slug(B.TURBO_LOSSLESS + ("tail",)) == "arc3-m2-turbo-lossless-tail"
    assert B.kernel_slug(B.TURBO + ("tail",)) == "arc3-m2-turbo-tail"
    assert B.kernel_slug(("audit",)) == "arc3-m2-lm-audit"
    assert B.kernel_slug(B.TURBO_LOSSLESS + ("audit",)) == "arc3-m2-turbo-lossless-audit"


def test_cli_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    assert B.main(["--turbo-lossless", "--prio-tail", "--strategy-audit"]) == 0
    assert (tmp_path / DIRNAME / "notebook" / f"{SLUG}.ipynb").exists()


def test_committed_kernel_is_current(tmp_path, monkeypatch):
    outs = []
    for i in range(2):
        monkeypatch.setattr(B, "ROOT", tmp_path / str(i))
        outs.append(B.build(VARIANTS).read_text(encoding="utf-8"))
    monkeypatch.undo()
    assert outs[0] == outs[1], "builder is not deterministic"
    assert outs[0] == NB.read_text(encoding="utf-8"), f"{SLUG} is stale: rerun the build"
    assert check_notebook(NB) == []
    meta = json.loads((NB.parent / "kernel-metadata.json").read_text())
    inc = json.loads((ROOT / "kaggle_submission_m2_level_memory" / "notebook" / "kernel-metadata.json").read_text())
    assert meta["id"] == f"calamitychasm/{SLUG}" and meta["code_file"] == NB.name
    for k in ("id", "title", "code_file"):
        meta.pop(k), inc.pop(k)
    assert meta.pop("docker_image_pinning_type") == "original"
    assert meta == inc


def test_kernel_differs_from_turbo_lossless_tail_only_in_header_audit_cell_and_dump():
    a, b = _cells(BASE_NB), _cells(NB)
    assert len(b) == len(a) + 1
    i = next(k for k, c in enumerate(b) if "STRATEGY_AUDIT installed" in c and "_SA_SRC" in c)
    rest = b[:i] + b[i + 1:]
    diff = [k for k, (x, y) in enumerate(zip(a, rest)) if x != y]
    assert diff[0] == 0 and len(diff) == 2, diff
    run = rest[diff[1]]
    assert run.replace(B.SA_DUMP, "") == a[diff[1]]
    # the audit cell comes after the level-memory install cell and the history-cache cell, before the run cell
    lm_cell = next(k for k, c in enumerate(b) if "LEVEL_MEMORY.install(_lm_ta.ToolAgent)" in c)
    hc_cell = next(k for k, c in enumerate(b) if "HISTORY_CACHE installed" in c and "_HC_SRC" in c)
    run_cell = next(k for k, c in enumerate(b) if c.startswith("print('Starting benchmark...')"))
    assert lm_cell < hc_cell < i < run_cell
    assert f"'{B.SA_TOKENS}'" in b[i] and B.SA_SRC in b[i]


def test_audit_cell_executes_against_a_stub_tool_agent(monkeypatch):
    import types
    cell = next(c for c in _cells(NB) if "_SA_SRC" in c)

    class ToolAgent:
        _lm_installed = True

        def _build_user_prompt(self, action_num, **kw):
            return "BASE"

    pkg, sub = types.ModuleType("inference"), types.ModuleType("inference.agent")
    pkg.__path__, sub.__path__ = [], []
    mod = types.ModuleType("inference.agent.tool_agent")
    mod.ToolAgent = ToolAgent
    for n, m in (("inference", pkg), ("inference.agent", sub), ("inference.agent.tool_agent", mod)):
        monkeypatch.setitem(sys.modules, n, m)
    monkeypatch.setitem(sys.modules, "strategy_audit", None)
    monkeypatch.setenv(sa.AUDIT_TOKENS_ENV, "restored-after-the-test")
    ns = {"json": json, "__name__": "cell"}
    exec(compile(cell, "audit_cell", "exec"), ns)
    assert os.environ[sa.AUDIT_TOKENS_ENV] == str(B.SA_TOKENS)
    assert ToolAgent._build_user_prompt.__module__ == "strategy_audit"
