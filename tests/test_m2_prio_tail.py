"""Priority-gate variant `tail` (arc3-m2-lm-tail, arc3-m2-turbo-tail, arc3-m2-turbo-lossless-tail): h = 60 in A's
efficiency proxy and B = 5 for a game on its last level, installed at runtime. The install lines are executed here
against the real upstream priority_scheduler.py, extracted from the harness patch in the upstream notebook."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_m2_level_memory_kernel as B  # noqa: E402
from _check_notebook_cell import check_notebook  # noqa: E402

KERNELS = {
    ("tail",): ("arc3-m2-lm-tail", "kaggle_submission_m2_lm_tail"),
    B.TURBO + ("tail",): ("arc3-m2-turbo-tail", "kaggle_submission_m2_turbo_tail"),
    B.TURBO_LOSSLESS + ("tail",): ("arc3-m2-turbo-lossless-tail", "kaggle_submission_m2_turbo_lossless_tail"),
}


def _cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def upstream_priority_scheduler() -> types.ModuleType:
    """priority_scheduler.py as the upstream harness patch creates it (a new file: every line is '+')."""
    src = [s for s in _cells(B.UPSTREAM) if "b/ARC3-Inference/inference/agent/priority_scheduler.py" in s]
    assert len(src) == 1
    text = src[0]
    lines = text[text.index("+++ b/ARC3-Inference/inference/agent/priority_scheduler.py"):].split("\n")
    assert lines[1].startswith("@@ -0,0 +1,"), lines[1]
    body = []
    for line in lines[2:]:
        if line.startswith("diff --git"):
            break
        assert line.startswith("+"), line[:60]
        body.append(line[1:])
    assert len(body) == int(lines[1].split(",")[-1].split()[0])
    mod = types.ModuleType("priority_scheduler_upstream")
    sys.modules[mod.__name__] = mod      # dataclasses resolve string annotations through sys.modules
    try:
        exec(compile("\n".join(body), "priority_scheduler.py", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(mod.__name__, None)
    return mod


def fake_tool_agent(ps: types.ModuleType) -> types.ModuleType:
    """The three tool_agent names the install lines read, with upstream's env handling."""
    ta = types.ModuleType("fake_tool_agent")
    ta.priority_value = ps.priority_value
    ta._priority_human_actions = lambda: max(1.0, float(os.environ.get("ARC3_PRIORITY_HUMAN_ACTIONS", "25")))
    ta._priority_level_options = lambda: {
        "normalize_score": os.environ.get("ARC3_PRIORITY_SCORE_NORMALIZATION") == "1",
        "tail_lookup": "remaining" if os.environ.get("ARC3_PRIORITY_TAIL_LOOKUP", "").lower() == "remaining"
        else False,
        "tail_efficiency": 0.8,
    }
    return ta


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setenv("ARC3_PRIORITY_HUMAN_ACTIONS", "25")       # restored after the test
    monkeypatch.setenv("ARC3_PRIORITY_TAIL_LOOKUP", "remaining")   # the incumbent's setup cell
    monkeypatch.setenv("ARC3_PRIORITY_SCORE_NORMALIZATION", "1")
    ps = upstream_priority_scheduler()
    before = {n: v for n, v in ps.TAIL_LOOKUP_REMAINING.items()}
    ns = {"_lm_ta": fake_tool_agent(ps), "__name__": "install_cell"}
    for name in ("inference", "inference.agent"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, "inference.agent.priority_scheduler", ps)
    exec(compile(B.VARIANT_INSTALL_LINES["tail"], "install_cell", "exec"), ns)
    return ps, before, ns


# ---------------------------------------------------------------------------------------------- builder

def test_slugs_dirs_and_markers():
    for variants, (slug, dirname) in KERNELS.items():
        assert B.kernel_slug(variants) == slug
        assert B.kernel_slug(tuple(reversed(variants))) == slug
        assert B.kernel_dir(variants).relative_to(B.ROOT).as_posix() == f"{dirname}/notebook"
        assert "PRIORITY_TAIL installed" in B.kernel_markers(variants)
    assert set(B.kernel_markers(B.TURBO)) < set(B.kernel_markers(B.TURBO + ("tail",)))
    assert B.kernel_counters(B.TURBO + ("tail",)) == B.kernel_counters(B.TURBO)
    # unchanged slugs of the existing kernels
    assert B.kernel_slug(B.TURBO) == "arc3-m2-turbo" and B.kernel_slug(()) == "arc3-m2-level-memory"
    assert B.kernel_slug(("histcache", "tail")) == "arc3-m2-lm-histcache-tail"


def test_cli_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    assert B.main(["--turbo", "--prio-tail"]) == 0
    assert (tmp_path / "kaggle_submission_m2_turbo_tail" / "notebook" / "arc3-m2-turbo-tail.ipynb").exists()


def test_committed_tail_kernels_are_current(tmp_path, monkeypatch):
    for variants, (slug, dirname) in KERNELS.items():
        committed = ROOT / dirname / "notebook" / f"{slug}.ipynb"
        outs = []
        for i in range(2):
            monkeypatch.setattr(B, "ROOT", tmp_path / str(i))
            outs.append(B.build(variants).read_text(encoding="utf-8"))
        monkeypatch.undo()
        assert outs[0] == outs[1], "builder is not deterministic"
        assert outs[0] == committed.read_text(encoding="utf-8"), f"{slug} is stale: rerun the build"
        assert check_notebook(committed) == []
        meta = json.loads((committed.parent / "kernel-metadata.json").read_text())
        inc = json.loads((ROOT / "kaggle_submission_m2_level_memory" / "notebook" / "kernel-metadata.json").read_text())
        assert meta["id"] == f"calamitychasm/{slug}" and meta["code_file"] == committed.name
        for k in ("id", "title", "code_file"):
            meta.pop(k), inc.pop(k)
        assert meta == inc


def test_tail_kernel_differs_from_its_base_only_in_header_and_install_cell():
    pairs = [("kaggle_submission_m2_turbo/notebook/arc3-m2-turbo.ipynb",
              "kaggle_submission_m2_turbo_tail/notebook/arc3-m2-turbo-tail.ipynb"),
             ("kaggle_submission_m2_turbo_lossless/notebook/arc3-m2-turbo-lossless.ipynb",
              "kaggle_submission_m2_turbo_lossless_tail/notebook/arc3-m2-turbo-lossless-tail.ipynb")]
    for base, tail in pairs:
        a, b = _cells(ROOT / base), _cells(ROOT / tail)
        assert len(a) == len(b)
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        assert len(diff) == 2 and diff[0] == 0, diff
        i = diff[1]
        assert "LEVEL_MEMORY.install(_lm_ta.ToolAgent)" in b[i]
        assert b[i].replace(B.VARIANT_INSTALL_LINES["tail"], "") == a[i]
    # the setup cell still carries the incumbent's priority knobs (the variant overrides at runtime only)
    setup = [c for c in _cells(ROOT / pairs[0][1]) if "'ARC3_PRIORITY_TAIL_LOOKUP': 'remaining'" in c]
    assert len(setup) == 1 and "ARC3_PRIORITY_HUMAN_ACTIONS" not in setup[0]


def test_install_lines_come_after_tool_agent_import():
    cell = [c for c in _cells(ROOT / "kaggle_submission_m2_lm_tail/notebook/arc3-m2-lm-tail.ipynb")
            if "PRIORITY_TAIL installed" in c]
    assert len(cell) == 1
    c = cell[0]
    assert c.index("import inference.agent.tool_agent as _lm_ta") < c.index("_ps.TAIL_LOOKUP_REMAINING")
    assert c.index("PRIORITY_TAIL installed") < c.index("print('LEVEL_MEMORY installed'")


# ------------------------------------------------------------------------- runtime, on upstream's scheduler

def test_upstream_table_is_what_the_variant_expects():
    ps = upstream_priority_scheduler()
    assert ps.TAIL_LOOKUP_REMAINING[7] == (8.0, 8.0, 8.0, 8.0, 7.0, 5.0, 0.0)
    assert all(v[-2:] == (5.0, 0.0) for v in ps.TAIL_LOOKUP_REMAINING.values())


def test_install_sets_final_b_and_h(installed):
    ps, before, ns = installed
    assert os.environ["ARC3_PRIORITY_HUMAN_ACTIONS"] == "60"
    assert ns["_lm_ta"]._priority_human_actions() == 60.0
    for n, v in ps.TAIL_LOOKUP_REMAINING.items():
        assert v == before[n][:-1] + (5.0,)
    assert ps.TAIL_LOOKUP_REMAINING[6] == (8.0, 8.0, 8.0, 7.0, 5.0, 5.0)


def test_install_refuses_to_run_twice(installed):
    ps, _, ns = installed      # the fixture's sys.modules stubs are still in place here
    with pytest.raises(AssertionError):
        exec(compile(B.VARIANT_INSTALL_LINES["tail"], "install_cell", "exec"), ns)


def test_final_level_no_longer_parked_behind_mid_game(installed):
    """N = 7, a waiting game at a trim having spent 50 actions / 40k tokens on its level (C identical)."""
    ps, before, _ = installed
    kw = dict(tail_fraction=1.0, normalize_score=True, tail_lookup="remaining")

    def p(level, h):
        return ps.priority_value(ps.PrioritySnapshot(level, 50, 40000, 1.0, 7), human_actions=h, **kw)

    new_final, new_mid = p(7, 60.0), p(4, 60.0)
    ps.TAIL_LOOKUP_REMAINING.update(before)                   # upstream table and h
    old_final, old_mid = p(7, 25.0), p(4, 25.0)
    assert old_final < 0.2 * old_mid                          # upstream: 119 vs 693, parked
    assert 0.8 * new_mid < new_final < new_mid               # variant: 710 vs 807, it competes
    # a fresh level is priced as before: h only changes how fast A decays with actions
    fresh = ps.PrioritySnapshot(3, 0, 0, 1.0, 7)
    assert ps.priority_value(fresh, human_actions=60.0, **kw) == ps.priority_value(fresh, human_actions=25.0, **kw)
