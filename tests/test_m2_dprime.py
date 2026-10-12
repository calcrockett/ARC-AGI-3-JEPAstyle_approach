"""D' slot-priority variant `dprime` (arc3-m2-turbo-lossless-dprime): shiiin9's module verbatim, installed by its own
install_d, plus our call/error counter and the 2400 s boot grace. The install lines are executed against the real
patched tool_agent / solver when the reconstructed milestone-2 src is available (tests/m2_harness.py)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _build_m2_level_memory_kernel as B  # noqa: E402
import m2_harness  # noqa: E402
from _check_notebook_cell import check_notebook  # noqa: E402

VARIANTS = B.TURBO_LOSSLESS + ("dprime",)
SLUG, DIRNAME = "arc3-m2-turbo-lossless-dprime", "kaggle_submission_m2_turbo_lossless_dprime"
COMMITTED = ROOT / DIRNAME / "notebook" / f"{SLUG}.ipynb"
BASE = ROOT / "kaggle_submission_m2_turbo_lossless" / "notebook" / "arc3-m2-turbo-lossless.ipynb"
DP_FILE = ROOT / "kaggle_submission_milestone2_fork" / "dprime" / "ours_form_priority.py"
DP_SHA = "6456efdd7500d67c2851d2b6474b12bd1c41b5138c0b6f3de603299e63d9fcb2"
JA_COPY = Path("/home/user/ext/JustAdev742_Arc-Agi-3-Kaggle-comp/kaggle/dprime/affectify-arc-31-54-in-a-single-sub.ipynb")
NB_SHA = "f649d005040ee367bec580c180b3712c81583f622b5690c8e0e1a745dc0858c7"


def _cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def _dp_module() -> types.ModuleType:
    mod = types.ModuleType("ours_form_priority_test")
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(DP_FILE.read_text(encoding="utf-8"), "ours_form_priority.py", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(mod.__name__, None)
    return mod


# ------------------------------------------------------------------------------------------- vendored module

def test_vendored_module_is_pinned_and_verbatim():
    assert hashlib.sha256(DP_FILE.read_bytes()).hexdigest() == DP_SHA
    if not JA_COPY.exists():
        pytest.skip("JustAdev742's copy of the D' notebook is not checked out here")
    import ast
    raw = JA_COPY.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == NB_SHA
    cell = "".join(json.loads(raw)["cells"][21]["source"])
    line = next(ln for ln in cell.split("\n") if ln.startswith("_FP_SOURCE = "))
    assert ast.literal_eval(line[len("_FP_SOURCE = "):]) == DP_FILE.read_text(encoding="utf-8")


def test_formula_matches_the_published_description():
    dp = _dp_module()
    assert dp.D_PRIME["b1"] == 10 and dp.D_PRIME["b2"] == 14 and dp.D_PRIME["b3"] == 16 and dp.D_PRIME["fade"] == 0.4
    # l = 3 of N = 7, a = 40, t = 50k, pace p = 45k tokens per cleared level, phi = 0.5
    n, l, a, t, p, phi = 7, 3, 40.0, 50000.0, 45000.0, 0.5
    A = (1 + 0.5 * (l - 1)) * 55 / (n * (n + 1) / 2) * (300 / (300 + a)) ** 2.5
    M = min(4.0, max(0.25, (30000 / p) ** 0.4))
    T = 225000 * min(2.0, max(0.5, p / 30000)) ** 0.5
    C = 0.1 * max(0.1, 1 - a / 115) + 0.9 * max(0.1, 1 - t / T)
    want = A * M * C + 16 * phi                    # 4 levels remain -> B = 16
    assert dp.d_value(l, a, t, p, n, phi) == pytest.approx(want, rel=1e-12)
    assert dp.d_parts(7, 0, 0, 0, 7)[1] == 0.0 and dp.d_parts(6, 0, 0, 0, 7)[1] == 10.0
    # a fresh game: l = 1, a = t = 0, no pace (M = 1, T at the reference)
    assert dp.d_value(1, 0, 0, 0, 7, 1.0) == pytest.approx(55 / 28 + 16)


# ------------------------------------------------------------------------------------------------- builder

def test_slug_dir_markers_counters():
    assert B.kernel_slug(VARIANTS) == SLUG and B.kernel_slug(tuple(reversed(VARIANTS))) == SLUG
    assert B.kernel_dir(VARIANTS).relative_to(B.ROOT).as_posix() == f"{DIRNAME}/notebook"
    m = B.kernel_markers(VARIANTS)
    assert "PRIORITY_DPRIME installed" in m and "PRIORITY_TAIL installed" not in m
    assert set(B.kernel_markers(B.TURBO_LOSSLESS)) | {"PRIORITY_DPRIME installed"} == set(m)
    assert B.kernel_counters(VARIANTS) == ["level_memory_summary.json", "history_cache_summary.json",
                                           "dprime_summary.json", "timeout_fix_summary.json"]
    with pytest.raises(SystemExit):
        B.variant_names(("tail", "dprime"))
    assert B.kernel_slug(("dprime",)) == "arc3-m2-lm-dprime"


def test_cli_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    assert B.main(["--turbo-lossless", "--dprime"]) == 0
    assert (tmp_path / DIRNAME / "notebook" / f"{SLUG}.ipynb").exists()
    with pytest.raises(SystemExit):
        B.main(["--turbo-lossless", "--dprime", "--prio-tail"])


def test_committed_kernel_is_current(tmp_path, monkeypatch):
    outs = []
    for i in range(2):
        monkeypatch.setattr(B, "ROOT", tmp_path / str(i))
        outs.append(B.build(VARIANTS).read_text(encoding="utf-8"))
    monkeypatch.undo()
    assert outs[0] == outs[1], "builder is not deterministic"
    assert outs[0] == COMMITTED.read_text(encoding="utf-8"), f"{SLUG} is stale: rerun the build"
    assert check_notebook(COMMITTED) == []
    meta = json.loads((COMMITTED.parent / "kernel-metadata.json").read_text())
    base = json.loads((BASE.parent / "kernel-metadata.json").read_text())
    assert meta["id"] == f"calamitychasm/{SLUG}" and meta["code_file"] == COMMITTED.name
    for k in ("id", "title", "code_file"):
        meta.pop(k), base.pop(k)
    assert meta == base


def test_differs_from_turbo_lossless_only_where_intended():
    a, b = _cells(BASE), _cells(COMMITTED)
    assert len(a) == len(b)
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(diff) == 4 and diff[0] == 0, diff
    setup, install, run = (b[i] for i in diff[1:])
    assert a[diff[1]].count(B.BOOT_GRACE_OLD) == 1 and setup == a[diff[1]].replace(B.BOOT_GRACE_OLD, B.BOOT_GRACE_NEW)
    assert "LEVEL_MEMORY.install(_lm_ta.ToolAgent)" in install
    assert install.replace(B.VARIANT_INSTALL_LINES["dprime"], "") == a[diff[2]]
    assert run.replace(B.DP_DUMP, "") == a[diff[3]]
    assert install.index("import inference.agent.tool_agent as _lm_ta") < install.index("_dp_mod.install_d(")
    assert install.index("PRIORITY_DPRIME installed") < install.index("print('LEVEL_MEMORY installed'")
    # the arm-B kernels on Kaggle keep the old grace; only this new kernel carries 2400
    assert all("'2400'" not in c for c in a)


# ------------------------------------------------------------------------------- runtime, on the real harness

M2_SRC = m2_harness.m2_src()


@pytest.fixture()
def real(monkeypatch):
    if not M2_SRC:
        pytest.skip("milestone-2 src unavailable (ARC3_M2_SRC / ARC3_M2_BASE)")
    if "inference.agent.tool_agent" not in sys.modules:
        for k, v in m2_harness.notebook_env().items():
            monkeypatch.setenv(k, v)
    monkeypatch.setenv("LOCAL_ANALYZER_BASE_URL", os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1"))
    monkeypatch.setenv("LOCAL_ANALYZER_MODEL_ID", os.environ.get("LOCAL_ANALYZER_MODEL_ID", "flashnext"))
    ta, _, _, solver = m2_harness.import_harness(M2_SRC)
    if solver is None:
        pytest.skip("solver not importable here")
    for k in ("ARC3_PRIORITY_PACE", "ARC3_PRIORITY_REFRESH_QUEUE", "ARC3_PRIORITY_TAIL_FADE",
              "ARC3_PRIORITY_TAIL_FADE_FRACTION"):
        monkeypatch.setenv(k, m2_harness.notebook_env()[k])          # the incumbent's values; restored after
    monkeypatch.setattr(ta, "priority_value", ta.priority_value)
    monkeypatch.setattr(ta, "ProgressPace", ta.ProgressPace)
    monkeypatch.setattr(ta._PriorityGate, "acquire", ta._PriorityGate.acquire)
    monkeypatch.setattr(solver._HarnessGameSession, "play", solver._HarnessGameSession.play)
    monkeypatch.delitem(sys.modules, "ours_form_priority", raising=False)
    upstream = ta.priority_value
    ns = {"_lm_ta": ta, "_sys": sys, "_types": types, "__name__": "install_cell"}
    exec(compile(B.VARIANT_INSTALL_LINES["dprime"], "install_cell", "exec"), ns)
    return ta, solver, ns, upstream


def test_install_replaces_priority_pace_and_fresh_queueing(real):
    ta, solver, ns, upstream = real
    dp = ns["_dp_mod"]
    assert ta.priority_value is ns["_dp_counted"] and ta.ProgressPace is dp.FormPace
    assert os.environ["ARC3_PRIORITY_PACE"] == "1" and os.environ["ARC3_PRIORITY_TAIL_FADE_FRACTION"] == "0.4"
    gate = ta._PriorityGate(2)
    now = time.monotonic()
    gate.configure_clock(now, now + 1000.0)
    snap = ta.PrioritySnapshot(3, 40, 50000.0, 45000.0, 7)
    assert gate._snapshot_priority(snap, now) == dp.d_priority(snap, tail_fraction=1000.0 / 400.0)
    # inside the fade window (last 40% of the shared clock): phi = remaining / window
    assert gate._snapshot_priority(snap, now + 800.0) == dp.d_priority(snap, tail_fraction=200.0 / 400.0)
    # a never-started game is priced by the formula (l = 1, a = t = 0), not queued in the 2,000,000 band
    gate.acquire(ta._PRIORITY_UNTRIMMED_BASE - 3)
    assert dp.FRESH["replaced"] == 1 and ns["DPRIME_STATS"]["errors"] == 0 and ns["DPRIME_STATS"]["calls"] >= 3
    # a trimmed game's handover priority still goes through the original acquire path
    assert dp.d_priority(ta.PrioritySnapshot(1, 0, 0.0, 0.0, 7)) == int((55 / 28 + 16) * 1000)
    pace = ta.ProgressPace()
    pace.record_completion(20000.0, 1.0)
    pace.record_completion(40000.0, 1.0)
    assert pace.cost_multiplier() == 30000.0
    assert ns["dprime_summary"]()["fresh_replaced"] == 1


def test_counter_falls_back_to_upstream_on_error(real, monkeypatch):
    ta, _, ns, upstream = real

    def boom(state, **kw):
        raise ValueError("x")

    monkeypatch.setattr(ns["_dp_mod"], "d_priority", boom)
    snap = ta.PrioritySnapshot(2, 10, 1000.0, 30000.0, 7)
    got = ta.priority_value(snap, tail_fraction=1.0)
    assert got == upstream(ta.PrioritySnapshot(2, 10, 1000.0, 1.0, 7), tail_fraction=1.0)
    assert ns["DPRIME_STATS"]["errors"] == 1 and "ValueError" in ns["DPRIME_STATS"]["last_error"]


# ----------------------------------------------------------------------------------------------- simulator

def test_simulator_runs_dprime_from_the_vendored_module():
    import random

    import sim_m2_priority_gate as S
    snap_args = (3, 40, 50000.0, 45000.0, 7)
    assert S.DP.d_value(*snap_args, 0.5) == _dp_module().d_value(*snap_args, 0.5)
    world = S.World(random.Random("d/0"), n_games=14, hard=1.5)
    a = S.run(world, slots=4, minutes=90.0, prio="dprime")
    assert a == S.run(world, slots=4, minutes=90.0, prio="dprime")
    assert 0.0 <= a["score"] <= 100.0 and a["levels"] > 0 and 0 <= a["starved"] <= 14
    b = S.run(world, slots=4, minutes=90.0, prio="dprime", fresh_first=True)
    assert b["starved"] == 0 or b["starved"] <= a["starved"]
    assert S.run(world, slots=4, minutes=90.0)["starved"] >= 0
