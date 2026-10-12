"""Fresh-first hedge on D' (`--dprime-fresh-first`, kernel arc3-m2-turbo-lossless-dprime-ff): D' installed verbatim, plus
a wrapper of _PriorityGate.acquire that queues never-started games in upstream's band (above every priced game) instead of
pricing them by the D' formula. Wrapper logic is executed against the real patched harness when it is available."""

from __future__ import annotations

import json
import os
import sys
import threading
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

VARIANTS = B.TURBO_LOSSLESS + ("dprime", "ff")
SLUG, DIRNAME = "arc3-m2-turbo-lossless-dprime-ff", "kaggle_submission_m2_turbo_lossless_dprime_ff"
COMMITTED = ROOT / DIRNAME / "notebook" / f"{SLUG}.ipynb"
DPRIME_NB = ROOT / "kaggle_submission_m2_turbo_lossless_dprime" / "notebook" / "arc3-m2-turbo-lossless-dprime.ipynb"
DP_FILE = ROOT / "kaggle_submission_milestone2_fork" / "dprime" / "ours_form_priority.py"
DP_SHA = "6456efdd7500d67c2851d2b6474b12bd1c41b5138c0b6f3de603299e63d9fcb2"


def _cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


# ------------------------------------------------------------------------------------------------- builder

def test_slug_dir_markers_counters():
    assert B.kernel_slug(VARIANTS) == SLUG and B.kernel_slug(tuple(reversed(VARIANTS))) == SLUG
    assert B.kernel_dir(VARIANTS).relative_to(B.ROOT).as_posix() == f"{DIRNAME}/notebook"
    m = B.kernel_markers(VARIANTS)
    assert "PRIORITY_DPRIME installed" in m and "DPRIME_FRESH_FIRST installed" in m
    assert "PRIORITY_TAIL installed" not in m
    assert set(B.kernel_markers(B.TURBO_LOSSLESS + ("dprime",))) | {"DPRIME_FRESH_FIRST installed"} == set(m)
    assert B.kernel_counters(VARIANTS) == B.kernel_counters(B.TURBO_LOSSLESS + ("dprime",))
    assert "dprime_summary.json" in B.kernel_counters(VARIANTS)
    assert B.kernel_slug(("dprime", "ff")) == "arc3-m2-lm-dprime-ff"


def test_ff_needs_dprime_and_excludes_tail():
    with pytest.raises(SystemExit):
        B.variant_names(("ff",))
    with pytest.raises(SystemExit):
        B.variant_names(("tail", "dprime", "ff"))


def test_cli_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    assert B.main(["--turbo-lossless", "--dprime-fresh-first"]) == 0
    assert (tmp_path / DIRNAME / "notebook" / f"{SLUG}.ipynb").exists()
    # --dprime together with it is the same build
    assert B.main(["--turbo-lossless", "--dprime", "--dprime-fresh-first"]) == 0
    with pytest.raises(SystemExit):
        B.main(["--turbo-lossless", "--dprime-fresh-first", "--prio-tail"])


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
    base = json.loads((DPRIME_NB.parent / "kernel-metadata.json").read_text())
    assert meta["id"] == f"calamitychasm/{SLUG}" and meta["code_file"] == COMMITTED.name
    for k in ("id", "title", "code_file"):
        meta.pop(k), base.pop(k)
    assert meta == base


def test_differs_from_dprime_only_where_intended():
    a, b = _cells(DPRIME_NB), _cells(COMMITTED)
    assert len(a) == len(b)
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(diff) == 2 and diff[0] == 0, diff          # header blurb + the install cell, nothing else
    assert "fresh-first" in b[0].lower()
    ff = B.VARIANT_INSTALL_LINES["ff"]
    # the ff lines sit right after D''s lines and before the LEVEL_MEMORY installed print
    assert b[diff[1]] == a[diff[1]].replace("print('LEVEL_MEMORY installed'", ff + "print('LEVEL_MEMORY installed'")
    assert b[diff[1]].index("PRIORITY_DPRIME installed") < b[diff[1]].index("DPRIME_FRESH_FIRST installed")


def test_dprime_module_embedded_verbatim_and_unedited():
    import hashlib
    assert hashlib.sha256(DP_FILE.read_bytes()).hexdigest() == DP_SHA
    install = next(c for c in _cells(COMMITTED) if "_DP_SRC = " in c)
    assert f"_DP_SRC = {DP_FILE.read_text(encoding='utf-8')!r}\n" in install
    assert B.DP_SRC == DP_FILE.read_text(encoding="utf-8")


# ------------------------------------------------------------------------------- runtime, on the real harness

M2_SRC = m2_harness.m2_src()


def _install(monkeypatch, with_ff: bool):
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
        monkeypatch.setenv(k, m2_harness.notebook_env()[k])
    monkeypatch.setattr(ta, "priority_value", ta.priority_value)
    monkeypatch.setattr(ta, "ProgressPace", ta.ProgressPace)
    monkeypatch.setattr(ta._PriorityGate, "acquire", ta._PriorityGate.acquire)
    monkeypatch.setattr(solver._HarnessGameSession, "play", solver._HarnessGameSession.play)
    monkeypatch.delitem(sys.modules, "ours_form_priority", raising=False)
    ns = {"_lm_ta": ta, "_sys": sys, "_types": types, "__name__": "install_cell"}
    exec(compile(B.VARIANT_INSTALL_LINES["dprime"], "install_cell", "exec"), ns)
    if with_ff:
        exec(compile(B.VARIANT_INSTALL_LINES["ff"], "install_cell", "exec"), ns)
    return ta, ns


def _admission_order(ta, entries):
    """One slot held by the caller; every entry (name, priority) queues from its own thread; release the slot once per
    entry and return the names in the order they were admitted."""
    gate = ta._PriorityGate(1)
    now = time.monotonic()
    gate.configure_clock(now, now + 1000.0)
    gate.acquire(ta._PRIORITY_UNTRIMMED_BASE)         # takes the only slot (a fresh game in both builds)
    order, lock = [], threading.Lock()

    def waiter(name, prio):
        gate.acquire(prio)
        with lock:
            order.append(name)

    threads = [threading.Thread(target=waiter, args=e, daemon=True) for e in entries]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with gate._cond:
            if len(gate._waiting) == len(entries):
                break
        time.sleep(0.01)
    else:
        raise AssertionError("waiters did not queue")
    for i in range(len(entries)):
        gate.release()
        deadline = time.monotonic() + 10
        while len(order) < i + 1 and time.monotonic() < deadline:
            time.sleep(0.01)
    for t in threads:
        t.join(timeout=5)
    return order


def test_fresh_games_admitted_first_then_d_prime_order(monkeypatch):
    ta, ns = _install(monkeypatch, with_ff=True)
    dp = ns["_dp_mod"]
    # three started games waiting (D' values of different states), two never-started games arriving later
    snap = ta.PrioritySnapshot
    hi = dp.d_priority(snap(5, 0, 0.0, 0.0, 10))           # 6 levels remain: B = 16
    mid = dp.d_priority(snap(9, 10, 1000.0, 20000.0, 10))  # 1 level remains: B = 10
    lo = dp.d_priority(snap(10, 300, 150000.0, 40000.0, 10))   # last level, spent: B = 0
    fresh_price = dp.d_priority(snap(1, 0, 0.0, 0.0, 10))
    assert lo < mid < hi and hi > fresh_price                  # the started game outranks a *priced* fresh game
    base = ta._PRIORITY_UNTRIMMED_BASE
    order = _admission_order(ta, [("started-mid", mid), ("started-hi", hi), ("started-lo", lo),
                                  ("fresh-2", base - 2), ("fresh-1", base - 1)])
    assert order == ["fresh-1", "fresh-2", "started-hi", "started-mid", "started-lo"]
    assert ns["FF_STATS"]["fresh_first_admissions"] == 3       # the slot holder + the two waiters
    assert ns["dprime_summary"]()["fresh_first_admissions"] == 3
    assert ns["dprime_summary"]()["fresh_replaced"] == 0 and ns["dprime_summary"]()["errors"] == 0


def _queued_priority_of_a_fresh_game(ta):
    gate = ta._PriorityGate(1)
    now = time.monotonic()
    gate.configure_clock(now, now + 1000.0)
    gate.acquire(ta._PRIORITY_UNTRIMMED_BASE)                     # the only slot is taken
    t = threading.Thread(target=gate.acquire, args=(ta._PRIORITY_UNTRIMMED_BASE - 1,), daemon=True)
    t.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with gate._cond:
            if gate._waiting:
                queued = -gate._waiting[0][0]
                break
        time.sleep(0.01)
    else:
        raise AssertionError("the fresh game did not queue")
    gate.release()
    t.join(timeout=5)
    return queued


def test_control_plain_dprime_prices_a_fresh_game(monkeypatch):
    ta, ns = _install(monkeypatch, with_ff=False)
    priced = _queued_priority_of_a_fresh_game(ta)
    assert priced == ns["_dp_mod"].d_priority(ta.PrioritySnapshot(1, 0, 0.0, 0.0, None), tail_fraction=2.5)
    assert priced < ta._PRIORITY_UNTRIMMED_BASE - ta._PRIORITY_BAND
    assert ns["_dp_mod"].FRESH["replaced"] == 2        # the slot holder and the waiter


def test_ff_queues_a_fresh_game_in_the_band_without_pricing_it(monkeypatch):
    ta, ns = _install(monkeypatch, with_ff=True)
    assert _queued_priority_of_a_fresh_game(ta) == ta._PRIORITY_UNTRIMMED_BASE - 1
    assert ns["_dp_mod"].FRESH["replaced"] == 0


def test_started_games_still_use_dprime_acquire_and_install_is_idempotent_marked(monkeypatch):
    ta, ns = _install(monkeypatch, with_ff=True)
    assert getattr(ta._PriorityGate.acquire, "_ours_ff", False) and getattr(ta._PriorityGate.acquire, "_ours_d", False)
    # a priority below the band reaches D''s acquire (and through it, upstream's), not the fresh counter
    gate = ta._PriorityGate(2)
    gate.acquire(12345)
    assert ns["FF_STATS"]["fresh_first_admissions"] == 0
    gate.acquire(ta._PRIORITY_UNTRIMMED_BASE - 7)
    assert ns["FF_STATS"]["fresh_first_admissions"] == 1


# ----------------------------------------------------------------------------------------------- simulator

def test_simulator_fresh_first_has_no_starvation_and_matches_alias():
    import random

    import sim_m2_priority_gate as S
    assert S.ALIASES["dprime-ff"] == "dprime (fresh first)"
    assert S.VARIANTS["dprime (fresh first)"] == dict(prio="dprime", fresh_first=True)
    world = S.World(random.Random("ff/0"), n_games=14, hard=2.5)
    d = S.run(world, slots=4, minutes=60.0, prio="dprime")
    f = S.run(world, slots=4, minutes=60.0, prio="dprime", fresh_first=True)
    assert f["starved"] == 0 and f["starved"] <= d["starved"]
