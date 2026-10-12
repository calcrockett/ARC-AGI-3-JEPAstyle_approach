"""History cache (variant A') for the milestone-2 harness: zero behavior change, flat cost.

Runs the REAL patched ToolAgent and the REAL sandbox subprocess (base bundle + dfranzen's
harness patch, reconstructed by tests/m2_harness.py) against a synthetic 64x64 game. Every
scenario is played twice in the same process -- once with the original functions, once with
the cache installed -- and must produce identical:
  * sandbox-visible state: everything a snippet can read (history, transitions, frames incl.
    their raw grids and segmentation, results, last_* globals) -- printed by the snippet itself;
  * the tool payload text handed back to the model;
  * the next turn's user prompt (built from load_runtime_state, as analyze() does);
  * the state file's JSON content and the agent's step summary.
Scenarios cover N = 1, 50, 1000 history entries, deltas after action(), a new sandbox per call,
a sandbox killed by the tool timeout, RESETs and level changes, and a snippet that mutates
history objects before acting (they must come back fresh, as when the history was re-parsed).
Skipped when the milestone-2 src cannot be found or rebuilt.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "kaggle_submission_milestone2_fork" / "history_cache"))

import m2_harness as h  # noqa: E402
import history_cache as hc  # noqa: E402

SRC = h.m2_src()
TIMEOUT = "Tool timed out after"


@pytest.fixture(scope="module")
def harness():
    if SRC is None:
        pytest.skip("milestone-2 src unavailable (ARC3_M2_SRC / ARC3_M2_BASE)")
    saved = dict(os.environ)
    os.environ.update(h.notebook_env())
    try:
        TA, RS, SB, S = h.import_harness(SRC)
        if S is None:
            pytest.skip("solver module needs taaf's dependencies (matplotlib, arc_agi, imageio, scipy)")
        assert not hc.STATS["installed"]
        yield TA, RS, SB, S
    finally:
        hc.uninstall()
        os.environ.clear()
        os.environ.update(saved)


OBS = r'''
import json
def _fv(f):
    if f is None:
        return None
    return [f.ascii, f.step, f.level, list(f.shape), f._grid, repr(f), str(f)]
def _hv(e):
    return [e.action, _fv(e.frame), e.result, repr(e)]
def _obs(tag, full=False):
    d = {"tag": tag, "n": len(history), "current": _fv(current_frame),
         "latest_is_current": latest_frame is current_frame, "previous": _fv(previous_frame),
         "last_action_frame": _fv(last_action_frame), "last_action": last_action,
         "valid_actions": valid_actions, "last_call": last_action_call_result,
         "death_ledger": death_ledger, "n_transitions": len(transitions),
         "tail": [_hv(e) for e in history[-3:]],
         "tail_tr": [[t.action, _fv(t.before_frame), _fv(t.after_frame), t.result] for t in transitions[-2:]],
         "last_transition": None if last_transition is None else repr(last_transition),
         "seg": len(json.dumps(current_frame.segmentation, sort_keys=True, default=str)),
         "diff": json.dumps(frame_diff(), sort_keys=True, default=str)[:4000]}
    if full:
        d["history"] = [_hv(e) for e in history]
        d["transitions"] = [[t.action, t.before_frame.step if t.before_frame else None, t.after_frame.step]
                            for t in transitions]
        d["seg0"] = len(json.dumps(history[0].frame.segmentation, sort_keys=True, default=str))
    print("OBS " + json.dumps(d, sort_keys=True))
'''

MUTATE = r'''
history[0].frame._grid[0][0] = 99
history[-1].frame._grid[1][1] = 77
history[-1].result["mutated"] = True
if history[-1].result.get("valid_actions"):
    history[-1].result["valid_actions"].append("BOGUS")
current_frame._grid[0][0] = 55
'''

# (code, timeout) per python tool call
SCENARIO_FULL = [
    # mutate history objects, then act with the no-op ACTION5 (never terminal): the refreshed
    # history must be fresh objects again, as when the whole history was re-sent and re-parsed
    (OBS + '_obs("start", True)\n' + MUTATE + 'action("ACTION5")\n_obs("a5", True)\n'
     + 'print("pristine", history[0].frame._grid[0][0] != 99)\n'
     + 'action("ACTION1")\n_obs("a1")\naction("ACTION2")\n_obs("a2", True)\n'
     + 'action("ACTION3")\n_obs("a3")\naction("ACTION4")\n_obs("a4", True)\n', None),
    # a new sandbox process: full payload again
    (OBS + '_obs("restart", True)\nfor _a in ["ACTION1", "ACTION2", "ACTION5", "ACTION3"]:\n'
     '    action(_a)\n    _obs(_a)\n_obs("end", True)\n', None),
    (OBS + 'action("RESET")\n_obs("reset", True)\naction("ACTION4")\n_obs("after-reset", True)\n', None),
    # killed by the tool timeout after acting
    (OBS + 'action("ACTION1")\n_obs("before-kill")\nwhile True:\n    pass\n', 2),
    (OBS + '_obs("after-kill", True)\nfor _a in ["ACTION2", "ACTION3", "ACTION4", "ACTION1", "ACTION2"]:\n'
     '    action(_a)\n    _obs(_a)\n_obs("end2", True)\n', None),
]
SCENARIO_LONG = [SCENARIO_FULL[0], SCENARIO_FULL[1], SCENARIO_FULL[3], SCENARIO_FULL[4]]


def play(harness, tmp_path, *, cached, n, scenario, tag):
    TA, RS, SB, S = harness
    sp = tmp_path / f"{tag}_{'c' if cached else 'b'}" / f"g_{n}_tool_runtime_state.json"
    sp.parent.mkdir(parents=True)
    if cached:
        assert hc.install()
        hc.reset_stats()
    raw = []
    inner = TA.run_sandboxed_python

    def capture(*a, **kw):
        r = inner(*a, **kw)
        raw.append({k: r.get(k) for k in ("stdout", "error", "result", "action_results")})
        return r

    TA.run_sandboxed_python = capture
    try:
        game = h.FakeGame(RS, S, sp, level_every=6, die_every=4)
        game.seed()
        game.advance(max(0, n - 1), write_each=False)
        agent = h.make_agent(TA, sp, game)
        contents = []
        for code, timeout in scenario:
            agent._python_timeout = timeout or 30
            res = agent._run_python_tool(sp, {"code": code})
            contents.append((res.content, res.step_executed, res.made_progress))
        cf, hist = TA.load_runtime_state(sp)
        prompt = agent._build_user_prompt(game.n, valid_actions=list(agent._current_valid_actions),
                                          current_frame=cf, history_entries=hist,
                                          previous_step_summary=agent._last_step_summary)
        out = dict(contents=contents, raw=raw, prompt=prompt, summary=agent._last_step_summary,
                   file=json.loads(sp.read_text(encoding="utf-8")), n_hist=len(game.history_entries),
                   loaded=(cf, hist))
        if cached:
            out["stats"] = hc.summary()
        return out
    finally:
        TA.run_sandboxed_python = inner
        if cached:
            hc.uninstall()


def assert_same(base, cache, expect_timeouts=1):
    assert len(base["contents"]) == len(cache["contents"])
    for run in (base, cache):
        # not vacuous: every call but the deliberate kill ran to the end and printed its observations
        timeouts = sum(TIMEOUT in (r["error"] or "") for r in run["raw"])
        assert timeouts == expect_timeouts, [r["error"] for r in run["raw"]]
        assert all("OBS " in (r["stdout"] or "") for r in run["raw"] if TIMEOUT not in (r["error"] or ""))
    for i, (b, c) in enumerate(zip(base["raw"], cache["raw"])):
        assert b["stdout"] == c["stdout"], f"sandbox-visible state differs in call {i}"
        assert b["error"] == c["error"], f"sandbox error differs in call {i}"
        assert b["result"] == c["result"] and b["action_results"] == c["action_results"], i
    for i, (b, c) in enumerate(zip(base["contents"], cache["contents"])):
        assert b == c, f"model-visible tool payload differs in call {i}"
    assert base["prompt"] == cache["prompt"], "next user prompt differs"
    assert base["summary"] == cache["summary"]
    assert base["file"] == cache["file"], "state file content differs"
    assert base["loaded"] == cache["loaded"], "load_runtime_state differs"
    assert base["n_hist"] == cache["n_hist"]


@pytest.mark.parametrize("n", [1, 50])
def test_sandbox_and_prompts_identical(harness, tmp_path, n):
    base = play(harness, tmp_path, cached=False, n=n, scenario=SCENARIO_FULL, tag="full")
    cache = play(harness, tmp_path, cached=True, n=n, scenario=SCENARIO_FULL, tag="full")
    assert_same(base, cache)
    s = cache["stats"]
    assert s["errors"] == 0, s["last_error"]
    # one full payload per sandbox process, every later state as a delta
    assert s["payload_full"] == len(SCENARIO_FULL)
    # every action() reply after the first payload of a process went out as a delta
    replies = sum(len(r["action_results"] or []) for r in cache["raw"])
    assert s["payload_delta"] == replies >= 5 and s["payload_plain"] == 0
    assert s["sandbox_timeout_kills"] == 1
    assert s["loads_cached"] > 0 and s["loads_file"] == 0 and s["loads_stale"] == 0
    assert s["view_misses"] == 0 and s["writes_appended"] > 0
    stdout = "".join(r["stdout"] or "" for r in cache["raw"])
    assert "pristine True" in stdout
    # the scenario really crossed RESETs and level changes
    actions = [e["action"] for e in cache["file"]["history"]]
    assert "RESET" in actions
    assert len({e["frame"]["level"] for e in cache["file"]["history"]}) > 1


def test_identical_at_1000(harness, tmp_path):
    base = play(harness, tmp_path, cached=False, n=1000, scenario=SCENARIO_LONG, tag="long")
    cache = play(harness, tmp_path, cached=True, n=1000, scenario=SCENARIO_LONG, tag="long")
    assert_same(base, cache)
    s = cache["stats"]
    assert s["errors"] == 0 and s["payload_full"] == len(SCENARIO_LONG) and s["payload_delta"] >= 5
    assert s["max_history"] >= 1000


def test_file_is_valid_json_equal_to_the_original_writer(harness, tmp_path):
    TA, RS, SB, S = harness
    orig_dir, cache_dir = tmp_path / "o", tmp_path / "c"
    orig_dir.mkdir()
    cache_dir.mkdir()
    g1 = h.FakeGame(RS, S, orig_dir / "s.json", seed=3)
    g1.seed()
    g1.advance(60)
    assert hc.install()
    try:
        g2 = h.FakeGame(RS, S, cache_dir / "s.json", seed=3)
        g2.seed()
        for k in (4, 8, 48):          # appended in place across several writes (multiples of 4 keep
            #                           advance()'s action cycle aligned with g1's)
            g2.advance(k)
            assert json.loads((cache_dir / "s.json").read_text()) == json.loads(
                json.dumps({"current_frame": RS.frame_to_payload(g2.current_frame()),
                            "history": [RS.history_entry_to_payload(e) for e in g2.history_entries]}))
        assert json.loads((orig_dir / "s.json").read_text()) == json.loads((cache_dir / "s.json").read_text())
        # a file rewritten behind the cache's back is detected and parsed instead
        RS.write_runtime_state(cache_dir / "s.json", current_frame=None, history=g2.history_entries[:3])
        cf, hist = TA.load_runtime_state(cache_dir / "s.json")
        assert cf is None and len(hist) == 3 and hc.STATS["loads_stale"] >= 1
        # and the next solver write starts over from a full write
        g2.advance(1)
        assert json.loads((cache_dir / "s.json").read_text())["history"][-1]["action"] == "ACTION1"
        assert TA.load_runtime_state(cache_dir / "s.json") == RS.load_runtime_state(cache_dir / "s.json")
    finally:
        hc.uninstall()


def test_a_replaced_history_falls_back_to_a_full_rebuild(harness, tmp_path):
    TA, RS, SB, S = harness
    assert hc.install()
    try:
        hc.reset_stats()
        g = h.FakeGame(RS, S, tmp_path / "s.json")
        g.seed()
        g.advance(10)
        g.history_entries = list(g.history_entries[:4])      # not an extension of what was cached
        g.advance(2)
        assert hc.STATS["rebuilds"] == 1
        assert TA.load_runtime_state(tmp_path / "s.json") == RS.load_runtime_state(tmp_path / "s.json")
    finally:
        hc.uninstall()


def test_batch_writes_collapse_to_one(tmp_path):
    calls = []

    class Sess:
        def __init__(self):
            self.state_path = tmp_path / "x.json"

        def write_runtime_state(self):
            calls.append("w")

        def step_env(self, arguments):
            for _ in arguments["actions"]:
                self.write_runtime_state()
            if arguments.get("boom"):
                raise ValueError("boom")
            return {"ok": True}

        def play(self):
            return "played"

    hc.wrap_session_class(Sess)
    s = Sess()
    assert s.step_env({"actions": [1, 2, 3]}) == {"ok": True} and calls == ["w"]
    with pytest.raises(ValueError):
        s.step_env({"actions": [1, 2], "boom": True})
    assert calls == ["w", "w"], "the batch's state is still written when it raises"
    s.write_runtime_state()
    assert calls == ["w", "w", "w"], "outside a batch every write goes through"
    assert s.step_env({"actions": []}) == {"ok": True} and len(calls) == 3


def test_installs_on_the_real_classes_and_uninstalls(harness):
    TA, RS, SB, S = harness
    originals = (TA.load_runtime_state, TA._ascii_history_view_payload, SB._send_json_line,
                 SB._SANDBOX_BOOTSTRAP, S.write_runtime_state, S._HarnessGameSession.step_env,
                 S._HarnessGameSession.write_runtime_state, S._HarnessGameSession.play,
                 TA.ToolAgent._run_python_tool, TA.run_sandboxed_python)
    assert hc.install() and not hc.install()
    try:
        assert TA.load_runtime_state.__module__ == "history_cache"
        assert S.write_runtime_state.__module__ == "history_cache"
        assert S._HarnessGameSession.step_env.__module__ == "history_cache"
        assert "_hc_history(state_payload)" in SB._SANDBOX_BOOTSTRAP
        compile(SB._SANDBOX_BOOTSTRAP, "bootstrap", "exec")
    finally:
        hc.uninstall()
    assert originals == (TA.load_runtime_state, TA._ascii_history_view_payload, SB._send_json_line,
                         SB._SANDBOX_BOOTSTRAP, S.write_runtime_state, S._HarnessGameSession.step_env,
                         S._HarnessGameSession.write_runtime_state, S._HarnessGameSession.play,
                         TA.ToolAgent._run_python_tool, TA.run_sandboxed_python)


def _per_action(harness, tmp_path, n, cached, k=8):
    TA, RS, SB, S = harness
    sp = tmp_path / f"t{n}{'c' if cached else 'b'}" / "g_tool_runtime_state.json"
    sp.parent.mkdir(parents=True)
    if cached:
        hc.install()
    try:
        g = h.FakeGame(RS, S, sp, level_every=10**9, die_every=10**9)
        g.seed()
        g.advance(n - 1, write_each=False)
        agent = h.make_agent(TA, sp, g)
        agent._run_python_tool(sp, {"code": f'for _a in ["ACTION5"] * {k}:\n    action(_a)\n'})
        iv = sorted(b - a for a, b in zip(g.calls, g.calls[1:]))
        return iv[len(iv) // 2]
    finally:
        if cached:
            hc.uninstall()


def test_per_action_overhead_is_roughly_flat_in_history_length(harness, tmp_path):
    cached = {n: _per_action(harness, tmp_path, n, True) for n in (50, 300, 1000)}
    base300 = _per_action(harness, tmp_path, 300, False, k=4)
    print("per-action seconds, cache:", {n: round(v, 4) for n, v in cached.items()}, "baseline@300:", round(base300, 3))
    # baseline grows ~4 ms per history entry; with the cache the only O(N) left is the sandbox
    # rebuilding fresh frame objects (~0.1 ms per entry), so 20x the history costs well under 20x
    assert cached[1000] < 0.6
    assert cached[1000] < base300, "the cache at 1000 entries must beat the baseline at 300"
    assert cached[1000] - cached[50] < 0.1 * (base300 / 300) * 950, "slope must be < 10% of the baseline's"


def _build_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    import _build_m2_level_memory_kernel as build
    return build


def test_the_built_install_cells_compose_on_the_real_harness(harness, capsys):
    """histcache + triedfacts: the kernel's level-memory install cell and history-cache cell, executed
    verbatim in notebook order on the real harness, install both (and print every gate marker)."""
    TA, RS, SB, S = harness
    build = _build_module()
    variants = ("histcache", "triedfacts")
    cells = ["".join(build.install_cell(variants)["source"]), "".join(build.history_cache_cell()["source"])]
    cls = TA.ToolAgent
    saved_lm = {n: cls.__dict__[n] for n in ("_build_user_prompt", "_trim_messages_for_context", "_ensure_session")}
    saved_mods = {k: sys.modules.get(k) for k in ("history_cache", "level_memory")}
    saved_env = os.environ.get("LEVEL_MEMORY_TRIED_FACTS")
    ns = {"json": json}
    try:
        for i, cell in enumerate(cells):
            exec(compile(cell, f"install_cell_{i}", "exec"), ns)
        out = capsys.readouterr().out
        for marker in ["LEVEL_MEMORY installed"] + [build.VARIANT_MARKERS[v] for v in variants]:
            assert marker in out, marker
        assert ns["HISTORY_CACHE"].STATS["installed"] and ns["LEVEL_MEMORY"].tried_facts_enabled()
        assert cls._build_user_prompt.__module__ == "level_memory"
        assert cls._run_python_tool.__module__ == "history_cache"
        assert TA.load_runtime_state.__module__ == "history_cache"
        assert ns["HISTORY_CACHE"].summary()["errors"] == 0 and ns["LEVEL_MEMORY"].summary()["errors"] == 0
    finally:
        if "HISTORY_CACHE" in ns:
            ns["HISTORY_CACHE"].uninstall()
        for n, f in saved_lm.items():
            setattr(cls, n, f)
        if "_lm_installed" in cls.__dict__:
            del cls._lm_installed
        for k, m in saved_mods.items():
            if m is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = m
        if saved_env is None:
            os.environ.pop("LEVEL_MEMORY_TRIED_FACTS", None)
        else:
            os.environ["LEVEL_MEMORY_TRIED_FACTS"] = saved_env
    assert TA.load_runtime_state.__module__ != "history_cache"


@pytest.mark.parametrize("variants", [("histcache",), ("histcache", "triedfacts"), ("triedfacts",)])
def test_built_kernels_match_the_incumbent_except_the_additions(variants):
    """Every committed variant kernel is current (equal to a fresh build) and differs from the
    committed incumbent only by its own additions; its metadata only by id/title/code_file."""
    build = _build_module()
    from _check_notebook_cell import check_notebook
    slug = build.kernel_slug(variants)
    inc_dir = ROOT / "kaggle_submission_m2_level_memory" / "notebook"
    new_dir = ROOT / ("kaggle_submission_m2_lm_" + "_".join(variants)) / "notebook"
    inc = json.loads((inc_dir / "arc3-m2-level-memory.ipynb").read_text(encoding="utf-8"))
    new = json.loads((new_dir / f"{slug}.ipynb").read_text(encoding="utf-8"))
    assert check_notebook(new_dir / f"{slug}.ipynb") == []
    import _m2_input_resolver
    _m2_input_resolver.apply_input_resolver(inc)     # variants carry the mount-layout input resolver
    src_inc = ["".join(c["source"]) for c in inc["cells"]]
    src_new = ["".join(c["source"]) for c in new["cells"]]
    extra = [build.VARIANT_CELLS[v]() for v in variants if v in build.VARIANT_CELLS]
    extra_src = ["".join(c["source"]) for c in extra]
    for s in extra_src:
        assert s in src_new, "committed kernel is stale: rerun the build"
    lm_new = "".join(build.install_cell(variants)["source"])
    assert lm_new in src_new, "committed kernel is stale: rerun the build"
    rest = [s for s in src_new if s not in extra_src]
    assert len(rest) == len(src_inc)
    dump = build.RUN_DUMP + "".join(build.VARIANT_RUN_DUMP.get(v, "") for v in variants)
    lm_inc = "".join(build.install_cell(())["source"])
    for a, b in zip(src_inc[1:], rest[1:]):          # [0] is the markdown header
        if a == b:
            continue
        if b == lm_new:
            # the incumbent was built from an earlier level_memory.py; only the variant lines are new
            assert a.startswith(lm_inc.split("_LM_SRC = ")[0]), "install cell header changed"
            continue
        assert b == a.replace(build.RUN_DUMP, dump), "only the run cell and the install cell may differ"
    if "histcache" in variants:
        i_lm, i_hc, i_run = (src_new.index(lm_new), src_new.index(extra_src[0]),
                             next(i for i, s in enumerate(src_new) if s.startswith("print('Starting benchmark...')")))
        assert i_lm < i_hc < i_run == i_hc + 1
    m_inc = json.loads((inc_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    m_new = json.loads((new_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert m_new["id"] == build.kernel_id(variants) and m_new["title"] == slug and m_new["code_file"] == f"{slug}.ipynb"
    for k in ("id", "title", "code_file"):
        m_inc.pop(k), m_new.pop(k)
    assert m_new.pop("docker_image_pinning_type") == build.DOCKER_PINNING   # variants pin the image
    assert m_inc == m_new


def test_built_kernels_are_current(tmp_path, monkeypatch):
    """A fresh build of each variant is byte-identical to the committed notebook."""
    build = _build_module()
    for variants in [("histcache",), ("histcache", "triedfacts"), ("triedfacts",)]:
        slug = build.kernel_slug(variants)
        d = ROOT / ("kaggle_submission_m2_lm_" + "_".join(variants)) / "notebook"
        committed = (d / f"{slug}.ipynb").read_text(encoding="utf-8")
        monkeypatch.setattr(build, "ROOT", tmp_path)
        path = build.build(variants)
        monkeypatch.undo()
        assert path.read_text(encoding="utf-8") == committed, f"{slug}: committed kernel is stale, rerun the build"
