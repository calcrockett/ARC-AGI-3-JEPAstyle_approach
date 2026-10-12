"""Build the momentum-time kernels from the anim notebook.

Two notebooks, one policy source (kaggle_submission_duck_nvfp4_anim_momentum/
momentum_time.py, inlined verbatim into both):

  notebook/        arc3-duck-nvfp4-anim-momentum -- the REAL submission kernel:
                   the anim notebook unchanged except (a) one cell after the
                   customization hook installs the policy at production
                   settings, (b) the policy's decisions are written out after
                   the run.
  notebook_haiku/  arc3-momentum-haiku-smoke -- a cheap end-to-end test on a CPU
                   kernel with internet: no vLLM; the analyzer is Claude Haiku via
                   Anthropic's OpenAI-compatible endpoint, under the SAME harness
                   settings the served model runs with (32,768-token context,
                   same tool/yield/temperature knobs), so it sees only what the
                   harness shows. 6 games on 3 slots so the queue is exercised,
                   with the policy's clocks scaled down so every rule can fire.

Every edit is an exact-anchor replacement that must match once, and both
notebooks must pass the use-before-definition check, or the build refuses.

    python scripts/_build_momentum_kernels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

ANIM_DIR = ROOT / "kaggle_submission_duck_nvfp4_anim" / "notebook"
OUT = ROOT / "kaggle_submission_duck_nvfp4_anim_momentum"
POLICY_SRC = (OUT / "momentum_time.py").read_text(encoding="utf-8")
assert "'''" not in POLICY_SRC, "policy source must not contain ''' (it is inlined in r'''...''')"

# Production settings, from the simulation in experiments/stage7_momentum_time.md.
PROD = {"momentum_s": 45 * 60.0, "stall_s": 100 * 60.0, "hard_mult": 1.5}
# Haiku smoke: a 600 s base so a short run reaches every rule.
SMOKE = {"base": 600.0, "momentum_s": 240.0, "stall_s": 420.0, "hard_mult": 1.5, "concurrency": 3}
SMOKE_GAMES = ["ft09-0d8bbf25", "vc33-5430563c", "tu93-0768757b",   # usually level early
               "dc22-fdcac232", "sc25-635fd71a", "sk48-d8078629"]   # usually stall
HAIKU_MODEL = "claude-haiku-4-5"


def install_cell(params_expr: str) -> dict:
    src = (
        "# [calamitychasm] MOMENTUM TIME ALLOCATION -- see experiments/stage7_momentum_time.md.\n"
        "# Replaces _HarnessGameSession.runtime_limit_reached / timing_payload; no bundle file is edited.\n"
        "import sys as _sys, types as _types\n"
        f"_MT_SRC = r'''{POLICY_SRC}'''\n"
        "_mt = _types.ModuleType('momentum_time')\n"
        "_sys.modules['momentum_time'] = _mt   # dataclasses resolve annotations through sys.modules\n"
        "exec(compile(_MT_SRC, 'momentum_time.py', 'exec'), _mt.__dict__)\n"
        "import inference.framework.solver as _solver_mod\n"
        f"MOMENTUM_POLICY = _mt.MomentumTimePolicy({params_expr})\n"
        "_mt.install(_solver_mod._HarnessGameSession, MOMENTUM_POLICY)\n"
        "assert _solver_mod._HarnessGameSession._momentum_policy is MOMENTUM_POLICY\n"
        "print('MOMENTUM_TIME installed', MOMENTUM_POLICY.summary()['params'], flush=True)\n"
    )
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(True)}


SUMMARY_DUMP = (
    "    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True)\n",
    "    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True)\n"
    "    try:   # [calamitychasm] momentum-time decisions, for the write-up\n"
    "        (WORKING_DIR / 'momentum_time_summary.json').write_text(json.dumps(MOMENTUM_POLICY.summary(), indent=2))\n"
    "        print('MOMENTUM_TIME decisions', json.dumps(MOMENTUM_POLICY.summary()['decisions']), flush=True)\n"
    "    except Exception as _exc:\n"
    "        print('MOMENTUM_TIME summary failed', repr(_exc), flush=True)\n",
)


def sub(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"REFUSING TO BUILD -- anchor for {what!r} found {n}x")
    return text.replace(old, new)


def cells_of(nb):
    return ["".join(c["source"]) for c in nb["cells"]]


def set_src(nb, i, text):
    nb["cells"][i]["source"] = text.splitlines(True)


def build_real() -> Path:
    nb = json.loads((ANIM_DIR / "arc3-duck-nvfp4-anim.ipynb").read_text(encoding="utf-8"))
    src = cells_of(nb)
    assert src[13].startswith("# Exact public-25 and competition settings."), "cell 13 moved"
    set_src(nb, 15, sub(src[15], *SUMMARY_DUMP, "bm.run summary dump"))
    nb["cells"].insert(14, install_cell(
        "base_s=float(bm.solver.max_runtime_s_per_game), "
        + ", ".join(f"{k}={v!r}" for k, v in PROD.items())))
    d = OUT / "notebook"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "arc3-duck-nvfp4-anim-momentum.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = json.loads((ANIM_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(id="calamitychasm/arc3-duck-nvfp4-anim-momentum", title="arc3-duck-nvfp4-anim-momentum",
                code_file=path.name)
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def build_haiku() -> Path:
    nb = json.loads((ANIM_DIR / "arc3-duck-nvfp4-anim.ipynb").read_text(encoding="utf-8"))
    src = cells_of(nb)

    # --- cell 1: upstream GPU diagnostic (runs nvidia-smi) -- diagnostic only; no GPU here
    assert "nvidia-smi" in src[1], "cell 1 moved"
    set_src(nb, 1, "# [calamitychasm] Haiku smoke test runs on CPU: the upstream nvidia-smi diagnostic is skipped.\n"
                   "print('HAIKU_SMOKE cpu run: no GPU diagnostic', flush=True)\n")

    # --- cell 9: no vLLM boot; the analyzer is Haiku under the served model's harness settings
    cut = src[9].index("# Solver setup commands (wheels, vLLM server startup, ...) run before the benchmark loads.")
    haiku_env = {
        # identical to the served run (taaf_setup_env.json of a real anim run)
        "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
        "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
        "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
        "LOCAL_ANALYZER_TOOL_STEPS": "0",
        "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
        "LOCAL_ANALYZER_TEMPERATURE": "0.6",
        "LOCAL_ANALYZER_TOP_P": "0.95",
        "LOCAL_ANALYZER_TOP_K": "20",
        "LOCAL_ANALYZER_SEED": "20260825",
        "LOCAL_ANALYZER_YIELD_SECONDS": "180",
        "MULTIMODAL_CONTEXT": "current_grid",
        "MULTIMODAL_UPSCALE": "4",
        # the substitution
        "LOCAL_ANALYZER_PROVIDER": "anthropic",   # not 'openai': that maps to vLLM mode (top_k, chat_template_kwargs)
        "LOCAL_ANALYZER_BASE_URL": "https://api.anthropic.com/v1",
        "LOCAL_ANALYZER_MODEL_ID": HAIKU_MODEL,
        "LOCAL_ANALYZER_MAX_OUTPUT": "4096",      # the served run leaves it unset; Anthropic requires one
        "LOCAL_ANALYZER_ENABLE_THINKING": "false",
    }
    cell9 = src[9][:cut] + (
        "# [calamitychasm] HAIKU SMOKE TEST: no vLLM. The analyzer is Claude Haiku through Anthropic's\n"
        "# OpenAI-compatible endpoint, under the served model's harness settings (32,768-token context).\n"
        "from kaggle_secrets import UserSecretsClient\n"
        "try:\n"
        "    _ANTHROPIC_KEY = UserSecretsClient().get_secret('ANTHROPIC_API_KEY')\n"
        "except Exception as _exc:\n"
        "    raise RuntimeError('Attach the Kaggle secret ANTHROPIC_API_KEY to this notebook '\n"
        "                       '(Add-ons > Secrets) and run it again.') from _exc\n"
        f"_HAIKU_ENV = {json.dumps(haiku_env, indent=4)}\n"
        "_persisted = json.loads(SETUP_ENV_PATH.read_text())\n"
        "_persisted.update(_HAIKU_ENV)                    # the key is NOT written to disk\n"
        "SETUP_ENV_PATH.write_text(json.dumps(_persisted, indent=2, sort_keys=True) + '\\n')\n"
        "os.environ.update(_HAIKU_ENV)\n"
        "os.environ['LOCAL_ANALYZER_API_KEY'] = _ANTHROPIC_KEY\n"
        "assert 'inference' not in sys.modules and 'taaf' not in sys.modules, 'solver imported before env was set'\n"
        "import inference.agent.tool_agent as _tool_agent\n"
        "import taaf as _taaf\n"
        "for _m in (_tool_agent, _taaf):\n"
        "    assert str(Path(_m.__file__).resolve()).startswith(str(ANIM_BUNDLE_DIR.resolve())), (_m.__name__, _m.__file__)\n"
        "assert _tool_agent._LOCAL_ANALYZER_CONTEXT_WINDOW == 32768\n"
        "# Anthropic rejects temperature and top_p together on Claude 4.x models: keep temperature.\n"
        "import requests as _rq\n"
        "_orig_post = _rq.post\n"
        "HAIKU_CALLS = {'n': 0, 'errors': 0}\n"
        "def _anthropic_post(url, *a, **kw):\n"
        "    if 'api.anthropic.com' in str(url) and isinstance(kw.get('json'), dict):\n"
        "        _p = dict(kw['json'])\n"
        "        if 'temperature' in _p:\n"
        "            _p.pop('top_p', None)\n"
        "        kw['json'] = _p\n"
        "        HAIKU_CALLS['n'] += 1\n"
        "        _r = _orig_post(url, *a, **kw)\n"
        "        if getattr(_r, 'status_code', 200) >= 400:\n"
        "            HAIKU_CALLS['errors'] += 1\n"
        "            print('HAIKU_HTTP_ERROR', _r.status_code, _r.text[:500], flush=True)\n"
        "        return _r\n"
        "    return _orig_post(url, *a, **kw)\n"
        "_rq.post = _anthropic_post\n"
        f"print('HAIKU_SMOKE analyzer={HAIKU_MODEL} context_window=32768 (served-model setting)', flush=True)\n"
    )
    set_src(nb, 9, cell9)

    # --- cell 11: the pickled solver is the anim one, as in production
    # --- cell 13: smoke settings (queue: 6 games on 3 slots; scaled clocks; Haiku, not a local server)
    assert src[13].startswith("# Exact public-25 and competition settings."), "cell 13 moved"
    set_src(nb, 13,
        "# [calamitychasm] HAIKU SMOKE settings: 6 games on 3 slots (the queue is exercised), short clocks.\n"
        f"bm.solver.max_runtime_s_per_game = {SMOKE['base']!r}\n"
        "bm.solver.analyzer_timeout = 300.0\n"
        f"bm.solver.concurrency = {SMOKE['concurrency']}\n"
        "bm.solver.max_actions_per_game = None\n"
        "bm.solver.save_request_logs = False\n"
        "bm.solver.start_local_server = False   # no vLLM\n"
        "bm.solver.model = 'local'              # resolve the analyzer from LOCAL_ANALYZER_* (Haiku)\n"
        "print('HAIKU_SMOKE settings', bm.solver.max_runtime_s_per_game, bm.solver.concurrency, flush=True)\n")

    # --- cell 15: subset of games, no vLLM watchdog, subset audit
    c15 = src[15]
    c15 = sub(c15, "    bm.games = [offline_by_id[game_id] for game_id in PUBLIC_GAME_IDS]\n",
              f"    bm.games = [offline_by_id[game_id] for game_id in {SMOKE_GAMES!r}]\n", "subset games")
    c15 = sub(c15, "    if len(bm.games) != 25:\n        raise RuntimeError(f'Expected 25 public games, got {len(bm.games)}.')\n",
              f"    assert len(bm.games) == {len(SMOKE_GAMES)}, len(bm.games)\n", "25-game assert")
    c15 = sub(c15, "    print(f'PUBLIC25_SELECTION games={len(bm.games)} passes=1', flush=True)\n",
              "    print(f'HAIKU_SMOKE_SELECTION games={[g.env_name for g in bm.games]}', flush=True)\n", "selection print")
    ws = c15.index("import vllm_server_watchdog as vllm_watchdog")
    we = c15.index("# Play the benchmark; watchdog stop and teardown run even if it raises.")
    c15 = c15[:ws] + (
        "class _NoWatchdog:   # [calamitychasm] no vLLM to watch in the Haiku smoke test\n"
        "    @staticmethod\n"
        "    def stop_background(timeout_seconds=0.0):\n"
        "        return None\n"
        "vllm_watchdog = _NoWatchdog()\n\n") + c15[we:]
    c15 = sub(c15, *SUMMARY_DUMP, "bm.run summary dump")
    audit_start = c15.index("        public_runs = list(bm.game_runs)")
    audit_end = c15.index("finally:\n    try:\n        vllm_watchdog.stop_background")
    c15 = c15[:audit_start] + (
        "        # [calamitychasm] smoke audit: every game finalized, no crash, policy consulted.\n"
        "        _runs = list(bm.game_runs)\n"
        "        _rows = [(r.game_id, r.state, r.levels_completed, len(r.history), r.final_score) for r in _runs]\n"
        "        for _row in _rows:\n"
        "            print('HAIKU_SMOKE_RUN', _row, flush=True)\n"
        "        _summary = MOMENTUM_POLICY.summary()\n"
        "        _report = {'runs': _rows, 'momentum': _summary, 'haiku_calls': HAIKU_CALLS}\n"
        "        (WORKING_DIR / 'haiku_smoke_report.json').write_text(json.dumps(_report, indent=2, default=str))\n"
        "        _problems = []\n"
        "        if len(_runs) != len(bm.games): _problems.append(f'runs {len(_runs)} != games {len(bm.games)}')\n"
        "        if any(r.state == 'crashed' for r in _runs): _problems.append('crashed runs')\n"
        "        if any(r.state not in {'won', 'gave_up', 'cancelled'} or r.final_score is None for r in _runs):\n"
        "            _problems.append('unfinalized runs')\n"
        "        if sum(len(r.history) for r in _runs) <= 0: _problems.append('no actions')\n"
        "        if _summary['games_seen'] != len(_runs): _problems.append('policy did not see every game')\n"
        "        if HAIKU_CALLS['n'] == 0: _problems.append('no Haiku calls')\n"
        "        print('HAIKU_SMOKE_REPORT', json.dumps({'decisions': _summary['decisions'], 'haiku_calls': HAIKU_CALLS, 'problems': _problems}), flush=True)\n"
        "        if _problems:\n"
        "            raise RuntimeError(f'HAIKU SMOKE FAILED: {_problems}')\n"
        "        print('HAIKU_SMOKE_PASSED', flush=True)\n"
    ) + c15[audit_end:]
    set_src(nb, 15, c15)

    nb["cells"].insert(14, install_cell(
        f"base_s={SMOKE['base']!r}, momentum_s={SMOKE['momentum_s']!r}, "
        f"stall_s={SMOKE['stall_s']!r}, hard_mult={SMOKE['hard_mult']!r}"))
    nb["cells"][0]["source"] = [
        "# arc3-momentum-haiku-smoke\n", "\n",
        "CPU + internet smoke test of the momentum time policy, NOT a submission. The anim notebook with the vLLM\n",
        "boot removed and Claude Haiku as the analyzer, under the served model's harness settings. 6 games on 3 slots.\n",
    ]
    d = OUT / "notebook_haiku"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "arc3-momentum-haiku-smoke.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = json.loads((ANIM_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(id="calamitychasm/arc3-momentum-haiku-smoke", title="arc3-momentum-haiku-smoke",
                code_file=path.name, enable_gpu=False, enable_internet=True, model_sources=[],
                keywords=[])
    meta.pop("machine_shape", None)
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def main() -> int:
    from _check_notebook_cell import check_notebook

    ok = True
    for build in (build_real, build_haiku):
        path = build()
        problems = check_notebook(path)
        print(f"wrote {path.relative_to(ROOT)}: {'OK' if not problems else problems[:5]}")
        ok &= not problems
    if not ok:
        print("REFUSING TO SHIP -- use-before-definition problems")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
