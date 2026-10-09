"""Build arc3-m2-level-memory: the verbatim milestone-2 fork + solved-level memory.

Two edits to the upstream notebook (kaggle_submission_milestone2_fork/upstream/), nothing else:
  (a) a cell right before the run cell installs level_memory.py (inlined) on the patched
      ToolAgent -- after every env knob and the benchmark unpickle, so the harness sees exactly
      the configuration the upstream run sees;
  (b) the memory's counters are printed and written out after bm.run.

    python scripts/_build_m2_level_memory_kernel.py                  # incumbent: arc3-m2-level-memory
    python scripts/_build_m2_level_memory_kernel.py --tried-facts   # arc3-m2-lm-triedfacts
    python scripts/_build_m2_level_memory_kernel.py --history-cache # arc3-m2-lm-histcache
    python scripts/_build_m2_level_memory_kernel.py --history-cache --tried-facts   # arc3-m2-lm-histcache-triedfacts

Variant flags compose. Each one adds a suffix, in the fixed order of VARIANT_ORDER, to the kernel slug and
output dir (e.g. --history-cache --tried-facts -> arc3-m2-lm-histcache-triedfacts); with none, the build
is the incumbent. A variant that changes behaviour does it through install-cell lines (VARIANT_INSTALL_LINES)
and/or a cell of its own inserted after the install cell (VARIANT_CELLS), optional lines appended to the
run cell's counter dump (VARIANT_RUN_DUMP), and a marker the run is checked for (VARIANT_MARKERS) -- never
by editing upstream cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))



def sub(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"REFUSING TO BUILD -- anchor for {what!r} found {n}x")
    return text.replace(old, new)


def cells_of(nb):
    return ["".join(c["source"]) for c in nb["cells"]]


def set_src(nb, i, text):
    nb["cells"][i]["source"] = text.splitlines(True)


FORK = ROOT / "kaggle_submission_milestone2_fork"
UPSTREAM = FORK / "upstream" / "arc-agi-3-milestone-2-solution.ipynb"
LM_SRC = (FORK / "level_memory" / "level_memory.py").read_text(encoding="utf-8")
assert "'''" not in LM_SRC, "module source must not contain ''' (it is inlined in r'''...''')"
HC_SRC = (FORK / "history_cache" / "history_cache.py").read_text(encoding="utf-8")
assert "'''" not in HC_SRC, "module source must not contain ''' (it is inlined in r'''...''')"
INCUMBENT = "arc3-m2-level-memory"
# flag name -> slug suffix; the slug and the checks below follow this order whatever the CLI order is
VARIANT_ORDER = ("histcache", "triedfacts")
VARIANT_INSTALL_LINES = {
    # run after the module is exec'd and installed, before the 'LEVEL_MEMORY installed' print
    "triedfacts": (
        "import os as _os\n"
        "_os.environ['LEVEL_MEMORY_TRIED_FACTS'] = '1'   # variant: pin current-level facts at eviction\n"
        "assert LEVEL_MEMORY.tried_facts_enabled()\n"
        "print('TRIED_FACTS installed', LEVEL_MEMORY.summary(), flush=True)\n"
    ),
}
VARIANT_MARKERS = {"histcache": "HISTORY_CACHE installed", "triedfacts": "TRIED_FACTS installed"}
VARIANT_BLURB = {
    "histcache": "A history cache (in-memory game history, compact batched state writes, a per-frame ascii "
                 "cache, incremental sandbox payloads), ported from sirikilohit's patch M86 and extended; "
                 "installed by its own cell. It changes the cost of each action, not what the model or a "
                 "snippet sees.",
    "triedfacts": "At the same eviction point it also pins a facts-only block about the current unsolved "
                  "level (actions spent, game-over counts, last actions of recent fatal runs, tail of the "
                  "model's last reasoning; <= 3 KB).",
}


def variant_names(variants) -> tuple[str, ...]:
    unknown = set(variants) - set(VARIANT_ORDER)
    assert not unknown, unknown
    return tuple(v for v in VARIANT_ORDER if v in set(variants))


def kernel_slug(variants) -> str:
    names = variant_names(variants)
    return "arc3-m2-lm-" + "-".join(names) if names else INCUMBENT


def kernel_id(variants) -> str:
    return "calamitychasm/" + kernel_slug(variants)


RUN_ANCHOR = ("await bm.run(soft_end_time=soft_end, runtime_environment=target, "
              "minimal_diagnostics=TRUE_SUBMISSION)\n")
RUN_DUMP = RUN_ANCHOR + (
    "try:   # [calamitychasm] solved-level memory counters\n"
    "    (WORKING_DIR / 'level_memory_summary.json').write_text(json.dumps(LEVEL_MEMORY.summary(), indent=2))\n"
    "    print('LEVEL_MEMORY summary', json.dumps(LEVEL_MEMORY.summary()), flush=True)\n"
    "except Exception as _exc:\n"
    "    print('LEVEL_MEMORY summary failed', repr(_exc), flush=True)\n"
)


HC_DUMP = (
    "try:   # [calamitychasm] history cache counters\n"
    "    (WORKING_DIR / 'history_cache_summary.json').write_text(json.dumps(HISTORY_CACHE.summary(), indent=2))\n"
    "    print('HISTORY_CACHE summary', json.dumps(HISTORY_CACHE.summary()), flush=True)\n"
    "except Exception as _exc:\n"
    "    print('HISTORY_CACHE summary failed', repr(_exc), flush=True)\n"
)


def history_cache_cell() -> dict:
    src = (
        "# [calamitychasm] HISTORY CACHE (variant histcache) -- see experiments/stage7_m2_level_memory.md.\n"
        "# Replaces the per-action O(history) state-file rewrite/re-parse and full-history sandbox\n"
        "# payload with an in-memory cache + incremental deltas; what the model sees is unchanged.\n"
        "# Ported from sirikilohit's Milestone-2 patch M86 (Apache-2.0) and adapted to this harness.\n"
        "import sys as _sys, types as _types\n"
        f"_HC_SRC = r'''{HC_SRC}'''\n"
        "HISTORY_CACHE = _types.ModuleType('history_cache')\n"
        "_sys.modules['history_cache'] = HISTORY_CACHE\n"
        "exec(compile(_HC_SRC, 'history_cache.py', 'exec'), HISTORY_CACHE.__dict__)\n"
        "assert HISTORY_CACHE.install()\n"
        "import inference.agent.tool_agent as _hc_ta\n"
        "import inference.agent.python_tool_sandbox as _hc_sb\n"
        "import inference.framework.solver as _hc_s\n"
        "for _f in (_hc_ta.load_runtime_state, _hc_ta._ascii_history_view_payload, _hc_sb._send_json_line,\n"
        "           _hc_s.write_runtime_state, _hc_s._HarnessGameSession.step_env):\n"
        "    assert _f.__module__ == 'history_cache', _f\n"
        "assert '_hc_history(state_payload)' in _hc_sb._SANDBOX_BOOTSTRAP\n"
        "print('HISTORY_CACHE installed', json.dumps(HISTORY_CACHE.summary()), flush=True)\n"
    )
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(True)}


# variant -> builder of a cell inserted right after the level-memory install cell (before the run cell)
VARIANT_CELLS = {"histcache": history_cache_cell}
# variant -> lines appended to the run cell's counter dump
VARIANT_RUN_DUMP = {"histcache": HC_DUMP}


def install_cell(variants=()) -> dict:
    src = (
        "# [calamitychasm] SOLVED-LEVEL MEMORY -- see experiments/stage7_m2_level_memory.md.\n"
        "# Wraps ToolAgent._build_user_prompt / _trim_messages_for_context / _ensure_session at runtime;\n"
        "# the harness patch and bundle are untouched.\n"
        "import sys as _sys, types as _types\n"
        f"_LM_SRC = r'''{LM_SRC}'''\n"
        "LEVEL_MEMORY = _types.ModuleType('level_memory')\n"
        "_sys.modules['level_memory'] = LEVEL_MEMORY\n"
        "exec(compile(_LM_SRC, 'level_memory.py', 'exec'), LEVEL_MEMORY.__dict__)\n"
        "import inference.agent.tool_agent as _lm_ta\n"
        "LEVEL_MEMORY.install(_lm_ta.ToolAgent)\n"
        "assert _lm_ta.ToolAgent._lm_installed\n"
        "for _n in ('_build_user_prompt', '_trim_messages_for_context', '_ensure_session'):\n"
        "    assert getattr(_lm_ta.ToolAgent, _n).__module__ == 'level_memory', _n\n"
        + "".join(VARIANT_INSTALL_LINES.get(v, "") for v in variant_names(variants))
        + "print('LEVEL_MEMORY installed', LEVEL_MEMORY.summary(), flush=True)\n"
    )
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(True)}


def build(variants=()) -> Path:
    variants = variant_names(variants)
    nb = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    src = cells_of(nb)
    run_idx = [i for i, s in enumerate(src) if s.startswith("print('Starting benchmark...')")]
    assert len(run_idx) == 1, run_idx
    r = run_idx[0]
    dump = RUN_DUMP + "".join(VARIANT_RUN_DUMP.get(v, "") for v in variants)
    set_src(nb, r, sub(src[r], RUN_ANCHOR, dump, "bm.run summary dump"))
    for v in reversed(variants):      # inserted at r in reverse: they end up in VARIANT_ORDER
        if v in VARIANT_CELLS:
            nb["cells"].insert(r, VARIANT_CELLS[v]())
    nb["cells"].insert(r, install_cell(variants))
    md = ("## [calamitychasm] fork: milestone-2 solution + solved-level memory\n\n"
          "Verbatim fork of `dfranzen/arc-agi-3-milestone-2-solution` (Apache-2.0; full credit to Daniel Franzen, "
          "Jeroen Cottaar and Tufa Labs). One addition: solved-level memory, ported from sirikilohit's "
          "Milestone-2 patch M85 and adapted to this harness's prefix cache. Installed by the cell before the run.\n"
          + "".join(f"\nVariant `{v}`: {VARIANT_BLURB[v]}\n" for v in variants))
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": md.splitlines(True)})
    slug = kernel_slug(variants)
    d = ROOT / ("kaggle_submission_m2_level_memory" if not variants else
                "kaggle_submission_m2_lm_" + "_".join(variants)) / "notebook"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{slug}.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = json.loads((FORK / "notebook" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(id=kernel_id(variants), title=slug, code_file=path.name, is_private=True)
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def main(argv=None) -> int:
    from _check_notebook_cell import check_notebook
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history-cache", action="store_true",
                    help="cache the game history in memory, send the sandbox deltas (kernel arc3-m2-lm-histcache)")
    ap.add_argument("--tried-facts", action="store_true",
                    help="pin a facts-only block about the current level at eviction (kernel arc3-m2-lm-triedfacts)")
    args = ap.parse_args(argv)
    variants = ["histcache"] * args.history_cache + ["triedfacts"] * args.tried_facts
    path = build(variants)
    problems = check_notebook(path)
    if problems:
        print("REFUSING TO SHIP -- use-before-definition problems")
        for p in problems:
            print(" ", p)
        return 1
    print("built", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
