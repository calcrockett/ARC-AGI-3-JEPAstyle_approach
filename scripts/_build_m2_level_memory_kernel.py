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
    python scripts/_build_m2_level_memory_kernel.py --input-resolver  # arc3-m2-lm-inputs: incumbent + input resolver only
    python scripts/_build_m2_level_memory_kernel.py --turbo         # arc3-m2-turbo (see below)
    python scripts/_build_m2_level_memory_kernel.py --turbo-lossless  # arc3-m2-turbo-lossless (turbo, acceptance stays 1.0)
    python scripts/_build_m2_level_memory_kernel.py --reap --streams 14   # a subset: arc3-m2-lm-reap-s14

Input mount layouts: Kaggle mounts inputs either at /kaggle/input/{datasets/<owner>,competitions}/<slug> or at
/kaggle/input/<slug>. EVERY build with at least one variant (and the speed and vLLM kernels, which derive from
the incumbent notebook) carries scripts/_m2_input_resolver.py's resolver: INPUT_RESOLVED lines, INPUT_MISSING +
error if absent. The no-variant incumbent (v1, submitted) is built without it and stays byte-identical.

Variant flags compose. Each one adds a suffix, in the fixed order of VARIANT_ORDER, to the kernel slug and
output dir (e.g. --history-cache --tried-facts -> arc3-m2-lm-histcache-triedfacts); with none, the build
is the incumbent. A harness variant changes behaviour through install-cell lines (VARIANT_INSTALL_LINES)
and/or a cell of its own inserted after the install cell (VARIANT_CELLS), optional lines appended to the
run cell's counter dump (VARIANT_RUN_DUMP), and a marker the run is checked for (VARIANT_MARKERS) -- never
by editing upstream harness code.

Serving / "turbo" variants (ported from JustAdev742's Milestone-2 work, Apache-2.0; provenance and
vendored files in kaggle_submission_milestone2_fork/turbo/, see its NOTICE.md). These edit the SGLang
launcher cell, the stream knob in the setup cell or the check-run game list, each on an anchor that must
occur exactly once:

    --timeout-fix      a timed-out python call keeps the retained functions; `time` importable (harness, runtime)
    --reap             REAP-448: prune 64 of 512 routed experts per layer at load (frees ~7.3 GiB of weights)
    --spec-accept X    MTP acceptance thresholds single = acc = X (incumbent 1.0 = lossless; JustAdev742 ran 0.5)
    --arc-hotmap       ARC-tuned FR-Spec hot-token map for the MTP draft (+ TOKEN_MAP_SHA updated)
    --streams N        N concurrent streams: MAXREQ, CUDA-graph max bs, Mamba cache 6N, ARC3_MAX_ACTIVE_STREAMS
    --check-all25      the (non-submission) check run plays all 25 public games, the speed kernels' shape
    --turbo            all of: --history-cache --timeout-fix --reap --spec-accept 0.5 --arc-hotmap
                       --streams 14 --check-all25  ->  calamitychasm/arc3-m2-turbo (kaggle_submission_m2_turbo/)
    --turbo-lossless   turbo without --spec-accept (acceptance thresholds stay the incumbent's lossless 1.0)
                       ->  calamitychasm/arc3-m2-turbo-lossless (kaggle_submission_m2_turbo_lossless/)
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from _m2_input_resolver import MARKER as INPUT_MARKER, apply_input_resolver  # noqa: E402



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
# variant kind -> slug suffix order; the slug and the checks below follow this order whatever the CLI order is.
# Kinds "acc" and "streams" carry a value in their token: acc50 = acceptance 0.5, s14 = 14 streams.
VARIANT_ORDER = ("histcache", "triedfacts", "timeoutfix", "reap", "acc", "hotmap", "streams", "all25", "inputs")
_FIXED_KINDS = ("histcache", "triedfacts", "timeoutfix", "reap", "hotmap", "all25", "inputs")
TURBO = ("histcache", "timeoutfix", "reap", "acc50", "hotmap", "s14", "all25")
TURBO_SLUG = "arc3-m2-turbo"
# turbo without the lossy MTP acceptance (upstream SPEC_ACCEPT_SINGLE / SPEC_ACCEPT_ACC stay 1.0)
TURBO_LOSSLESS = tuple(v for v in TURBO if v != "acc50")
TURBO_LOSSLESS_SLUG = "arc3-m2-turbo-lossless"
MAX_STREAMS = 14              # 16 was measured by nobody on this stack and adds retractions at long contexts
MAX_STREAMS_WITHOUT_REAP = 12  # without REAP's freed 7.3 GiB the 1.01M-token KV pool is oversubscribed past 12
VARIANT_INSTALL_LINES = {
    # run after the module is exec'd and installed, before the 'LEVEL_MEMORY installed' print
    "triedfacts": (
        "import os as _os\n"
        "_os.environ['LEVEL_MEMORY_TRIED_FACTS'] = '1'   # variant: pin current-level facts at eviction\n"
        "assert LEVEL_MEMORY.tried_facts_enabled()\n"
        "print('TRIED_FACTS installed', LEVEL_MEMORY.summary(), flush=True)\n"
    ),
}
VARIANT_MARKERS = {"histcache": "HISTORY_CACHE installed", "triedfacts": "TRIED_FACTS installed",
                   "timeoutfix": "TIMEOUT_FIX installed", "reap": "REAP448 applied kept=448"}
VARIANT_BLURB = {
    "histcache": "A history cache (in-memory game history, compact batched state writes, a per-frame ascii "
                 "cache, incremental sandbox payloads), ported from sirikilohit's patch M86 and extended; "
                 "installed by its own cell. It changes the cost of each action, not what the model or a "
                 "snippet sees.",
    "triedfacts": "At the same eviction point it also pins a facts-only block about the current unsolved "
                  "level (actions spent, game-over counts, last actions of recent fatal runs, tail of the "
                  "model's last reasoning; <= 3 KB).",
    "timeoutfix": "Sandbox-timeout fix (JustAdev742's ours-sandbox-timeout-keeps-work.patch, Apache-2.0, installed "
                  "at runtime): a python call that times out no longer wipes every retained helper function; "
                  "`time` is importable in the sandbox.",
    "reap": "REAP-448 at load (JustAdev742's sglang_reap_patch.py, Apache-2.0): the target is served with 448 of "
            "its 512 routed experts per layer (the experts the public REAP-k448 build keeps, verified against "
            "the router sha256s); frees ~7.3 GiB for KV cache. The MTP draft keeps all 512.",
    "hotmap": "MTP draft FR-Spec map tuned to ARC harness outputs (JustAdev742's hot_tokens_64k_arc.pt, "
              "Apache-2.0) instead of Pennyroyal's generic 64k map.",
    "all25": "The non-submission check run plays all 25 public games (the speed kernels' shape); the "
             "competition rerun is unchanged.",
    "inputs": "Input paths are resolved in either Kaggle mount layout (/kaggle/input/datasets/<owner>/<slug> or "
              "/kaggle/input/<slug>, plus a directory-name glob) before anything reads them; every build with a "
              "variant does this (INPUT_RESOLVED lines in the log, INPUT_MISSING and an error if absent). "
              "This variant is the incumbent with only that change.",
}


def _kind(tok: str) -> str:
    if tok in _FIXED_KINDS:
        return tok
    if re.fullmatch(r"acc[1-9][0-9]", tok):
        return "acc"
    if re.fullmatch(r"s1[1-9]", tok):
        return "streams"
    raise SystemExit(f"REFUSING TO BUILD -- unknown variant {tok!r}")


def variant_names(variants) -> tuple[str, ...]:
    toks = list(dict.fromkeys(variants))
    kinds = [_kind(t) for t in toks]
    if len(set(kinds)) != len(kinds):
        raise SystemExit(f"REFUSING TO BUILD -- two values for one variant: {toks}")
    return tuple(sorted(toks, key=lambda t: VARIANT_ORDER.index(_kind(t))))


def accept_value(variants) -> float | None:
    tok = next((t for t in variant_names(variants) if _kind(t) == "acc"), None)
    return None if tok is None else int(tok[3:]) / 100


def streams_value(variants) -> int | None:
    tok = next((t for t in variant_names(variants) if _kind(t) == "streams"), None)
    return None if tok is None else int(tok[1:])


def accept_token(x: float) -> str:
    n = round(x * 100)
    if not (0 < x < 1 and abs(n - x * 100) < 1e-9 and 10 <= n <= 99):
        raise SystemExit(f"REFUSING TO BUILD -- --spec-accept {x}: must be in 0.10..0.99 in steps of 0.01 "
                         "(1.0 is the incumbent's lossless setting: omit the flag)")
    return f"acc{n}"


def mamba_cache(streams: int) -> int:
    return 6 * streams        # the incumbent's 60 / 10; >= 5 per running request is SGLang's floor


def check_serving(variants) -> None:
    names = variant_names(variants)
    n = streams_value(names)
    if n is not None:
        if not 10 < n <= MAX_STREAMS:
            raise SystemExit(f"REFUSING TO BUILD -- --streams {n}: allowed 11..{MAX_STREAMS}")
        if n > MAX_STREAMS_WITHOUT_REAP and "reap" not in names:
            raise SystemExit(f"REFUSING TO BUILD -- --streams {n} needs --reap (KV pool oversubscribed without it)")
        if mamba_cache(n) // 5 < n:
            raise SystemExit("REFUSING TO BUILD -- Mamba cache // 5 < streams: SGLang would cap running requests")


def kernel_slug(variants) -> str:
    names = variant_names(variants)
    if set(names) == set(TURBO):
        return TURBO_SLUG
    if set(names) == set(TURBO_LOSSLESS):
        return TURBO_LOSSLESS_SLUG
    return "arc3-m2-lm-" + "-".join(names) if names else INCUMBENT


def kernel_dir(variants, root: Path | None = None) -> Path:
    root = ROOT if root is None else root
    names = variant_names(variants)
    if set(names) == set(TURBO):
        return root / "kaggle_submission_m2_turbo" / "notebook"
    if set(names) == set(TURBO_LOSSLESS):
        return root / "kaggle_submission_m2_turbo_lossless" / "notebook"
    return root / ("kaggle_submission_m2_level_memory" if not names else
                   "kaggle_submission_m2_lm_" + "_".join(names)) / "notebook"


def kernel_id(variants) -> str:
    return "calamitychasm/" + kernel_slug(variants)


def kernel_markers(variants) -> list[str]:
    """Every log marker the check run of this kernel must show (the incumbent's three first)."""
    names = variant_names(variants)
    out = ["LEVEL_MEMORY installed", "priority gate active", "harness patch applied successfully"]
    if resolves_inputs(names):
        out.append(INPUT_MARKER)
    for v in names:
        k = _kind(v)
        if k in VARIANT_MARKERS:
            out.append(VARIANT_MARKERS[k])
        elif k == "acc":
            out.append(f"SPEC_ACCEPT {accept_value(names)}")
        elif k == "hotmap":
            out.append(f"ARC_HOTMAP sha={hot_map()['sha']}")
        elif k == "streams":
            out.append(f"priority gate active: {streams_value(names)} concurrent streams")
    return out


def resolves_inputs(variants) -> bool:
    """Every build except the incumbent v1 (no variants, byte-identical to what was submitted) resolves
    /kaggle/input in either mount layout. The `inputs` variant is that and nothing else: a hardened
    incumbent under its own slug (arc3-m2-lm-inputs)."""
    return bool(variant_names(variants))


def kernel_counters(variants) -> list[str]:
    names = variant_names(variants)
    return (["level_memory_summary.json"] + ["history_cache_summary.json"] * ("histcache" in names)
            + ["timeout_fix_summary.json"] * ("timeoutfix" in names))


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


# ------------------------------------------------------------------ turbo variants (JustAdev742, Apache-2.0)
TURBO_DIR = FORK / "turbo"
REAP_FILES = {"script": "/kaggle/arc3-reap-patch.py", "kept": "/kaggle/arc3-reap-kept.json",
              "meta": "/kaggle/arc3-reap-kept.meta.json"}   # meta = sglang_reap_patch.meta_path(kept)
REAP_ENV = "ARC3_REAP_KEPT_EXPERTS"
REAP_TOTAL = 512
REAP_KEPT = 448
REAP_KEPT_SHA256 = "e7e6a28b27b16032851313fe9edc611934529e11a8eb4b6ec1538537da8812fa"   # their kept list, verbatim
HOT_FILE = "/kaggle/arc3-hot-tokens.pt"
HOT_PT_SHA256 = "9c77419ad5757ce3e01229af834900263eef3f658dc26862e55c89cdd3c70ab9"     # their torch.save'd map
BASE_TOKEN_MAP_SHA = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"  # Pennyroyal's generic map

# launcher-cell (upstream cell 12) anchors, each must occur exactly once
L_CFG = "CFG = dict(\n"
L_NVCC = 'run([str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], env=env)\n'
L_PREFETCH = 'if CFG["PREFETCH_CHECKPOINTS"]: args += ["--weight-loader-prefetch-checkpoints"]\n'
L_LAUNCH = "# ---- launch detached and wait for health ----\n"
L_TOKMAP_ARG = '        args += ["--speculative-token-map", str(tok)]\n'
L_TOKMAP_FIND = ('        tok = find_unique(WHEELHOUSE_DIR, "hot_tokens_64k.pt", required=False)\n'
                 '        if tok is None: tok = find_unique(WHEELHOUSE_DIR, "flash-next-64k.pt")\n')
L_TOKMAP_SHA = f'TOKEN_MAP_SHA = "{BASE_TOKEN_MAP_SHA}"\n'
L_END = ('    print(f"\\nDEADLINE at {int(time.time()-NOTEBOOK_START_TIME)}s from notebook start; "\n'
         '          f"releasing the benchmark with the server still loading")')
SETUP_STREAMS = "    'ARC3_MAX_ACTIVE_STREAMS': 10,\n"
DEMO_LINE = ("demo_excluded_games = [] if TRUE_SUBMISSION else ['bp35', 'cd82', 'cn04', 'dc22', 'g50t', 'ka59', "
             "'lf52', 'ls20', 'm0r0', 's5i5', 'sk48', 'sp80', 'su15', 'tn36', 'wa30']\n")
TB = "# [calamitychasm turbo]"


def _src(name: str) -> str:
    text = (TURBO_DIR / name).read_text(encoding="utf-8")
    assert "'''" not in text and not text.endswith("\\"), f"{name} cannot be inlined in r'''...'''"
    return text


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(True)}


TF_DUMP = (
    "try:   # [calamitychasm] sandbox-timeout fix counters\n"
    "    (WORKING_DIR / 'timeout_fix_summary.json').write_text(json.dumps(TIMEOUT_FIX.summary(), indent=2))\n"
    "    print('TIMEOUT_FIX summary', json.dumps(TIMEOUT_FIX.summary()), flush=True)\n"
    "except Exception as _exc:\n"
    "    print('TIMEOUT_FIX summary failed', repr(_exc), flush=True)\n"
)


def timeout_fix_cell() -> dict:
    src = (
        "# [calamitychasm] SANDBOX-TIMEOUT FIX (variant timeoutfix) -- see experiments/stage7_m2_speed.md (turbo).\n"
        "# A python call that times out no longer wipes every retained function; `time` joins SAFE_MODULES.\n"
        "# Port of JustAdev742's ours-sandbox-timeout-keeps-work.patch (Apache-2.0), installed by wrapping\n"
        "# ToolAgent._record_retained_functions; the harness patch above stays verbatim. Raises on a missing anchor.\n"
        "import sys as _sys, types as _types\n"
        f"_TF_SRC = r'''{_src('timeout_fix.py')}'''\n"
        "TIMEOUT_FIX = _types.ModuleType('timeout_fix')\n"
        "_sys.modules['timeout_fix'] = TIMEOUT_FIX\n"
        "exec(compile(_TF_SRC, 'timeout_fix.py', 'exec'), TIMEOUT_FIX.__dict__)\n"
        "assert TIMEOUT_FIX.install()\n"
        "import inference.agent.tool_agent as _tf_ta\n"
        "import inference.agent.python_tool_sandbox as _tf_sb\n"
        "assert _tf_ta.ToolAgent._record_retained_functions.__module__ == 'timeout_fix'\n"
        "assert '\"time\",' in _tf_sb._SANDBOX_BOOTSTRAP\n"
        "print('TIMEOUT_FIX installed', json.dumps(TIMEOUT_FIX.summary()), flush=True)\n"
    )
    return _code_cell(src)


def reap_files_cell() -> dict:
    """Writes the REAP patch script, the kept list and its meta before the launcher reads them. The kept list
    travels as the 64 pruned ids per layer and is rebuilt byte-identical to JustAdev742's file (sha256 checked)."""
    script = _src("sglang_reap_patch.py")
    meta = _src("reap448_kept_experts.meta.json")
    kept_raw = (TURBO_DIR / "reap448_kept_experts.json").read_bytes()
    assert _sha(kept_raw) == REAP_KEPT_SHA256, "vendored kept list changed"
    kept = json.loads(kept_raw)
    pruned = {k: [e for e in range(REAP_TOTAL) if e not in set(v)] for k, v in kept.items()}
    assert all(len(v) == REAP_KEPT for v in kept.values()) and len(kept) == 48
    rebuilt = (json.dumps({k: [e for e in range(REAP_TOTAL) if e not in set(p)] for k, p in pruned.items()},
                          separators=(",", ":")) + "\n").encode("utf-8")
    assert rebuilt == kept_raw, "kept list does not round-trip through the pruned encoding"
    layers = json.loads(meta)["layers"]
    assert sorted(int(k) for k in layers) == sorted(int(k) for k in kept)
    assert {v["num_experts"] for v in layers.values()} == {REAP_TOTAL}
    src = (
        "# [calamitychasm] REAP-448 FILES (variant reap) -- see experiments/stage7_m2_speed.md (turbo).\n"
        "# JustAdev742's sglang_reap_patch.py (Apache-2.0, verbatim), the kept-expert list (rebuilt here from the\n"
        "# 64 pruned ids per layer; byte-identical to their reap448_kept_experts.json) and its router-sha256 meta.\n"
        "import hashlib as _rp_h, json as _rp_json\n"
        "from pathlib import Path as _rp_Path\n"
        "def _rp_write(path, data, sha):\n"
        "    if _rp_h.sha256(data).hexdigest() != sha:\n"
        "        raise RuntimeError(f'{path}: corrupted in the notebook')\n"
        "    _rp_Path(path).write_bytes(data)\n"
        f"_RP_SCRIPT = r'''{script}'''\n"
        f"_rp_write({REAP_FILES['script']!r}, _RP_SCRIPT.encode('utf-8'), {_sha(script.encode('utf-8'))!r})\n"
        f"_RP_PRUNED = _rp_json.loads({json.dumps(pruned, separators=(',', ':'))!r})\n"
        f"_rp_kept = {{k: [e for e in range({REAP_TOTAL}) if e not in set(p)] for k, p in _RP_PRUNED.items()}}\n"
        f"_rp_write({REAP_FILES['kept']!r}, (_rp_json.dumps(_rp_kept, separators=(',', ':')) + '\\n').encode('utf-8'),\n"
        f"          {REAP_KEPT_SHA256!r})\n"
        f"_RP_META = r'''{meta}'''\n"
        f"_rp_write({REAP_FILES['meta']!r}, _RP_META.encode('utf-8'), {_sha(meta.encode('utf-8'))!r})\n"
        f"print('REAP448 files written: kept {REAP_KEPT} of {REAP_TOTAL} routed experts in each of '\n"
        f"      f'{{len(_rp_kept)}} layers', flush=True)\n"
    )
    return _code_cell(src)


# The FR-Spec map is shipped as runs of consecutive ids (varint pairs, base64; no zlib, so the payload is the
# same on every build machine) and written as torch.save's zip layout with fixed records, names, times and
# create_system, so the file's sha256 is known at build time and replaces TOKEN_MAP_SHA (the launcher's own
# assert then checks it). The record layout is JustAdev742's HOT_WRITER (scripts/build_franzen_nb.py), which
# Pennyroyal loaded in their runs (exp-077 onward).
HOT_WRITER = '''
def _tb_write_hot_tokens(runs_b64, ids_sha, path):
    import base64, hashlib, pickle, zipfile
    raw = base64.b64decode(runs_b64)
    nums, i = [], 0
    while i < len(raw):
        value, shift = 0, 0
        while True:
            byte = raw[i]
            i += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        nums.append(value)
    ids, prev = [], -1
    for gap, extra in zip(nums[0::2], nums[1::2]):
        start = prev + 1 + gap
        ids.extend(range(start, start + extra + 1))
        prev = ids[-1]
    if hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest() != ids_sha:
        raise RuntimeError("FR-Spec map corrupted in the notebook")
    records = [("data.pkl", pickle.dumps(ids, protocol=2)), (".format_version", b"1"), (".storage_alignment", b"64"),
               ("byteorder", b"little"), ("version", b"3\\n"), (".data/serialization_id", b"0" * 40)]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in records:
            info = zipfile.ZipInfo("arc3_hot_tokens/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            z.writestr(info, data)
    return len(ids)
'''


def _read_hot_ids(path: Path) -> list[int]:
    """The id list in a torch.save'd FR-Spec map (a zip with one pickled list), read without torch or globals."""
    import pickle
    import zipfile

    class _NoGlobals(pickle.Unpickler):
        def find_class(self, module, name):
            raise SystemExit(f"{path}: refusing pickled global {module}.{name}")

    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith("/data.pkl")]
        assert len(names) == 1, names
        ids = _NoGlobals(z.open(names[0])).load()
    assert isinstance(ids, list) and ids == sorted(set(ids)) and all(isinstance(i, int) and i >= 0 for i in ids)
    return ids


@functools.lru_cache(maxsize=None)
def hot_map() -> dict:
    """{'ids', 'payload', 'ids_sha', 'sha'}: sha is the sha256 of the file the notebook will write."""
    import base64
    import tempfile

    raw_pt = (TURBO_DIR / "hot_tokens_64k_arc.pt").read_bytes()
    assert _sha(raw_pt) == HOT_PT_SHA256, "vendored FR-Spec map changed"
    ids = _read_hot_ids(TURBO_DIR / "hot_tokens_64k_arc.pt")
    runs, start, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i != prev + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    out, last = bytearray(), -1
    for a, b in runs:
        for value in (a - last - 1, b - a):
            while True:
                byte, value = value & 0x7F, value >> 7
                out.append(byte | (0x80 if value else 0))
                if not value:
                    break
        last = b
    payload = base64.b64encode(bytes(out)).decode()
    ids_sha = _sha(",".join(map(str, ids)).encode())
    ns: dict = {}
    exec(HOT_WRITER, ns)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "hot.pt"
        assert ns["_tb_write_hot_tokens"](payload, ids_sha, str(f)) == len(ids)
        assert _read_hot_ids(f) == ids, "notebook writer does not round-trip the ids"
        sha = _sha(f.read_bytes())
    return {"ids": len(ids), "payload": payload, "ids_sha": ids_sha, "sha": sha}


def hotmap_cell() -> dict:
    h = hot_map()
    lines = "\n".join(h["payload"][i:i + 120] for i in range(0, len(h["payload"]), 120))
    src = (
        "# [calamitychasm] ARC FR-SPEC MAP (variant hotmap) -- see experiments/stage7_m2_speed.md (turbo).\n"
        "# JustAdev742's hot_tokens_64k_arc.pt (Apache-2.0; same 65,536-id size as Pennyroyal's generic map, covers\n"
        "# 99.9% of this harness's output tokens vs 98.5-98.8%). The launcher's own sha256 assert checks this file."
        + HOT_WRITER
        + f"_tb_n = _tb_write_hot_tokens(\"\"\"\n{lines}\n\"\"\", {h['ids_sha']!r}, {HOT_FILE!r})\n"
        f"print(f'ARC FR-Spec map written to {HOT_FILE}: {{_tb_n}} ids', flush=True)\n"
    )
    return _code_cell(src)


def _launcher_edits(cell: str, names) -> str:
    """The SGLang launcher cell (upstream cell 12) with the serving variants applied."""
    acc, n = accept_value(names), streams_value(names)
    if n is not None:
        for key, old, new in (("MAXREQ", 10, n), ("CUDAGRAPH_MAXBS", 10, n), ("MAMBA_CACHE", 60, mamba_cache(n))):
            cell = sub(cell, f"    {key}={old},\n", f"    {key}={new},   {TB} streams\n", key)
    if acc is not None:
        for key in ("SPEC_ACCEPT_SINGLE", "SPEC_ACCEPT_ACC"):
            cell = sub(cell, f"    {key}=1.0,\n", f"    {key}={acc!r},   {TB} lossy MTP acceptance\n", key)
    if "hotmap" in names:
        cell = sub(cell, L_TOKMAP_SHA, f'TOKEN_MAP_SHA = "{hot_map()["sha"]}"   {TB} ARC FR-Spec map\n',
                   "TOKEN_MAP_SHA")
        cell = sub(cell, L_TOKMAP_FIND, f"        tok = Path({HOT_FILE!r})   {TB} ARC FR-Spec map\n", "map find")
        cell = sub(cell, L_TOKMAP_ARG, L_TOKMAP_ARG
                   + f'        print(f"ARC_HOTMAP sha={{sha256(tok)}} file={{tok}}", flush=True)   {TB}\n',
                   "map marker")
    if "reap" in names:
        apply = (
            f"{TB} REAP-448 at load (JustAdev742's sglang_reap_patch.py, Apache-2.0). Patch the installed sglang\n"
            "# (raises -- the cell stops before the server starts -- if qwen4_exp.py is not the analysed file or an\n"
            "# anchor is missing), then tell the server the kept list.\n"
            '_reap_model = sorted(Path(VENV).glob("lib/python*/site-packages/sglang/srt/models/qwen4_exp.py"))\n'
            "if len(_reap_model) != 1:\n"
            '    raise RuntimeError(f"REAP448: expected one installed sglang qwen4_exp.py, found {_reap_model}")\n'
            f'run([sys.executable, "-I", "{REAP_FILES["script"]}", "apply", "--site-packages", '
            'str(_reap_model[0].parents[3]),\n'
            f'     "--kept", "{REAP_FILES["kept"]}"])\n'
            f'env["{REAP_ENV}"] = "{REAP_FILES["kept"]}"\n'
            f'print("REAP448 patch installed; {REAP_ENV} set for the server", flush=True)\n')
        cell = sub(cell, L_NVCC, apply + L_NVCC, "reap apply")
        override = json.dumps({"text_config": {"num_experts": REAP_KEPT}})
        args = (
            f"{TB} REAP-448: the target is built with {REAP_KEPT} routed experts per layer; the MTP draft keeps its\n"
            "# own config (unset, it would inherit --json-model-override-args and be built 448 wide).\n"
            f"args += [\"--json-model-override-args\", {override!r},\n"
            "         \"--speculative-draft-model-override-args\", \"{}\"]\n")
        cell = sub(cell, L_PREFETCH, args + L_PREFETCH, "reap args")
    if acc is not None or n is not None:
        marks = f"{TB} serving markers (the submit gate checks them)\n"
        if acc is not None:
            marks += ('assert args[args.index("--speculative-accept-threshold-single") + 1] == str(CFG["SPEC_ACCEPT_SINGLE"])\n'
                      'assert args[args.index("--speculative-accept-threshold-acc") + 1] == str(CFG["SPEC_ACCEPT_ACC"])\n'
                      'print(f"SPEC_ACCEPT {CFG[\'SPEC_ACCEPT_SINGLE\']} acc={CFG[\'SPEC_ACCEPT_ACC\']}", flush=True)\n')
        if n is not None:
            marks += ('print(f"STREAMS max_running_requests={CFG[\'MAXREQ\']} cuda_graph_bs={graph_bs} '
                      'mamba_cache={CFG[\'MAMBA_CACHE\']}", flush=True)\n')
        cell = sub(cell, L_LAUNCH, marks + L_LAUNCH, "markers")
    if "reap" in names:
        assert cell.endswith(L_END), "launcher cell end changed"
        cell += (
            f"\n\n{TB} REAP-448: confirm from serve.log that the server loaded the pruned target (a READY server has\n"
            "# finished loading). Never raises here: a rerun must not stop after the server started.\n"
            "_rp_line = next((ln for ln in Path(LOG).read_text(errors='replace').splitlines()\n"
            f"                 if 'ARC3 REAP: kept {REAP_KEPT} of' in ln), None)\n"
            "if _rp_line:\n"
            f"    print('REAP448 applied kept={REAP_KEPT} |', _rp_line.split('ARC3 REAP: ', 1)[1][:240], flush=True)\n"
            "else:\n"
            "    print('REAP448 NOT CONFIRMED: no \"ARC3 REAP: kept\" line in serve.log (server not ready, or died)',\n"
            "          flush=True)")
    return cell


def serving_edits(nb, names) -> None:
    """Apply the serving variants to the upstream cells in place (before any cell is inserted)."""
    names = variant_names(names)
    if not ({_kind(v) for v in names} & {"reap", "acc", "hotmap", "streams", "all25"}):
        return
    src = cells_of(nb)

    def one(needle):
        hits = [i for i, s in enumerate(src) if needle in s]
        assert len(hits) == 1, (needle, hits)
        return hits[0]

    i = one(L_CFG)
    set_src(nb, i, _launcher_edits(src[i], names))
    n = streams_value(names)
    if n is not None:
        i = one(SETUP_STREAMS)
        set_src(nb, i, sub(src[i], SETUP_STREAMS, f"    'ARC3_MAX_ACTIVE_STREAMS': {n},   {TB} streams\n", "gate"))
    if "all25" in names:
        i = one(DEMO_LINE)
        set_src(nb, i, sub(src[i], DEMO_LINE, f"demo_excluded_games = []   {TB} check run: all 25 public games "
                                              "(the competition rerun used [] already)\n", "demo games"))


# cells inserted right before the launcher cell, in this order
VARIANT_PRE_LAUNCH_CELLS = {"reap": reap_files_cell, "hotmap": hotmap_cell}
VARIANT_CELLS["timeoutfix"] = timeout_fix_cell
VARIANT_RUN_DUMP["timeoutfix"] = TF_DUMP


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


def _blurb(v: str, names) -> str:
    k = _kind(v)
    if k == "acc":
        return (f"MTP speculative acceptance thresholds single = acc = {accept_value(names)} (incumbent 1.0, "
                "lossless): a draft token is also accepted when the target is confident enough, as in "
                "JustAdev742's measured config (Apache-2.0). Lossy by design.")
    if k == "streams":
        n = streams_value(names)
        return (f"{n} concurrent streams: SGLang max running requests and CUDA-graph max batch {n}, Mamba cache "
                f"{mamba_cache(n)}, harness priority gate {n} (incumbent 10).")
    return VARIANT_BLURB[k]


TURBO_NOTE = ("\nServing and harness changes marked [calamitychasm turbo] are ported from JustAdev742's Milestone-2 "
              "work (github.com/JustAdev742/Arc-Agi-3-Kaggle-comp, Apache-2.0), measured by them on this exact "
              "stack (same Pennyroyal v253 wheel, same Intel W4A16 checkpoint and MTP draft): REAP-448 + 14 "
              "streams +14% output tok/s, + acceptance 0.5 +28%, + ARC FR-Spec map ~+2%. Provenance: "
              "kaggle_submission_milestone2_fork/turbo/NOTICE.md in our repo.\n")

TURBO_NOTE_LOSSLESS = ("\nServing and harness changes marked [calamitychasm turbo] are ported from JustAdev742's "
                       "Milestone-2 work (github.com/JustAdev742/Arc-Agi-3-Kaggle-comp, Apache-2.0), measured by them "
                       "on this exact stack (same Pennyroyal v253 wheel, same Intel W4A16 checkpoint and MTP draft): "
                       "REAP-448 + 14 streams +14% output tok/s, ARC FR-Spec map ~+2%. MTP acceptance stays the "
                       "incumbent's lossless 1.0 (their +28% step to 0.5 is not applied). Provenance: "
                       "kaggle_submission_milestone2_fork/turbo/NOTICE.md in our repo.\n")


def build(variants=()) -> Path:
    variants = variant_names(variants)
    check_serving(variants)
    nb = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    serving_edits(nb, variants)      # upstream cells only, before any insertion
    if resolves_inputs(variants):
        apply_input_resolver(nb)
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
    pre = [VARIANT_PRE_LAUNCH_CELLS[v]() for v in variants if v in VARIANT_PRE_LAUNCH_CELLS]
    if pre:                           # the launcher is before the run cell: indices above r are unaffected
        launch = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and L_CFG in "".join(c["source"])]
        assert len(launch) == 1, launch
        nb["cells"][launch[0]:launch[0]] = pre
    md = ("## [calamitychasm] fork: milestone-2 solution + solved-level memory\n\n"
          "Verbatim fork of `dfranzen/arc-agi-3-milestone-2-solution` (Apache-2.0; full credit to Daniel Franzen, "
          "Jeroen Cottaar and Tufa Labs). One addition: solved-level memory, ported from sirikilohit's "
          "Milestone-2 patch M85 and adapted to this harness's prefix cache. Installed by the cell before the run.\n"
          + "".join(f"\nVariant `{v}`: {_blurb(v, variants)}\n" for v in variants)
          + (("" if not ({_kind(v) for v in variants} - {"histcache", "triedfacts", "inputs"}) else
             TURBO_NOTE if accept_value(variants) is not None else TURBO_NOTE_LOSSLESS)))
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": md.splitlines(True)})
    slug = kernel_slug(variants)
    d = kernel_dir(variants)
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
    ap.add_argument("--timeout-fix", action="store_true",
                    help="a timed-out python call keeps the retained functions; `time` importable (JustAdev742)")
    ap.add_argument("--reap", action="store_true", help="REAP-448 expert pruning at load (JustAdev742)")
    ap.add_argument("--spec-accept", type=float, default=None, metavar="X",
                    help="MTP acceptance thresholds single = acc = X, e.g. 0.5 (incumbent 1.0)")
    ap.add_argument("--arc-hotmap", action="store_true", help="ARC-tuned FR-Spec map for the MTP draft (JustAdev742)")
    ap.add_argument("--streams", type=int, default=None, metavar="N",
                    help=f"N concurrent streams (11..{MAX_STREAMS}; > {MAX_STREAMS_WITHOUT_REAP} needs --reap)")
    ap.add_argument("--check-all25", action="store_true",
                    help="the non-submission check run plays all 25 public games (speed-kernel shape)")
    ap.add_argument("--input-resolver", action="store_true",
                    help="the incumbent plus only the mount-layout input resolver (kernel arc3-m2-lm-inputs); "
                         "every other variant already includes the resolver")
    ap.add_argument("--turbo", action="store_true",
                    help="all of: " + " ".join(TURBO) + f" -> calamitychasm/{TURBO_SLUG}")
    ap.add_argument("--turbo-lossless", action="store_true",
                    help="turbo minus --spec-accept: " + " ".join(TURBO_LOSSLESS) + f" -> calamitychasm/{TURBO_LOSSLESS_SLUG}")
    args = ap.parse_args(argv)
    if args.turbo and args.turbo_lossless:
        raise SystemExit("REFUSING TO BUILD -- --turbo and --turbo-lossless are different presets")
    variants = (["histcache"] * args.history_cache + ["triedfacts"] * args.tried_facts
                + ["timeoutfix"] * args.timeout_fix + ["reap"] * args.reap + ["hotmap"] * args.arc_hotmap
                + ["all25"] * args.check_all25 + ["inputs"] * args.input_resolver)
    if args.spec_accept is not None and args.spec_accept != 1.0:
        variants.append(accept_token(args.spec_accept))
    if args.streams is not None and args.streams != 10:
        variants.append(f"s{args.streams}")
    if args.turbo_lossless:
        clash = [v for v in variants if _kind(v) in ("acc", "streams") and v not in TURBO_LOSSLESS]
        if clash:
            raise SystemExit(f"REFUSING TO BUILD -- --turbo-lossless fixes {TURBO_LOSSLESS} (acceptance stays 1.0); "
                             f"conflicting {clash}")
        variants += [v for v in TURBO_LOSSLESS if v not in variants]
    if args.turbo:
        clash = [v for v in variants if _kind(v) in ("acc", "streams") and v not in TURBO]
        if clash:
            raise SystemExit(f"REFUSING TO BUILD -- --turbo fixes {TURBO}; conflicting {clash}")
        variants += [v for v in TURBO if v not in variants]
    path = build(variants)
    problems = check_notebook(path)
    if problems:
        print("REFUSING TO SHIP -- use-before-definition problems")
        for p in problems:
            print(" ", p)
        return 1
    print("built", path)
    names = variant_names(variants)
    if names:
        print("markers:", "; ".join(kernel_markers(names)))
        print("counters:", " ".join(kernel_counters(names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
