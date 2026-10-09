"""Turbo kernel (arc3-m2-turbo): JustAdev742's serving changes + the sandbox-timeout fix on the incumbent.

Covers the builder (determinism, committed kernel current, existing kernels untouched by the new flags, only
the expected cells differ from the histcache kernel, flag/marker/env/CFG values, refusals), the REAP patch
against the real Pennyroyal d00d88e qwen4_exp.py (vendored gzipped in tests/fixtures; sha256 = the analysed
file), the notebook's own REAP apply block and file-writer cells executed against a fake install, the FR-Spec
map writer, and the timeout fix on the REAL harness (tests/m2_harness.py; skipped when unavailable).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TURBO_DIR = ROOT / "kaggle_submission_milestone2_fork" / "turbo"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TURBO_DIR))

import _build_m2_level_memory_kernel as B  # noqa: E402
import m2_harness as h  # noqa: E402
import sglang_reap_patch as reap  # noqa: E402
import timeout_fix as tf  # noqa: E402
from _check_notebook_cell import check_notebook  # noqa: E402

COMMITTED = ROOT / "kaggle_submission_m2_turbo" / "notebook"
QWEN4 = Path(__file__).resolve().parent / "fixtures" / "pennyroyal_d00d88e_qwen4_exp.py.gz"


def _cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def _one(cells, needle):
    hits = [c for c in cells if needle in c]
    assert len(hits) == 1, (needle, len(hits))
    return hits[0]


@pytest.fixture(scope="module")
def turbo_cells():
    return _cells(COMMITTED / "arc3-m2-turbo.ipynb")


# ---------------------------------------------------------------------------------------------- builder

def test_turbo_preset_slug_and_dir():
    assert B.kernel_slug(B.TURBO) == "arc3-m2-turbo"
    assert B.kernel_id(reversed(B.TURBO)) == "calamitychasm/arc3-m2-turbo"
    assert B.kernel_dir(B.TURBO).relative_to(ROOT).as_posix() == "kaggle_submission_m2_turbo/notebook"
    assert B.kernel_slug(("s14", "reap")) == "arc3-m2-lm-reap-s14"
    assert B.kernel_slug(("histcache", "triedfacts")) == "arc3-m2-lm-histcache-triedfacts"   # unchanged
    assert B.accept_value(B.TURBO) == 0.5 and B.streams_value(B.TURBO) == 14


def test_build_is_deterministic_and_committed_kernel_is_current(tmp_path, monkeypatch):
    committed = (COMMITTED / "arc3-m2-turbo.ipynb").read_text(encoding="utf-8")
    outs = []
    for i in range(2):
        monkeypatch.setattr(B, "ROOT", tmp_path / str(i))
        outs.append(B.build(B.TURBO).read_text(encoding="utf-8"))
    monkeypatch.undo()
    assert outs[0] == outs[1], "builder is not deterministic"
    assert outs[0] == committed, "committed turbo kernel is stale: rerun the build"
    assert check_notebook(COMMITTED / "arc3-m2-turbo.ipynb") == []
    assert len(committed.encode("utf-8")) < 900_000, "Kaggle refuses notebooks near 1 MB"


def test_existing_kernels_unchanged_by_the_new_variant_system(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    for variants in [("histcache",), ("triedfacts",), ("histcache", "triedfacts")]:
        new = B.build(variants)
        rel = new.relative_to(tmp_path)
        assert new.read_text(encoding="utf-8") == (ROOT / rel).read_text(encoding="utf-8"), rel
        assert (new.parent / "kernel-metadata.json").read_text() == (ROOT / rel.parent / "kernel-metadata.json").read_text()


def test_metadata_matches_incumbent_except_identity():
    inc = json.loads((ROOT / "kaggle_submission_m2_level_memory" / "notebook" / "kernel-metadata.json").read_text())
    new = json.loads((COMMITTED / "kernel-metadata.json").read_text())
    assert new["id"] == "calamitychasm/arc3-m2-turbo" and new["code_file"] == "arc3-m2-turbo.ipynb"
    for k in ("id", "title", "code_file"):
        inc.pop(k), new.pop(k)
    assert new.pop("docker_image_pinning_type") == "original"   # variants pin the incumbent's image
    assert inc == new          # same GPU, image, datasets, models; no internet


def test_only_the_expected_cells_differ_from_the_histcache_kernel(turbo_cells):
    hc = _cells(ROOT / "kaggle_submission_m2_lm_histcache" / "notebook" / "arc3-m2-lm-histcache.ipynb")
    added = [c for c in turbo_cells if c not in hc]
    # header, setup (streams), REAP files, FR-Spec writer, launcher, customization (all25), timeout fix, run dump
    assert len(added) == 8, [a[:60] for a in added]
    assert len(turbo_cells) == len(hc) + 3
    starts = ["## [calamitychasm] fork", "import json\nimport os", "# [calamitychasm] REAP-448 FILES",
              "# [calamitychasm] ARC FR-SPEC MAP", "# Paste this entire file", "# Make one-off changes",
              "# [calamitychasm] SANDBOX-TIMEOUT FIX", "print('Starting benchmark...')"]
    assert [next(s for s in starts if a.startswith(s)) for a in added] == starts
    i_files, i_hot, i_launch = (turbo_cells.index(added[k]) for k in (2, 3, 4))
    assert i_files + 1 == i_hot and i_hot + 1 == i_launch            # files are written right before the launcher
    i_tf, i_run = turbo_cells.index(added[6]), turbo_cells.index(added[7])
    i_hc = turbo_cells.index(_one(turbo_cells, "# [calamitychasm] HISTORY CACHE"))
    assert i_hc + 1 == i_tf and i_tf + 1 == i_run


def test_serving_values_markers_and_env(turbo_cells):
    launch = _one(turbo_cells, "CFG = dict(\n")
    for line in ("    MAXREQ=14,", "    CUDAGRAPH_MAXBS=14,", "    MAMBA_CACHE=84,", "    SPEC_ACCEPT_SINGLE=0.5,",
                 "    SPEC_ACCEPT_ACC=0.5,", "    MEMFRAC=0.96,", "    SPEC_STEPS=3,"):
        assert line in launch, line
    assert 84 // 5 >= 14
    assert f'TOKEN_MAP_SHA = "{B.hot_map()["sha"]}"' in launch and B.BASE_TOKEN_MAP_SHA not in launch
    assert "tok = Path('/kaggle/arc3-hot-tokens.pt')" in launch and "hot_tokens_64k.pt" not in launch
    assert 'env["ARC3_REAP_KEPT_EXPERTS"] = "/kaggle/arc3-reap-kept.json"' in launch
    assert '\'{"text_config": {"num_experts": 448}}\'' in launch
    assert '"--speculative-draft-model-override-args", "{}"' in launch
    # order: REAP apply after env.update and before nvcc/launch; override args before the launch
    assert launch.index("env.update({") < launch.index("_reap_model = ") < launch.index(B.L_NVCC)
    assert launch.index("--json-model-override-args") < launch.index(B.L_LAUNCH)
    assert launch.index("REAP448 applied kept=448") > launch.index("READY after")
    setup = _one(turbo_cells, "setup_env = {")
    assert "'ARC3_MAX_ACTIVE_STREAMS': 14," in setup
    assert "demo_excluded_games = []   #" in _one(turbo_cells, "bm.solver.max_runtime_s_per_game")
    run = _one(turbo_cells, "print('Starting benchmark...')")
    for f in ("level_memory_summary.json", "history_cache_summary.json", "timeout_fix_summary.json"):
        assert f in run
    markers = B.kernel_markers(B.TURBO)
    assert markers[:3] == ["LEVEL_MEMORY installed", "priority gate active", "harness patch applied successfully"]
    for m in ("HISTORY_CACHE installed", "TIMEOUT_FIX installed", "REAP448 applied kept=448", "SPEC_ACCEPT 0.5",
              "priority gate active: 14 concurrent streams"):
        assert m in markers
    assert f"ARC_HOTMAP sha={B.hot_map()['sha']}" in markers
    printed = "\n".join(turbo_cells)
    for m in ("TIMEOUT_FIX installed", "REAP448 applied kept=", "SPEC_ACCEPT {CFG", "ARC_HOTMAP sha={sha256(tok)}"):
        assert m in printed, m
    assert B.kernel_counters(B.TURBO) == ["level_memory_summary.json", "history_cache_summary.json",
                                          "timeout_fix_summary.json"]


def test_markers_print_what_the_gate_expects():
    """The marker strings printed by the launcher lines, evaluated, contain the gate's markers."""
    CFG = {"SPEC_ACCEPT_SINGLE": 0.5, "SPEC_ACCEPT_ACC": 0.5}
    assert "SPEC_ACCEPT 0.5" in f"SPEC_ACCEPT {CFG['SPEC_ACCEPT_SINGLE']} acc={CFG['SPEC_ACCEPT_ACC']}"
    line = ("ARC3 REAP: kept 448 of 512 routed experts in each of 48 layers (193536 expert tensors loaded, 27648 "
            "pruned tensors skipped); routers sliced to 448 rows, router sha256 verified; list x")
    assert "REAP448 applied kept=448" in " ".join(["REAP448 applied kept=448 |", line.split("ARC3 REAP: ", 1)[1]])


REAP_KERNELS = [("kaggle_submission_m2_turbo", "arc3-m2-turbo"),
                ("kaggle_submission_m2_turbo_lossless", "arc3-m2-turbo-lossless"),
                ("kaggle_submission_m2_turbo_tail", "arc3-m2-turbo-tail"),
                ("kaggle_submission_m2_turbo_lossless_tail", "arc3-m2-turbo-lossless-tail")]
REAP_LOG_LINE = ("[2026-10-09 01:02:03] ARC3 REAP: kept 448 of 512 routed experts in each of 48 layers (193536 expert "
                 "tensors loaded, 27648 pruned tensors skipped); routers sliced to 448 rows, router sha256 verified")


def _post_run_reap(run_cell: str) -> str:
    start = run_cell.index("try:   # [calamitychasm turbo] REAP-448 confirmation")
    end = run_cell.index("    print('REAP448 post-run check failed', repr(_exc), flush=True)\n", start)
    return run_cell[start:end] + "    print('REAP448 post-run check failed', repr(_exc), flush=True)\n"


@pytest.mark.parametrize("dirname,slug", REAP_KERNELS)
def test_reap_marker_is_rechecked_after_the_run(dirname, slug, tmp_path, capsys):
    """Regression: the launcher's REAP check runs right after its health loop; a server still loading at the
    12-min startup deadline (normal boot ~9 min) used to leave a healthy run without the gate's REAP marker.
    The run cell re-reads serve.log after bm.run and prints the marker the gate expects."""
    cells = _cells(ROOT / dirname / "notebook" / f"{slug}.ipynb")
    run = _one(cells, "print('Starting benchmark...')")
    assert run.index(B.RUN_ANCHOR) < run.index("REAP-448 confirmation from serve.log")
    code = _post_run_reap(run)
    assert code == B.REAP_DUMP
    log = tmp_path / "serve.log"
    marker = next(m for m in B.kernel_markers([v for v in B.variant_names(B.TURBO)]) if m.startswith("REAP448"))
    for content, expect in ((None, "REAP448 post-run check failed"),
                            ("loading...\n", "REAP448 NOT CONFIRMED after the run"),
                            ("loading...\n" + REAP_LOG_LINE + "\nready\n", marker)):
        if content is not None:
            log.write_text(content)
        exec(compile(code, "run_cell_reap", "exec"), {"Path": Path, "LOG": str(log)})   # never raises
        out = capsys.readouterr().out
        assert expect in out, (expect, out)
    assert "REAP448 applied kept=448 (post-run) | kept 448 of 512" in out


def test_variant_kernels_pin_the_docker_image_and_the_incumbent_is_untouched():
    """Kaggle's latest image moved to Python 3.13 (cp312-only wheelhouse): every variant pins the image."""
    inc = json.loads((ROOT / "kaggle_submission_m2_level_memory" / "notebook" / "kernel-metadata.json").read_text())
    assert "docker_image_pinning_type" not in inc and "@sha256:" in inc["docker_image"]
    dirs = sorted(p for p in ROOT.glob("kaggle_submission_m2_*/notebook/kernel-metadata.json")
                  if p.parent.parent.name.startswith(("kaggle_submission_m2_lm_", "kaggle_submission_m2_turbo")))
    assert len(dirs) >= 9
    for p in dirs:
        meta = json.loads(p.read_text())
        assert meta["docker_image_pinning_type"] == "original" and meta["docker_image"] == inc["docker_image"], p


def test_every_committed_m2_kernel_but_the_incumbent_pins_the_docker_image():
    """All committed kaggle_submission_m2_* kernels (lm, turbo, speed, spd, vllm) pin the incumbent's image; the
    incumbent itself must stay byte-for-byte as pushed (no docker_image_pinning_type)."""
    incumbent = ROOT / "kaggle_submission_m2_level_memory"
    inc = json.loads((incumbent / "notebook" / "kernel-metadata.json").read_text())
    assert "docker_image_pinning_type" not in inc and "@sha256:" in inc["docker_image"]
    paths = sorted(p for p in ROOT.glob("kaggle_submission_m2_*/**/kernel-metadata.json")
                   if incumbent not in p.parents)
    assert len(paths) >= 19, [str(p) for p in paths]
    for p in paths:
        meta = json.loads(p.read_text())
        assert meta["docker_image_pinning_type"] == "original" and meta["docker_image"] == inc["docker_image"], p


@pytest.mark.parametrize("argv,slug", [
    (["--timeout-fix"], "arc3-m2-lm-timeoutfix"),
    (["--reap", "--streams", "14"], "arc3-m2-lm-reap-s14"),
    (["--spec-accept", "0.5", "--arc-hotmap"], "arc3-m2-lm-acc50-hotmap"),
    (["--turbo"], "arc3-m2-turbo"),
    (["--history-cache", "--timeout-fix", "--reap", "--spec-accept", "0.5", "--arc-hotmap", "--streams", "14",
      "--check-all25"], "arc3-m2-turbo"),
])
def test_cli_subsets_build(tmp_path, monkeypatch, capsys, argv, slug):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    assert B.main(argv) == 0
    out = capsys.readouterr().out
    assert f"{slug}.ipynb" in out
    path = next(tmp_path.rglob(f"{slug}.ipynb"))
    assert check_notebook(path) == []
    cells = _cells(path)
    launch = _one(cells, "CFG = dict(\n")
    assert ("_reap_model" in launch) == ("--reap" in argv or "--turbo" in argv)
    assert ("SPEC_ACCEPT_SINGLE=0.5," in launch) == ("--spec-accept" in argv or "--turbo" in argv)


@pytest.mark.parametrize("argv,why", [
    (["--streams", "14"], "needs --reap"),
    (["--reap", "--streams", "16"], "allowed 11..14"),
    (["--spec-accept", "1.5"], "0.10..0.99"),
    (["--turbo", "--streams", "12"], "conflicting"),
])
def test_refusals(tmp_path, monkeypatch, argv, why):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as exc:
        B.main(argv)
    assert why in str(exc.value)


def test_launcher_anchor_missing_fails_the_build(monkeypatch):
    monkeypatch.setattr(B, "L_NVCC", "this anchor does not exist\n")
    with pytest.raises(SystemExit, match="REFUSING TO BUILD"):
        B._launcher_edits(_one(h._cells(), "CFG = dict(\n"), ("reap",))


# ---------------------------------------------------------------- REAP patch vs real Pennyroyal source

def _fake_site(tmp_path: Path) -> Path:
    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    models = site / "sglang" / "srt" / "models"
    models.mkdir(parents=True)
    (models / "qwen4_exp.py").write_bytes(gzip.decompress(QWEN4.read_bytes()))
    (models / "__pycache__").mkdir()
    (models / "__pycache__" / "qwen4_exp.cpython-312.pyc").write_bytes(b"stale")
    return site


def _exec_files_cell(cell: str, tmp_path: Path) -> dict:
    ns: dict = {}
    exec(compile(cell.replace("/kaggle/", f"{tmp_path}/"), "reap_files", "exec"), ns)
    return ns


def test_vendored_pennyroyal_file_is_the_analysed_one():
    text = gzip.decompress(QWEN4.read_bytes())
    assert hashlib.sha256(text).hexdigest() == reap.MODEL_FILE_SHA256


def test_files_cell_writes_the_vendored_files_byte_identically(tmp_path, turbo_cells):
    _exec_files_cell(_one(turbo_cells, "# [calamitychasm] REAP-448 FILES"), tmp_path)
    assert (tmp_path / "arc3-reap-patch.py").read_bytes() == (TURBO_DIR / "sglang_reap_patch.py").read_bytes()
    assert (tmp_path / "arc3-reap-kept.json").read_bytes() == (TURBO_DIR / "reap448_kept_experts.json").read_bytes()
    assert ((tmp_path / "arc3-reap-kept.meta.json").read_bytes()
            == (TURBO_DIR / "reap448_kept_experts.meta.json").read_bytes())
    kept = reap.load_kept(tmp_path / "arc3-reap-kept.json")
    assert len(kept) == 48 and {len(v) for v in kept.values()} == {448}
    assert sorted(reap.load_router_sha256(tmp_path / "arc3-reap-kept.json")) == list(range(48))


def test_notebook_reap_apply_block_patches_the_real_file(tmp_path, turbo_cells, capsys):
    """The launcher's REAP block, executed verbatim with the launcher's own run(check=True), patches the real
    d00d88e qwen4_exp.py (both anchors), sets the server env, and is a no-op on a rerun."""
    _exec_files_cell(_one(turbo_cells, "# [calamitychasm] REAP-448 FILES"), tmp_path)
    site = _fake_site(tmp_path)
    launch = _one(turbo_cells, "CFG = dict(\n")
    block = launch[launch.index("# [calamitychasm turbo] REAP-448 at load"):launch.index(B.L_NVCC)]
    run_src = launch[launch.index("def run(cmd"):launch.index("def find_unique")]
    ns = {"Path": Path, "sys": sys, "subprocess": subprocess, "shlex": shlex, "VENV": str(tmp_path / "venv"),
          "env": {}}
    exec(run_src, ns)
    block = block.replace("/kaggle/", f"{tmp_path}/")
    for attempt in ("patched", "already patched"):
        exec(block, ns)
        out = capsys.readouterr().out
        assert f"qwen4_exp.py {attempt}" in out and "48 layers x 448 experts, router sha256 for every layer" in out
        assert "REAP448 patch installed" in out
    assert ns["env"]["ARC3_REAP_KEPT_EXPERTS"] == f"{tmp_path}/arc3-reap-kept.json"
    patched = (site / "sglang/srt/models/qwen4_exp.py").read_text()
    assert patched.count(reap.MARK) == 2 and "weights = _arc3_reap.wrap_weights(weights, self.config)" in patched
    compile(patched, "qwen4_exp.py", "exec")
    assert not list((site / "sglang/srt/models/__pycache__").glob("qwen4_exp.*.pyc"))
    assert (site / "sglang/srt/arc3_reap.py").read_text() == (TURBO_DIR / "sglang_reap_patch.py").read_text()


def test_reap_apply_fails_loudly_on_a_moved_anchor(tmp_path, turbo_cells):
    _exec_files_cell(_one(turbo_cells, "# [calamitychasm] REAP-448 FILES"), tmp_path)
    site = _fake_site(tmp_path)
    f = site / "sglang/srt/models/qwen4_exp.py"
    text = f.read_text().replace("            return None\n", "            return  None\n", 1)
    f.write_text(text)
    launch = _one(turbo_cells, "CFG = dict(\n")
    block = launch[launch.index("# [calamitychasm turbo] REAP-448 at load"):launch.index(B.L_NVCC)]
    ns = {"Path": Path, "sys": sys, "subprocess": subprocess, "shlex": shlex, "VENV": str(tmp_path / "venv"),
          "env": {}}
    exec(launch[launch.index("def run(cmd"):launch.index("def find_unique")], ns)
    with pytest.raises(RuntimeError, match="Command failed"):
        exec(block.replace("/kaggle/", f"{tmp_path}/"), ns)
    assert "ARC3_REAP_KEPT_EXPERTS" not in ns["env"]
    # anchor check without the hash check too (as with --allow-other-version)
    with pytest.raises(reap.PatchError, match="anchor"):
        reap.patch_text(text, check_hash=False)


def test_wrap_weights_renumbers_experts_and_slices_routers(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    kept = {0: [0, 2, 3], 1: [1, 2, 3]}
    path = tmp_path / "kept.json"
    path.write_text(json.dumps({str(k): v for k, v in kept.items()}))
    monkeypatch.setenv(reap.ENV, str(path))

    class Cfg:
        num_experts, num_hidden_layers = 3, 2

    stream = []
    for layer in (0, 1):
        stream.append((f"model.language_model.layers.{layer}.mlp.gate.weight", np.arange(8).reshape(4, 2) + layer))
        for e in range(4):
            for t in ("gate_proj.qweight", "down_proj.scales"):
                stream.append((f"model.language_model.layers.{layer}.mlp.experts.{e}.{t}", np.full(2, 10 * e)))
    stream.append(("mtp.layers.0.mlp.experts.3.gate_proj.qweight", np.zeros(1)))
    out = dict(reap.wrap_weights(iter(stream), Cfg))
    assert out["model.language_model.layers.0.mlp.gate.weight"].tolist() == [[0, 1], [4, 5], [6, 7]]
    assert out["model.language_model.layers.0.mlp.experts.1.gate_proj.qweight"].tolist() == [20, 20]
    assert out["model.language_model.layers.1.mlp.experts.0.gate_proj.qweight"].tolist() == [10, 10]
    assert "model.language_model.layers.0.mlp.experts.3.gate_proj.qweight" not in out
    assert "mtp.layers.0.mlp.experts.3.gate_proj.qweight" in out     # the draft's experts pass through
    Cfg.num_experts = 4
    with pytest.raises(RuntimeError, match="num_experts"):
        reap.wrap_weights(iter(stream), Cfg)


# ------------------------------------------------------------------------------------- FR-Spec map writer

def test_hotmap_cell_writes_the_file_the_launcher_asserts(tmp_path, turbo_cells):
    cell = _one(turbo_cells, "# [calamitychasm] ARC FR-SPEC MAP")
    ns: dict = {}
    exec(compile(cell.replace("/kaggle/arc3-hot-tokens.pt", str(tmp_path / "hot.pt")), "hot", "exec"), ns)
    data = (tmp_path / "hot.pt").read_bytes()
    launch = _one(turbo_cells, "CFG = dict(\n")
    assert f'TOKEN_MAP_SHA = "{hashlib.sha256(data).hexdigest()}"' in launch
    assert B._read_hot_ids(tmp_path / "hot.pt") == B._read_hot_ids(TURBO_DIR / "hot_tokens_64k_arc.pt")
    assert ns["_tb_n"] == 65536
    meta = json.loads((TURBO_DIR / "hot_tokens_64k_arc.meta.json").read_text())
    assert meta["base_map"]["sha256"] == B.BASE_TOKEN_MAP_SHA and meta["tokenizer_sha256"] in launch


def test_hotmap_writer_detects_corruption(tmp_path):
    ns: dict = {}
    exec(B.HOT_WRITER, ns)
    hm = B.hot_map()
    with pytest.raises(RuntimeError, match="corrupted"):
        ns["_tb_write_hot_tokens"](hm["payload"], "0" * 64, str(tmp_path / "x.pt"))


# ------------------------------------------------------------------------- timeout fix on the real harness

SRC = h.m2_src()


@pytest.fixture(scope="module")
def harness():
    if SRC is None:
        pytest.skip("milestone-2 src unavailable (ARC3_M2_SRC / ARC3_M2_BASE)")
    saved = dict(os.environ)
    os.environ.update(h.notebook_env())
    try:
        TA, RS, SB, S = h.import_harness(SRC)
        if S is None:
            pytest.skip("solver module needs taaf's dependencies")
        assert not tf.STATS["installed"]
        yield TA, RS, SB, S
    finally:
        tf.uninstall()
        os.environ.clear()
        os.environ.update(saved)


DEFINE = "def f(x):\n    return x + 1\ndef g(y):\n    return y * 2\nprint('def', f(1), g(2))\n"
SPIN = "while True:\n    pass\n"
USE = "print('use', f(3), g(4))\n"
TIME = "import time\nprint('time', type(time.time()).__name__)\n"


def _play(harness, tmp_path, scenario, tag):
    TA, RS, SB, S = harness
    sp = tmp_path / tag / "g_tool_runtime_state.json"
    sp.parent.mkdir(parents=True)
    game = h.FakeGame(RS, S, sp)
    game.seed()
    agent = h.make_agent(TA, sp, game)
    out = []
    for code, timeout in scenario:
        agent._python_timeout = timeout
        res = agent._run_python_tool(sp, {"code": code})
        out.append((json.loads(res.content), sorted(agent._kept_functions)))
    return out


def test_timeout_keeps_previously_retained_functions(harness, tmp_path):
    scenario = [(DEFINE, 30), (SPIN, 2), (USE, 30), (TIME, 30)]
    base = _play(harness, tmp_path, scenario, "base")
    assert base[1][1] == [] and "no longer retained" in base[1][0]["function_retention"]       # the bug
    assert "NameError" in base[2][0]["error"]
    assert "not allowed" in base[3][0]["error"]
    assert tf.install()
    tf.reset_stats()
    try:
        fixed = _play(harness, tmp_path, scenario, "fixed")
        stats = tf.summary()
    finally:
        tf.uninstall()
    assert fixed[0] == base[0]                                     # a normal call is untouched
    assert fixed[1][1] == ["f", "g"]
    assert fixed[1][0]["error"].startswith("Tool timed out after 2s")
    assert "2 previously retained functions are still available" in fixed[1][0]["function_retention"]
    assert fixed[2][0]["stdout"] == "use 4 8\n"
    assert fixed[3][0]["stdout"] == "time float\n"
    assert stats["timeouts_seen"] == stats["timeouts_kept"] == 1 and stats["functions_kept_max"] == 2
    assert stats["errors"] == 0
    TA = harness[0]
    assert TA.ToolAgent._record_retained_functions.__module__ != "timeout_fix"


def test_timeout_with_nothing_retained_and_non_timeouts_go_to_the_original(harness, tmp_path):
    scenario = [(SPIN, 2), (DEFINE, 30), ("raise ValueError('x')\n", 30), (USE, 30)]
    base = _play(harness, tmp_path, scenario, "base")
    assert tf.install()
    tf.reset_stats()
    try:
        fixed = _play(harness, tmp_path, scenario, "fixed")
        stats = tf.summary()
    finally:
        tf.uninstall()
    assert fixed == base
    assert stats["timeouts_seen"] == 1 and stats["timeouts_kept"] == 0 and stats["errors"] == 0


def test_built_cells_compose_on_the_real_harness(harness, tmp_path, turbo_cells, capsys):
    """Level-memory install, history-cache and timeout-fix cells of the turbo kernel, executed verbatim in
    notebook order, install all three; a timeout then keeps the functions with the history cache active."""
    TA = harness[0]
    cells = [_one(turbo_cells, "# [calamitychasm] SOLVED-LEVEL MEMORY"),
             _one(turbo_cells, "# [calamitychasm] HISTORY CACHE"),
             _one(turbo_cells, "# [calamitychasm] SANDBOX-TIMEOUT FIX")]
    cls = TA.ToolAgent
    saved_lm = {n: cls.__dict__[n] for n in ("_build_user_prompt", "_trim_messages_for_context", "_ensure_session")}
    saved_mods = {k: sys.modules.get(k) for k in ("history_cache", "level_memory", "timeout_fix")}
    ns = {"json": json}
    try:
        for i, cell in enumerate(cells):
            exec(compile(cell, f"cell_{i}", "exec"), ns)
        out = capsys.readouterr().out
        for marker in ("LEVEL_MEMORY installed", "HISTORY_CACHE installed", "TIMEOUT_FIX installed"):
            assert marker in out
        assert cls._run_python_tool.__module__ == "history_cache"
        assert cls._record_retained_functions.__module__ == "timeout_fix"
        res = _play(harness, tmp_path, [(DEFINE, 30), (SPIN, 2), (USE, 30), (TIME, 30)], "composed")
        assert res[1][1] == ["f", "g"] and res[2][0]["stdout"] == "use 4 8\n" and res[3][0]["stdout"] == "time float\n"
        assert ns["TIMEOUT_FIX"].summary()["errors"] == 0 and ns["HISTORY_CACHE"].summary()["errors"] == 0
        assert ns["TIMEOUT_FIX"].summary()["timeouts_kept"] == 1
    finally:
        if "TIMEOUT_FIX" in ns:
            ns["TIMEOUT_FIX"].uninstall()
        if "HISTORY_CACHE" in ns:
            ns["HISTORY_CACHE"].uninstall()
        for n, fn in saved_lm.items():
            setattr(cls, n, fn)
        if "_lm_installed" in cls.__dict__:
            del cls._lm_installed
        for k, m in saved_mods.items():
            if m is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = m
    assert '"time",' not in harness[2]._SANDBOX_BOOTSTRAP


def test_timeout_fix_refuses_an_unknown_bootstrap():
    with pytest.raises(RuntimeError, match="SAFE_MODULES anchor"):
        tf.patch_bootstrap("SAFE_MODULES = set()\n")
