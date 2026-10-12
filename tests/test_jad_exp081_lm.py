"""Arm C+LM (calamitychasm/arc3-jad-exp081-lm): arm C v2's notebook plus our solved-level memory install cell and its
counter dump. Pins the committed notebook, the exact diff against arm C (one inserted cell, five lines in the run cell,
three metadata fields), and, on the real exp-081 harness tree, that the memory composes with their win ledger."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_jad_exp081_kernel as C  # noqa: E402
import _build_jad_exp081_lm_kernel as B  # noqa: E402
import _build_m2_level_memory_kernel as LMB  # noqa: E402
import kaggle_ops  # noqa: E402

NB = B.OUT / "arc3-jad-exp081-lm.ipynb"
META = B.OUT / "kernel-metadata.json"
NB_SHA = "906f12cf3643d3a575ae25563dca2880fd65fb6544757eb4912fb5c87aa3750f"
META_SHA = "7bc5bfdbc60494b56fe882ba82cf8600e0006778812b8616b9b20076becc9fa3"
LM_PY = ROOT / "kaggle_submission_milestone2_fork" / "level_memory" / "level_memory.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def test_committed_files_are_pinned():
    assert _sha(NB) == NB_SHA
    assert _sha(META) == META_SHA
    assert NB.stat().st_size < 900_000  # Kaggle refuses notebooks near 1 MB


def test_arm_c_is_untouched_and_is_the_source():
    assert _sha(B.ARM_C_NB) == B.ARM_C_NB_SHA  # a scheduled submit pins arm C version 2
    import test_jad_exp081 as TC  # noqa: PLC0415
    assert B.ARM_C_NB_SHA == TC.NB_SHA


def test_rebuild_is_byte_identical_and_idempotent(tmp_path):
    B.build(tmp_path)
    B.build(tmp_path)
    assert _sha(tmp_path / "arc3-jad-exp081-lm.ipynb") == NB_SHA
    assert _sha(tmp_path / "kernel-metadata.json") == META_SHA


def test_diff_vs_arm_c_is_the_install_cell_and_the_counter_dump():
    a, b = _nb(B.ARM_C_NB), _nb(NB)
    assert a["metadata"] == b["metadata"] and a["nbformat"] == b["nbformat"]
    assert a["nbformat_minor"] == b["nbformat_minor"]
    ca, cb = a["cells"], b["cells"]
    assert len(cb) == len(ca) + 1
    run = next(i for i, c in enumerate(ca) if _src(c).startswith(B.RUN_CELL_START))
    assert "install_d(_fp_ta, _fp_solver)" in _src(ca[run - 1])  # after D''s install cell
    assert cb[:run] == ca[:run]
    assert cb[run] == LMB.install_cell()
    assert cb[run + 2:] == ca[run + 1:]
    ra, rb = _src(ca[run]), _src(cb[run + 1])
    assert ra.count(LMB.RUN_ANCHOR) == 1 and rb == ra.replace(LMB.RUN_ANCHOR, LMB.RUN_DUMP)
    assert len(rb.splitlines()) == len(ra.splitlines()) + 5
    assert {k: v for k, v in cb[run + 1].items() if k != "source"} == {k: v for k, v in ca[run].items() if k != "source"}


def test_install_cell_ships_level_memory_verbatim():
    cell = _src(_nb(NB)["cells"][next(i for i, c in enumerate(_nb(NB)["cells"])
                                      if _src(c).startswith("# [calamitychasm] SOLVED-LEVEL MEMORY"))])
    assert f"_LM_SRC = r'''{LM_PY.read_text(encoding='utf-8')}'''\n" in cell
    assert "LEVEL_MEMORY.install(_lm_ta.ToolAgent)" in cell
    assert "print('LEVEL_MEMORY installed', LEVEL_MEMORY.summary(), flush=True)" in cell
    assert "LEVEL_MEMORY_TRIED_FACTS" not in cell.split("'''")[-1]  # tried facts stay off
    text = NB.read_text(encoding="utf-8")
    assert "(WORKING_DIR / 'level_memory_summary.json').write_text(" in text


def test_arm_c_settings_are_kept():
    cells = [_src(c) for c in _nb(NB)["cells"]]
    setup = next(c for c in cells if "setup_env = {" in c)
    assert "os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '2400'  # ours (--env)" in setup
    for key in ("OURS_BUDGET_METER", "OURS_WIN_LEDGER", "OURS_SEARCH_HELPER", "OURS_LEVEL_MEM", "OURS_PERCEPTION"):
        assert f"'{key}': 1,  # ours (--env-add)" in setup
    text = "\n".join(cells)
    assert "bm.solver.max_runtime_s_per_game = 121.0*60  # ours (--full25)" in text
    assert "#OURS_FORM ok version=d_prime" in text
    for p in C.PATCHES:
        assert _sha(C.VENDOR / "kaggle" / "franzen" / "patches" / p) in text, p


def test_metadata_differs_from_arm_c_only_in_identity():
    a = json.loads(B.ARM_C_META.read_text())
    b = json.loads(META.read_text())
    assert b["id"] == "calamitychasm/arc3-jad-exp081-lm"
    assert b["title"] == "arc3-jad-exp081-lm" and b["code_file"] == "arc3-jad-exp081-lm.ipynb"
    for k in ("id", "title", "code_file"):
        a.pop(k), b.pop(k)
    assert a == b
    assert "scottmahony" not in META.read_text()


def test_kaggle_ops_treats_it_as_a_gpu_kernel():
    assert kaggle_ops.is_gpu_family("calamitychasm/arc3-jad-exp081-lm")


@pytest.mark.skipif(not (C.DEFAULT_HIS_REPO / "ARC3-Inference").is_dir(),
                    reason="needs Franzen's repo to build the exp-081 harness tree")
def test_level_memory_composes_with_the_win_ledger_on_the_real_exp081_tree(tmp_path):
    run = subprocess.run([sys.executable, str(ROOT / "tests" / "jad_lm_probe.py"), str(NB), str(tmp_path / "tree"),
                          str(C.DEFAULT_HIS_REPO)], capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["ledger_patch"] and out["ledger_on"]
    assert set(out["wrapped"].values()) == {"level_memory"}
    # a level-up opener carries their exact win record, then our rule request
    assert out["opener_win_record"] and out["opener_rule_request"] and out["opener_order_ok"]
    assert out["rule"].startswith("ACTION1 moves the piece right")
    # the first eviction pins our block at the end of the system prompt; a later trim pins their ledger right after it
    assert out["first_trim_evicted"] and out["first_trim_block"]
    assert out["second_trim_ledger_after_system"] and out["second_trim_block"]
    assert "Level 1: won in 2 actions" in out["ledger_text"]
    s = out["summary"]
    assert s["errors"] == 0 and s["blocks_applied"] == 1 and s["rules_captured"] == 1 and s["levels_recorded"] == 1
