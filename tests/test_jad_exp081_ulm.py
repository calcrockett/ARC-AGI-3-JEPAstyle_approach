"""Bundle C+UL (calamitychasm/arc3-jad-exp081-ulm): arm C v2's notebook plus C+U's cells (ours-10 I1 with OURS_UNTRIED=i1,
the stale-latch fix, the 8-patch apply loop, the C+U install cell and counter dump) and C+LM's cells (the level-memory
install cell and its counter dump). Pins the committed notebook; checks that the diff against arm C is exactly the union
of the two variants' cells (each added cell byte-identical to the variant notebook's), that the -u and -lm notebooks are
untouched, and, on the real built harness tree, that the two mechanisms compose in ToolAgent's prompt building
(tests/jad_ulm_probe.py)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_jad_exp081_kernel as C  # noqa: E402
import _build_jad_exp081_lm_kernel as LM  # noqa: E402
import _build_jad_exp081_u_kernel as U  # noqa: E402
import _build_jad_exp081_ulm_kernel as B  # noqa: E402
import _build_m2_level_memory_kernel as LMB  # noqa: E402
import kaggle_ops  # noqa: E402

NB = B.OUT / "arc3-jad-exp081-ulm.ipynb"
META = B.OUT / "kernel-metadata.json"
NB_SHA = "99219f34fcfae2f968372fb1b519122b9cc847a7b58b13a11c973f5fd1a04c6c"
META_SHA = "1ab67137c6b1dedf9e434d53cf4df56802650cc5b4d851e8198b27cbeb25ce90"
U_NB = U.OUT / "arc3-jad-exp081-u.ipynb"
LM_NB = LM.OUT / "arc3-jad-exp081-lm.ipynb"
HAS_HIS = (C.DEFAULT_HIS_REPO / "ARC3-Inference").is_dir()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _index(srcs: list[str]) -> tuple[int, int, int]:
    """(arm C's last patch cell, the setup cell, the run cell) in a list of cell sources."""
    last = next(i for i, s in enumerate(srcs) if "/kaggle/ours-06-ours-08b-perception" in s.split("\n", 1)[0])
    setup = next(i for i, s in enumerate(srcs) if "setup_env = {" in s)
    run = next(i for i, s in enumerate(srcs) if s.startswith(B.RUN_CELL_START))
    return last, setup, run


def test_committed_files_are_pinned():
    assert _sha(NB) == NB_SHA
    assert _sha(META) == META_SHA
    assert NB.stat().st_size < 900_000  # Kaggle refuses notebooks near 1 MB


def test_arm_c_and_both_variants_are_untouched():
    import test_jad_exp081 as TC  # noqa: PLC0415
    import test_jad_exp081_lm as TL  # noqa: PLC0415
    import test_jad_exp081_u as TU  # noqa: PLC0415
    assert _sha(U.ARM_C_NB) == U.ARM_C_NB_SHA == LM.ARM_C_NB_SHA == TC.NB_SHA  # arm C v2 (scheduled submits pin it)
    assert _sha(U_NB) == TU.NB_SHA and _sha(U.OUT / "kernel-metadata.json") == TU.META_SHA  # C+U v1 check run
    assert _sha(LM_NB) == TL.NB_SHA and _sha(LM.OUT / "kernel-metadata.json") == TL.META_SHA  # C+LM v1 check run


def test_rebuild_is_byte_identical_and_idempotent(tmp_path):
    B.build(tmp_path)
    B.build(tmp_path)
    assert _sha(tmp_path / "arc3-jad-exp081-ulm.ipynb") == NB_SHA
    assert _sha(tmp_path / "kernel-metadata.json") == META_SHA


def test_diff_vs_arm_c_is_exactly_the_union_of_the_two_variants():
    a, u, lm, b = _nb(U.ARM_C_NB), _nb(U_NB), _nb(LM_NB), _nb(NB)
    for other in (u, lm):
        assert other["metadata"] == a["metadata"] == b["metadata"]
    assert (b["nbformat"], b["nbformat_minor"]) == (a["nbformat"], a["nbformat_minor"])
    ca, cu, cl, cb = a["cells"], u["cells"], lm["cells"], b["cells"]
    sa, su, sl, sb = ([_src(c) for c in cs] for cs in (ca, cu, cl, cb))
    assert len(cu) == len(ca) + 3 and len(cl) == len(ca) + 1
    assert len(cb) == len(ca) + 4
    last, setup, run = _index(sa)
    assert last < setup < run
    ulast, usetup, urun = _index(su)
    assert (ulast, usetup, urun) == (last, setup + 2, run + 3)
    assert _index(sl)[2] == run + 1
    # C+U's cells, byte for byte: everything up to and including the C+U install cell (the 2 patch cells, the setup cell)
    assert cb[:run + 3] == cu[:run + 3]
    assert cb[run + 2] == U.install_cell() == cu[run + 2]
    # C+LM's install cell, byte for byte, right before the run cell (after D''s and C+U's install cells)
    assert cb[run + 3] == cl[run] == LMB.install_cell()
    assert "install_d(_fp_ta, _fp_solver)" in sb[run + 1]
    # after the run cell: arm C's cells unchanged
    assert cb[run + 5:] == ca[run + 1:] == cu[run + 4:] == cl[run + 2:]
    # the run cell: arm C's plus both dumps after bm.run (LM's, then C+U's), and nothing else
    ra, rb = sa[run], sb[run + 4]
    tail_lm = LMB.RUN_DUMP[len(LMB.RUN_ANCHOR):]
    tail_u = U.RUN_DUMP[len(U.RUN_ANCHOR):]
    assert rb == ra.replace(LMB.RUN_ANCHOR, LMB.RUN_ANCHOR + tail_lm + tail_u)
    added = Counter(rb.splitlines()) - Counter(ra.splitlines())
    assert added == (Counter(su[urun].splitlines()) - Counter(ra.splitlines())) + \
        (Counter(sl[run + 1].splitlines()) - Counter(ra.splitlines()))
    assert not Counter(ra.splitlines()) - Counter(rb.splitlines())
    assert {k: v for k, v in cb[run + 4].items() if k != "source"} == {k: v for k, v in ca[run].items() if k != "source"}
    # every cell is arm C's, C+U's or C+LM's, except the run cell, which is the union of the two run cells
    pool = sa + su + sl
    assert [i for i, s in enumerate(sb) if s not in pool] == [run + 4]


def test_both_variants_markers_and_settings_are_present():
    cells = [_src(c) for c in _nb(NB)["cells"]]
    text = "\n".join(cells)
    setup = next(c for c in cells if "setup_env = {" in c)
    assert "os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '2400'  # ours (--env)" in setup
    assert "'ARC3_STALE_STATE_BLOCK': '1'," in setup and "'ARC3_MAX_ACTIVE_STREAMS': 14,  # ours (--env)" in setup
    for key in ("OURS_BUDGET_METER", "OURS_WIN_LEDGER", "OURS_SEARCH_HELPER", "OURS_LEVEL_MEM", "OURS_PERCEPTION"):
        assert f"'{key}': 1,  # ours (--env-add)" in setup
    assert "    'OURS_UNTRIED': 'i1',  # ours (--env-add)\n" in setup
    assert "print('our harness patches applied successfully: 8')" in setup
    assert "bm.solver.max_runtime_s_per_game = 121.0*60  # ours (--full25)" in text
    assert "#OURS_FORM ok version=d_prime" in text and '{"text_config": {"num_experts": 448}}' in text
    for cfg in ("MAXREQ=14", "CUDAGRAPH_MAXBS=14", "MAMBA_CACHE=84", "SPEC_ACCEPT_SINGLE=0.5", "SPEC_ACCEPT_ACC=0.5"):
        assert f"{cfg},  # ours (--cfg)" in text
    for p in C.PATCHES:
        assert _sha(C.VENDOR / "kaggle" / "franzen" / "patches" / p) in text, p
    for marker in ("LEVEL_MEMORY installed", "LEVEL_MEMORY summary", "UNTRIED_NOTICE summary", "STALE_LATCH_FIX summary",
                   "(WORKING_DIR / 'level_memory_summary.json').write_text("):
        assert marker in text, marker
    assert "LEVEL_MEMORY_TRIED_FACTS" not in _src(LMB.install_cell()).split("'''")[-1]  # tried facts stay off
    BF = U.their_builder()
    written = dict(BF.franzen_tree.our_patch_cells(NB))
    assert len(written) == 8
    assert written["/kaggle/ours-07-ours-10-untried.patch"] == U.UNTRIED_PATCH.read_text(encoding="utf-8")
    assert written["/kaggle/ours-08-calamitychasm-stale-latch-fix.patch"] == U.LATCH_PATCH.read_text(encoding="utf-8")


def test_metadata_differs_from_arm_c_only_in_identity():
    a = json.loads(U.ARM_C_META.read_text())
    b = json.loads(META.read_text())
    assert b["id"] == "calamitychasm/arc3-jad-exp081-ulm"
    assert b["title"] == "arc3-jad-exp081-ulm" and b["code_file"] == "arc3-jad-exp081-ulm.ipynb"
    for k in ("id", "title", "code_file"):
        a.pop(k), b.pop(k)
    assert a == b
    assert "scottmahony" not in META.read_text()


def test_kaggle_ops_treats_it_as_a_gpu_kernel():
    assert "arc3-jad-" in kaggle_ops.GPU_KERNEL_PREFIXES
    assert kaggle_ops.is_gpu_family("calamitychasm/arc3-jad-exp081-ulm")


@pytest.mark.skipif(not HAS_HIS, reason="needs Franzen's repo to build the exp-081 harness tree")
def test_the_two_mechanisms_compose_on_the_real_tree(tmp_path):
    run = subprocess.run([sys.executable, str(ROOT / "tests" / "jad_ulm_probe.py"), str(NB), str(tmp_path / "tree"),
                          str(C.DEFAULT_HIS_REPO)], capture_output=True, text=True, cwd=ROOT, timeout=900)
    assert run.returncode == 0, run.stderr[-3000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    # the tree and both install cells, in notebook order
    assert len(out["patches"]) == 8 and out["untried_mode"] == ["i1"]
    assert out["flags"] == {"OURS_WIN_LEDGER": "1", "OURS_LEVEL_MEM": "1", "OURS_UNTRIED": "i1",
                            "ARC3_STALE_STATE_BLOCK": "1", "ARC3_MAX_ACTIVE_STREAMS": "14"}
    assert out["install_order"] == ["cu", "lm"] and out["ledger_on"]
    for marker in ("STALE_LATCH_FIX installed bootstraps=4 releases_per_snippet=3", "UNTRIED_NOTICE installed mode=i1",
                   "LEVEL_MEMORY installed"):
        assert marker in out["install_stdout"], marker
    assert out["wrapped"] == {"_build_user_prompt": "level_memory", "_trim_messages_for_context": "level_memory",
                              "_ensure_session": "level_memory", "_run_python_tool": "cu_counters",
                              "prompt_lines": "cu_counters", "stamp": "cu_counters"}
    # the latch fix still works with LM installed
    assert "FIRST True False\\nSECOND refused\\nTHIRD True True" in out["catch_content"] and out["catch_executed"] == 2
    assert "LOOP 4 6" in out["loop_content"] and out["loop_executed"] == 4
    s = out["stale_summary"]
    assert s["errors"] == 0 and s["stale_refused_after_released"] == 0 and s["stale_refused_after_cap"] == 2
    assert s["executed_after_stale_refusal"] == 4
    # prompt layouts: each block at most once, in a sensible order (I1 before the tool line; LM's request last)
    p, ask_end = out["prompts"], out["ask_at_end"]
    assert p["l1_first"]["order"] == ["tool_line"]
    assert p["l1_stalled"]["order"] == ["i1", "tool_line"]
    assert p["l2_opener"]["order"] == ["win_record", "tool_line", "lm_ask"] and ask_end["l2_opener"]
    assert p["l2_stalled_ask2"]["order"] == ["i1", "tool_line", "lm_ask"] and ask_end["l2_stalled_ask2"]
    assert p["l2_after_rule"]["order"] == ["tool_line"]
    assert p["l2_after_trims"]["order"] == ["i1", "tool_line"]
    assert all(c <= 1 for lay in p.values() for c in lay["counts"].values())
    i1 = ("Not yet tried on this level: MOUSE never used; no click has hit g r3-5 c3-5, G r9-10 c9-12. "
          "Each is a one-action test of whether it is interactive.")
    for name in ("l1_stalled", "l2_stalled_ask2", "l2_after_trims"):
        assert out["i1_lines"][name] == [i1], name
    # LM: the rule is captured (second ask), the block sits at the end of the system prompt; 03b's ledger right after
    assert out["rule"].startswith("UP moves nothing; MOUSE on the 3x3 g block")
    assert out["first_trim_evicted"]
    kinds = ["system+lm_block", "ledger_03b", "user:history", "assistant", "user:l2_opener", "assistant",
             "user:l2_stalled_ask2", "assistant", "user:l2_after_rule"]
    assert out["trims"]["second"]["kinds"] == kinds and out["trims"]["third"]["kinds"] == kinds  # idempotent
    for t in ("first", "second", "third"):
        head = out["trims"][t]["head"][0]
        assert head["starts_with_base"] and head["lm_header"] == 1 and head["lm_footer"] == 1
        assert head["ends_with_lm_footer"] and head["rule_body"] == 1
        assert out["trims"][t]["lm_headers"] == 1
    assert out["trims"]["first"]["ledgers"] == 0
    assert out["trims"]["second"]["ledgers"] == out["trims"]["third"]["ledgers"] == 1
    assert "- Level 1: won in 3 actions (steps 1-3), no game over." in out["trims"]["second"]["ledger_text"]
    # counters: nothing unmatched across the system-prompt rewrite, no errors anywhere
    us = out["untried_summary"]
    assert us["errors"] == 0 and us["tracker_errors"] == 0 and us["i2_lines"] == 0 and us["i1_unmatched"] == 0
    assert us["i1_lines"] == 3 and us["i1_resolved"] == 3 and us["i1_clicked_named_within_3"] == 1
    assert out["untried_firings"][0]["hit"] == 2
    lm = out["lm_summary"]
    assert lm["errors"] == 0 and lm["levels_recorded"] == 1 and lm["rules_captured"] == 1
    assert lm["asks"] == 2 and lm["blocks_applied"] == 1
