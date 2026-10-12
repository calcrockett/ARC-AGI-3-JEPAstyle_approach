"""Arm C+U (calamitychasm/arc3-jad-exp081-u): arm C v2's notebook plus JustAdev742's ours-10 I1 notice (OURS_UNTRIED=i1)
and our stale-state latch fix. Pins the committed notebook and the vendored patch, the exact diff against arm C (two
patch cells, three setup-cell lines, one install cell, the counter dump, three metadata fields), that their own builder
ships ours-10 the same way, and, on the real built harness tree, that the fix lets a caught refusal act again (and that
without it the latch refuses every later action), that the per-snippet cap holds, and that the notice renders and is
counted."""

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
import _build_jad_exp081_u_kernel as B  # noqa: E402
import kaggle_ops  # noqa: E402

NB = B.OUT / "arc3-jad-exp081-u.ipynb"
META = B.OUT / "kernel-metadata.json"
NB_SHA = "cdb10604eb800b979f6879a412a4de94d637fa5f822e830688e287e17010593a"
META_SHA = "5d2e6cafac02c771375c40ce3a8fd5ab92541483c4c86d22aa7e76a924566ef8"
LATCH_SHA = "ee856f8074cb8fed48df6cce73533acbf87814b909b1282e7bab19a6827683b4"
UPSTREAM = Path("/home/user/ext/JustAdev742_Arc-Agi-3-Kaggle-comp")
HAS_HIS = (C.DEFAULT_HIS_REPO / "ARC3-Inference").is_dir()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def test_committed_files_are_pinned():
    assert _sha(NB) == NB_SHA
    assert _sha(META) == META_SHA
    assert _sha(B.LATCH_PATCH) == LATCH_SHA
    assert NB.stat().st_size < 900_000  # Kaggle refuses notebooks near 1 MB


def test_vendored_files_are_theirs_unmodified():
    vendor = B.KDIR / "vendor"
    listed = {}
    for line in filter(None, (vendor / "SHA256SUMS").read_text().split("\n")):
        sha, name = line.split(maxsplit=1)
        listed[name.removeprefix("./")] = sha
    on_disk = {str(p.relative_to(vendor)) for p in vendor.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    assert on_disk == set(listed) == {"LICENSE", "kaggle/franzen/NOTICE.md", "kaggle/franzen/patches/ours-10-untried.patch"}
    for name, sha in listed.items():
        assert _sha(vendor / name) == sha, name
    assert listed["kaggle/franzen/patches/ours-10-untried.patch"] == B.UNTRIED_PATCH_SHA
    # LICENSE and the Franzen NOTICE are the same files arm C vendors from c7a462b
    for name in ("LICENSE", "kaggle/franzen/NOTICE.md"):
        assert listed[name] == _sha(C.VENDOR / name), name
    if (UPSTREAM / "kaggle" / "franzen" / "patches" / "ours-10-untried.patch").is_file():
        for name, sha in listed.items():
            assert _sha(UPSTREAM / name) == sha, name


def test_arm_c_is_untouched_and_is_the_source():
    assert _sha(B.ARM_C_NB) == B.ARM_C_NB_SHA  # a scheduled submit pins arm C version 2
    import test_jad_exp081 as TC  # noqa: PLC0415
    assert B.ARM_C_NB_SHA == TC.NB_SHA


def test_rebuild_is_byte_identical_and_idempotent(tmp_path):
    B.build(tmp_path)
    B.build(tmp_path)
    assert _sha(tmp_path / "arc3-jad-exp081-u.ipynb") == NB_SHA
    assert _sha(tmp_path / "kernel-metadata.json") == META_SHA


def test_diff_vs_arm_c_is_exactly_the_intended_cells():
    a, b = _nb(B.ARM_C_NB), _nb(NB)
    assert a["metadata"] == b["metadata"] and a["nbformat"] == b["nbformat"]
    assert a["nbformat_minor"] == b["nbformat_minor"]
    ca, cb = a["cells"], b["cells"]
    assert len(cb) == len(ca) + 3
    sa, sb = [_src(c) for c in ca], [_src(c) for c in cb]
    last = next(i for i, s in enumerate(sa) if "/kaggle/ours-06-ours-08b-perception" in s.split("\n", 1)[0])
    setup = next(i for i, s in enumerate(sa) if "setup_env = {" in s)
    run = next(i for i, s in enumerate(sa) if s.startswith(B.RUN_CELL_START))
    assert last < setup < run
    # cells 0..last unchanged (cell 0 included), then the two patch cells
    assert cb[:last + 1] == ca[:last + 1]
    assert sb[last + 1].startswith("# >>> ours (--compact): /kaggle/ours-07-ours-10-untried.patch, 22845 bytes")
    assert sb[last + 2].startswith("%%writefile /kaggle/ours-08-calamitychasm-stale-latch-fix.patch\n")
    # unchanged up to the setup cell, which changes in exactly three lines
    assert cb[last + 3:setup + 2] == ca[last + 1:setup]
    la, lb = sa[setup].splitlines(), sb[setup + 2].splitlines()
    assert len(lb) == len(la) + 1
    changed = [x for x in lb if x not in la]
    assert changed == [next(x for x in lb if x.startswith("for _ours_patch in [")),
                       "print('our harness patches applied successfully: 8')",
                       "    'OURS_UNTRIED': 'i1',  # ours (--env-add)"]
    assert [x for x in la if x not in lb] == [next(x for x in la if x.startswith("for _ours_patch in [")),
                                              "print('our harness patches applied successfully: 6')"]
    assert changed[0].endswith("'/kaggle/ours-07-ours-10-untried.patch', "
                               "'/kaggle/ours-08-calamitychasm-stale-latch-fix.patch']:")
    # unchanged up to the run cell; the install cell before it; the run cell gains only the counter dump
    assert cb[setup + 3:run + 2] == ca[setup + 1:run]
    assert cb[run + 2] == B.install_cell()
    assert sb[run + 3] == sa[run].replace(B.RUN_ANCHOR, B.RUN_DUMP)
    assert len(sb[run + 3].splitlines()) == len(sa[run].splitlines()) + 7
    assert cb[run + 4:] == ca[run + 1:]


def test_patch_cells_write_the_patches_byte_for_byte():
    BF = B.their_builder()
    written = dict(BF.franzen_tree.our_patch_cells(NB))
    assert len(written) == 8
    assert written["/kaggle/ours-07-ours-10-untried.patch"] == B.UNTRIED_PATCH.read_text(encoding="utf-8")
    assert written["/kaggle/ours-08-calamitychasm-stale-latch-fix.patch"] == B.LATCH_PATCH.read_text(encoding="utf-8")
    for i, p in enumerate(C.PATCHES, start=1):
        assert written[f"/kaggle/{BF._patch_name(i, Path(p))}"] == \
            (C.VENDOR / "kaggle" / "franzen" / "patches" / p).read_text(encoding="utf-8")


def test_their_builder_ships_ours10_the_same_way(tmp_path):
    """Their own builder (c7a462b) with exp-081's flags + --env-add OURS_UNTRIED=i1 + --patch ours-10 writes the same
    ours-07 cell and the same setup cell, except for our 8th patch in the apply loop."""
    args = C.builder_args(tmp_path)
    at = args.index("--patch")
    args[at:at] = ["--env-add", "OURS_UNTRIED=i1"]
    at = args.index("--compact")
    args[at:at] = ["--patch", str(B.UNTRIED_PATCH)]
    run = subprocess.run([sys.executable, "-I", "-B", str(C.VENDOR / "scripts" / "build_franzen_nb.py"), *args,
                          "--no-apply-check"], cwd=C.VENDOR, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    theirs = [_src(c) for c in _nb(tmp_path / f"{C.SLUG}.ipynb")["cells"]]
    ours = [_src(c) for c in _nb(NB)["cells"]]
    cell07 = next(s for s in ours if s.startswith("# >>> ours (--compact): /kaggle/ours-07-"))
    assert cell07 in theirs
    t_setup = next(s for s in theirs if "setup_env = {" in s)
    o_setup = next(s for s in ours if "setup_env = {" in s)
    assert o_setup == t_setup.replace("'/kaggle/ours-07-ours-10-untried.patch']:",
                                      "'/kaggle/ours-07-ours-10-untried.patch', "
                                      "'/kaggle/ours-08-calamitychasm-stale-latch-fix.patch']:") \
                             .replace("applied successfully: 7')", "applied successfully: 8')")


def test_arm_c_settings_are_kept():
    cells = [_src(c) for c in _nb(NB)["cells"]]
    setup = next(c for c in cells if "setup_env = {" in c)
    assert "os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '2400'  # ours (--env)" in setup
    assert "'ARC3_STALE_STATE_BLOCK': '1'," in setup and "'ARC3_MAX_ACTIVE_STREAMS': 14,  # ours (--env)" in setup
    for key in ("OURS_BUDGET_METER", "OURS_WIN_LEDGER", "OURS_SEARCH_HELPER", "OURS_LEVEL_MEM", "OURS_PERCEPTION"):
        assert f"'{key}': 1,  # ours (--env-add)" in setup
    text = "\n".join(cells)
    assert "bm.solver.max_runtime_s_per_game = 121.0*60  # ours (--full25)" in text
    assert "#OURS_FORM ok version=d_prime" in text and '{"text_config": {"num_experts": 448}}' in text
    for cfg in ("MAXREQ=14", "CUDAGRAPH_MAXBS=14", "MAMBA_CACHE=84", "SPEC_ACCEPT_SINGLE=0.5", "SPEC_ACCEPT_ACC=0.5"):
        assert f"{cfg},  # ours (--cfg)" in text
    assert "LEVEL_MEMORY" not in text  # C+U is not C+LM


def test_install_cell_ships_the_counters_verbatim():
    cell = _src(B.install_cell())
    assert f"_CU_SRC = r'''{B.COUNTERS.read_text(encoding='utf-8')}'''\n" in cell
    assert "for _cu_line in CU_COUNTERS.install(_cu_sb, _cu_ta, _cu_ut):" in cell
    compile(cell, "install", "exec")


def test_latch_patch_is_small_and_marked():
    text = B.LATCH_PATCH.read_text(encoding="utf-8")
    added = [x for x in text.splitlines() if x.startswith("+") and not x.startswith("+++")]
    code = [x for x in added if not x.lstrip("+").strip().startswith("#")]
    assert [x.strip("+ ") for x in code] == ["_stale_releases = [3]", "if _stale_releases[0] > 0:",
                                              "_stale_releases[0] -= 1", "del _stale_state[:]"]
    assert not [x for x in text.splitlines() if x.startswith("-") and not x.startswith("---")]
    assert "calamitychasm STALE_LATCH_FIX" in text


def test_metadata_differs_from_arm_c_only_in_identity():
    a = json.loads(B.ARM_C_META.read_text())
    b = json.loads(META.read_text())
    assert b["id"] == "calamitychasm/arc3-jad-exp081-u"
    assert b["title"] == "arc3-jad-exp081-u" and b["code_file"] == "arc3-jad-exp081-u.ipynb"
    for k in ("id", "title", "code_file"):
        a.pop(k), b.pop(k)
    assert a == b
    assert "scottmahony" not in META.read_text()


def test_kaggle_ops_treats_it_as_a_gpu_kernel():
    assert "arc3-jad-" in kaggle_ops.GPU_KERNEL_PREFIXES
    assert kaggle_ops.is_gpu_family("calamitychasm/arc3-jad-exp081-u")


def _probe(tmp_path, *extra) -> dict:
    run = subprocess.run([sys.executable, str(ROOT / "tests" / "jad_u_probe.py"), str(NB), str(tmp_path / "tree"),
                          str(C.DEFAULT_HIS_REPO), *extra], capture_output=True, text=True, cwd=ROOT, timeout=900)
    assert run.returncode == 0, run.stderr[-3000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not HAS_HIS, reason="needs Franzen's repo to build the exp-081 harness tree")
def test_latch_fix_and_notice_on_the_real_tree(tmp_path):
    out = _probe(tmp_path)
    assert len(out["patches"]) == 8 and out["untried_mode"] == ["i1"] and out["stale_block"] == "1"
    assert out["install_error"] == ""
    assert "STALE_LATCH_FIX installed bootstraps=4 releases_per_snippet=3" in out["install_stdout"]
    assert "UNTRIED_NOTICE installed mode=i1" in out["install_stdout"]
    assert set(out["wrapped"].values()) == {"cu_counters"}
    # a caught refusal: the next, state-changing action executes
    assert "FIRST True False\\nSECOND refused\\nTHIRD True True" in out["catch_content"]
    assert out["catch_executed"] == 2
    # a caught loop of no-ops: 3 releases per snippet, then the latch holds as before
    assert "LOOP 4 6" in out["loop_content"] and out["loop_executed"] == 4
    s = out["final_stale_summary"]
    assert s["errors"] == 0 and s["snippets"] == 2 and s["stale_refusals"] == 7
    assert s["stale_refused_after_released"] == 0 and s["stale_refused_after_cap"] == 2
    assert s["executed_after_stale_refusal"] == 4 and s["max_stale_refusals_in_snippet"] == 6
    # the notice: nothing before 10 min, the I1 line at 11.7 min, before the tool line
    assert out["notice_first"] == []
    assert out["notice_second"] == ["Not yet tried on this level: MOUSE never used; no click has hit g r3-5 c3-5, "
                                    "G r9-10 c9-12. Each is a one-action test of whether it is interactive."]
    assert out["notice_before_tool_line"]
    u = out["untried_summary"]
    assert u["errors"] == 0 and u["tracker_errors"] == 0 and u["i2_lines"] == 0
    assert u["i1_lines"] == 1 and u["i1_resolved"] == 1 and u["i1_clicked_named_within_3"] == 1
    assert u["i1_used_named_action_within_3"] == 1 and u["i1_clicked_named_within_3_share"] == 1.0
    assert out["untried_firings"][0]["hit"] == 2 and out["untried_firings"][0]["window"] == 3


@pytest.mark.skipif(not HAS_HIS, reason="needs Franzen's repo to build the exp-081 harness tree")
def test_without_the_fix_the_latch_refuses_every_later_action(tmp_path):
    out = _probe(tmp_path, "--without-latch-fix")
    assert len(out["patches"]) == 7
    assert out["install_error"].startswith("RuntimeError: STALE_LATCH_FIX missing")  # the install cell refuses
    assert "SECOND refused\\nTHIRD refused" in out["catch_content"] and out["catch_executed"] == 1
    assert "LOOP 1 9" in out["loop_content"] and out["loop_executed"] == 1
