"""Arm C+hic16 (calamitychasm/arc3-jad-exp081-hic16): arm C v2's notebook plus the 32 GB system-RAM KV tier and 16
streams. Pins the committed notebook, the exact diff against arm C (one setup-cell line, the launcher's three CFG lines,
the tier flags, the serving markers and the serve.log check, the post-run check in the run cell, three metadata
fields), that arm C itself is untouched, and that the census the pass criteria read is arm C's own."""

from __future__ import annotations

import difflib
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_jad_exp081_hic16_kernel as B  # noqa: E402
import _build_jad_exp081_kernel as C  # noqa: E402
import _build_m2_level_memory_kernel as LMB  # noqa: E402
import kaggle_ops  # noqa: E402
from _build_m2_speed_kernels import hicache_line  # noqa: E402

NB = B.OUT / "arc3-jad-exp081-hic16.ipynb"
META = B.OUT / "kernel-metadata.json"
NB_SHA = "80bc62bdebc1ac7fa4f10d41b0bfd9102aa3aca83368d1c9d7fb6604c63bb08c"
META_SHA = "327873c73de88b53b1b648d4a7fe5b3c29d0fca3836e2acb592d9fa6194686cb"
TB_HIC16 = ROOT / "kaggle_submission_m2_turbo_tail_hic16" / "notebook" / "arc3-m2-turbo-tail-hic16.ipynb"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _changed(a: str, b: str) -> tuple[list[str], list[str]]:
    """(removed lines, added lines) between two cell sources."""
    rem, add = [], []
    for ln in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0):
        if ln.startswith(("---", "+++", "@@")):
            continue
        (rem if ln.startswith("-") else add).append(ln[1:])
    return rem, add


def test_committed_files_are_pinned():
    assert _sha(NB) == NB_SHA
    assert _sha(META) == META_SHA
    assert NB.stat().st_size < 900_000  # Kaggle refuses notebooks near 1 MB


def test_arm_c_is_untouched_and_is_the_source():
    assert _sha(B.ARM_C_NB) == B.ARM_C_NB_SHA  # a scheduled submit pins arm C version 2
    import test_jad_exp081 as TC  # noqa: PLC0415
    assert B.ARM_C_NB_SHA == TC.NB_SHA
    assert _sha(B.ARM_C_META) == TC.META_SHA


def test_rebuild_is_byte_identical_and_idempotent(tmp_path):
    B.build(tmp_path)
    B.build(tmp_path)
    assert _sha(tmp_path / "arc3-jad-exp081-hic16.ipynb") == NB_SHA
    assert _sha(tmp_path / "kernel-metadata.json") == META_SHA
    assert _sha(B.ARM_C_NB) == B.ARM_C_NB_SHA  # building never writes arm C


def test_diff_vs_arm_c_is_exactly_the_intended_lines():
    a, b = _nb(B.ARM_C_NB), _nb(NB)
    assert {k: v for k, v in a.items() if k != "cells"} == {k: v for k, v in b.items() if k != "cells"}
    ca, cb = a["cells"], b["cells"]
    assert len(ca) == len(cb)
    sa, sb = [_src(c) for c in ca], [_src(c) for c in cb]
    setup = next(i for i, s in enumerate(sa) if "setup_env = {" in s)
    launcher = next(i for i, s in enumerate(sa) if s.startswith("# Paste this entire file into the launcher cell."))
    run = next(i for i, s in enumerate(sa) if s.startswith(B.RUN_CELL_START))
    changed = {i for i in range(len(ca)) if ca[i] != cb[i]}
    assert changed == {setup, launcher, run}
    for i in changed:  # only the source differs (no outputs, metadata or execution counts)
        assert {k: v for k, v in ca[i].items() if k != "source"} == {k: v for k, v in cb[i].items() if k != "source"}
    assert sa[0] == sb[0]  # cell 0 (the fork note) is byte-identical

    rem, add = _changed(sa[setup], sb[setup])
    assert rem == ["    'ARC3_MAX_ACTIVE_STREAMS': 14,  # ours (--env)"]
    assert add == ["    'ARC3_MAX_ACTIVE_STREAMS': 16,  # [calamitychasm hic16] streams (arm C: 14)"]

    rem, add = _changed(sa[launcher], sb[launcher])
    assert rem == ["    MAXREQ=14,  # ours (--cfg)", "    CUDAGRAPH_MAXBS=14,  # ours (--cfg)",
                   "    MAMBA_CACHE=84,  # ours (--cfg)"]
    assert add[:3] == ["    MAXREQ=16,  # [calamitychasm hic16] streams (arm C: 14)",
                       "    CUDAGRAPH_MAXBS=16,  # [calamitychasm hic16] streams (arm C: 14)",
                       "    MAMBA_CACHE=96,  # [calamitychasm hic16] streams (arm C: 84)"]
    tail = LMB.HICACHE_CHECK.format(when="", indent="").splitlines()
    assert add[3:] == (["# [calamitychasm hic16] system-RAM KV tier (the hic16 kernels' flags)",
                        hicache_line(32).rstrip("\n"),
                        "# [calamitychasm hic16] serving markers (the submit gate checks them)",
                        "print(f\"STREAMS max_running_requests={CFG['MAXREQ']} cuda_graph_bs={graph_bs} "
                        "mamba_cache={CFG['MAMBA_CACHE']}\", flush=True)",
                        'assert args[args.index("--hicache-size") + 1] == "32" and "--enable-hierarchical-cache" in args',
                        'print("HICACHE requested --hicache-size 32 GB (write_through, kernel io)", flush=True)',
                        "",
                        "# [calamitychasm hicache] confirm from serve.log that SGLang attached the host tier to the "
                        "radix cache",
                        "# (mem_cache/registry.py logs 'Tree cache initialized: ... hicache_attached=<bool>'). "
                        "Never raises."] + tail)
    # placement: the tier right after the prefetch flag, the markers right before the launch, the check at the end
    assert LMB.L_PREFETCH + "# [calamitychasm hic16] system-RAM KV tier" in sb[launcher]
    assert "flush=True)\n" + LMB.L_LAUNCH in sb[launcher]
    assert sb[launcher].startswith(sa[launcher].split("    MAXREQ=")[0])
    assert sb[launcher].endswith(tail[-1])

    rem, add = _changed(sa[run], sb[run])
    assert rem == [] and add == LMB.HIC_DUMP.rstrip("\n").splitlines()
    assert B.RUN_ANCHOR + LMB.HIC_DUMP in sb[run]


def test_tier_and_streams_match_our_hic16_kernels():
    hic = "\n".join(_src(c) for c in _nb(TB_HIC16)["cells"])
    text = NB.read_text(encoding="utf-8")
    cells = [_src(c) for c in _nb(NB)["cells"]]
    launcher = next(c for c in cells if c.startswith("# Paste this entire file into the launcher cell."))
    # the same tier flags (write policy, io backend, size) and the same Mamba-per-stream rule as turbo-tail-hic16
    assert hicache_line(32) in launcher and hicache_line(32) in hic
    assert "MAXREQ=16," in hic and "MAMBA_CACHE=96," in hic
    assert 96 // 5 >= 16 and 84 == 6 * 14 and B.MAMBA_PER_STREAM * B.STREAMS == 96
    # SGLang rejects the host tier with these; neither arm C nor this kernel sets them
    for flag in ("--disable-radix-cache", "--enable-int8-mamba-checkpoint"):
        assert flag not in launcher
    # the launcher's own graph-batch list covers 16 and its assert holds (CUDAGRAPH_MAXBS >= MAXREQ)
    assert 'graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})' in launcher
    # the flag line, the assert and the request print; nowhere else
    assert text.count("--hicache-size") == 3 and launcher.count("--hicache-size") == 3
    assert text.count("--enable-hierarchical-cache") == 2 and launcher.count("--enable-hierarchical-cache") == 2


def test_markers_are_printed_by_the_notebook():
    text = "\n".join(_src(c) for c in _nb(NB)["cells"])
    for m in ("harness patch applied successfully", "our harness patches applied successfully: 6",
              "#OURS_FORM ok version=d_prime", "ours: FR-Spec map hot_tokens_64k_arc.pt written",
              "HICACHE_TIER attached hicache_attached=True", "HICACHE_TIER attached hicache_attached=True (post-run)",
              "STREAMS max_running_requests=", "HICACHE requested --hicache-size 32 GB"):
        assert m in text, m
    # the gate's line comes from Franzen's patch, formatted with the env value
    assert 'log.warning("priority gate active: %d concurrent streams", slots)' in text
    assert "'ARC3_MAX_ACTIVE_STREAMS': 16," in text and "'ARC3_MAX_ACTIVE_STREAMS': 14," not in text
    assert B.kernel_markers()[3] == "priority gate active: 16 concurrent streams"
    assert LMB.HICACHE_MARKER in B.kernel_markers()
    assert "48 layers x 448 experts, router sha256 for every layer" in B.kernel_markers()


def test_arm_c_settings_are_kept():
    cells = [_src(c) for c in _nb(NB)["cells"]]
    setup = next(c for c in cells if "setup_env = {" in c)
    assert "os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '2400'  # ours (--env)" in setup
    for key in ("OURS_BUDGET_METER", "OURS_WIN_LEDGER", "OURS_SEARCH_HELPER", "OURS_LEVEL_MEM", "OURS_PERCEPTION"):
        assert f"'{key}': 1,  # ours (--env-add)" in setup
    text = "\n".join(cells)
    assert "bm.solver.max_runtime_s_per_game = 121.0*60  # ours (--full25)" in text
    assert "SERVER_STARTUP_TIMEOUT = 12 * 60" in text
    for cfg in ("SPEC_ACCEPT_SINGLE=0.5", "SPEC_ACCEPT_ACC=0.5", "MEMFRAC=0.96"):
        assert cfg in text
    assert '{"text_config": {"num_experts": 448}}' in text
    assert "#OURS_FORM ok version=d_prime" in text and "install_d(_fp_ta, _fp_solver)" in text
    for p in C.PATCHES:
        assert _sha(C.VENDOR / "kaggle" / "franzen" / "patches" / p) in text, p


def test_census_is_arm_cs_and_parses():
    a = [_src(c) for c in _nb(B.ARM_C_NB)["cells"]]
    b = [_src(c) for c in _nb(NB)["cells"]]
    census = next(c for c in b if c.startswith("# --- system census"))
    assert census in a  # unchanged
    assert 'out.append(f"RAM {(mem[\'MemTotal\']-mem[\'MemAvailable\'])/GB:.1f}/{mem[\'MemTotal\']/GB:.1f}")' in census
    assert "_stop_monitor = start_monitor(2)" in census
    d = kaggle_ops.sys_ram([("[sys] RAM 115.3/176.9 | swap none | proc 3.1 GB, 412 threads", None),
                            ("[sys] RAM 120.0/176.9 | swap none", None)])
    assert d["count"] == 2 and d["min_gib"] == 56.9 and d["all_ge_10"] is True


def test_metadata_differs_from_arm_c_only_in_identity():
    a = json.loads(B.ARM_C_META.read_text())
    b = json.loads(META.read_text())
    assert b["id"] == "calamitychasm/arc3-jad-exp081-hic16"
    assert b["title"] == "arc3-jad-exp081-hic16" and b["code_file"] == "arc3-jad-exp081-hic16.ipynb"
    for k in ("id", "title", "code_file"):
        a.pop(k), b.pop(k)
    assert a == b
    assert "scottmahony" not in META.read_text()


def test_kaggle_ops_treats_it_as_a_gpu_kernel():
    assert kaggle_ops.is_gpu_family("calamitychasm/arc3-jad-exp081-hic16")
