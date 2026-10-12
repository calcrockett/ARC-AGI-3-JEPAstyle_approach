"""arc3-m2-turbo-lossless-tail-hic16: arm B (turbo-lossless-tail) + 32 GB host KV tier + 16 streams.
Pins the committed notebook; the host-tier mechanics are tested in test_m2_turbo_tail_hic.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_m2_level_memory_kernel as B  # noqa: E402
from _check_notebook_cell import check_notebook  # noqa: E402

VARIANTS = tuple(v for v in B.TURBO_LOSSLESS if v != "s14") + ("tail", "hic32", "s16")
SLUG, DIRNAME = "arc3-m2-turbo-lossless-tail-hic16", "kaggle_submission_m2_turbo_lossless_tail_hic16"
COMMITTED = ROOT / DIRNAME / "notebook" / f"{SLUG}.ipynb"
ARM_B = ROOT / "kaggle_submission_m2_turbo_lossless_tail" / "notebook" / "arc3-m2-turbo-lossless-tail.ipynb"
ARM_A_HIC = ROOT / "kaggle_submission_m2_turbo_tail_hic16" / "notebook" / "arc3-m2-turbo-tail-hic16.ipynb"


def _cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def test_slug_dir_markers_and_counters():
    assert B.kernel_slug(VARIANTS) == SLUG
    assert B.kernel_dir(VARIANTS).relative_to(B.ROOT).as_posix() == f"{DIRNAME}/notebook"
    markers = B.kernel_markers(VARIANTS)
    for m in ("HICACHE_TIER attached hicache_attached=True", "priority gate active: 16 concurrent streams",
              "STREAMS max_running_requests=16 cuda_graph_bs=", "PRIORITY_TAIL installed", "REAP448 applied kept=448"):
        assert m in markers, m
    assert "SPEC_ACCEPT 0.5" not in markers and "priority gate active: 14 concurrent streams" not in markers
    assert B.kernel_counters(VARIANTS) == B.kernel_counters(B.TURBO_LOSSLESS)


def test_committed_kernel_is_current_and_deterministic(tmp_path, monkeypatch):
    outs = []
    for i in range(2):
        monkeypatch.setattr(B, "ROOT", tmp_path / str(i))
        outs.append(B.build(VARIANTS).read_text(encoding="utf-8"))
    assert outs[0] == outs[1], "builder is not deterministic"
    assert outs[0] == COMMITTED.read_text(encoding="utf-8"), f"{SLUG} is stale: rerun the build"
    assert check_notebook(COMMITTED) == []


def test_metadata_matches_incumbent_except_identity():
    meta = json.loads((COMMITTED.parent / "kernel-metadata.json").read_text())
    inc = json.loads((ROOT / "kaggle_submission_m2_level_memory" / "notebook" / "kernel-metadata.json").read_text())
    assert meta["id"] == f"calamitychasm/{SLUG}" and meta["code_file"] == COMMITTED.name and meta["title"] == SLUG
    for k in ("id", "title", "code_file"):
        meta.pop(k), inc.pop(k)
    assert meta.pop("docker_image_pinning_type") == "original"
    assert meta == inc


def test_differs_from_arm_b_only_in_streams_tier_header_and_run_dump():
    a, b = _cells(ARM_B), _cells(COMMITTED)
    assert len(a) == len(b)
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    setup = next(i for i, c in enumerate(a) if "'ARC3_MAX_ACTIVE_STREAMS': 14," in c)
    launch = next(i for i, c in enumerate(a) if B.L_CFG in c)
    run = next(i for i, c in enumerate(a) if c.startswith("print('Starting benchmark...')"))
    assert diff == [0, setup, launch, run]
    assert b[setup].replace("'ARC3_MAX_ACTIVE_STREAMS': 16,", "'ARC3_MAX_ACTIVE_STREAMS': 14,") == a[setup]
    assert b[run] == a[run].replace(B.REAP_DUMP, B.REAP_DUMP + B.HIC_DUMP)
    for key, new in (("MAXREQ", 16), ("CUDAGRAPH_MAXBS", 16), ("MAMBA_CACHE", 96)):
        assert f"    {key}={new}," in b[launch]
    assert "--enable-hierarchical-cache" in b[launch] and "--hicache-size" in b[launch]


def test_differs_from_turbo_tail_hic16_only_by_the_lossy_acceptance():
    a, b = _cells(ARM_A_HIC), _cells(COMMITTED)
    assert len(a) == len(b)
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    launch = next(i for i, c in enumerate(a) if B.L_CFG in c)
    assert diff == [0, launch]
    assert "SPEC_ACCEPT_SINGLE=0.5" in a[launch] and "SPEC_ACCEPT_SINGLE=1.0" in b[launch]
    assert "SPEC_ACCEPT_ACC=1.0" in b[launch] and "lossy MTP acceptance" not in b[launch]
    assert "print(f\"SPEC_ACCEPT" not in b[launch] and "Variant `acc50`" not in b[0]
