"""Arm C (calamitychasm/arc3-jad-exp081): JustAdev742's exp-081 built by their own vendored builder with exp-081's
flags. Pins the committed notebook, the vendored files, and the exact set of deviations (2400 s grace, our kernel id,
the fork note)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_jad_exp081_kernel as B  # noqa: E402

KDIR = ROOT / "kaggle_submission_jad_exp081"
NB = KDIR / "notebook" / "arc3-jad-exp081.ipynb"
META = KDIR / "notebook" / "kernel-metadata.json"
NB_SHA = "d108fa761fb66dfef828df668de947841b857634756568954541d4f77151a26a"
META_SHA = "edb7c1c108f1ffd6d337414a8aad8cc6043d99c73a03131551687891acbc3cc8"
UPSTREAM = Path("/home/user/ext/JustAdev742_Arc-Agi-3-Kaggle-comp")
PUBLIC_SOURCES = {
    "dataset_sources": ["dfranzen/pennyroyal-v253", "dfranzen/taaf-kaggle-source-bundle-copy"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": ["dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
                      "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"],
    "kernel_sources": [],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def test_committed_notebook_is_pinned():
    assert _sha(NB) == NB_SHA
    assert _sha(META) == META_SHA
    assert NB.stat().st_size < 900_000  # Kaggle refuses notebooks near 1 MB (their lesson 0033)


def test_vendored_files_are_unmodified():
    sums = (KDIR / "vendor" / "SHA256SUMS").read_text().split("\n")
    listed = {}
    for line in filter(None, sums):
        sha, name = line.split(maxsplit=1)
        listed[name.removeprefix("./")] = sha
    on_disk = {str(p.relative_to(KDIR / "vendor")) for p in (KDIR / "vendor").rglob("*")
               if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts}
    assert on_disk == set(listed)
    for name, sha in listed.items():
        assert _sha(KDIR / "vendor" / name) == sha, name
    if (UPSTREAM / "scripts" / "build_franzen_nb.py").is_file():
        for name in listed:  # the clone may be past c7a462b; only check files unchanged there
            up = UPSTREAM / name
            if up.is_file() and name.startswith(("kaggle/", "LICENSE")):
                assert _sha(up) == listed[name], name


def test_rebuild_is_byte_identical(tmp_path):
    B.build(tmp_path, no_apply_check=True)
    assert _sha(tmp_path / "arc3-jad-exp081.ipynb") == NB_SHA
    assert _sha(tmp_path / "kernel-metadata.json") == META_SHA


@pytest.mark.skipif(not (B.DEFAULT_HIS_REPO / "ARC3-Inference").is_dir(),
                    reason="needs Franzen's repo for the apply check")
def test_rebuild_with_apply_check(tmp_path):
    B.build(tmp_path, his_repo=B.DEFAULT_HIS_REPO)
    assert _sha(tmp_path / "arc3-jad-exp081.ipynb") == NB_SHA


def test_metadata_is_ours_private_and_public_inputs_only():
    meta = json.loads(META.read_text())
    assert meta["id"] == "calamitychasm/arc3-jad-exp081"
    assert meta["title"] == "arc3-jad-exp081" and meta["code_file"] == "arc3-jad-exp081.ipynb"
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is False
    assert meta["machine_shape"] == "NvidiaRtxPro6000"
    for key, value in PUBLIC_SOURCES.items():
        assert meta[key] == value, key
    assert "scottmahony" not in META.read_text()


def test_exp081_mechanisms_and_our_single_code_deviation():
    cells = _cells(NB)
    text = "\n".join(cells)
    setup = next(c for c in cells if "setup_env = {" in c)
    # our deviation: the 2400 s first-request grace, and nothing else of ours in code cells
    assert "os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '2400'  # ours (--env)" in setup
    assert "'900'" not in setup
    # exp-081's flags
    for key in ("OURS_BUDGET_METER", "OURS_WIN_LEDGER", "OURS_SEARCH_HELPER", "OURS_LEVEL_MEM", "OURS_PERCEPTION"):
        assert f"'{key}': 1,  # ours (--env-add)" in setup
    assert "'ARC3_MAX_ACTIVE_STREAMS': 14,  # ours (--env)" in setup
    assert "print('our harness patches applied successfully: 6')" in setup
    assert "print('harness patch applied successfully')" in setup
    assert "EXPOSE_RESET" not in setup.split("########## ours (--env-add)")[1]
    for cfg in ("MAXREQ=14", "CUDAGRAPH_MAXBS=14", "MAMBA_CACHE=84", "SPEC_ACCEPT_SINGLE=0.5", "SPEC_ACCEPT_ACC=0.5"):
        assert f"{cfg},  # ours (--cfg)" in text
    assert '{"text_config": {"num_experts": 448}}' in text
    assert "#OURS_FORM ok version=d_prime" in text
    assert "ours: FR-Spec map hot_tokens_64k_arc.pt written" in text
    assert "health_deadline_s=3300" in text  # fail-fast (Save & Run only), c7a462b builder
    assert "--draft" not in text.split("**Our arm", 1)[0]
    assert "arc3-mtp-session-a" not in text  # no private draft
    # every shipped patch is the vendored file (the notebook checks the sha256 of what it writes)
    for p in B.PATCHES:
        assert _sha(KDIR / "vendor" / "kaggle" / "franzen" / "patches" / p) in text, p
    # the Save & Run plays all 25 public games at 121 min per game (their full-length check-run shape)
    assert "bm.solver.max_runtime_s_per_game = 121.0*60  # ours (--full25)" in text
    assert "demo_excluded_games = []  # ours (--full25): all 25 public games" in text
