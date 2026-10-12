"""Arm C+D (calamitychasm/arc3-jad-exp081-draft): arm C built by their vendored builder plus their --draft with OUR
session-A manifest. The builder refuses to write anything without a real manifest; with a fake one (temp dir only)
the notebook differs from arm C v2 only in the drafter cell (cell 4's DRAFT_MODEL_DIR) and markdown cell 0, and the
metadata only in the slug, the stock-draft model source and the session kernel source.

The committed notebook was built from session A v1's real manifest (2026-10-12, drafter gate PASS) and is pinned: it
rebuilds byte-identically from the committed manifest, and its draft cell checks that manifest's and dense shard's
sha256 in either mount layout of the session kernel's output."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_jad_exp081_draft_kernel as D  # noqa: E402
import kaggle_ops  # noqa: E402

ARM_C = ROOT / "kaggle_submission_jad_exp081" / "notebook"
ARM_C_NB_SHA = "d108fa761fb66dfef828df668de947841b857634756568954541d4f77151a26a"
# The committed C+D build (session A v1's manifest; see experiments/stage7_milestone2_improvements.md section 15.8).
NB_SHA = "0e2a7599a5b3958dd5071210c0b5b8950df27961e88de0fe4bdab241866d4bf6"
META_SHA = "ca4d0266c0ce0e3c79b57a522f0f239b7e930e2621422a39f112a74b09817bdd"
MANIFEST_SHA = "6f408d58fd813bce415edc55802f174ebac73fef292455ac388104c404dc1d45"
DENSE_SHA = "b677e0c4234c6344fe29f147a972b192de241e557e83a12452409da13967a985"
DRAFT_LINE = "DRAFT_MODEL_DIR   = _ours_input('/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/"


def _manifest(tmp: Path, **change) -> Path:
    man = {"format": "arc3-mtp-draft", "source": "/kaggle/input/x/runtime/mtp-int4-g32",
           "trained_file": "/kaggle/working/mtp-train/trained-dense.safetensors", "trained_sha256": "0" * 64,
           "replaced": ["mtp.fc.weight", "mtp.pre_fc_norm_hidden.weight"], "copied_extra": [],
           "files": {"config.json": "1" * 64, "model.safetensors.index.json": "3" * 64,
                     "mtp-dense.safetensors": "2" * 64}}
    man.update(change)
    path = tmp / "arc3-draft-manifest.json"
    path.write_text(json.dumps(man, indent=1) + "\n")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("draft")
    manifest = _manifest(tmp)
    out = tmp / "nb"
    info = D.build(manifest, out, expect_sha=_sha(manifest), no_apply_check=True)
    return manifest, out, info


@pytest.mark.parametrize("change, why", [
    ({"format": "something-else"}, "format"),
    ({"replaced": []}, "replaced"),
    ({"files": {"config.json": "1" * 64}}, "mtp-dense.safetensors"),
    ({"files": {"mtp-dense.safetensors": "2" * 64}}, "frozen"),
    ({"not_for_push": True}, "placeholder"),
    ({"placeholder": "fill after session A"}, "placeholder"),
])
def test_refuses_a_placeholder_or_foreign_manifest(tmp_path, change, why):
    with pytest.raises(SystemExit, match=why):
        D.build(_manifest(tmp_path, **change), tmp_path / "nb", no_apply_check=True)
    assert not (tmp_path / "nb").exists()


def test_refuses_without_a_manifest_or_with_the_wrong_sha(tmp_path):
    with pytest.raises(SystemExit, match="required"):
        D.build(None, tmp_path / "nb", no_apply_check=True)
    with pytest.raises(SystemExit, match="unreadable"):
        D.build(tmp_path / "missing.json", tmp_path / "nb", no_apply_check=True)
    with pytest.raises(SystemExit, match="the session printed"):
        D.build(_manifest(tmp_path), tmp_path / "nb", expect_sha="f" * 64, no_apply_check=True)
    assert not (tmp_path / "nb").exists()


def test_no_notebook_is_committed_without_a_real_manifest():
    """The C+D notebook is committed only together with the manifest it was built from (and then rebuilds exactly)."""
    if not (D.OUT / f"{D.SLUG}.ipynb").exists():
        assert not (D.KDIR / "arc3-draft-manifest.json").exists()
        return
    manifest = D.KDIR / "arc3-draft-manifest.json"
    assert manifest.is_file()
    D.check_manifest(manifest)


def test_committed_build_is_pinned_and_rebuilds_byte_identically(tmp_path):
    """Session A v1's manifest (2998 bytes, json.dumps(obj, indent=1) + newline), the notebook built from it, and its
    metadata are pinned; a rebuild from the committed manifest gives the same bytes."""
    manifest = D.KDIR / "arc3-draft-manifest.json"
    assert _sha(manifest) == MANIFEST_SHA and manifest.stat().st_size == 2998
    man = json.loads(manifest.read_text())
    assert manifest.read_text() == json.dumps(man, indent=1) + "\n"
    assert man["files"][D.DENSE] == DENSE_SHA and len(man["replaced"]) == 26 and len(man["files"]) == 12
    info = D.check_manifest(manifest, MANIFEST_SHA)
    assert info == {"sha256": MANIFEST_SHA, "dense_sha256": DENSE_SHA, "replaced": 26}
    assert _sha(D.OUT / f"{D.SLUG}.ipynb") == NB_SHA
    assert _sha(D.OUT / "kernel-metadata.json") == META_SHA
    D.build(manifest, tmp_path / "nb", expect_sha=MANIFEST_SHA, no_apply_check=True)
    assert _sha(tmp_path / "nb" / f"{D.SLUG}.ipynb") == NB_SHA
    assert _sha(tmp_path / "nb" / "kernel-metadata.json") == META_SHA


def test_committed_notebook_checks_the_real_draft_in_both_mount_layouts():
    """The committed notebook differs from arm C v2 only in cell 0 and the drafter line; the draft block waits for the
    session's mtp-draft/ under /kaggle/input/notebooks/<owner>/<slug>/ or /kaggle/input/<slug>/ and checks the real
    manifest and dense sha256 before anything reads the draft."""
    a, b = _cells(ARM_C / "arc3-jad-exp081.ipynb"), _cells(D.OUT / f"{D.SLUG}.ipynb")
    setup = next(i for i, c in enumerate(a) if "setup_env = {" in "".join(c["source"]))
    assert [i for i, (x, y) in enumerate(zip(a, b)) if x != y] == [0, setup] and len(a) == len(b)
    new = "".join(b[setup]["source"])
    block = new[new.index("# >>> ours (--draft)"):new.index("# <<< ours (--draft)")]
    call = next(ln for ln in block.splitlines() if ln.startswith("DRAFT_MODEL_DIR   = _ours_draft_dir("))
    assert call == ("DRAFT_MODEL_DIR   = _ours_draft_dir(['/kaggle/input/notebooks/calamitychasm/arc3-mtp-session-a/"
                    "mtp-draft', '/kaggle/input/arc3-mtp-session-a/mtp-draft'], "
                    f"{MANIFEST_SHA!r}, {DENSE_SHA!r})")
    assert "'arc3-draft-manifest.json'" in block and "'mtp-dense.safetensors'" in block and "timeout_s=600.0" in block
    # the wait and the sha check run before the precache thread and the draft view read DRAFT_MODEL_DIR
    assert new.index("DRAFT_MODEL_DIR   = _ours_draft_dir(") < new.index("_ours_need = [")
    assert "albucino-qwen3-8-flash-next-drafter/transformers" not in new.split("# >>> ours (--draft)")[1]


def test_differs_from_arm_c_only_in_the_drafter_cells(built):
    manifest, out, info = built
    assert _sha(ARM_C / "arc3-jad-exp081.ipynb") == ARM_C_NB_SHA
    a, b = _cells(ARM_C / "arc3-jad-exp081.ipynb"), _cells(out / f"{D.SLUG}.ipynb")
    assert len(a) == len(b)
    differ = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    setup = next(i for i, c in enumerate(a) if "setup_env = {" in "".join(c["source"]))
    assert differ == [0, setup]
    old, new = "".join(a[setup]["source"]), "".join(b[setup]["source"])
    line = next(ln for ln in old.splitlines(keepends=True) if ln.startswith(DRAFT_LINE))
    begin, end = new.index("# >>> ours (--draft)"), new.index("# <<< ours (--draft)\n") + len("# <<< ours (--draft)\n")
    assert new[:begin] + line + new[end:] == old  # the stock-draft line, replaced by the draft block, nothing else
    block = new[begin:end]
    assert info["sha256"] == _sha(manifest) and repr(info["sha256"]) in block and repr("2" * 64) in block
    assert "'/kaggle/input/notebooks/calamitychasm/arc3-mtp-session-a/mtp-draft'" in block
    assert "ours: MTP draft {path} after" in block and "timeout_s=600.0" in block
    assert "calamitychasm/arc3-mtp-session-a" in "".join(b[0]["source"])


def test_metadata_swaps_the_stock_draft_for_the_session_kernel(built):
    _, out, _ = built
    arm_c = json.loads((ARM_C / "kernel-metadata.json").read_text())
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["id"] == "calamitychasm/arc3-jad-exp081-draft" and meta["title"] == "arc3-jad-exp081-draft"
    assert meta["code_file"] == "arc3-jad-exp081-draft.ipynb"
    assert meta["kernel_sources"] == ["calamitychasm/arc3-mtp-session-a"]
    assert meta["model_sources"] == [m for m in arm_c["model_sources"] if "drafter" not in m]
    same = {k: v for k, v in arm_c.items() if k not in ("id", "title", "code_file", "model_sources", "kernel_sources")}
    assert {k: meta[k] for k in same} == same  # machine shape, image, datasets, competition, private, GPU
    assert "scottmahony" not in (out / "kernel-metadata.json").read_text()
    assert kaggle_ops.is_gpu_family(meta["id"])


@pytest.mark.skipif(not (D.C.DEFAULT_HIS_REPO / "ARC3-Inference").is_dir(),
                    reason="needs Franzen's repo for the apply check")
def test_apply_check_gives_the_same_bytes(built, tmp_path):
    manifest, out, _ = built
    D.build(manifest, tmp_path / "nb", his_repo=D.C.DEFAULT_HIS_REPO)
    assert _sha(tmp_path / "nb" / f"{D.SLUG}.ipynb") == _sha(out / f"{D.SLUG}.ipynb")
