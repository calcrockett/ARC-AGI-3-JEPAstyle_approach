"""Arm C+D (calamitychasm/arc3-jad-exp081-draft): arm C built by their vendored builder plus their --draft with OUR
session-A manifest. The builder refuses to write anything without a real manifest; with a fake one (temp dir only)
the notebook differs from arm C v2 only in the drafter cell (cell 4's DRAFT_MODEL_DIR) and markdown cell 0, and the
metadata only in the slug, the stock-draft model source and the session kernel source."""

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
