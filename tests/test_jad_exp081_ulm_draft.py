"""Bundle + draft (calamitychasm/arc3-jad-exp081-ulm-draft): the bundle C+UL serving our session-A MTP draft, i.e. the
bundle notebook with exactly arm C+D's edits on top (markdown cell 0, the setup cell's ``# >>> ours (--draft)`` block in
place of the stock DRAFT_MODEL_DIR line, the metadata's model-source swap plus kernel_sources). Pins the committed build
and its manifest; checks the C+D guard; a byte-identical rebuild; that the cell-wise composition is bundle U C+D with
nothing else changed versus arm C v2; and that the interaction probe of the bundle gives the same findings on it."""

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

import _build_jad_exp081_draft_kernel as D  # noqa: E402
import _build_jad_exp081_u_kernel as U  # noqa: E402
import _build_jad_exp081_ulm_draft_kernel as BD  # noqa: E402
import _build_jad_exp081_ulm_kernel as B  # noqa: E402
import kaggle_ops  # noqa: E402

NB = BD.OUT / "arc3-jad-exp081-ulm-draft.ipynb"
META = BD.OUT / "kernel-metadata.json"
MANIFEST = BD.KDIR / "arc3-draft-manifest.json"
NB_SHA = "e2814c700ce302bd32b6ddb815a7d77cb24d9bed20028679835e2e9d4493a55e"
META_SHA = "1342c86cac608faf714487524fed079d16230db0c48d336614b3480d748decae"
MANIFEST_SHA = "6f408d58fd813bce415edc55802f174ebac73fef292455ac388104c404dc1d45"  # session A v1 (= C+D's)
DENSE_SHA = "b677e0c4234c6344fe29f147a972b192de241e557e83a12452409da13967a985"
ULM_NB = B.OUT / "arc3-jad-exp081-ulm.ipynb"
DRAFT_NB = D.OUT / "arc3-jad-exp081-draft.ipynb"
HAS_HIS = (D.C.DEFAULT_HIS_REPO / "ARC3-Inference").is_dir()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _setup(cells: list[dict]) -> int:
    return next(i for i, c in enumerate(cells) if "setup_env = {" in _src(c))


def _lines(text: str) -> Counter:
    return Counter(text.splitlines())


def test_committed_files_are_pinned():
    assert _sha(NB) == NB_SHA and _sha(META) == META_SHA
    assert _sha(MANIFEST) == MANIFEST_SHA == _sha(D.KDIR / "arc3-draft-manifest.json")
    assert NB.stat().st_size < 900_000  # Kaggle refuses notebooks near 1 MB


def test_inputs_are_the_pinned_bundle_and_draft_builds():
    import test_jad_exp081_draft as TD  # noqa: PLC0415
    import test_jad_exp081_ulm as TB  # noqa: PLC0415
    assert _sha(ULM_NB) == TB.NB_SHA and _sha(B.OUT / "kernel-metadata.json") == TB.META_SHA
    assert _sha(DRAFT_NB) == TD.NB_SHA and _sha(D.OUT / "kernel-metadata.json") == TD.META_SHA
    assert TD.MANIFEST_SHA == MANIFEST_SHA and TD.DENSE_SHA == DENSE_SHA


def test_rebuild_is_byte_identical_and_idempotent(tmp_path):
    for _ in range(2):
        info = BD.build(MANIFEST, tmp_path, expect_sha=MANIFEST_SHA)
    assert info == {"sha256": MANIFEST_SHA, "dense_sha256": DENSE_SHA, "replaced": 26}
    assert _sha(tmp_path / NB.name) == NB_SHA and _sha(tmp_path / META.name) == META_SHA
    assert not (tmp_path / "arc3-draft-manifest.json").exists()  # the manifest is copied only next to OUT


def test_refuses_without_a_real_manifest(tmp_path):
    with pytest.raises(SystemExit, match="required"):
        BD.build(None, tmp_path / "nb")
    with pytest.raises(SystemExit, match="the session printed"):
        BD.build(MANIFEST, tmp_path / "nb", expect_sha="f" * 64)
    fake = json.loads(MANIFEST.read_text())
    fake["not_for_push"] = True
    (tmp_path / "m.json").write_text(json.dumps(fake, indent=1) + "\n")
    with pytest.raises(SystemExit, match="placeholder"):
        BD.build(tmp_path / "m.json", tmp_path / "nb")
    assert not (tmp_path / "nb").exists()


def test_composition_is_the_bundle_with_exactly_the_c_plus_d_edits():
    a, u, d, n = _nb(U.ARM_C_NB), _nb(ULM_NB), _nb(DRAFT_NB), _nb(NB)
    assert {k: v for k, v in n.items() if k != "cells"} == {k: v for k, v in u.items() if k != "cells"}
    ca, cu, cd, cn = a["cells"], u["cells"], d["cells"], n["cells"]
    sa, sd = _setup(ca), _setup(cd)
    su, sn = _setup(cu), _setup(cn)
    assert len(cn) == len(cu) == len(ca) + 4 and len(cd) == len(ca) and sd == sa and sn == su == sa + 2
    # C+D vs arm C: cell 0 and the setup cell only; the bundle vs arm C: cell 0 untouched
    assert [i for i, (x, y) in enumerate(zip(ca, cd)) if x != y] == [0, sa]
    assert cu[0] == ca[0]
    # the new notebook vs the bundle: cell 0 and the setup cell only
    assert [i for i, (x, y) in enumerate(zip(cu, cn)) if x != y] == [0, su]
    assert cn[0] == cd[0]  # C+D's markdown cell 0, byte for byte
    # the setup cell: the bundle's with the one stock drafter line replaced by C+D's block, the same edit C+D makes
    old_c, new_d, old_u, new_n = (_src(c) for c in (ca[sa], cd[sd], cu[su], cn[sn]))
    line = next(ln for ln in old_c.splitlines(True) if ln.startswith(BD.STOCK_DRAFT))
    block = new_d[new_d.index(BD.BLOCK_BEGIN):new_d.index(BD.BLOCK_END) + len(BD.BLOCK_END)]
    assert old_c.replace(line, block) == new_d and old_u.count(line) == 1
    assert new_n == old_u.replace(line, block)
    assert {k: v for k, v in cn[sn].items() if k != "source"} == {k: v for k, v in cu[su].items() if k != "source"}
    # line-level: setup changes vs arm C = the bundle's (env line, 8-patch loop) + C+D's (the block), nothing else
    assert _lines(new_n) - _lines(old_c) == (_lines(old_u) - _lines(old_c)) + (_lines(new_d) - _lines(old_c))
    assert _lines(old_c) - _lines(new_n) == (_lines(old_c) - _lines(old_u)) + (_lines(old_c) - _lines(new_d))
    # every cell is the bundle's or C+D's, except the setup cell (the union of the two setup cells)
    pool = [_src(c) for c in cu + cd]
    assert [i for i, c in enumerate(cn) if _src(c) not in pool] == [sn]


def test_draft_block_is_the_real_session_a_check():
    new = _src(_nb(NB)["cells"][_setup(_nb(NB)["cells"])])
    block = new[new.index(BD.BLOCK_BEGIN):new.index(BD.BLOCK_END)]
    call = next(ln for ln in block.splitlines() if ln.startswith("DRAFT_MODEL_DIR   = _ours_draft_dir("))
    assert call == ("DRAFT_MODEL_DIR   = _ours_draft_dir(['/kaggle/input/notebooks/calamitychasm/arc3-mtp-session-a/"
                    "mtp-draft', '/kaggle/input/arc3-mtp-session-a/mtp-draft'], "
                    f"{MANIFEST_SHA!r}, {DENSE_SHA!r})")
    assert "ours: MTP draft {path} after" in block and "timeout_s=600.0" in block
    assert new.index("DRAFT_MODEL_DIR   = _ours_draft_dir(") < new.index("_ours_need = [")
    assert "albucino-qwen3-8-flash-next-drafter/transformers" not in new
    # the bundle's own setup edits are still there
    assert "    'OURS_UNTRIED': 'i1',  # ours (--env-add)\n" in new
    assert "print('our harness patches applied successfully: 8')" in new
    text = "\n".join(_src(c) for c in _nb(NB)["cells"])
    for marker in ("LEVEL_MEMORY installed", "LEVEL_MEMORY summary", "UNTRIED_NOTICE summary", "STALE_LATCH_FIX summary",
                   "#OURS_FORM ok version=d_prime", "bm.solver.max_runtime_s_per_game = 121.0*60  # ours (--full25)"):
        assert marker in text, marker


def test_metadata_is_the_bundles_with_c_plus_d_sources():
    arm_c, ulm = json.loads(U.ARM_C_META.read_text()), json.loads((B.OUT / "kernel-metadata.json").read_text())
    draft, meta = json.loads((D.OUT / "kernel-metadata.json").read_text()), json.loads(META.read_text())
    assert meta["id"] == "calamitychasm/arc3-jad-exp081-ulm-draft" and meta["title"] == "arc3-jad-exp081-ulm-draft"
    assert meta["code_file"] == "arc3-jad-exp081-ulm-draft.ipynb"
    assert meta["kernel_sources"] == draft["kernel_sources"] == ["calamitychasm/arc3-mtp-session-a"]
    assert meta["model_sources"] == draft["model_sources"] == [m for m in arm_c["model_sources"] if "drafter" not in m]
    ident = ("id", "title", "code_file")
    strip = lambda m, extra=(): {k: v for k, v in m.items() if k not in ident + extra}  # noqa: E731
    assert list(meta) == list(ulm) == list(draft)
    assert strip(meta) == strip(draft)  # the bundle's metadata is arm C's, so: C+D's except identity
    assert strip(meta, ("model_sources", "kernel_sources")) == strip(ulm, ("model_sources", "kernel_sources"))
    assert "scottmahony" not in META.read_text()
    assert kaggle_ops.is_gpu_family(meta["id"])


@pytest.mark.skipif(not HAS_HIS, reason="needs Franzen's repo to build the exp-081 harness tree")
def test_bundle_probe_gives_the_same_findings(tmp_path):
    """The draft edit is serving-only (a cell-4 path), so the bundle's interaction probe (tests/jad_ulm_probe.py:
    8-patch tree, setup_env, both install cells, prompts, trims, latch, counters) must find exactly the same on both."""
    outs = []
    for name, nb in (("bundle", ULM_NB), ("draft", NB)):
        run = subprocess.run([sys.executable, str(ROOT / "tests" / "jad_ulm_probe.py"), str(nb),
                              str(tmp_path / name), str(D.C.DEFAULT_HIS_REPO)],
                             capture_output=True, text=True, cwd=ROOT, timeout=900)
        assert run.returncode == 0, run.stderr[-3000:]
        outs.append(json.loads(run.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1]
    assert outs[1]["install_order"] == ["cu", "lm"] and len(outs[1]["patches"]) == 8
