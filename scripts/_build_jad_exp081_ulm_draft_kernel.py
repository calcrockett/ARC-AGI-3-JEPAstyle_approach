#!/usr/bin/env python
"""Build the serving swap of the bundle: C+U+LM (calamitychasm/arc3-jad-exp081-ulm) serving OUR fine-tuned MTP draft
(arm C+D's edit), as calamitychasm/arc3-jad-exp081-ulm-draft (experiments/stage7_milestone2_improvements.md 15.9).

    python scripts/_build_jad_exp081_ulm_draft_kernel.py --draft-manifest FILE [--expect-sha SHA256]
        # writes kaggle_submission_jad_exp081_ulm_draft/notebook/ and copies FILE to
        # kaggle_submission_jad_exp081_ulm_draft/arc3-draft-manifest.json
    python scripts/_build_jad_exp081_ulm_draft_kernel.py --draft-manifest FILE --out DIR    # elsewhere (tests)

FILE is session A's manifest (today: kaggle_submission_jad_exp081_draft/arc3-draft-manifest.json, sha256
6f408d58fd81...). The guard is arm C+D's own (_build_jad_exp081_draft_kernel.check_manifest): no default, no placeholder,
the exporter's format only, ``--expect-sha`` checked.

Composed, not copied, the way the -ulm builder composes C+U and C+LM:

  1. scripts/_build_jad_exp081_ulm_kernel.build_notebook() makes the bundle from the sha-pinned arm C v2 notebook.
  2. scripts/_build_jad_exp081_draft_kernel.build() makes arm C+D (their vendored builder + their --draft) into a temp
     dir from FILE. Its diff against arm C v2 must be exactly cell 0 (the fork note) and, in the setup cell, the stock
     ``DRAFT_MODEL_DIR = _ours_input(...albucino...drafter...)`` line replaced by the ``# >>> ours (--draft)`` block;
     its metadata must differ from arm C's only in identity, ``model_sources`` (stock drafter dropped) and
     ``kernel_sources`` (the session kernel). Anything else stops the build.
  3. Those C+D edits on top of the bundle: cell 0 := C+D's cell 0; the bundle's setup cell with the same one-line
     replacement (the line occurs once there; C+U's setup edits, the env line and the 8-patch loop, are untouched);
     metadata := the bundle's with C+D's model_sources / kernel_sources and the new identity.

Every other cell is byte-identical to the bundle's; cell 0 is byte-identical to C+D's; the setup cell is the one cell in
neither notebook (bundle's setup cell + C+D's block, checked both ways). Draft serving is a cell-4 path change only:
the six patches, C+U's two patches, the install cells (D', C+U, LM), the run cell and its three counter dumps, REAP-448,
acceptance 0.5/0.5, 14 streams, 2400 s grace and the 25 x 121 min Save & Run shape are the bundle's.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_jad_exp081_draft_kernel as D  # noqa: E402  (C+D: check_manifest, build, SLUG)
import _build_jad_exp081_u_kernel as U  # noqa: E402  (ARM_C_NB / ARM_C_META pins, _dump)
import _build_jad_exp081_ulm_kernel as B  # noqa: E402  (the bundle: build_notebook)

KDIR = ROOT / "kaggle_submission_jad_exp081_ulm_draft"
OUT = KDIR / "notebook"
OWNER = "calamitychasm"
SLUG = "arc3-jad-exp081-ulm-draft"
BLOCK_BEGIN = "# >>> ours (--draft)"
BLOCK_END = "# <<< ours (--draft)\n"
STOCK_DRAFT = "DRAFT_MODEL_DIR   = _ours_input('/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/"
IDENTITY = ("id", "title", "code_file")
DRAFT_META_KEYS = ("model_sources", "kernel_sources")


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _setup_index(cells: list[dict]) -> int:
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "setup_env = {" in _src(c)]
    if len(hits) != 1:
        raise SystemExit(f"expected one setup cell, found {hits}")
    return hits[0]


def draft_edit(arm_c_nb: dict, draft_nb: dict) -> tuple[dict, str, str]:
    """C+D's edit as (cell 0, the stock drafter line, the --draft block); SystemExit unless C+D is exactly that."""
    a, d = arm_c_nb["cells"], draft_nb["cells"]
    if len(a) != len(d) or {k: v for k, v in arm_c_nb.items() if k != "cells"} != \
            {k: v for k, v in draft_nb.items() if k != "cells"}:
        raise SystemExit("C+D adds/removes cells or changes notebook metadata")
    setup = _setup_index(a)
    differ = [i for i, (x, y) in enumerate(zip(a, d)) if x != y]
    if differ != [0, setup] or a[0]["cell_type"] != "markdown":
        raise SystemExit(f"C+D differs from arm C in cells {differ}, expected [0, {setup}]")
    if {k: v for k, v in a[setup].items() if k != "source"} != {k: v for k, v in d[setup].items() if k != "source"}:
        raise SystemExit("C+D changes the setup cell's non-source fields")
    old, new = _src(a[setup]), _src(d[setup])
    lines = [ln for ln in old.splitlines(keepends=True) if ln.startswith(STOCK_DRAFT)]
    if len(lines) != 1 or old.count(lines[0]) != 1:
        raise SystemExit(f"arm C's setup cell: stock drafter line found {len(lines)} times")
    if new.count(BLOCK_BEGIN) != 1 or new.count(BLOCK_END) != 1:
        raise SystemExit("C+D's setup cell: the --draft block markers not found exactly once")
    begin, end = new.index(BLOCK_BEGIN), new.index(BLOCK_END) + len(BLOCK_END)
    block = new[begin:end]
    if new[:begin] + lines[0] + new[end:] != old or STOCK_DRAFT in new:
        raise SystemExit("C+D's setup cell is not arm C's with the stock drafter line replaced by the block")
    return d[0], lines[0], block


def draft_metadata(arm_c_meta: dict, draft_meta: dict) -> dict:
    """C+D's metadata edit ({model_sources, kernel_sources}); SystemExit if it changes anything else."""
    rest = lambda m: {k: v for k, v in m.items() if k not in IDENTITY + DRAFT_META_KEYS}  # noqa: E731
    if list(arm_c_meta) != list(draft_meta) or rest(arm_c_meta) != rest(draft_meta):
        raise SystemExit("C+D's metadata differs from arm C's beyond identity / model_sources / kernel_sources")
    if draft_meta["kernel_sources"] != [D.SESSION_KERNEL] or any("drafter" in m for m in draft_meta["model_sources"]):
        raise SystemExit(f"C+D's metadata does not swap the stock drafter for {D.SESSION_KERNEL}")
    return {k: draft_meta[k] for k in DRAFT_META_KEYS}


def build_notebook(manifest: Path | None, *, expect_sha: str | None = None) -> tuple[str, str, dict]:
    info = D.check_manifest(manifest, expect_sha)  # the C+D guard, before anything is built
    nb_text, meta_text = B.build_notebook()  # the bundle, from the sha-pinned arm C v2 notebook
    with tempfile.TemporaryDirectory() as tmp:
        D.build(manifest, Path(tmp), expect_sha=expect_sha, no_apply_check=True)  # C+D, same manifest
        draft_nb = json.loads((Path(tmp) / f"{D.SLUG}.ipynb").read_text(encoding="utf-8"))
        draft_meta = json.loads((Path(tmp) / "kernel-metadata.json").read_text())
    arm_c_nb = json.loads(U.ARM_C_NB.read_text(encoding="utf-8"))
    arm_c_meta = json.loads(U.ARM_C_META.read_text())
    cell0, stock_line, block = draft_edit(arm_c_nb, draft_nb)
    nb = json.loads(nb_text)
    cells = nb["cells"]
    if cells[0] != arm_c_nb["cells"][0]:
        raise SystemExit("the bundle's cell 0 is not arm C's")
    setup = _setup_index(cells)
    body = _src(cells[setup])
    if body.count(stock_line) != 1 or BLOCK_BEGIN in body:
        raise SystemExit("the bundle's setup cell: stock drafter line not found exactly once (or already drafted)")
    cells[0] = cell0
    cells[setup]["source"] = body.replace(stock_line, block).splitlines(True)
    meta = json.loads(meta_text)
    if meta["id"] != f"{OWNER}/{B.SLUG}":
        raise SystemExit(f"unexpected bundle kernel id {meta['id']!r}")
    meta.update(draft_metadata(arm_c_meta, draft_meta))
    meta.update(id=f"{OWNER}/{SLUG}", title=SLUG, code_file=f"{SLUG}.ipynb")
    return U._dump(nb), json.dumps(meta, indent=1) + "\n", info


def build(manifest: Path | None, out: Path = OUT, *, expect_sha: str | None = None) -> dict:
    nb_text, meta_text, info = build_notebook(manifest, expect_sha=expect_sha)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{SLUG}.ipynb").write_text(nb_text, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(meta_text, encoding="utf-8")
    if Path(out).resolve() == OUT.resolve():
        shutil.copyfile(manifest, KDIR / "arc3-draft-manifest.json")
    print(f"built {out / (SLUG + '.ipynb')} (manifest sha256 {info['sha256'][:12]}, dense {info['dense_sha256'][:12]})")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--draft-manifest", type=Path, default=None, metavar="FILE",
                    help="session A's mtp-draft/arc3-draft-manifest.json, as pulled (required)")
    ap.add_argument("--expect-sha", default=None, metavar="SHA256", help="the manifest sha256 session A printed")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    build(a.draft_manifest, a.out, expect_sha=a.expect_sha)


if __name__ == "__main__":
    main()
