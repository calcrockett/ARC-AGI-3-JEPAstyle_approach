#!/usr/bin/env python
"""Build the bundle C+UL: arm C (calamitychasm/arc3-jad-exp081 v2, JustAdev742's exp-081 fork) plus C+U's two source
patches and C+LM's solved-level memory, as one kernel (calamitychasm/arc3-jad-exp081-ulm).

    python scripts/_build_jad_exp081_ulm_kernel.py              # writes kaggle_submission_jad_exp081_ulm/notebook/
    python scripts/_build_jad_exp081_ulm_kernel.py --out DIR    # elsewhere (the test rebuilds into a tmp dir)

Composed, not copied (experiments/stage7_milestone2_improvements.md sections 16.5 and 16.8):

  1. scripts/_build_jad_exp081_u_kernel.build_notebook() makes C+U from the COMMITTED arm C notebook (sha256-pinned
     there; a scheduled submit pins its version 2, so it is read, never rewritten): the ours-10 cell (OURS_UNTRIED=i1)
     and our stale-latch-fix cell after arm C's last patch cell, the setup cell's env line and 8-patch apply loop
     (``our harness patches applied successfully: 8``), the C+U install cell before the run cell, and the C+U counter
     dump after bm.run.
  2. On top of that, exactly C+LM's two edits, from the same generator C+LM uses
     (_build_m2_level_memory_kernel.install_cell / RUN_ANCHOR / RUN_DUMP): the level-memory install cell, inserted right
     before the run cell (so after D''s install cell and the C+U install cell), and the level_memory_summary.json dump,
     which lands right after bm.run, ahead of the C+U dump.

Every added cell is byte-identical to the cell the single-variant builders add. kernel-metadata.json: arm C's, with
id/title/code_file for the new slug. Everything else is byte-identical to arm C v2: cell 0, D', REAP-448, acceptance
0.5, 14 streams / Mamba 84, the ARC map, the six exp-081 patches, the 2400 s first-request grace and the full-length
Save & Run shape (25 x 121 min). The -u and -lm notebooks are not touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_jad_exp081_lm_kernel as LM  # noqa: E402  (C+LM: ARM_C_* pins, RUN_CELL_START)
import _build_jad_exp081_u_kernel as U  # noqa: E402  (C+U: build_notebook, install_cell, RUN_DUMP)
import _build_m2_level_memory_kernel as LMB  # noqa: E402  (install_cell, RUN_ANCHOR, RUN_DUMP)

OWNER = "calamitychasm"
SLUG = "arc3-jad-exp081-ulm"
OUT = ROOT / "kaggle_submission_jad_exp081_ulm" / "notebook"
RUN_CELL_START = U.RUN_CELL_START
LM_INSTALL_HEAD = "# [calamitychasm] SOLVED-LEVEL MEMORY"


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _check_composable() -> None:
    """Both variants are defined on the same arm C notebook and anchor on the same bm.run line."""
    if (U.ARM_C_NB, U.ARM_C_NB_SHA, U.ARM_C_META) != (LM.ARM_C_NB, LM.ARM_C_NB_SHA, LM.ARM_C_META):
        raise SystemExit("C+U and C+LM are not built from the same arm C notebook")
    if U.RUN_ANCHOR != LMB.RUN_ANCHOR or U.RUN_CELL_START != LM.RUN_CELL_START:
        raise SystemExit("C+U and C+LM anchor on different run-cell lines")
    if not (U.RUN_DUMP.startswith(U.RUN_ANCHOR) and LMB.RUN_DUMP.startswith(LMB.RUN_ANCHOR)):
        raise SystemExit("a counter dump does not start with the bm.run anchor")


def build_notebook() -> tuple[str, str]:
    _check_composable()
    nb_text, meta_text = U.build_notebook()  # C+U, from the sha-pinned arm C notebook
    nb = json.loads(nb_text)
    if U._dump(nb) != nb_text:
        raise SystemExit("C+U notebook does not round-trip through json.dumps(indent=1, ensure_ascii=False)")
    cells = nb["cells"]
    runs = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and _src(c).startswith(RUN_CELL_START)]
    if len(runs) != 1:
        raise SystemExit(f"expected one run cell, found {runs}")
    run = runs[0]
    if cells[run - 1] != U.install_cell() or "install_d(_fp_ta, _fp_solver)" not in _src(cells[run - 2]):
        raise SystemExit("the run cell is not preceded by D''s install cell and the C+U install cell")
    if any(_src(c).startswith(LM_INSTALL_HEAD) for c in cells):
        raise SystemExit("C+U already carries a level-memory cell")
    body = _src(cells[run])
    if body.count(LMB.RUN_ANCHOR) != 1 or body.count(U.RUN_DUMP) != 1:
        raise SystemExit("run cell: the bm.run anchor / C+U dump not found exactly once")
    # C+LM's edits, the same way its builder makes them: the dump right after bm.run, the install cell before the run
    cells[run]["source"] = body.replace(LMB.RUN_ANCHOR, LMB.RUN_DUMP).splitlines(True)
    cells.insert(run, LMB.install_cell())
    meta = json.loads(meta_text)
    if meta["id"] != f"{OWNER}/{U.SLUG}":
        raise SystemExit(f"unexpected C+U kernel id {meta['id']!r}")
    meta.update(id=f"{OWNER}/{SLUG}", title=SLUG, code_file=f"{SLUG}.ipynb")
    return U._dump(nb), json.dumps(meta, indent=1) + "\n"


def build(out: Path = OUT) -> None:
    nb_text, meta_text = build_notebook()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{SLUG}.ipynb").write_text(nb_text, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(meta_text, encoding="utf-8")
    print(f"built {out / (SLUG + '.ipynb')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
