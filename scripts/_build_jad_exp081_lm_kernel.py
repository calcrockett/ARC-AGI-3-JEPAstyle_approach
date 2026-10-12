#!/usr/bin/env python
"""Build arm C+LM: arm C (calamitychasm/arc3-jad-exp081 v2, JustAdev742's exp-081 fork) plus our solved-level memory.

    python scripts/_build_jad_exp081_lm_kernel.py              # writes kaggle_submission_jad_exp081_lm/notebook/
    python scripts/_build_jad_exp081_lm_kernel.py --out DIR    # elsewhere (the test rebuilds into a tmp dir)

Source: the COMMITTED arm C notebook (kaggle_submission_jad_exp081/notebook/, sha256-pinned below; a scheduled submit
pins its version 2, so it is read, never rewritten). Two edits, nothing else (experiments/stage7_milestone2_improvements.md
section 10):

  (a) one code cell, inserted right before the run cell (after D''s install cell, after every env knob and the
      benchmark unpickle), installs kaggle_submission_milestone2_fork/level_memory/level_memory.py on the patched
      ToolAgent. It is the incumbent's install cell, made by the same function
      (_build_m2_level_memory_kernel.install_cell): it wraps _build_user_prompt / _trim_messages_for_context /
      _ensure_session at runtime and prints ``LEVEL_MEMORY installed``. Their six source patches are untouched; the
      wrappers sit on top of their patched methods (checked on the real exp-081 tree by tests/test_jad_exp081_lm.py).
  (b) the run cell writes level_memory_summary.json and prints ``LEVEL_MEMORY summary`` after bm.run (the
      incumbent's counter dump, same text).

kernel-metadata.json: arm C's, with id/title/code_file for the new slug. Everything else is byte-identical, including
the 2400 s first-request grace (ARC3_HTTP_RETRY_INITIAL_SECONDS) and the full-length Save & Run shape (25 x 121 min).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_m2_level_memory_kernel as LMB  # noqa: E402  (install_cell, RUN_ANCHOR, RUN_DUMP)

ARM_C_DIR = ROOT / "kaggle_submission_jad_exp081" / "notebook"
ARM_C_NB = ARM_C_DIR / "arc3-jad-exp081.ipynb"
ARM_C_META = ARM_C_DIR / "kernel-metadata.json"
ARM_C_NB_SHA = "d108fa761fb66dfef828df668de947841b857634756568954541d4f77151a26a"  # arm C v1/v2 as pushed
OWNER = "calamitychasm"
SLUG = "arc3-jad-exp081-lm"
OUT = ROOT / "kaggle_submission_jad_exp081_lm" / "notebook"
RUN_CELL_START = "print('Starting benchmark...')"


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _dump(nb: dict) -> str:
    # their builder writes json.dumps(nb, indent=1, ensure_ascii=False) with no final newline; matching it keeps every
    # unchanged cell byte-identical in the file
    return json.dumps(nb, indent=1, ensure_ascii=False)


def build_notebook() -> tuple[str, str]:
    raw = ARM_C_NB.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != ARM_C_NB_SHA:
        raise SystemExit(f"arm C notebook changed (sha256 {sha}); arm C+LM is defined on {ARM_C_NB_SHA}")
    text = raw.decode("utf-8")
    nb = json.loads(text)
    if _dump(nb) != text:
        raise SystemExit("arm C notebook does not round-trip through json.dumps(indent=1, ensure_ascii=False)")
    runs = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and _src(c).startswith(RUN_CELL_START)]
    if len(runs) != 1:
        raise SystemExit(f"expected one run cell, found {runs}")
    run = runs[0]
    body = _src(nb["cells"][run])
    if body.count(LMB.RUN_ANCHOR) != 1:
        raise SystemExit("run cell: bm.run anchor not found exactly once")
    if "install_d(_fp_ta, _fp_solver)" not in _src(nb["cells"][run - 1]):
        raise SystemExit("the cell before the run cell is not D''s install cell")
    nb["cells"][run]["source"] = body.replace(LMB.RUN_ANCHOR, LMB.RUN_DUMP).splitlines(True)
    nb["cells"].insert(run, LMB.install_cell())
    meta = json.loads(ARM_C_META.read_text(encoding="utf-8"))
    if meta["id"] != f"{OWNER}/arc3-jad-exp081":
        raise SystemExit(f"unexpected arm C kernel id {meta['id']!r}")
    meta.update(id=f"{OWNER}/{SLUG}", title=SLUG, code_file=f"{SLUG}.ipynb")
    return _dump(nb), json.dumps(meta, indent=1) + "\n"


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
