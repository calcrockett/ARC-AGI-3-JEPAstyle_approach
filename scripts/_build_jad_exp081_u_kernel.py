#!/usr/bin/env python
"""Build arm C+U: arm C (calamitychasm/arc3-jad-exp081 v2, JustAdev742's exp-081 fork) plus the I1 notice and the
stale-state latch fix.

    python scripts/_build_jad_exp081_u_kernel.py              # writes kaggle_submission_jad_exp081_u/notebook/
    python scripts/_build_jad_exp081_u_kernel.py --out DIR    # elsewhere (the test rebuilds into a tmp dir)

Source: the COMMITTED arm C notebook (kaggle_submission_jad_exp081/notebook/, sha256-pinned; a scheduled submit pins
its version 2, so it is read, never rewritten). Changes (experiments/stage7_milestone2_improvements.md section 12):

  C1  JustAdev742's ours-10 patch (kaggle_submission_jad_exp081_u/vendor/, verbatim from their repo at c7a462b =
      ccb7214, Apache-2.0) as a 7th harness patch, shipped exactly as their builder ships a --patch with --compact
      (a zlib/base64 cell written by their own _file_cell, sha256-checked when the notebook writes it), plus the
      setup-cell line ``'OURS_UNTRIED': 'i1',  # ours (--env-add)`` (their --env-add format): the I1 line only.
  C2  Our stale-state latch fix (kaggle_submission_jad_exp081_u/patches/calamitychasm-stale-latch-fix.patch) as an
      8th patch, in a plain ``%%writefile`` cell so it reads in the notebook.
      Both patches join the notebook's own apply loop (their _apply_code: the same git apply, the same check that
      every file a patch names applied); its success line becomes ``our harness patches applied successfully: 8``.
  +   One install cell right before the run cell (after D''s install cell) embeds cu_counters.py: it checks both
      changes in the imported harness, prints ``STALE_LATCH_FIX installed ...`` and ``UNTRIED_NOTICE installed ...``,
      and installs count-only wrappers. The run cell writes untried_summary.json, stale_latch_summary.json and
      untried_firings.json after bm.run.

kernel-metadata.json: arm C's, with id/title/code_file for the new slug. Everything else is byte-identical, including
cell 0, the 2400 s first-request grace, D', REAP-448, 14 streams and the full-length Save & Run (25 x 121 min).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_jad_exp081_kernel as C  # noqa: E402  (VENDOR: their c7a462b builder, PATCHES: exp-081's six)

ARM_C_DIR = ROOT / "kaggle_submission_jad_exp081" / "notebook"
ARM_C_NB = ARM_C_DIR / "arc3-jad-exp081.ipynb"
ARM_C_META = ARM_C_DIR / "kernel-metadata.json"
ARM_C_NB_SHA = "d108fa761fb66dfef828df668de947841b857634756568954541d4f77151a26a"  # arm C v1/v2 as pushed
KDIR = ROOT / "kaggle_submission_jad_exp081_u"
OUT = KDIR / "notebook"
OWNER = "calamitychasm"
SLUG = "arc3-jad-exp081-u"
UNTRIED_PATCH = KDIR / "vendor" / "kaggle" / "franzen" / "patches" / "ours-10-untried.patch"
UNTRIED_PATCH_SHA = "a3a7d89fbb2cb253afcfc8eb9b4d43f763c45ea8b8624139db54949e7bc6a52e"  # their c7a462b == ccb7214
UNTRIED_SOURCE_COMMIT = "c7a462be8fe3a22910804d4d61614f43a510fdfb"  # unchanged at ccb7214862bbe7fab3a24a7e7e2ba6653d931cac
LATCH_PATCH = KDIR / "patches" / "calamitychasm-stale-latch-fix.patch"
COUNTERS = KDIR / "cu_counters.py"
UNTRIED_ENV = ("OURS_UNTRIED", "i1")
LAST_ENV_LINE = "    'OURS_PERCEPTION': 1,  # ours (--env-add)\n"
RUN_CELL_START = "print('Starting benchmark...')"
RUN_ANCHOR = ("await bm.run(soft_end_time=soft_end, runtime_environment=target, "
              "minimal_diagnostics=TRUE_SUBMISSION)\n")
RUN_DUMP = RUN_ANCHOR + (
    "try:   # [calamitychasm] C+U counters (ours-10 I1 notice, stale-latch fix)\n"
    "    for _cu_name, _cu_data in CU_COUNTERS.finish().items():\n"
    "        (WORKING_DIR / _cu_name).write_text(json.dumps(_cu_data, indent=2))\n"
    "    print('UNTRIED_NOTICE summary', json.dumps(CU_COUNTERS.untried_summary()), flush=True)\n"
    "    print('STALE_LATCH_FIX summary', json.dumps(CU_COUNTERS.stale_summary()), flush=True)\n"
    "except Exception as _exc:\n"
    "    print('C+U summary failed', repr(_exc), flush=True)\n"
)
INSTALL_HEAD = "# [calamitychasm] C+U INSTALL"


def their_builder():
    """JustAdev742's build_franzen_nb.py at c7a462b, vendored unmodified with arm C (imported, not run)."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(C.VENDOR / "scripts"))
    import build_franzen_nb as BF  # noqa: PLC0415
    return BF


def patch_names() -> tuple[list[str], str, str]:
    """(arm C's six notebook paths, the ours-10 path, the latch-fix path), by their builder's naming rule."""
    BF = their_builder()
    six = [f"/kaggle/{BF._patch_name(i, Path(p))}" for i, p in enumerate(C.PATCHES, start=1)]
    return six, f"/kaggle/{BF._patch_name(7, UNTRIED_PATCH)}", f"/kaggle/{BF._patch_name(8, LATCH_PATCH)}"


def install_cell() -> dict:
    src = COUNTERS.read_text(encoding="utf-8")
    if "'''" in src:
        raise SystemExit("cu_counters.py must not contain ''' (it is embedded in a raw string)")
    text = (
        f"{INSTALL_HEAD} -- arm C + the ours-10 I1 notice (OURS_UNTRIED=i1) + the stale-state latch fix; see\n"
        "# experiments/stage7_milestone2_improvements.md section 12. Both changes are source patches applied in the\n"
        "# setup cell; this cell checks them in the imported harness, prints their markers, and installs count-only\n"
        "# wrappers (cu_counters.py, embedded verbatim): what the model sees and what the harness does are unchanged.\n"
        "import sys as _sys, types as _types\n"
        f"_CU_SRC = r'''{src}'''\n"
        "CU_COUNTERS = _types.ModuleType('cu_counters')\n"
        "_sys.modules['cu_counters'] = CU_COUNTERS\n"
        "exec(compile(_CU_SRC, 'cu_counters.py', 'exec'), CU_COUNTERS.__dict__)\n"
        "import inference.agent.python_tool_sandbox as _cu_sb\n"
        "import inference.agent.tool_agent as _cu_ta\n"
        "import inference.utils.ours_untried as _cu_ut\n"
        "for _cu_line in CU_COUNTERS.install(_cu_sb, _cu_ta, _cu_ut):\n"
        "    print(_cu_line, flush=True)\n"
    )
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": text.splitlines(True)}


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _dump(nb: dict) -> str:
    # their builder writes json.dumps(nb, indent=1, ensure_ascii=False) with no final newline; matching it keeps every
    # unchanged cell byte-identical in the file
    return json.dumps(nb, indent=1, ensure_ascii=False)


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{what}: anchor found {text.count(old)} times (expected once)")
    return text.replace(old, new)


def build_notebook() -> tuple[str, str]:
    BF = their_builder()
    raw = ARM_C_NB.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARM_C_NB_SHA:
        raise SystemExit(f"arm C notebook changed; arm C+U is defined on {ARM_C_NB_SHA}")
    text = raw.decode("utf-8")
    nb = json.loads(text)
    if _dump(nb) != text:
        raise SystemExit("arm C notebook does not round-trip through json.dumps(indent=1, ensure_ascii=False)")
    untried = UNTRIED_PATCH.read_text(encoding="utf-8")
    if hashlib.sha256(untried.encode()).hexdigest() != UNTRIED_PATCH_SHA:
        raise SystemExit("vendored ours-10-untried.patch is not their a3a7d89f")
    latch = LATCH_PATCH.read_text(encoding="utf-8")
    if not latch.endswith("\n") or latch.count("\ndiff --git ") != 1:
        raise SystemExit("the latch-fix patch must name one file and end with a newline")
    six, p_untried, p_latch = patch_names()
    cells = nb["cells"]
    srcs = [_src(c) for c in cells]

    # C1 + C2 patch cells, right after arm C's last patch cell (their builder puts --patch cells there, in order)
    last = [i for i, s in enumerate(srcs) if (BF.franzen_tree.written_file(s) or ("",))[0] == six[-1]]
    if len(last) != 1 or srcs[last[0] + 1].startswith(BF.COMPACT_BEGIN):
        raise SystemExit(f"arm C's last patch cell not found once: {last}")
    added = []
    for path, body, compact in ((p_untried, untried, True), (p_latch, latch, False)):
        cell = copy.deepcopy(cells[last[0]])
        cell["outputs"], cell["execution_count"] = [], None
        cell["source"] = BF._file_cell(path, body, compact).splitlines(True)
        if BF.franzen_tree.written_file(_src(cell)) != (path, body):
            raise SystemExit(f"{path}: the cell does not write the patch byte for byte")
        added.append(cell)

    # setup cell: the env line and the apply loop over all eight patches (their _apply_code)
    setup = [i for i, s in enumerate(srcs) if "setup_env = {" in s]
    if len(setup) != 1:
        raise SystemExit(f"setup cell not found once: {setup}")
    s = srcs[setup[0]]
    if f"'{UNTRIED_ENV[0]}'" in s:
        raise SystemExit("arm C already sets OURS_UNTRIED")
    env_line = f"    '{UNTRIED_ENV[0]}': {BF._env_literal(UNTRIED_ENV[1])},  # ours (--env-add)\n"
    s = _replace_once(s, LAST_ENV_LINE, LAST_ENV_LINE + env_line, "--env-add block")
    s = _replace_once(s, BF._apply_code([p[len('/kaggle/'):] for p in six]),
                      BF._apply_code([p[len('/kaggle/'):] for p in six + [p_untried, p_latch]]), "apply loop")
    cells[setup[0]]["source"] = s.splitlines(True)

    # the install cell before the run cell, the counter dump after bm.run
    runs = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and srcs[i].startswith(RUN_CELL_START)]
    if len(runs) != 1 or "install_d(_fp_ta, _fp_solver)" not in srcs[runs[0] - 1]:
        raise SystemExit(f"run cell after D''s install cell not found once: {runs}")
    run = runs[0]
    cells[run]["source"] = _replace_once(srcs[run], RUN_ANCHOR, RUN_DUMP, "run cell").splitlines(True)
    cells.insert(run, install_cell())
    cells[last[0] + 1:last[0] + 1] = added

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
