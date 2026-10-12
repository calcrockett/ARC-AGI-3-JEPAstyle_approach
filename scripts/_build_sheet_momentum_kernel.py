"""Build arc3-sheetu12b-momentum: scottlegrand/taaf-flashnext-sheetu12b-0922 (public 5.19,
Apache-2.0, vendored verbatim in kaggle_submission_sheetu12b_momentum/upstream/) plus the
momentum time policy -- the same two edits the anim-momentum kernel makes, nothing else:

  (a) one cell after the customization hook installs MomentumTimePolicy at production
      settings (kaggle_submission_duck_nvfp4_anim_momentum/momentum_time.py, inlined);
  (b) the policy's decisions are written out after bm.run.

The upstream solver bundle (keithtyser duck smoke) has the same
_HarnessGameSession.runtime_limit_reached / timing_payload / should_stop as anim, and none
of upstream's agentfix patches touch them (checked 2026-09-28).

    python scripts/_build_sheet_momentum_kernel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from _build_momentum_kernels import PROD, SUMMARY_DUMP, cells_of, install_cell, set_src, sub  # noqa: E402

OUT = ROOT / "kaggle_submission_sheetu12b_momentum"
UPSTREAM = OUT / "upstream" / "taaf-flashnext-sheetu12b-0922.ipynb"
KERNEL_ID = "calamitychasm/arc3-sheetu12b-momentum"


def build() -> Path:
    nb = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    src = cells_of(nb)
    assert src[14].startswith("# Exact public-25 and competition settings."), "cell 14 moved"
    assert "AGENTFIX LIVE" in src[10] and "ARM P installed" in src[10], "agentfix cell moved"
    set_src(nb, 16, sub(src[16], *SUMMARY_DUMP, "bm.run summary dump"))
    nb["cells"].insert(15, install_cell(
        "base_s=float(bm.solver.max_runtime_s_per_game), "
        + ", ".join(f"{k}={v!r}" for k, v in PROD.items())))
    md = ("## [calamitychasm] fork: upstream + momentum time allocation\n\n"
          "Verbatim fork of `scottlegrand/taaf-flashnext-sheetu12b-0922` (Apache-2.0; full credit to "
          "Scott Le Grand, Jeroen Cottaar and Tufa Labs). Two additions only: the momentum time "
          "policy cell after the customization hook, and a dump of its decisions after the run.\n")
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": md.splitlines(True)})
    d = OUT / "notebook"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "arc3-sheetu12b-momentum.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = json.loads((OUT / "upstream" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.pop("id_no", None)
    meta.update(id=KERNEL_ID, title="arc3-sheetu12b-momentum", code_file=path.name, is_private=True)
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def main() -> int:
    from _check_notebook_cell import check_notebook
    path = build()
    problems = check_notebook(path)
    if problems:
        print("REFUSING TO SHIP -- use-before-definition problems")
        for p in problems:
            print(" ", p)
        return 1
    print("built", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
