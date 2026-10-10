#!/usr/bin/env python
"""Build arm C: a verbatim fork of JustAdev742's exp-081 (hidden-set draw 34.04) as calamitychasm/arc3-jad-exp081.

    python scripts/_build_jad_exp081_kernel.py                      # writes kaggle_submission_jad_exp081/notebook/
    python scripts/_build_jad_exp081_kernel.py --out DIR            # elsewhere (the test rebuilds into a tmp dir)
    python scripts/_build_jad_exp081_kernel.py --no-apply-check     # without Franzen's repo (same bytes)

It runs THEIR builder, vendored unmodified in kaggle_submission_jad_exp081/vendor/ (JustAdev742/Arc-Agi-3-Kaggle-comp
@ c7a462b, Apache-2.0; see the README there), with exp-081's flags (their scripts/build_candidates.sh exp-084 command
minus the fine-tuned draft), plus our deviations:

- ``--env ARC3_HTTP_RETRY_INITIAL_SECONDS=2400`` (our 2026-10-10 standing rule; their exp-083/084 do the same);
- the slug ``arc3-jad-exp081`` and a ``--note`` naming the fork (markdown cell 0 only);
- kernel-metadata.json ``id``/``title`` rewritten from their hard-coded ``scottmahony/`` to ``calamitychasm/``.

Inherited from the c7a462b builder (not ours, Save & Run only): the --fail-fast watchdog limit is 55 min after the
notebook start; exp-081 itself was built on 2026-10-09 with the 35-min limit of their builder at that time. A
competition rerun never starts that watchdog.

The apply check (their patches on top of Franzen's patch, on the exact tree the notebook builds) needs Franzen's repo
(da-fr/arc-agi-3-solution @ 10882e3): --his-repo, else $FRANZEN_REPO, else /home/user/ext/da-fr_arc-agi-3-solution.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KDIR = ROOT / "kaggle_submission_jad_exp081"
VENDOR = KDIR / "vendor"
OUT = KDIR / "notebook"
SLUG = "arc3-jad-exp081"
OWNER = "calamitychasm"
DEFAULT_HIS_REPO = Path("/home/user/ext/da-fr_arc-agi-3-solution")
GRACE_S = 2400
P = "kaggle/franzen/patches"
PATCHES = [  # exp-081's order (their build_candidates.sh exp084 line)
    "ours-sandbox-timeout-keeps-work.patch", "ours-02-budget-meter.patch", "ours-04-search-helper.patch",
    "ours-03b-win-ledger-on-02-04.patch", "ours-05-level-mem.patch", "ours-08b-perception-on-01-02-04-03b-05.patch",
]
NOTE = ("Fork by calamitychasm (arm C) of JustAdev742's exp-081 (scottmahony/arc3-dprime-r14a05-harness4-percept-full, "
        "hidden-set draw 34.04; github.com/JustAdev742/Arc-Agi-3-Kaggle-comp @ c7a462b, Apache-2.0), built with their "
        "builder and exp-081's flags. Deviation from exp-081: first-request grace 2400 s (ARC3_HTTP_RETRY_INITIAL_SECONDS; "
        "theirs 900); the Save & Run fail-fast limit is the c7a462b builder's 55 min (exp-081: 35 min)")


def builder_args(out: Path) -> list[str]:
    args = ["--base", "dprime", "--full25", "121", "--input-fallback", "--wait-inputs", "120",
            "--env", "ARC3_MAX_ACTIVE_STREAMS=14", "--env", f"ARC3_HTTP_RETRY_INITIAL_SECONDS={GRACE_S}",
            "--cfg", "MAXREQ=14", "--cfg", "CUDAGRAPH_MAXBS=14", "--cfg", "MAMBA_CACHE=84",
            "--cfg", "SPEC_ACCEPT_SINGLE=0.5", "--cfg", "SPEC_ACCEPT_ACC=0.5",
            "--reap-kept", "kaggle/franzen/reap448_kept_experts.json",
            "--hot-tokens", "kaggle/franzen/hot_tokens_64k_arc.pt", "--fail-fast",
            "--env-add", "OURS_BUDGET_METER=1", "--env-add", "OURS_WIN_LEDGER=1", "--env-add", "OURS_SEARCH_HELPER=1",
            "--env-add", "OURS_LEVEL_MEM=1", "--env-add", "OURS_PERCEPTION=1"]
    for p in PATCHES:
        args += ["--patch", f"{P}/{p}"]
    return args + ["--compact", "--out", str(out), "--slug", SLUG, "--note", NOTE]


def build(out: Path = OUT, *, his_repo: Path | None = None, no_apply_check: bool = False) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        args = [sys.executable, "-I", "-B", str(VENDOR / "scripts" / "build_franzen_nb.py"), *builder_args(Path(tmp))]
        if no_apply_check:
            args.append("--no-apply-check")
        else:
            repo = his_repo or Path(os.environ.get("FRANZEN_REPO") or DEFAULT_HIS_REPO)
            args += ["--his-repo", str(repo)]
        run = subprocess.run(args, cwd=VENDOR, capture_output=True, text=True, check=False)
        if run.returncode != 0:
            raise SystemExit(f"their builder failed ({run.returncode}):\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}")
        meta_path = Path(tmp) / "kernel-metadata.json"
        meta = json.loads(meta_path.read_text())
        if meta["id"] != f"scottmahony/{SLUG}":
            raise SystemExit(f"unexpected kernel id from their builder: {meta['id']!r}")
        meta["id"] = f"{OWNER}/{SLUG}"
        meta["title"] = SLUG
        meta_path.write_text(json.dumps(meta, indent=1) + "\n")
        out.mkdir(parents=True, exist_ok=True)
        for f in (f"{SLUG}.ipynb", "kernel-metadata.json"):
            shutil.copyfile(Path(tmp) / f, out / f)
    print(f"built {out / (SLUG + '.ipynb')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--his-repo", type=Path, default=None)
    ap.add_argument("--no-apply-check", action="store_true")
    a = ap.parse_args()
    build(a.out, his_repo=a.his_repo, no_apply_check=a.no_apply_check)


if __name__ == "__main__":
    main()
