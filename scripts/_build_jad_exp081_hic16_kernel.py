#!/usr/bin/env python
"""Build arm C+hic16: arm C (calamitychasm/arc3-jad-exp081 v2, JustAdev742's exp-081 fork) plus the 32 GB system-RAM KV
tier and 16 streams.

    python scripts/_build_jad_exp081_hic16_kernel.py              # writes kaggle_submission_jad_exp081_hic16/notebook/
    python scripts/_build_jad_exp081_hic16_kernel.py --out DIR    # elsewhere (the test rebuilds into a tmp dir)

Source: the COMMITTED arm C notebook (kaggle_submission_jad_exp081/notebook/, sha256-pinned; a scheduled submit pins its
version 2, so it is read, never rewritten). Edits, nothing else (experiments/stage7_milestone2_improvements.md section 14):

  setup cell     ARC3_MAX_ACTIVE_STREAMS 14 -> 16. It is the only stream knob of the harness: Franzen's patch builds
                 _PriorityGate(slots) from it and logs ``priority gate active: 16 concurrent streams``; D' replaces only
                 the priority value and the fresh-game queueing, never the slot count.
  launcher cell  MAXREQ / CUDAGRAPH_MAXBS 14 -> 16 and MAMBA_CACHE 84 -> 96 (arm C's own formula: 84 = 6 x 14, so
                 6 x 16; 96 // 5 >= 16, SGLang's cap on running requests); the host KV tier
                 ``--enable-hierarchical-cache --hicache-size 32 --hicache-write-policy write_through
                 --hicache-io-backend kernel`` (the hic16 kernels' line, _build_m2_speed_kernels.hicache_line); the
                 hic16 kernels' ``STREAMS max_running_requests=...`` print and the ``HICACHE requested`` assert + print
                 before the launch; and after the health loop their serve.log check, which prints
                 ``HICACHE_TIER attached hicache_attached=True`` only when SGLang logged it.
  run cell       the same serve.log check again after bm.run (``... (post-run)``): the launcher's check can run before
                 the boot finishes (it did on arm B's hic16 run).

The ``[sys] RAM used/total`` census (every 2 min outside a rerun) is arm C's own cell, the same census our hic16 kernels
print, so it is reused unchanged. kernel-metadata.json: arm C's, with id/title/code_file for the new slug. Everything
else is byte-identical, including cell 0, D', REAP-448, acceptance 0.5, the ARC map, the six patches, the 2400 s
first-request grace (ARC3_HTTP_RETRY_INITIAL_SECONDS) and the full-length Save & Run shape (25 x 121 min).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_m2_level_memory_kernel as LMB  # noqa: E402  (HICACHE_CHECK, HIC_DUMP, HICACHE_MARKER, L_* anchors)
from _build_m2_speed_kernels import hicache_line  # noqa: E402

ARM_C_DIR = ROOT / "kaggle_submission_jad_exp081" / "notebook"
ARM_C_NB = ARM_C_DIR / "arc3-jad-exp081.ipynb"
ARM_C_META = ARM_C_DIR / "kernel-metadata.json"
ARM_C_NB_SHA = "d108fa761fb66dfef828df668de947841b857634756568954541d4f77151a26a"  # arm C v1/v2 as pushed
OWNER = "calamitychasm"
SLUG = "arc3-jad-exp081-hic16"
OUT = ROOT / "kaggle_submission_jad_exp081_hic16" / "notebook"

STREAMS_C, STREAMS = 14, 16
MAMBA_PER_STREAM = 6          # arm C: MAMBA_CACHE 84 at 14 streams
MAMBA_SLOTS_PER_REQUEST = 5   # SGLang caps running requests at max_mamba_cache_size // 5 on this model
HICACHE_GB = 32
TAG = "# [calamitychasm hic16]"

SETUP_OLD = f"    'ARC3_MAX_ACTIVE_STREAMS': {STREAMS_C},  # ours (--env)\n"
SETUP_NEW = f"    'ARC3_MAX_ACTIVE_STREAMS': {STREAMS},  {TAG} streams (arm C: {STREAMS_C})\n"
CFG_EDITS = (("MAXREQ", STREAMS_C, STREAMS), ("CUDAGRAPH_MAXBS", STREAMS_C, STREAMS),
             ("MAMBA_CACHE", MAMBA_PER_STREAM * STREAMS_C, MAMBA_PER_STREAM * STREAMS))
RUN_CELL_START = "print('Starting benchmark...')"
RUN_ANCHOR = ("await bm.run(soft_end_time=soft_end, runtime_environment=target, "
              "minimal_diagnostics=TRUE_SUBMISSION)\n")


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


def _one(srcs: list[str], needle: str, what: str) -> int:
    hits = [i for i, s in enumerate(srcs) if needle in s]
    if len(hits) != 1:
        raise SystemExit(f"{what}: found in cells {hits} (expected one)")
    return hits[0]


def launcher_edits(cell: str) -> str:
    """Arm C's launcher cell with 16 streams and the host KV tier (the hic16 kernels' edits, on their text)."""
    if MAMBA_PER_STREAM * STREAMS // MAMBA_SLOTS_PER_REQUEST < STREAMS:
        raise SystemExit("Mamba cache // 5 < streams: SGLang would cap running requests")
    for key, old, new in CFG_EDITS:
        cell = _replace_once(cell, f"    {key}={old},  # ours (--cfg)\n",
                             f"    {key}={new},  {TAG} streams (arm C: {old})\n", key)
    if "hicache" in cell or "hierarchical" in cell:
        raise SystemExit("arm C's launcher already mentions the host tier")
    cell = _replace_once(cell, LMB.L_PREFETCH, LMB.L_PREFETCH
                         + f"{TAG} system-RAM KV tier (the hic16 kernels' flags)\n" + hicache_line(HICACHE_GB),
                         "hicache args")
    marks = (f"{TAG} serving markers (the submit gate checks them)\n"
             'print(f"STREAMS max_running_requests={CFG[\'MAXREQ\']} cuda_graph_bs={graph_bs} '
             'mamba_cache={CFG[\'MAMBA_CACHE\']}", flush=True)\n'
             f'assert args[args.index("--hicache-size") + 1] == "{HICACHE_GB}" and "--enable-hierarchical-cache" in args\n'
             f'print("HICACHE requested --hicache-size {HICACHE_GB} GB (write_through, kernel io)", flush=True)\n')
    cell = _replace_once(cell, LMB.L_LAUNCH, marks + LMB.L_LAUNCH, "markers")
    if not cell.endswith(LMB.L_END):
        raise SystemExit("arm C's launcher cell no longer ends with the health loop's DEADLINE print")
    return cell + ("\n\n" + f"{LMB.TH} confirm from serve.log that SGLang attached the host tier to the radix cache\n"
                   "# (mem_cache/registry.py logs 'Tree cache initialized: ... hicache_attached=<bool>'). Never raises.\n"
                   + LMB.HICACHE_CHECK.format(when="", indent=""))


def build_notebook() -> tuple[str, str]:
    raw = ARM_C_NB.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != ARM_C_NB_SHA:
        raise SystemExit(f"arm C notebook changed (sha256 {sha}); arm C+hic16 is defined on {ARM_C_NB_SHA}")
    text = raw.decode("utf-8")
    nb = json.loads(text)
    if _dump(nb) != text:
        raise SystemExit("arm C notebook does not round-trip through json.dumps(indent=1, ensure_ascii=False)")
    cells = nb["cells"]
    srcs = [_src(c) for c in cells]

    i = _one(srcs, SETUP_OLD, "setup cell (ARC3_MAX_ACTIVE_STREAMS)")
    cells[i]["source"] = _replace_once(srcs[i], SETUP_OLD, SETUP_NEW, "streams").splitlines(True)

    i = _one(srcs, LMB.L_CFG, "launcher cell")
    cells[i]["source"] = launcher_edits(srcs[i]).splitlines(True)

    runs = [k for k, c in enumerate(cells) if c["cell_type"] == "code" and srcs[k].startswith(RUN_CELL_START)]
    if len(runs) != 1:
        raise SystemExit(f"expected one run cell, found {runs}")
    run = runs[0]
    cells[run]["source"] = _replace_once(srcs[run], RUN_ANCHOR, RUN_ANCHOR + LMB.HIC_DUMP, "run cell").splitlines(True)

    meta = json.loads(ARM_C_META.read_text(encoding="utf-8"))
    if meta["id"] != f"{OWNER}/arc3-jad-exp081":
        raise SystemExit(f"unexpected arm C kernel id {meta['id']!r}")
    meta.update(id=f"{OWNER}/{SLUG}", title=SLUG, code_file=f"{SLUG}.ipynb")
    return _dump(nb), json.dumps(meta, indent=1) + "\n"


def kernel_markers() -> list[str]:
    """Submit-gate markers: arm C's six with 16 streams, plus the host tier and the server's stream count."""
    return ["harness patch applied successfully", "our harness patches applied successfully: 6",
            "#OURS_FORM ok version=d_prime", f"priority gate active: {STREAMS} concurrent streams",
            "48 layers x 448 experts, router sha256 for every layer", "ours: FR-Spec map hot_tokens_64k_arc.pt written",
            LMB.HICACHE_MARKER, f"STREAMS max_running_requests={STREAMS} cuda_graph_bs="]


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
