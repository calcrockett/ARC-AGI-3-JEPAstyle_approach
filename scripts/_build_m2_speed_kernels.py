"""Build the serving-speed test kernels from the incumbent (milestone-2 fork + level memory).

Each variant changes ONLY serving knobs, and every variant (baseline included) plays all 25
public games for 25 minutes each -- more games than streams, as in the hidden run -- so the
stream count is actually exercised. Never submitted: these are measurement kernels.

  base    spec 3 draft steps, mem 0.96, 10 streams              (the incumbent's serving)
  spec4   spec 4 draft steps                                     (lossless; speed only)
  s12     mem 0.98, 12 streams
  s14     mem 0.98, 14 streams, Mamba cache 72
  s14hic  s14 + 32 GB hierarchical KV cache in system RAM

    python scripts/_build_m2_speed_kernels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

SRC = ROOT / "kaggle_submission_m2_level_memory" / "notebook"
OUT = ROOT / "kaggle_submission_m2_speed"

HICACHE = ('args += ["--enable-hierarchical-cache", "--hicache-size", "32", '
           '"--hicache-write-policy", "write_through", "--hicache-io-backend", "kernel"]\n')

VARIANTS = {
    "base":   dict(),
    "spec4":  dict(SPEC_STEPS=4),
    "s12":    dict(MEMFRAC=0.98, MAXREQ=12, CUDAGRAPH_MAXBS=12),
    "s14":    dict(MEMFRAC=0.98, MAXREQ=14, CUDAGRAPH_MAXBS=14, MAMBA_CACHE=72),
    "s14hic": dict(MEMFRAC=0.98, MAXREQ=14, CUDAGRAPH_MAXBS=14, MAMBA_CACHE=72, HICACHE=True),
}
DEFAULTS = dict(MEMFRAC=0.96, MAXREQ=10, CUDAGRAPH_MAXBS=10, MAMBA_CACHE=60, SPEC_STEPS=3)


def sub(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"REFUSING TO BUILD -- anchor for {what!r} found {n}x")
    return text.replace(old, new)


def cell_index(src, needle):
    hits = [i for i, s in enumerate(src) if needle in s]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def build(name: str, knobs: dict) -> Path:
    nb = json.loads((SRC / "arc3-m2-level-memory.ipynb").read_text(encoding="utf-8"))
    src = ["".join(c["source"]) for c in nb["cells"]]

    i = cell_index(src, "CFG = dict(")
    s = src[i]
    for k, default in DEFAULTS.items():
        s = sub(s, f"    {k}={default},", f"    {k}={knobs.get(k, default)},", k)
    if knobs.get("HICACHE"):
        anchor = 'if CFG["PREFETCH_CHECKPOINTS"]: args += ["--weight-loader-prefetch-checkpoints"]\n'
        s = sub(s, anchor, anchor + "# [calamitychasm speed test] system-RAM KV tier\n" + HICACHE, "hicache")
    src[i] = s

    i = cell_index(src, "'ARC3_MAX_ACTIVE_STREAMS': 10,")
    src[i] = sub(src[i], "'ARC3_MAX_ACTIVE_STREAMS': 10,",
                 f"'ARC3_MAX_ACTIVE_STREAMS': {knobs.get('MAXREQ', 10)},", "streams")

    i = cell_index(src, "demo_excluded_games = [] if TRUE_SUBMISSION else")
    line = [ln for ln in src[i].splitlines() if ln.startswith("demo_excluded_games = ")][0]
    src[i] = sub(src[i], line, "demo_excluded_games = []   # [calamitychasm speed test] all 25 public games", "games")

    for c, text in zip(nb["cells"], src):
        c["source"] = text.splitlines(True)
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": [
        f"## [calamitychasm] SPEED TEST `{name}` -- not a submission\n\n",
        f"Incumbent (milestone-2 fork + level memory) with serving knobs {json.dumps(knobs)}; "
        "all 25 public games, 25 min each.\n"]})
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"arc3-m2-speed-{name}.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(id=f"calamitychasm/arc3-m2-speed-{name}", title=f"arc3-m2-speed-{name}",
                code_file=path.name, is_private=True)
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def main() -> int:
    from _check_notebook_cell import check_notebook
    for name, knobs in VARIANTS.items():
        path = build(name, knobs)
        problems = check_notebook(path)
        if problems:
            print("REFUSING TO SHIP", name, problems)
            return 1
        print("built", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
