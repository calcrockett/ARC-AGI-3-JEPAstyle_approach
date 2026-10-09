"""Build the serving-speed test kernels from the incumbent (milestone-2 fork + level memory).

Each variant changes ONLY serving knobs, and every variant (baseline included) plays all 25
public games for 25 minutes each -- more games than streams, as in the hidden run -- so the
stream count is actually exercised. Never submitted: these are measurement kernels.

  base    spec 3 draft steps, mem 0.96, 10 streams              (the incumbent's serving)
  spec4   spec 4 draft steps                                     (lossless; speed only)
  s12     mem 0.98, 12 streams
  s14     mem 0.98, 14 streams, Mamba cache 72
  s14hic  s14 + 32 GB hierarchical KV cache in system RAM

Round 3 ("spd" kernels; mem 0.98 OOMs, so these keep mem <= 0.97 and find KV room elsewhere;
written to kaggle_submission_m2_spd_<name>/notebook, see experiments/stage7_m2_speed.md):

  m97s12     mem 0.97, 12 streams, Mamba cache 72 (5 slots/request is SGLang's floor)
  m96s12hic  mem 0.96, 12 streams, Mamba cache 72, 32 GB hierarchical host KV cache
  m97s12hic  both

    python scripts/_build_m2_speed_kernels.py [name ...]     # no names: build every variant
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

SRC = ROOT / "kaggle_submission_m2_level_memory" / "notebook"

# Flags verified against the Pennyroyal v2.5.3 source (d00d88e, server_args.py) and against
# sirikilohit's working run (--hicache-size 48, same write policy / io backend, hicache_attached=True).
# --hicache-size is GB of host RAM for ALL host pools (KV + Mamba + QSA index keys, split by device
# pool bytes); the other three flags are the SGLang defaults, spelled out so the log shows them.
def hicache_line(gb: int) -> str:
    return (f'args += ["--enable-hierarchical-cache", "--hicache-size", "{gb}", '
            '"--hicache-write-policy", "write_through", "--hicache-io-backend", "kernel"]\n')


# SGLang caps max_running_requests at max_mamba_cache_size // 5 for this model (3 base slots per
# request + 2 for the overlap-schedule extra_buffer ping-pong; kv_cache_configurator.py,
# _calculate_mamba_ratio). Also the host-RAM ceiling for --hicache-size, see the experiment doc.
MAMBA_SLOTS_PER_REQUEST = 5
HICACHE_MAX_GB = 40

# s12/s14/s14hic were first created by pushes rejected at the 2-GPU-session limit and could
# never mount the bundle dataset afterwards (3 attempts each); they get fresh slugs (spd2).
VARIANTS = {
    "base":   dict(),
    "spec4":  dict(SPEC_STEPS=4),
    "s12":    dict(MEMFRAC=0.98, MAXREQ=12, CUDAGRAPH_MAXBS=12),
    "s14":    dict(MEMFRAC=0.98, MAXREQ=14, CUDAGRAPH_MAXBS=14, MAMBA_CACHE=72),
    "s14hic": dict(MEMFRAC=0.98, MAXREQ=14, CUDAGRAPH_MAXBS=14, MAMBA_CACHE=72, HICACHE_GB=32),
}
# 12 streams x 6 Mamba slots (the incumbent's 60 / 10) = 72; 60 would be exactly SGLang's 5-per-request
# floor and leave no slots for retained prefix checkpoints, confounding the speed test with a
# prefix-reuse regression.
S12 = dict(MAXREQ=12, CUDAGRAPH_MAXBS=12, MAMBA_CACHE=72)
SPD_VARIANTS = {
    "m97s12":    dict(MEMFRAC=0.97, **S12),
    "m96s12hic": dict(MEMFRAC=0.96, HICACHE_GB=32, **S12),
    "m97s12hic": dict(MEMFRAC=0.97, HICACHE_GB=32, **S12),
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


def layout(name: str, root: Path = ROOT) -> tuple[Path, str, str]:
    """(output dir, notebook file name, kernel slug) for a variant."""
    if name in SPD_VARIANTS:
        return (root / f"kaggle_submission_m2_spd_{name}" / "notebook",
                f"arc3-m2-spd-{name}.ipynb", f"arc3-m2-spd-{name}")
    slug = f"arc3-m2-speed-{name}" if name in ("base", "spec4") else f"arc3-m2-spd2-{name}"
    return root / "kaggle_submission_m2_speed" / name, f"arc3-m2-speed-{name}.ipynb", slug


def check_consistency(name: str, knobs: dict) -> None:
    """Everything the harness / SGLang sizes from the stream count must agree."""
    k = {**DEFAULTS, **knobs}
    problems = []
    if k["CUDAGRAPH_MAXBS"] < k["MAXREQ"]:
        problems.append("CUDAGRAPH_MAXBS < MAXREQ (the launcher asserts this)")
    if k["MAMBA_CACHE"] // MAMBA_SLOTS_PER_REQUEST < k["MAXREQ"]:
        problems.append(f"MAMBA_CACHE {k['MAMBA_CACHE']} // {MAMBA_SLOTS_PER_REQUEST} < MAXREQ "
                        f"{k['MAXREQ']}: SGLang would silently cap max_running_requests")
    if name in SPD_VARIANTS and not 0 < k["MEMFRAC"] <= 0.97:   # legacy s12/s14* are the known 0.98 OOMs
        problems.append("MEMFRAC must be <= 0.97 (0.98 OOMs at runtime, see stage7_m2_speed.md)")
    if not 0 <= k.get("HICACHE_GB", 0) <= HICACHE_MAX_GB:
        problems.append(f"HICACHE_GB must be 0..{HICACHE_MAX_GB} (BF16 PLE is ~95 GiB of 177 GiB host RAM)")
    if problems:
        raise SystemExit(f"REFUSING TO BUILD {name}: " + "; ".join(problems))


def build(name: str, knobs: dict, root: Path = ROOT) -> Path:
    check_consistency(name, knobs)
    nb = json.loads((SRC / "arc3-m2-level-memory.ipynb").read_text(encoding="utf-8"))
    src = ["".join(c["source"]) for c in nb["cells"]]

    i = cell_index(src, "CFG = dict(")
    s = src[i]
    for k, default in DEFAULTS.items():
        s = sub(s, f"    {k}={default},", f"    {k}={knobs.get(k, default)},", k)
    if knobs.get("HICACHE_GB"):
        anchor = 'if CFG["PREFETCH_CHECKPOINTS"]: args += ["--weight-loader-prefetch-checkpoints"]\n'
        s = sub(s, anchor, anchor + "# [calamitychasm speed test] system-RAM KV tier\n" + hicache_line(knobs["HICACHE_GB"]),
                "hicache")
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
    d, nb_name, slug = layout(name, root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / nb_name
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(id=f"calamitychasm/{slug}", title=slug,
                code_file=path.name, is_private=True)
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    from _check_notebook_cell import check_notebook
    every = {**VARIANTS, **SPD_VARIANTS}
    names = (sys.argv[1:] if argv is None else argv) or list(every)
    unknown = [n for n in names if n not in every]
    if unknown:
        print("unknown variant(s):", unknown, "known:", list(every))
        return 2
    for name in names:
        knobs = every[name]
        path = build(name, knobs)
        problems = check_notebook(path)
        if problems:
            print("REFUSING TO SHIP", name, problems)
            return 1
        print("built", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
