"""Host KV tier variant `hic<GB>` (arc3-m2-turbo-tail-hic16): turbo-tail + SGLang's system-RAM KV tier
(--enable-hierarchical-cache --hicache-size 32, the flags of the spd *hic kernels) + 16 streams / Mamba cache 96.
FP4 KV is not usable on this stack (experiments/stage7_m2_speed.md, "FP4 KV cache on this stack"); the host tier is
the next route to more streams."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_m2_level_memory_kernel as B  # noqa: E402
import _build_m2_speed_kernels as S  # noqa: E402
from _check_notebook_cell import check_notebook  # noqa: E402

HIC16 = tuple(v for v in B.TURBO if v != "s14") + ("tail", "hic32", "s16")
SLUG, DIRNAME = "arc3-m2-turbo-tail-hic16", "kaggle_submission_m2_turbo_tail_hic16"
COMMITTED = ROOT / DIRNAME / "notebook" / f"{SLUG}.ipynb"
TAIL = ROOT / "kaggle_submission_m2_turbo_tail" / "notebook" / "arc3-m2-turbo-tail.ipynb"
# sirikilohit's Pennyroyal serve log (ext/LohitSiriki_arc-agi-3-milestone2-solution/run/sglang-main.log)
ATTACHED = ("[2026-09-30 11:50:23] Tree cache initialized: source=default impl=UnifiedRadixCache hybrid_swa=False "
            "hybrid_ssm=True hicache_attached=True streaming_wrapped=False")


def _cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def _one(cells, needle):
    hits = [c for c in cells if needle in c]
    assert len(hits) == 1, needle
    return hits[0]


def test_slug_dir_markers_and_counters():
    assert B.kernel_slug(HIC16) == SLUG and B.kernel_slug(tuple(reversed(HIC16))) == SLUG
    assert B.kernel_dir(HIC16).relative_to(B.ROOT).as_posix() == f"{DIRNAME}/notebook"
    markers = B.kernel_markers(HIC16)
    base = B.kernel_markers(B.TURBO + ("tail",))
    assert [m for m in base if "concurrent streams" not in m] == [m for m in markers if m in base]
    for m in ("HICACHE_TIER attached hicache_attached=True", "priority gate active: 16 concurrent streams",
              "STREAMS max_running_requests=16 cuda_graph_bs=", "SPEC_ACCEPT 0.5", "PRIORITY_TAIL installed"):
        assert m in markers, m
    assert "priority gate active: 14 concurrent streams" not in markers
    assert B.kernel_counters(HIC16) == B.kernel_counters(B.TURBO)
    # other sizes / presets keep distinct slugs; the existing slugs are unchanged
    assert B.kernel_slug(tuple(v for v in HIC16 if v not in ("hic32", "s16")) + ("hic40", "s15")) == \
        "arc3-m2-turbo-tail-hic40gb-s15"
    lossless = tuple(v for v in B.TURBO_LOSSLESS if v != "s14") + ("tail", "hic32", "s16")
    assert B.kernel_slug(lossless) == "arc3-m2-turbo-lossless-tail-hic16"
    assert B.kernel_slug(("reap", "s16", "hic32")) == "arc3-m2-lm-reap-s16-hic32"
    assert B.kernel_slug(B.TURBO + ("tail",)) == "arc3-m2-turbo-tail" and B.kernel_slug(B.TURBO) == "arc3-m2-turbo"


def test_committed_kernel_is_current_and_deterministic(tmp_path, monkeypatch):
    outs = []
    for i in range(2):
        monkeypatch.setattr(B, "ROOT", tmp_path / str(i))
        outs.append(B.build(HIC16).read_text(encoding="utf-8"))
    assert outs[0] == outs[1], "builder is not deterministic"
    assert outs[0] == COMMITTED.read_text(encoding="utf-8"), f"{SLUG} is stale: rerun the build"
    assert check_notebook(COMMITTED) == []


def test_metadata_matches_incumbent_except_identity():
    meta = json.loads((COMMITTED.parent / "kernel-metadata.json").read_text())
    inc = json.loads((ROOT / "kaggle_submission_m2_level_memory" / "notebook" / "kernel-metadata.json").read_text())
    assert meta["id"] == f"calamitychasm/{SLUG}" and meta["code_file"] == COMMITTED.name and meta["title"] == SLUG
    for k in ("id", "title", "code_file"):
        meta.pop(k), inc.pop(k)
    assert meta.pop("docker_image_pinning_type") == "original"
    assert meta == inc


def test_only_streams_hicache_header_and_run_dump_differ_from_turbo_tail():
    a, b = _cells(TAIL), _cells(COMMITTED)
    assert len(a) == len(b)
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    setup = next(i for i, c in enumerate(a) if "'ARC3_MAX_ACTIVE_STREAMS': 14," in c)
    launch = next(i for i, c in enumerate(a) if B.L_CFG in c)
    run = next(i for i, c in enumerate(a) if c.startswith("print('Starting benchmark...')"))
    assert diff == [0, setup, launch, run]
    assert "'ARC3_MAX_ACTIVE_STREAMS': 16," in b[setup]
    assert b[setup].replace("'ARC3_MAX_ACTIVE_STREAMS': 16,", "'ARC3_MAX_ACTIVE_STREAMS': 14,") == a[setup]
    assert b[run] == a[run].replace(B.REAP_DUMP, B.REAP_DUMP + B.HIC_DUMP)
    la, lb = a[launch], b[launch]
    for key, old, new in (("MAXREQ", 14, 16), ("CUDAGRAPH_MAXBS", 14, 16), ("MAMBA_CACHE", 84, 96)):
        assert f"    {key}={new},   " in lb and f"    {key}={old},   " in la
        lb = lb.replace(f"    {key}={new},   ", f"    {key}={old},   ")
    # exactly three insertions: the tier's args, its request marker, and the serve.log check after the cell end
    exp = la.replace(B.L_PREFETCH, B.L_PREFETCH + f"{B.TH} system-RAM KV tier (same flags as the spd *hic kernels)\n"
                     + S.hicache_line(32))
    exp = exp.replace(B.L_LAUNCH, 'assert args[args.index("--hicache-size") + 1] == "32" and '
                      '"--enable-hierarchical-cache" in args\n'
                      'print("HICACHE requested --hicache-size 32 GB (write_through, kernel io)", flush=True)\n'
                      + B.L_LAUNCH)
    assert lb.startswith(exp), "unexpected launcher change"
    tail = lb[len(exp):]
    assert tail.startswith(f"\n\n{B.TH} confirm from serve.log") and tail.endswith(
        B.HICACHE_CHECK.format(when="", indent=""))
    assert tail.count("\n") == 11


def test_launcher_args_carry_the_tier_and_sixteen_streams():
    launch = _one(_cells(COMMITTED), B.L_CFG)
    assert launch.count(S.hicache_line(32)) == 1
    assert "    MAXREQ=16," in launch and "    CUDAGRAPH_MAXBS=16," in launch and "    MAMBA_CACHE=96," in launch
    assert B.mamba_cache(16) // 5 >= 16
    # the tier is added after the REAP / prefetch args and before launch; the request marker asserts the size
    assert launch.index(B.L_PREFETCH) < launch.index("--enable-hierarchical-cache") < launch.index(B.L_LAUNCH)
    assert 'assert args[args.index("--hicache-size") + 1] == "32"' in launch
    for flag in ("--disable-radix-cache", "--enable-int8-mamba-checkpoint"):   # SGLang rejects hicache with these
        assert flag not in launch
    # the launch args evaluate to a list SGLang accepts: --hicache-size value right after its flag
    ns = {"args": []}
    exec(compile(S.hicache_line(32), "hic", "exec"), ns)
    assert ns["args"][ns["args"].index("--hicache-size") + 1] == "32"
    assert "STREAMS max_running_requests=16 cuda_graph_bs=" in \
        "STREAMS max_running_requests={} cuda_graph_bs={} mamba_cache={}".format(16, [1, 16], 96)


def _launcher_check(cells) -> str:
    launch = _one(cells, B.L_CFG)
    return launch[launch.index("_hc_line = next("):]


@pytest.mark.parametrize("which", ["launcher", "post-run"])
def test_tier_check_prints_the_marker_only_when_sglang_attached_it(which, tmp_path, capsys):
    cells = _cells(COMMITTED)
    code = _launcher_check(cells) if which == "launcher" else B.HIC_DUMP
    if which == "post-run":
        assert B.HIC_DUMP in _one(cells, "print('Starting benchmark...')")
    log = tmp_path / "serve.log"
    marker = B.HICACHE_MARKER
    cases = [("loading...\n", "HICACHE_TIER NOT CONFIRMED"),
             ("x\n" + ATTACHED.replace("hicache_attached=True", "hicache_attached=False") + "\n", "NOT ATTACHED"),
             ("x\n" + ATTACHED + "\nready\n", marker)]
    if which == "post-run":
        cases.insert(0, (None, "HICACHE_TIER post-run check failed"))    # missing serve.log: never raises
    for content, expect in cases:
        if content is not None:
            log.write_text(content)
        exec(compile(code, "hic_check", "exec"), {"Path": Path, "LOG": str(log)})
        out = capsys.readouterr().out
        assert expect in out, (expect, out)
        assert (marker in out) == (expect == marker), out
    assert "impl=UnifiedRadixCache" in out and "hicache_attached=True" in out


@pytest.mark.parametrize("argv,slug", [
    (["--turbo", "--prio-tail", "--hicache-gb"], SLUG),
    (["--turbo", "--prio-tail", "--hicache-gb", "32", "--streams", "16"], SLUG),
    (["--turbo", "--prio-tail", "--hicache-gb", "40", "--streams", "15"], "arc3-m2-turbo-tail-hic40gb-s15"),
    (["--turbo-lossless", "--prio-tail", "--hicache-gb"], "arc3-m2-turbo-lossless-tail-hic16"),
])
def test_cli_builds(tmp_path, monkeypatch, capsys, argv, slug):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    assert B.main(argv) == 0
    out = capsys.readouterr().out
    assert f"{slug}.ipynb" in out and "HICACHE_TIER attached hicache_attached=True" in out
    path = next(tmp_path.rglob(f"{slug}.ipynb"))
    assert check_notebook(path) == []


@pytest.mark.parametrize("argv,why", [
    (["--turbo", "--prio-tail", "--hicache-gb", "48"], "allowed 8..40"),
    (["--turbo", "--prio-tail", "--hicache-gb", "--streams", "17"], "allowed 11..14"),
    (["--turbo", "--prio-tail", "--hicache-gb", "--spec-accept", "0.6"], "conflicting"),
    (["--turbo", "--streams", "16"], "conflicting"),
    (["--reap", "--streams", "16"], "with --hicache-gb and --reap"),
    (["--hicache-gb", "--streams", "16"], "needs --reap"),
])
def test_refusals(tmp_path, monkeypatch, argv, why):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as exc:
        B.main(argv)
    assert why in str(exc.value)


def test_hicache_flags_and_cap_are_the_speed_builders():
    assert B.hicache_line is S.hicache_line and B.HICACHE_MAX_GB == S.HICACHE_MAX_GB == 40
    assert B.HICACHE_DEFAULT_GB == 32 and B.MAX_STREAMS_WITH_HICACHE == 16 and B.MAX_STREAMS == 14


def test_speed_report_compares_against_turbo_tail():
    import m2_speed_report as rep
    assert rep.slug("turbo-tail-hic16") == SLUG and rep.slug("turbo-tail") == "arc3-m2-turbo-tail"
