"""The serving-speed kernel builder: flags present, stream-count couplings consistent, metadata."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _build_m2_speed_kernels as B  # noqa: E402
from _check_notebook_cell import check_notebook  # noqa: E402

INCUMBENT = ROOT / "kaggle_submission_m2_level_memory" / "notebook"


def _cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def _cfg_value(cells: list[str], key: str) -> str:
    (cell,) = [c for c in cells if "CFG = dict(" in c]
    return re.search(rf"^    {key}=([^,]+),", cell, re.M).group(1)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("spd")
    return {n: B.build(n, k, root) for n, k in B.SPD_VARIANTS.items()}


def test_three_requested_variants_exist():
    assert set(B.SPD_VARIANTS) == {"m97s12", "m96s12hic", "m97s12hic"}


@pytest.mark.parametrize("name", list(B.SPD_VARIANTS))
def test_stream_count_is_consistent_everywhere(built, name):
    cells = _cells(built[name])
    assert _cfg_value(cells, "MAXREQ") == "12"
    assert _cfg_value(cells, "CUDAGRAPH_MAXBS") == "12"
    assert "'ARC3_MAX_ACTIVE_STREAMS': 12," in "".join(cells)
    assert "'ARC3_MAX_ACTIVE_STREAMS': 10," not in "".join(cells)
    # SGLang caps running requests at max_mamba_cache_size // 5: 12 streams need >= 60, and the
    # incumbent's 6 slots per stream (retained checkpoints) means 72.
    assert int(_cfg_value(cells, "MAMBA_CACHE")) // B.MAMBA_SLOTS_PER_REQUEST >= 12
    assert _cfg_value(cells, "MAMBA_CACHE") == "72"


@pytest.mark.parametrize("name,frac", [("m97s12", "0.97"), ("m96s12hic", "0.96"), ("m97s12hic", "0.97")])
def test_mem_fraction(built, name, frac):
    assert _cfg_value(_cells(built[name]), "MEMFRAC") == frac


@pytest.mark.parametrize("name,hic", [("m97s12", False), ("m96s12hic", True), ("m97s12hic", True)])
def test_hicache_flags_only_where_asked(built, name, hic):
    text = "".join(_cells(built[name]))
    flags = ['"--enable-hierarchical-cache"', '"--hicache-size", "32"',
             '"--hicache-write-policy", "write_through"', '"--hicache-io-backend", "kernel"']
    for f in flags:
        assert (f in text) is hic, f
    if hic:
        # appended after the base argument list, before the server is launched
        assert text.index('"--hicache-size"') < text.index("precache_model_thread.start()")


@pytest.mark.parametrize("name", list(B.SPD_VARIANTS))
def test_everything_else_is_the_incumbent(built, name):
    inc = _cells(INCUMBENT / "arc3-m2-level-memory.ipynb")
    new = _cells(built[name])[1:]            # [0] is the added markdown banner
    assert len(inc) == len(new)
    changed = [(a, b) for a, b in zip(inc, new) if a != b]
    # launcher CFG(+hicache), streams env (also the input resolver's helper + path constants), game list --
    # plus the resolver's two other cells (offline wheel install, offline environment files)
    assert len(changed) == 3 + 2
    assert sum("resolve_input(" in b or "COMPETITION_WHEELS_DIR" in b for _, b in changed) >= 3
    joined = "\n".join(b for _, b in changed)
    assert any("LEVEL_MEMORY installed" in c for c in new)
    assert "demo_excluded_games = []" in joined


@pytest.mark.parametrize("name", list(B.SPD_VARIANTS))
def test_metadata_identical_except_id_title_code_file(built, name):
    inc = json.loads((INCUMBENT / "kernel-metadata.json").read_text(encoding="utf-8"))
    new = json.loads((built[name].parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    diff = {k for k in set(inc) | set(new) if inc.get(k) != new.get(k)}
    assert diff == {"id", "title", "code_file", "docker_image_pinning_type"}
    assert new["docker_image_pinning_type"] == "original" and new["docker_image"] == inc["docker_image"]
    assert new["id"] == f"calamitychasm/arc3-m2-spd-{name}" == f"calamitychasm/{new['title']}"
    assert new["code_file"] == built[name].name
    assert new["is_private"] and new["enable_gpu"] and new["machine_shape"] == "NvidiaRtxPro6000"


@pytest.mark.parametrize("name", list(B.SPD_VARIANTS))
def test_notebook_cells_pass_the_static_checker(built, name):
    assert check_notebook(built[name]) == []


def test_committed_notebooks_match_the_builder(built):
    for name, path in built.items():
        committed, nb_name, _ = B.layout(name)
        assert committed.parts[-2:] == (f"kaggle_submission_m2_spd_{name}", "notebook")
        assert (committed / nb_name).read_text(encoding="utf-8") == path.read_text(encoding="utf-8"), name


def test_inconsistent_knobs_are_refused():
    with pytest.raises(SystemExit):   # 12 streams on 55 Mamba slots would be capped silently
        B.check_consistency("m97s12", dict(MAXREQ=12, CUDAGRAPH_MAXBS=12, MAMBA_CACHE=55, MEMFRAC=0.97))
    with pytest.raises(SystemExit):   # 0.98 is the known runtime OOM
        B.check_consistency("m97s12", dict(MAXREQ=12, CUDAGRAPH_MAXBS=12, MAMBA_CACHE=72, MEMFRAC=0.98))
    with pytest.raises(SystemExit):   # host RAM ceiling
        B.check_consistency("m96s12hic", dict(MAXREQ=12, CUDAGRAPH_MAXBS=12, MAMBA_CACHE=72, HICACHE_GB=48))
    with pytest.raises(SystemExit):
        B.check_consistency("x", dict(MAXREQ=12, CUDAGRAPH_MAXBS=10, MAMBA_CACHE=72))


def test_legacy_variants_unchanged():
    assert B.layout("base")[2] == "arc3-m2-speed-base"
    assert B.layout("s14hic")[2] == "arc3-m2-spd2-s14hic"
    assert B.VARIANTS["s14hic"]["HICACHE_GB"] == 32
