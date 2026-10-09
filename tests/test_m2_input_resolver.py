"""The mount-layout input resolver baked into every derived milestone-2 notebook.

Kaggle sessions of one kernel get inputs at /kaggle/input/{datasets/<owner>,competitions,models/<owner>}/... or at
the older /kaggle/input/<slug>; a notebook that hardcodes one fails at random (JustAdev742: 5 of 9 sessions).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import _m2_input_resolver as R  # noqa: E402
import _build_m2_level_memory_kernel as LM  # noqa: E402

INCUMBENT_NB = ROOT / "kaggle_submission_m2_level_memory" / "notebook" / "arc3-m2-level-memory.ipynb"
WHEELHOUSE = "/kaggle/input/datasets/dfranzen/pennyroyal-v253"
MODEL = "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"
COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"


def load_helper(root):
    spec = importlib.util.spec_from_file_location("input_resolver_under_test", R.HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.INPUT_ROOT = str(root)
    return mod


@pytest.fixture
def mod(tmp_path):
    return load_helper(tmp_path)


def mk(tmp, rel):
    d = tmp / rel
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- the helper itself -------------------------------------------------------------------------------------

def test_new_layout_dataset_is_found_in_place(mod, tmp_path, capsys):
    d = mk(tmp_path, "datasets/dfranzen/pennyroyal-v253")
    assert mod.resolve_input("wheelhouse", WHEELHOUSE) == str(d)
    assert f"INPUT_RESOLVED wheelhouse -> {d} (via canonical)" in capsys.readouterr().out


def test_old_layout_dataset_is_found(mod, tmp_path, capsys):
    d = mk(tmp_path, "pennyroyal-v253")
    assert mod.resolve_input("wheelhouse", WHEELHOUSE) == str(d)
    assert f"INPUT_RESOLVED wheelhouse -> {d} (via older layout)" in capsys.readouterr().out


def test_prefers_the_canonical_path_when_both_exist(mod, tmp_path):
    new = mk(tmp_path, "datasets/dfranzen/pennyroyal-v253")
    mk(tmp_path, "pennyroyal-v253")
    assert mod.resolve_input("wheelhouse", WHEELHOUSE) == str(new)


def test_glob_fallback_finds_the_slug_anywhere_within_depth(mod, tmp_path, capsys):
    d = mk(tmp_path, "somewhere/else/Pennyroyal-V253")           # depth 3, different case, unknown parents
    assert mod.resolve_input("wheelhouse", WHEELHOUSE) == str(d)
    assert "(via glob)" in capsys.readouterr().out


def test_glob_fallback_respects_the_depth_limit(mod, tmp_path):
    mk(tmp_path, "a/b/c/d/pennyroyal-v253")                      # depth 5
    with pytest.raises(FileNotFoundError):
        mod.resolve_input("wheelhouse", WHEELHOUSE)


def test_glob_with_a_sub_path_requires_the_sub_path(mod, tmp_path):
    mk(tmp_path, "x/pennyroyal-v253/other")
    with pytest.raises(FileNotFoundError):
        mod.resolve_input("wheels", WHEELHOUSE + "/wheels")
    mk(tmp_path, "y/pennyroyal-v253/wheels")
    assert mod.resolve_input("wheels", WHEELHOUSE + "/wheels") == str(tmp_path / "y/pennyroyal-v253/wheels")


@pytest.mark.parametrize("rel", ["models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
                                 "intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
                                 "intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
                                 "models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"])
def test_model_layouts_and_framework_case(mod, tmp_path, rel):
    d = mk(tmp_path, rel)
    assert mod.resolve_input("model", MODEL) == str(d)


def test_competition_both_layouts(mod, tmp_path):
    new = mk(tmp_path, "competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels")
    assert mod.resolve_input("competition_wheels", COMP) == str(new)
    old_root = tmp_path / "old"
    old = mk(old_root, "arc-prize-2026-arc-agi-3/arc_agi_3_wheels")
    mod.INPUT_ROOT = str(old_root)
    assert mod.resolve_input("competition_wheels", COMP) == str(old)


def test_absent_input_fails_loudly_and_lists_what_was_searched(mod, tmp_path, capsys):
    mk(tmp_path, "unrelated-dataset/files")
    with pytest.raises(FileNotFoundError, match="INPUT_MISSING wheelhouse"):
        mod.resolve_input("wheelhouse", WHEELHOUSE)
    out = capsys.readouterr().out
    assert "INPUT_MISSING wheelhouse: expected " + WHEELHOUSE in out
    assert "INPUT_RESOLVED" not in out
    assert f"searched {tmp_path}/datasets/dfranzen/pennyroyal-v253" in out
    assert f"searched {tmp_path}/pennyroyal-v253" in out
    assert "unrelated-dataset" in out                              # the listing of what IS mounted


def test_no_input_root_at_all_fails_loudly(mod, tmp_path, capsys):
    mod.INPUT_ROOT = str(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        mod.resolve_input("wheelhouse", WHEELHOUSE)
    assert "INPUT_MISSING wheelhouse" in capsys.readouterr().out


def test_a_path_outside_kaggle_input_is_a_programming_error(mod):
    with pytest.raises(ValueError):
        mod.resolve_input("x", "/tmp/foo")


# ---- the builders --------------------------------------------------------------------------------------------

def all_variant_notebooks():
    out = sorted(ROOT.glob("kaggle_submission_m2_*/notebook/*.ipynb")) + sorted(ROOT.glob("kaggle_submission_m2_speed/*/*.ipynb"))
    return [p for p in out if p != INCUMBENT_NB]


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def test_the_incumbent_v1_is_untouched_and_has_no_resolver():
    text = INCUMBENT_NB.read_text(encoding="utf-8")
    assert "INPUT RESOLVER" not in text and "resolve_input" not in text
    assert "ORIG_BUNDLE_DIR   = '/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy'" in text


def test_only_variants_resolve_inputs():
    assert not LM.resolves_inputs(())
    assert R.MARKER not in LM.kernel_markers(())
    for v in (("histcache",), ("triedfacts",), ("inputs",), LM.TURBO):
        assert LM.resolves_inputs(v)
        assert LM.kernel_markers(v).count(R.MARKER) == 1
    assert LM.kernel_slug(("inputs",)) == "arc3-m2-lm-inputs"


def test_every_derived_notebook_carries_the_resolver_and_no_bare_input_literal():
    paths = all_variant_notebooks()
    assert len(paths) >= 14
    for p in paths:
        nb = load(p)
        cells = ["".join(c["source"]) for c in nb["cells"]]
        host = [s for s in cells if R.ANCHOR in s]
        assert len(host) == 1 and "def resolve_input(" in host[0], p
        assert host[0].index("def resolve_input(") < host[0].index(R.ANCHOR), p
        assert f"{R.COMPETITION_NAME} = resolve_input('competition_wheels', '{R.COMPETITION_WHEELS}')" in host[0], p
        assert len(re.findall(r"^[A-Z_]+_DIR += resolve_input\(", host[0], re.M)) >= 4, p
        launcher = {i for i, s in enumerate(cells)      # the vLLM setup step and launcher keep their own resolvers
                    if s.startswith("%%writefile /kaggle/arc3_vllm_setup.py") or "_first_dir(VLLM_RUNTIME_DIR" in s}
        assert R.unresolved_literals(nb, skip=launcher) == [], p
        assert not any(R.COMPETITION_WHEELS in s for i, s in enumerate(cells) if R.ANCHOR not in s), p


def test_marker_and_builder_agree_on_the_log_line():
    assert "INPUT_RESOLVED" in R.helper_source() and R.MARKER == "INPUT_RESOLVED"


def test_apply_refuses_a_notebook_without_the_anchor():
    nb = {"cells": [{"cell_type": "code", "source": ["x = 1\n"]}]}
    with pytest.raises(SystemExit):
        R.apply_input_resolver(nb)


def test_a_built_notebook_resolves_in_the_older_layout(tmp_path):
    """Run the notebook's own resolver lines against a fake older-layout /kaggle/input."""
    nb = load(ROOT / "kaggle_submission_m2_lm_histcache" / "notebook" / "arc3-m2-lm-histcache.ipynb")
    host = next("".join(c["source"]) for c in nb["cells"] if R.ANCHOR in "".join(c["source"]))
    a, b = host.index("# >>> [calamitychasm] INPUT RESOLVER"), host.index("# <<< [calamitychasm] INPUT RESOLVER")
    helper = host[a:b]
    consts = "\n".join(ln for ln in host.splitlines() if re.match(r"^[A-Z_]+ += resolve_input\(", ln))
    for rel in ("pennyroyal-v253/wheels", "taaf-kaggle-source-bundle-copy/src",
                "intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
                "albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
                "arc-prize-2026-arc-agi-3/arc_agi_3_wheels"):
        mk(tmp_path, rel)
    ns: dict = {}
    exec(compile(helper, "helper", "exec"), ns)
    ns["INPUT_ROOT"] = str(tmp_path)
    exec(compile(consts, "consts", "exec"), ns)
    assert ns["WHEELHOUSE_DIR"] == str(tmp_path / "pennyroyal-v253")
    assert ns["ORIG_BUNDLE_DIR"] == str(tmp_path / "taaf-kaggle-source-bundle-copy")
    assert ns["MODEL_DIR"].endswith("intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1")
    assert Path(ns["DRAFT_MODEL_DIR"]).is_dir()
    assert ns["COMPETITION_WHEELS_DIR"] == str(tmp_path / "arc-prize-2026-arc-agi-3" / "arc_agi_3_wheels")
    assert len(ns["RESOLVED_INPUTS"]) == 5


def test_a_built_notebook_fails_loudly_when_an_input_is_absent(tmp_path, capsys):
    nb = load(ROOT / "kaggle_submission_m2_speed" / "base" / "arc3-m2-speed-base.ipynb")
    host = next("".join(c["source"]) for c in nb["cells"] if R.ANCHOR in "".join(c["source"]))
    helper = host[host.index("# >>> [calamitychasm] INPUT RESOLVER"):host.index("# <<< [calamitychasm] INPUT RESOLVER")]
    consts = "\n".join(ln for ln in host.splitlines() if re.match(r"^[A-Z_]+ += resolve_input\(", ln))
    mk(tmp_path, "pennyroyal-v253")                      # the wheelhouse is there, the bundle is not
    ns: dict = {}
    exec(compile(helper, "helper", "exec"), ns)
    ns["INPUT_ROOT"] = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="INPUT_MISSING"):
        exec(compile(consts, "consts", "exec"), ns)
    out = capsys.readouterr().out
    assert "INPUT_RESOLVED wheelhouse" in out and "INPUT_MISSING" in out
