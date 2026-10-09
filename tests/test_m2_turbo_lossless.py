"""Turbo-lossless kernel (arc3-m2-turbo-lossless): turbo without the lossy MTP acceptance, and the request-log
metrics (output tokens per request, repeated assistant turns) that guard lossy acceptance on the check run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_m2_level_memory_kernel as B  # noqa: E402
import m2_speed_report as rep  # noqa: E402
from _check_notebook_cell import check_notebook  # noqa: E402

COMMITTED = ROOT / "kaggle_submission_m2_turbo_lossless" / "notebook"
NB = COMMITTED / "arc3-m2-turbo-lossless.ipynb"
TURBO_NB = ROOT / "kaggle_submission_m2_turbo" / "notebook" / "arc3-m2-turbo.ipynb"


def _cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def _one(cells, needle):
    hits = [c for c in cells if needle in c]
    assert len(hits) == 1, (needle, len(hits))
    return hits[0]


# ---------------------------------------------------------------------------------------------- builder

def test_preset_slug_dir_and_components():
    assert set(B.TURBO_LOSSLESS) == set(B.TURBO) - {"acc50"}
    assert B.kernel_slug(B.TURBO_LOSSLESS) == "arc3-m2-turbo-lossless"
    assert B.kernel_id(reversed(B.TURBO_LOSSLESS)) == "calamitychasm/arc3-m2-turbo-lossless"
    assert B.kernel_dir(B.TURBO_LOSSLESS).relative_to(ROOT).as_posix() == "kaggle_submission_m2_turbo_lossless/notebook"
    assert B.accept_value(B.TURBO_LOSSLESS) is None and B.streams_value(B.TURBO_LOSSLESS) == 14
    assert B.kernel_slug(B.TURBO) == "arc3-m2-turbo"          # unchanged
    assert B.kernel_slug(("reap", "s14")) == "arc3-m2-lm-reap-s14"


def test_build_is_deterministic_and_committed_kernel_is_current(tmp_path, monkeypatch):
    committed = NB.read_text(encoding="utf-8")
    outs = []
    for i in range(2):
        monkeypatch.setattr(B, "ROOT", tmp_path / str(i))
        outs.append(B.build(B.TURBO_LOSSLESS).read_text(encoding="utf-8"))
    monkeypatch.undo()
    assert outs[0] == outs[1], "builder is not deterministic"
    assert outs[0] == committed, "committed turbo-lossless kernel is stale: rerun the build"
    assert check_notebook(NB) == []
    assert len(committed.encode("utf-8")) < 900_000


def test_other_committed_kernels_still_rebuild_byte_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    for variants in [B.TURBO, ("histcache",), ("triedfacts",), ("histcache", "triedfacts"), ("inputs",)]:
        new = B.build(variants)
        rel = new.relative_to(tmp_path)
        assert new.read_text(encoding="utf-8") == (ROOT / rel).read_text(encoding="utf-8"), rel
        assert (new.parent / "kernel-metadata.json").read_text() == (ROOT / rel.parent / "kernel-metadata.json").read_text()


def test_metadata_matches_incumbent_except_identity():
    inc = json.loads((ROOT / "kaggle_submission_m2_level_memory" / "notebook" / "kernel-metadata.json").read_text())
    new = json.loads((COMMITTED / "kernel-metadata.json").read_text())
    assert new["id"] == "calamitychasm/arc3-m2-turbo-lossless" and new["code_file"] == "arc3-m2-turbo-lossless.ipynb"
    assert new["title"] == "arc3-m2-turbo-lossless"
    for k in ("id", "title", "code_file"):
        inc.pop(k), new.pop(k)
    assert inc == new


def test_acceptance_stays_lossless_and_markers_exclude_it():
    cells = _cells(NB)
    launch = _one(cells, "CFG = dict(\n")
    assert "    SPEC_ACCEPT_SINGLE=1.0,\n" in launch and "    SPEC_ACCEPT_ACC=1.0,\n" in launch
    assert "SPEC_ACCEPT_SINGLE=0.5" not in launch and "lossy MTP acceptance" not in launch
    assert "SPEC_ACCEPT" not in "\n".join(c for c in cells if c is not launch).replace("SPEC_ACCEPT_", "")
    markers = B.kernel_markers(B.TURBO_LOSSLESS)
    assert not any(m.startswith("SPEC_ACCEPT") for m in markers)
    assert markers[:3] == ["LEVEL_MEMORY installed", "priority gate active", "harness patch applied successfully"]
    for m in ("HISTORY_CACHE installed", "TIMEOUT_FIX installed", "REAP448 applied kept=448",
              "priority gate active: 14 concurrent streams", B.INPUT_MARKER):
        assert m in markers
    assert f"ARC_HOTMAP sha={B.hot_map()['sha']}" in markers
    assert markers == [m for m in B.kernel_markers(B.TURBO) if m != "SPEC_ACCEPT 0.5"]
    assert B.kernel_counters(B.TURBO_LOSSLESS) == B.kernel_counters(B.TURBO)


def test_serving_values_are_turbos_apart_from_acceptance():
    launch = _one(_cells(NB), "CFG = dict(\n")
    for line in ("    MAXREQ=14,", "    CUDAGRAPH_MAXBS=14,", "    MAMBA_CACHE=84,", "    MEMFRAC=0.96,",
                 "    SPEC_STEPS=3,"):
        assert line in launch, line
    assert 'env["ARC3_REAP_KEPT_EXPERTS"]' in launch and "'ARC3_MAX_ACTIVE_STREAMS': 14," in _one(_cells(NB), "setup_env = {")


def test_only_the_expected_cells_differ_from_turbo():
    a, b = _cells(TURBO_NB), _cells(NB)
    assert len(a) == len(b)
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(diff) == 2, diff
    header, launcher = diff
    assert a[header].startswith("## [calamitychasm] fork") and b[header].startswith("## [calamitychasm] fork")
    assert "Variant `acc50`" in a[header] and "acc50" not in b[header] and "acceptance stays" in b[header]
    assert "CFG = dict(\n" in a[launcher] and "CFG = dict(\n" in b[launcher]
    # the launcher differs only in the three acceptance edits
    xs, ys = a[launcher].splitlines(), b[launcher].splitlines()
    removed = [x for x in xs if x not in ys]
    added = [y for y in ys if y not in xs]
    assert len(removed) == 5 and len(added) == 2, (removed, added)
    assert sum("SPEC_ACCEPT" in r or "accept-threshold" in r for r in removed) == 5
    assert added == ["    SPEC_ACCEPT_SINGLE=1.0,", "    SPEC_ACCEPT_ACC=1.0,"]


def test_cli_preset_builds_and_refuses_conflicts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    assert B.main(["--turbo-lossless"]) == 0
    assert "arc3-m2-turbo-lossless.ipynb" in capsys.readouterr().out
    assert next(tmp_path.rglob("arc3-m2-turbo-lossless.ipynb")).read_text(encoding="utf-8") == NB.read_text(encoding="utf-8")
    # the equivalent explicit flags land on the same preset
    assert B.main(["--history-cache", "--timeout-fix", "--reap", "--arc-hotmap", "--streams", "14",
                   "--check-all25"]) == 0
    for argv, why in ((["--turbo-lossless", "--spec-accept", "0.5"], "conflicting"),
                      (["--turbo-lossless", "--streams", "12"], "conflicting"),
                      (["--turbo-lossless", "--turbo"], "different presets")):
        with pytest.raises(SystemExit) as exc:
            B.main(argv)
        assert why in str(exc.value)


# ---------------------------------------------------------------------------------- request-log metrics

def _resp(messages, completion=100, prompt=1000):
    return json.dumps({"event": "response", "messages": messages, "tools": [], "finish_reason": "stop",
                       "usage": {"prompt_tokens": prompt, "completion_tokens": completion}})


def _req(messages):
    return json.dumps({"event": "request", "messages": messages, "tools": []})


def _a(text, **kw):
    return {"role": "assistant", "content": text, **kw}


U = {"role": "user", "content": "go"}
LONG = "x" * 300


def _write(d: Path, game: str, lines: list[str]) -> None:
    (d / f"{game}_p0_requests.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_request_log_mean_completion_tokens(tmp_path):
    _write(tmp_path, "aa11", [_req([U]), _resp([U], 100), _resp([U, _a("one")], 300)])
    _write(tmp_path, "bb22", [_resp([U], 200)])
    m = rep.request_log_metrics(tmp_path)
    assert m["requests"] == 3 and m["completion_tokens_per_request"] == 200.0
    assert m["request_log_games"] == 2 and m["prompt_tokens_per_request"] == 1000.0


def test_no_repeats_in_a_growing_history_with_trimming_and_retries(tmp_path):
    h1, h2, h3, h4 = (_a(f"turn {i} {LONG}") for i in range(4))
    lines = [_resp([U, h1]), _resp([U, h1]),                    # a retried request: same history, not a new turn
             _resp([U, h1, h2]), _resp([U, h1, h2, h3]),
             _resp([U, h2, h3, h4])]                            # oldest turn trimmed away
    _write(tmp_path, "g", lines)
    m = rep.request_log_metrics(tmp_path)
    assert m["assistant_turns"] == 4 and m["repeated_assistant_turns"] == 0
    assert m["repeated_assistant_turns_long"] == 0 and m["repeat_examples"] == []


def test_exact_repeat_of_the_previous_turn_is_counted(tmp_path):
    h1, h2 = _a("alpha " + LONG), _a("beta " + LONG)
    loop = _a("stuck " + LONG)
    lines = [_resp([U, h1]), _resp([U, h1, h2]), _resp([U, h1, h2, loop]),
             _resp([U, h1, h2, loop, loop]),                    # the model emitted the same turn again
             _resp([U, h1, h2, loop, loop, loop])]              # and again
    _write(tmp_path, "g", lines)
    m = rep.request_log_metrics(tmp_path)
    assert m["assistant_turns"] == 5
    assert m["repeated_assistant_turns"] == 2 and m["repeated_assistant_turns_long"] == 2
    assert len(m["repeat_examples"]) == 2 and m["repeat_examples"][0].startswith("g_p0_requests.jsonl")


def test_repeat_is_per_game_and_short_repeats_are_reported_separately(tmp_path):
    same = _a("same " + LONG)
    _write(tmp_path, "g1", [_resp([U, same])])
    _write(tmp_path, "g2", [_resp([U, same])])                  # same text in another game: not a repeat
    assert rep.request_log_metrics(tmp_path)["repeated_assistant_turns"] == 0
    tool = _a("", tool_calls=[{"function": {"name": "act", "arguments": '{"a": 1}'}}])
    _write(tmp_path, "g3", [_resp([U, tool]), _resp([U, tool, tool])])
    m = rep.request_log_metrics(tmp_path)
    assert m["repeated_assistant_turns"] == 1 and m["repeated_assistant_turns_long"] == 0


def test_request_log_tolerates_junk_and_missing_files(tmp_path):
    assert rep.request_log_metrics(tmp_path) == {
        "request_log_games": 0, "requests": 0, "completion_tokens_per_request": None,
        "prompt_tokens_per_request": None, "assistant_turns": 0, "repeated_assistant_turns": 0,
        "repeated_assistant_turns_long": 0, "repeat_examples": []}
    _write(tmp_path, "g", ['{"event": "response", "messages": [', _resp([U, _a("ok")], 50),
                           _resp([U], 70).replace('"completion_tokens": 70', '"completion_tokens": null')])
    m = rep.request_log_metrics(tmp_path)
    assert m["requests"] == 1 and m["completion_tokens_per_request"] == 50.0 and m["assistant_turns"] == 1


def test_digest_reports_request_log_metrics(tmp_path):
    import kaggle_ops as ko
    h1 = _a("one " + LONG)
    _write(tmp_path, "g", [_resp([U, h1], 120), _resp([U, h1, h1], 80)])
    d = ko.digest(tmp_path, [])
    assert d["request_log"]["completion_tokens_per_request"] == 100.0
    assert d["request_log"]["repeated_assistant_turns"] == 1


def test_speed_report_knows_the_turbo_slugs():
    assert rep.slug("turbo") == "arc3-m2-turbo" and rep.slug("turbo-lossless") == "arc3-m2-turbo-lossless"
