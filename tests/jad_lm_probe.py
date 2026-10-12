"""Probe for tests/test_jad_exp081_lm.py, run in a subprocess (it imports a harness tree other than tests/m2_harness.py's).

    python tests/jad_lm_probe.py NOTEBOOK TREE_DIR HIS_REPO

Builds /kaggle/taaf-kaggle-source-share exactly as NOTEBOOK's cells build it (Franzen's patch, then the six exp-081 patches
shipped in the notebook's --compact cells, via JustAdev742's vendored franzen_tree.py), turns on exp-081's harness flags,
executes NOTEBOOK's own level-memory install cell against that tree, plays a level-up and two context trims on the real
ToolAgent, and prints one JSON line of findings.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "kaggle_submission_jad_exp081" / "vendor"
EXP081_FLAGS = ("OURS_BUDGET_METER", "OURS_WIN_LEDGER", "OURS_SEARCH_HELPER", "OURS_LEVEL_MEM", "OURS_PERCEPTION")


def main(notebook: Path, tree: Path, his_repo: Path) -> dict:
    sys.path.insert(0, str(VENDOR / "scripts"))
    sys.path.insert(0, str(REPO / "tests"))
    import franzen_tree as FT
    import m2_harness

    if not (tree / "src" / "ARC3-Inference").is_dir():
        FT.notebook_bundle(tree, FT.our_patch_cells(notebook), his_repo=his_repo, his_patch=FT.his_patch_text(notebook))
    cells = FT.notebook_cells(notebook)
    install = [c for c in cells if c.startswith("# [calamitychasm] SOLVED-LEVEL MEMORY")]
    assert len(install) == 1, len(install)

    os.environ.update(m2_harness.notebook_env())         # Franzen's knobs (module constants read them at import)
    os.environ.update({k: "1" for k in EXP081_FLAGS})   # exp-081's --env-add flags
    os.environ["ARC3_MAX_ACTIVE_STREAMS"] = "14"
    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")
    os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "flashnext")
    TA, RS, _, _ = m2_harness.import_harness(tree / "src")
    ns: dict = {}
    exec(compile(install[0], "lm_install_cell", "exec"), ns)  # the shipped cell, verbatim
    lm = ns["LEVEL_MEMORY"]
    cls = TA.ToolAgent
    out = {"ledger_patch": callable(getattr(cls, "_ours_ledger_pin", None)),
           "ledger_on": TA._ours_win_ledger_enabled(),
           "wrapped": {n: getattr(cls, n).__module__ for n in ("_build_user_prompt", "_trim_messages_for_context",
                                                              "_ensure_session")}}

    agent = cls(model="local")
    agent._ensure_session(Path(tempfile.mkdtemp()) / "state.json")
    base = agent._system_prompt

    def frame(level, step):
        return RS.Frame(grid=[[0] * 8 for _ in range(8)], step=step, level=level)

    H = RS.HistoryEntry
    hist = [H(action="", frame=frame(1, 0)), H(action="ACTION1", frame=frame(1, 1), result={"executed": True}),
            H(action="ACTION2", frame=frame(2, 2), result={"executed": True, "level_completed": True})]
    opener = agent._build_user_prompt(2, valid_actions=["ACTION1", "ACTION2"], current_frame=frame(2, 2),
                                      history_entries=hist,
                                      previous_step_summary={"level_transition": True, "level": 2,
                                                             "executed_actions": ["ACTION2"]})
    out["opener_win_record"] = "Level 1 record (exact facts from the recorded boards)" in opener
    out["opener_rule_request"] = "starts exactly with 'Rule for level 1:'" in opener
    out["opener_order_ok"] = opener.find("Level 1 record") < opener.find("LEVEL 1 SOLVED.")
    rule = "Rule for level 1: ACTION1 moves the piece right; ACTION2 then clears the level."
    agent._history_messages.append({"role": "assistant", "content": "ok", "reasoning_content": rule})
    agent._build_user_prompt(3, valid_actions=["ACTION1"], current_frame=frame(2, 3),
                             history_entries=hist + [H(action="ACTION1", frame=frame(2, 3), result={"executed": True})],
                             previous_step_summary={"level": 2})
    out["rule"] = agent._lm_state["levels"][0]["rule"]

    filler = "x " * 40000
    turns = []
    for i in range(46):
        turns += [{"role": "user", "content": f"turn {i} {filler}"}, {"role": "assistant", "content": f"ok {i}"}]
    agent._context_was_trimmed = False
    first = agent._trim_messages_for_context([{"role": "system", "content": base}, *turns[:80]])
    out["first_trim_evicted"] = len(first) < 81
    out["first_trim_block"] = lm.HEADER in first[0]["content"] and first[0]["content"].startswith(base)
    # a later request: history has grown, the head moves, so the ledger is pinned right after the system prompt
    second = agent._trim_messages_for_context([{"role": "system", "content": agent._system_prompt}, *turns])
    out["second_trim_ledger_after_system"] = TA._ours_is_ledger(second[1])
    out["second_trim_block"] = lm.HEADER in second[0]["content"]
    out["ledger_text"] = second[1]["content"] if TA._ours_is_ledger(second[1]) else ""
    out["summary"] = lm.summary()
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))))
