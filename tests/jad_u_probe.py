"""Probe for tests/test_jad_exp081_u.py, run in a subprocess (it imports a harness tree other than tests/m2_harness.py's).

    python tests/jad_u_probe.py NOTEBOOK TREE_DIR HIS_REPO [--without-latch-fix]

Builds /kaggle/taaf-kaggle-source-share exactly as NOTEBOOK's cells build it (Franzen's patch, then every
/kaggle/ours-*.patch the notebook writes, in cell order, via JustAdev742's vendored franzen_tree.py; with
--without-latch-fix the last one, our stale-latch fix, is left out, i.e. arm C's tree plus ours-10), sets the
notebook's own setup_env, executes NOTEBOOK's C+U install cell verbatim, and drives the REAL ToolAgent + sandbox:

- latch: one snippet whose first action is a no-op, whose second is refused and caught, and whose third changes the
  board; a second snippet that catches refusals in a loop of no-ops (the per-snippet cap);
- notice: two user prompts 700 s apart on a level with objects, the second carrying the I1 line, then three more
  actions of which the second clicks a named object.

Prints one JSON line of findings.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "kaggle_submission_jad_exp081" / "vendor"
INSTALL_HEAD = "# [calamitychasm] C+U INSTALL"


def setup_env(cells: list[str]) -> dict[str, str]:
    """The notebook's own setup_env dict + priority block (as m2_harness.notebook_env reads the upstream cell)."""
    cell = next(s for s in cells if "setup_env = {" in s)
    body = cell[cell.index("setup_env = {") + len("setup_env = "):]
    body = body[: body.index("\n}\n") + 2]
    env = eval(body, {"__builtins__": {}}, {"SERVER_BASE_URL": "http://127.0.0.1:9/v1",  # noqa: S307 (our notebook)
                                            "SERVED_MODEL_NAME": "flashnext"})
    start = cell.index("setup_env.update(") + len("setup_env.update(")
    env.update(eval(cell[start: cell.index("})", start) + 1], {"__builtins__": {}}, {}))  # noqa: S307
    return {k: str(v) for k, v in env.items()}


def main(notebook: Path, tree: Path, his_repo: Path, with_fix: bool) -> dict:
    sys.path.insert(0, str(VENDOR / "scripts"))
    sys.path.insert(0, str(REPO / "tests"))
    import franzen_tree as FT
    import m2_harness

    patches = FT.our_patch_cells(notebook)
    if not with_fix:
        assert patches[-1][0].endswith("calamitychasm-stale-latch-fix.patch"), patches[-1][0]
        patches = patches[:-1]
    if not (tree / "src" / "ARC3-Inference").is_dir():
        FT.notebook_bundle(tree, patches, his_repo=his_repo, his_patch=FT.his_patch_text(notebook))
    cells = FT.notebook_cells(notebook)
    os.environ.update(setup_env(cells))
    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")
    TA, RS, SB, S = m2_harness.import_harness(tree / "src")
    import inference.utils.ours_untried as UT

    out: dict = {"patches": [p for p, _ in patches], "untried_mode": sorted(UT.mode()),
                 "stale_block": os.environ.get("ARC3_STALE_STATE_BLOCK")}
    install = [c for c in cells if c.startswith(INSTALL_HEAD)]
    assert len(install) == 1, len(install)
    ns: dict = {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(install[0], "cu_install_cell", "exec"), ns)  # the shipped cell, verbatim
        out["install_error"] = ""
    except Exception as exc:  # noqa: BLE001 - the unfixed tree must refuse to install
        out["install_error"] = f"{type(exc).__name__}: {exc}"
    out["install_stdout"] = buf.getvalue()
    cu = ns.get("CU_COUNTERS")

    # --- latch, on the real ToolAgent + sandbox -------------------------------------------------------------------
    class StaleGame(m2_harness.FakeGame):
        """FakeGame plus the solver's stale-state rule (guards active): a call carrying stale_after is refused."""

        def step_env(self, arguments):
            if arguments.get("stale_after") and str(arguments.get("query") or "") != "animation":
                name = str((arguments.get("actions") or [{}])[0].get("action"))
                return {"executed": False, "stop_reason": "stale_state", "error": "stale",
                        "stop_detail": f"{name} was not executed (stale_after {arguments['stale_after']})",
                        "requested_count": len(arguments.get("actions") or []), "executed_count": 0,
                        "stopped_early": True, "requested_actions": [name], "skipped_actions": [name],
                        "blocked_action": name}
            return super().step_env(arguments)

    tmp = Path(tempfile.mkdtemp())
    sp = tmp / "latch" / "g_tool_runtime_state.json"
    sp.parent.mkdir(parents=True)
    game = StaleGame(RS, S, sp, level_every=10**9, die_every=10**9)
    game.seed()
    agent = m2_harness.make_agent(TA, sp, game)
    agent._python_timeout = 60
    catch = ('r = action("ACTION5")\nprint("FIRST", r.get("executed"), r.get("gameplay_changed"))\n'
             'try:\n    action("ACTION1")\n    print("SECOND ran")\nexcept StaleStateActionError:\n'
             '    print("SECOND refused")\n'
             'try:\n    r = action("ACTION2")\n    print("THIRD", r.get("executed"), r.get("gameplay_changed"))\n'
             'except StaleStateActionError:\n    print("THIRD refused")\n')
    n0 = game.n
    res = agent._run_python_tool(sp, {"code": catch})
    out["catch_content"] = str(res.content)[-1500:]
    out["catch_executed"] = game.n - n0
    loop = ('ex = ref = 0\nfor _ in range(10):\n    try:\n        r = action("ACTION5")\n        ex += 1\n'
            '    except StaleStateActionError:\n        ref += 1\nprint("LOOP", ex, ref)\n')
    n0 = game.n
    res = agent._run_python_tool(sp, {"code": loop})
    out["loop_content"] = str(res.content)[-1500:]
    out["loop_executed"] = game.n - n0
    out["stale_summary"] = cu.stale_summary() if cu else None

    # --- the I1 notice and its click counter ----------------------------------------------------------------------
    grid = [[0] * 16 for _ in range(16)]
    for r in range(3, 6):
        for c in range(3, 6):
            grid[r][c] = 2  # a 3x3 block of colour 2
    for r in range(9, 11):
        for c in range(9, 13):
            grid[r][c] = 3  # a 2x4 bar of colour 3
    frame = lambda step: RS.Frame(grid=tuple(tuple(x) for x in grid), step=step, level=1)  # noqa: E731
    H = RS.HistoryEntry
    clock = [1000.0]
    UT._clock = lambda: clock[0]
    na = tmp / "notice" / "g_tool_runtime_state.json"
    na.parent.mkdir(parents=True)
    ag = TA.ToolAgent(model="local")
    ag._ensure_session(na)
    valid = ["ACTION1", "ACTION6"]
    hist = [H(action="", frame=frame(0)), H(action="ACTION1", frame=frame(1), result={"executed": True})]
    first = ag._build_user_prompt(2, valid_actions=valid, current_frame=frame(1), history_entries=hist,
                                  previous_step_summary={"level": 1})
    clock[0] = 1700.0
    second = ag._build_user_prompt(2, valid_actions=valid, current_frame=frame(1), history_entries=hist,
                                   previous_step_summary={"level": 1})
    out["notice_first"] = [ln for ln in first.splitlines() if ln.startswith("Not yet tried")]
    out["notice_second"] = [ln for ln in second.splitlines() if ln.startswith("Not yet tried")]
    out["notice_before_tool_line"] = second.find("Not yet tried") < second.find("Only tool: `python`")
    hist += [H(action="ACTION1", frame=frame(2), result={"executed": True}),
             H(action="MOUSE(row=4, col=4)", frame=frame(3), result={"executed": True}),
             H(action="ACTION1", frame=frame(4), result={"executed": True})]
    clock[0] = 1710.0
    ag._build_user_prompt(5, valid_actions=valid, current_frame=frame(4), history_entries=hist,
                          previous_step_summary={"level": 1})
    files = cu.finish() if cu else {}
    out["untried_summary"] = files.get("untried_summary.json")
    out["untried_firings"] = files.get("untried_firings.json")
    out["final_stale_summary"] = files.get("stale_latch_summary.json")
    out["wrapped"] = {"prompt_lines": UT.prompt_lines.__module__, "stamp": UT.stamp.__module__,
                      "_run_python_tool": TA.ToolAgent._run_python_tool.__module__}
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    print(json.dumps(main(Path(args[0]), Path(args[1]), Path(args[2]), "--without-latch-fix" not in sys.argv),
                     default=str))
