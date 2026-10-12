"""Probe for tests/test_jad_exp081_ulm.py, run in a subprocess (it imports a harness tree other than tests/m2_harness.py's).

    python tests/jad_ulm_probe.py NOTEBOOK TREE_DIR HIS_REPO

The interaction probe of the bundle C+UL (arm C + ours-10 I1 + the stale-latch fix + our solved-level memory). Builds
/kaggle/taaf-kaggle-source-share exactly as NOTEBOOK's cells build it (Franzen's patch, then all eight /kaggle/ours-*.patch
the notebook writes, in cell order, via JustAdev742's vendored franzen_tree.py), sets the notebook's own setup_env,
executes NOTEBOOK's C+U install cell and then its level-memory install cell verbatim (notebook order, one namespace),
and drives the REAL ToolAgent + sandbox. Modelled on tests/jad_lm_probe.py and tests/jad_u_probe.py. The two mechanisms
meet in ToolAgent's prompt building (ours-10's call site inside the patched _build_user_prompt body; LM's wrappers
around _build_user_prompt / _trim_messages_for_context / _ensure_session), so one agent plays them together:

  t=1000  level 1, first prompt                          -> nothing extra
  t=1700  level 1 stalled 700 s                          -> the I1 line (level 1)
          a click on a named object, then the level-up
  t=1710  the level-2 opener                             -> 03b's win record + LM's rule request; no I1 (new clock)
          a reply without a rule
  t=2400  level 2 stalled 690 s                          -> the I1 line (level 2) + LM's second rule request
          a reply with the rule
  t=2410  next prompt                                    -> the rule is captured; nothing extra
          trim 1 (eviction)                              -> LM's block at the end of the system prompt
          trim 2 (head moved)                            -> 03b's ledger right after the system prompt
          trim 3 (same history again)                    -> idempotent: one block, one ledger
  t=3100  level 2, 700 s after its last I1               -> the I1 line still renders after the trims

plus the latch (a separate agent on a fake game that applies the solver's stale rule): a caught refusal acts again,
and a caught loop of no-ops gets 3 releases. Prints one JSON line of findings, including the prompt layouts.
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
CU_HEAD = "# [calamitychasm] C+U INSTALL"
LM_HEAD = "# [calamitychasm] SOLVED-LEVEL MEMORY"
I1 = "Not yet tried on this level"
WIN = "Level 1 record (exact facts from the recorded boards)"
TOOL = "Only tool: `python`"
ASK = "LEVEL 1 SOLVED."
RULE = "Rule for level 1: UP moves nothing; MOUSE on the 3x3 g block clears the level."
PROMPT_MARKS = {"win_record": WIN, "i1": I1, "tool_line": TOOL, "lm_ask": ASK}


def layout(text: str, marks: dict) -> dict:
    """{"order": [names in order of first appearance], "counts": {name: occurrences}}."""
    found = sorted((text.find(m), n) for n, m in marks.items() if m in text)
    return {"order": [n for _, n in found], "counts": {n: text.count(m) for n, m in marks.items()}}


def main(notebook: Path, tree: Path, his_repo: Path) -> dict:
    sys.path.insert(0, str(VENDOR / "scripts"))
    sys.path.insert(0, str(REPO / "tests"))
    import franzen_tree as FT
    import m2_harness
    from jad_u_probe import setup_env

    patches = FT.our_patch_cells(notebook)
    if not (tree / "src" / "ARC3-Inference").is_dir():
        FT.notebook_bundle(tree, patches, his_repo=his_repo, his_patch=FT.his_patch_text(notebook))
    cells = FT.notebook_cells(notebook)
    os.environ.update(setup_env(cells))
    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")
    TA, RS, _, S = m2_harness.import_harness(tree / "src")
    import inference.utils.ours_untried as UT

    out: dict = {"patches": [p for p, _ in patches], "untried_mode": sorted(UT.mode()),
                 "flags": {k: os.environ.get(k) for k in ("OURS_WIN_LEDGER", "OURS_LEVEL_MEM", "OURS_UNTRIED",
                                                          "ARC3_STALE_STATE_BLOCK", "ARC3_MAX_ACTIVE_STREAMS")}}
    order = [i for i, c in enumerate(cells) if c.startswith((CU_HEAD, LM_HEAD))]
    out["install_order"] = ["cu" if cells[i].startswith(CU_HEAD) else "lm" for i in order]
    ns: dict = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for i in order:  # the shipped cells, verbatim, in notebook order, in one namespace (as the kernel runs them)
            exec(compile(cells[i], f"install_cell_{i}", "exec"), ns)
    out["install_stdout"] = buf.getvalue()
    cu, lm = ns["CU_COUNTERS"], ns["LEVEL_MEMORY"]
    cls = TA.ToolAgent
    out["wrapped"] = {n: getattr(cls, n).__module__ for n in ("_build_user_prompt", "_trim_messages_for_context",
                                                              "_ensure_session", "_run_python_tool")}
    out["wrapped"].update(prompt_lines=UT.prompt_lines.__module__, stamp=UT.stamp.__module__)
    out["ledger_on"] = TA._ours_win_ledger_enabled()

    # --- latch, on the real ToolAgent + sandbox (as tests/jad_u_probe.py) ------------------------------------------
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
    out["catch_content"] = str(agent._run_python_tool(sp, {"code": catch}).content)[-1500:]
    out["catch_executed"] = game.n - n0
    loop = ('ex = ref = 0\nfor _ in range(10):\n    try:\n        r = action("ACTION5")\n        ex += 1\n'
            '    except StaleStateActionError:\n        ref += 1\nprint("LOOP", ex, ref)\n')
    n0 = game.n
    out["loop_content"] = str(agent._run_python_tool(sp, {"code": loop}).content)[-1500:]
    out["loop_executed"] = game.n - n0

    # --- the prompt-building interaction: I1 + 03b + LM on one agent ------------------------------------------------
    grid = [[0] * 16 for _ in range(16)]
    for r in range(3, 6):
        for c in range(3, 6):
            grid[r][c] = 2  # a 3x3 block of colour 2 ("g r3-5 c3-5")
    for r in range(9, 11):
        for c in range(9, 13):
            grid[r][c] = 3  # a 2x4 bar of colour 3 ("G r9-10 c9-12")
    board = tuple(tuple(x) for x in grid)

    def frame(level, step):
        return RS.Frame(grid=board, step=step, level=level)

    H = RS.HistoryEntry
    clock = [1000.0]
    UT._clock = lambda: clock[0]
    na = tmp / "notice" / "g_tool_runtime_state.json"
    na.parent.mkdir(parents=True)
    ag = cls(model="local")
    ag._ensure_session(na)
    base = ag._system_prompt
    valid = ["ACTION1", "ACTION6"]
    prompts: dict = {}

    def prompt(name, t, action_num, current, hist, summary):
        clock[0] = t
        prompts[name] = ag._build_user_prompt(action_num, valid_actions=valid, current_frame=current,
                                              history_entries=hist, previous_step_summary=summary)
        return prompts[name]

    def reply(text, reasoning=""):
        ag._history_messages.append({"role": "assistant", "content": text, "reasoning_content": reasoning})

    hist = [H(action="", frame=frame(1, 0)), H(action="ACTION1", frame=frame(1, 1), result={"executed": True})]
    prompt("l1_first", 1000.0, 2, frame(1, 1), hist, {"level": 1})
    prompt("l1_stalled", 1700.0, 2, frame(1, 1), hist, {"level": 1})
    hist += [H(action="ACTION1", frame=frame(1, 2), result={"executed": True}),
             H(action="MOUSE(row=4, col=4)", frame=frame(2, 3), result={"executed": True, "level_completed": True})]
    prompt("l2_opener", 1710.0, 4, frame(2, 3), hist,
           {"level_transition": True, "level": 2, "executed_actions": ["MOUSE(row=4, col=4)"]})
    reply("I will test the bar next.")
    hist += [H(action="ACTION1", frame=frame(2, 4), result={"executed": True})]
    prompt("l2_stalled_ask2", 2400.0, 5, frame(2, 4), hist, {"level": 2})
    reply("ok", RULE)
    hist += [H(action="ACTION1", frame=frame(2, 5), result={"executed": True})]
    prompt("l2_after_rule", 2410.0, 6, frame(2, 5), hist, {"level": 2})
    out["rule"] = (ag._lm_state["levels"][0].get("rule") or "") if ag._lm_state["levels"] else ""

    filler = "x " * 40000
    turns = []
    for i in range(46):
        turns += [{"role": "user", "content": f"turn {i} {filler}"}, {"role": "assistant", "content": f"ok {i}"}]
    turns += [{"role": "user", "content": prompts["l2_opener"]}, {"role": "assistant", "content": "x"},
              {"role": "user", "content": prompts["l2_stalled_ask2"]}, {"role": "assistant", "content": RULE},
              {"role": "user", "content": prompts["l2_after_rule"]}]
    ag._context_was_trimmed = False
    first = ag._trim_messages_for_context([{"role": "system", "content": base}, *turns[:80]])
    second = ag._trim_messages_for_context([{"role": "system", "content": ag._system_prompt}, *turns])
    third = ag._trim_messages_for_context([{"role": "system", "content": ag._system_prompt}, *turns])
    out["first_trim_evicted"] = len(first) < 81
    hist += [H(action="ACTION1", frame=frame(2, 6), result={"executed": True})]
    prompt("l2_after_trims", 3100.0, 7, frame(2, 6), hist, {"level": 2})

    # prompt layouts and duplicate/missing text
    out["prompts"] = {n: layout(p, PROMPT_MARKS) for n, p in prompts.items()}
    out["i1_lines"] = {n: [ln for ln in p.splitlines() if ln.startswith(I1)] for n, p in prompts.items()}
    out["prompt_text"] = {n: prompts[n] for n in ("l2_opener", "l2_stalled_ask2")}
    out["ask_at_end"] = {n: p.rstrip().endswith("It is pinned for every later level.") for n, p in prompts.items()}

    def msgs(messages):
        rows = []
        for m in messages[:3]:
            c = str(m.get("content", ""))
            rows.append({"role": m.get("role"), "ledger": TA._ours_is_ledger(m), "len": len(c),
                         "starts_with_base": c.startswith(base), "lm_header": c.count(lm.HEADER),
                         "lm_footer": c.count(lm.FOOTER), "ends_with_lm_footer": c.rstrip().endswith(lm.FOOTER),
                         "rule_body": c.count(RULE.split(": ", 1)[1])})
        def kind(m):
            c = str(m.get("content", ""))
            if m.get("role") == "system":
                return "system+lm_block" if lm.HEADER in c else "system"
            if TA._ours_is_ledger(m):
                return "ledger_03b"
            for name, p in prompts.items():
                if c == p:
                    return f"user:{name}"
            return "assistant" if m.get("role") == "assistant" else "user:history"

        return {"n": len(messages), "head": rows, "kinds": [kind(m) for m in messages], "ledgers": sum(bool(TA._ours_is_ledger(m)) for m in messages),
                "lm_headers": sum(str(m.get("content", "")).count(lm.HEADER) for m in messages),
                "ledger_text": next((m["content"] for m in messages if TA._ours_is_ledger(m)), "")}

    out["trims"] = {"first": msgs(first), "second": msgs(second), "third": msgs(third)}
    out["system_prompt_tail"] = ag._system_prompt[len(base):]
    files = cu.finish()
    out["untried_summary"] = files.get("untried_summary.json")
    out["untried_firings"] = files.get("untried_firings.json")
    out["stale_summary"] = files.get("stale_latch_summary.json")
    out["lm_summary"] = lm.summary()
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])), default=str))
