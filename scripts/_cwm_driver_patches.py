"""Patches applied to the CodeWorldAgent diag driver for the goal_hint run.

The driver is base64-embedded in the diag notebook's cell 1 (see
`kaggle_submission_llm_world_engine/notebook_diag/`). Added 2026-09-25: the
question is no longer only "does a model pass replay" but "is the installed
model USED to choose actions" -- which the driver never recorded.

Pre-registered measures (recorded in EVIDENCE["summary"]["goal_hint_run"]):

  mechanism  plan calls the model decides vs stalls, in games with a model.
             Every planner call also records whether the OLD absolute-0.05
             rule would have stalled on that same call -- an exact,
             noise-free counterfactual from one run.
  gate       candidates whose predict() passes but whose goal_hint is
             rejected, and whether repair recovers them.
  outcome    levels completed across the run (the 2026-09-23 run: 0).

Every patch must match its anchor exactly once, or the build refuses.
"""

from __future__ import annotations

import base64
import re

DRIVER_PATCHES: dict[str, str] = {}

# -- 1. fail fast on a stale mount ---------------------------------------
# A stale dataset would run every game and report numbers about the OLD
# agent, indistinguishable from a real null.
DRIVER_PATCHES[
    '''    "DRAFT_MAX_TOKENS >= 4096": getattr(drafting, "DRAFT_MAX_TOKENS", 0) >= 4096,
}
for k, v in checks.items():
    say(("  OK   " if v else "  FAIL "), k)
EVIDENCE["fixed_code_checks"] = checks
'''
] = '''    "DRAFT_MAX_TOKENS >= 4096": getattr(drafting, "DRAFT_MAX_TOKENS", 0) >= 4096,
}
from llm_engine import planner as _planner_mod  # noqa: E402
GOAL_HINT_CHECKS = {
    "planner stall test is scale-free (hints_tied)":
        "hints_tied(first_ply_hints" in open(_planner_mod.__file__, encoding="utf-8").read(),
    "replay gates goal_hint (check_goal_hint)": hasattr(replay_mod, "check_goal_hint"),
    "drafting prompt demands non-constant goal_hint": "must NOT be constant" in _draft_src,
    # 2026-09-26: the counterfactual (playfield-response) gate, not the
    # history-variation one that a step counter passed.
    "replay gate is counterfactual (_transplant)": hasattr(replay_mod, "_transplant"),
    "drafting prompt warns colour counts are blind to movement": "do not just count colours" in _draft_src,
}
checks.update(GOAL_HINT_CHECKS)
for k, v in checks.items():
    say(("  OK   " if v else "  FAIL "), k)
EVIDENCE["fixed_code_checks"] = checks
if not all(GOAL_HINT_CHECKS.values()):
    say("FATAL: mounted engine is NOT the goal_hint code -- refusing to run games on it")
    EVIDENCE["errors"].append("stale engine mounted: " + json.dumps(GOAL_HINT_CHECKS))
    flush_evidence()
    sys.exit(2)
'''

# -- 2. record both halves of the (now two-part) replay gate --------------
DRIVER_PATCHES[
    '''        "passed": res.passed,
'''
] = '''        "passed": res.passed,
        "predict_passed": getattr(res, "predict_passed", res.passed),
        "goal_hint_ok": getattr(getattr(res, "goal_hint", None), "ok", None),
        "goal_hint_distinct": getattr(getattr(res, "goal_hint", None), "distinct_values", None),
        "goal_hint_problem": (getattr(getattr(res, "goal_hint", None), "problem", None) or "")[:400],
        "goal_hint_informative_boards": getattr(getattr(res, "goal_hint", None), "informative_boards", None),
        "predict_distinguishes_actions": getattr(getattr(res, "goal_hint", None), "predict_distinguishes_actions", None),
'''

# -- 3. per planner call: new decision + old rule's decision ---------------
# Old rule: best_levels == 0 and no first-ply level and spread < 0.05.
# predicted_levels_gained == 0 stands in for the first two.
DRIVER_PATCHES[
    '''cwa_mod.repair_world_model = _wrap_round("repair", _orig_repair_fn)
'''
] = '''cwa_mod.repair_world_model = _wrap_round("repair", _orig_repair_fn)

EVIDENCE["plan_calls"] = []
_orig_next_action = cwa_mod.next_action


def _instr_next_action(model, state, *a, **kw):  # type: ignore[no-untyped-def]
    action, res = _orig_next_action(model, state, *a, **kw)
    EVIDENCE["plan_calls"].append({
        "thread": threading.current_thread().name,
        "stalled": bool(res.stalled),
        "spread": res.goal_hint_spread,
        "levels_pred": res.predicted_levels_gained,
        "old_rule_would_stall": bool(res.predicted_levels_gained == 0 and res.goal_hint_spread < 0.05),
        "action": None if action is None else action.name,
    })
    return action, res


cwa_mod.next_action = _instr_next_action
say("patched code_world_agent.next_action (plan-call telemetry)")
'''

# -- 4. per game: was the model USED ---------------------------------------
DRIVER_PATCHES[
    '''        "model_source": src,
    }
    EVIDENCE["games"].append(rec)
'''
] = '''        "model_source": src,
        "plan_stats": getattr(a, "plan_stats", None),
        "levels_seen": getattr(a, "levels_seen", None),
    }
    EVIDENCE["games"].append(rec)
    say("plan stats               :", rec["plan_stats"], "levels_seen:", rec["levels_seen"])
'''

# -- 5. the pre-registered summary ------------------------------------------
DRIVER_PATCHES[
    '''    "wall_clock_seconds": round(time.time() - T0, 1),
}
'''
] = '''    "wall_clock_seconds": round(time.time() - T0, 1),
}
_rr = EVIDENCE["replay_results"]
_pc = EVIDENCE.get("plan_calls", [])
EVIDENCE["summary"]["goal_hint_run"] = {
    "candidates_predict_passed": sum(1 for r in _rr if r.get("predict_passed")),
    "predict_passed_but_goal_rejected": sum(
        1 for r in _rr if r.get("predict_passed") and r.get("goal_hint_ok") is False
    ),
    "candidates_accepted_both_halves": sum(1 for r in _rr if r.get("passed")),
    "goal_rejected_cannot_tell_actions_apart": sum(
        1 for r in _rr if r.get("predict_passed") and "cannot tell actions apart" in (r.get("goal_hint_problem") or "")
    ),
    "accepted_but_predict_never_distinguishes": sum(
        1 for r in _rr if r.get("passed") and r.get("predict_distinguishes_actions") is False
    ),
    "plan_calls": len(_pc),
    "plan_calls_stalled_new_rule": sum(1 for c in _pc if c["stalled"]),
    "plan_calls_old_rule_would_stall": sum(1 for c in _pc if c["old_rule_would_stall"]),
    "plan_calls_rescued_by_new_rule": sum(
        1 for c in _pc if c["old_rule_would_stall"] and not c["stalled"]
    ),
    "levels_completed_total": sum((g.get("levels_completed") or 0) for g in EVIDENCE["games"]),
    "games_with_installed_model": sum(1 for g in EVIDENCE["games"] if g.get("world_model_installed")),
}
say("GOAL-HINT RUN:", json.dumps(EVIDENCE["summary"]["goal_hint_run"], indent=2))
'''

#: Probes that must appear in the patched driver.
DRIVER_REQUIRED = [
    ("driver: stale-engine fail-fast", "GOAL_HINT_CHECKS"),
    ("driver: replay records goal_hint", '"goal_hint_ok"'),
    ("driver: plan-call counterfactual", '"old_rule_would_stall"'),
    ("driver: per-game plan_stats", '"plan_stats"'),
    ("driver: pre-registered summary", '"goal_hint_run"'),
]

_B64_RE = re.compile(r'DRIVER_B64 = "([^"]+)"')


def patch_driver_cell(cell_source: str) -> str:
    """Decode the embedded driver, apply DRIVER_PATCHES, re-encode."""
    m = _B64_RE.search(cell_source)
    if m is None:
        return cell_source
    driver = base64.b64decode(m.group(1)).decode("utf-8")
    for old, new in DRIVER_PATCHES.items():
        n = driver.count(old)
        if n != 1:
            raise SystemExit(f"REFUSING TO SHIP -- driver patch anchor found {n}x: {old[:70]!r}")
        driver = driver.replace(old, new)
    # A syntax error here would only surface after vLLM boots on the GPU.
    compile(driver, "diag_driver.py", "exec")
    return cell_source.replace(m.group(1), base64.b64encode(driver.encode("utf-8")).decode("ascii"))


def driver_text(notebook: dict) -> str:
    for cell in notebook["cells"]:
        m = _B64_RE.search("".join(cell["source"]))
        if m:
            return base64.b64decode(m.group(1)).decode("utf-8")
    return ""
