"""Strategy audit for the milestone-2 (dfranzen) Duck harness.

Port of lordhansolo's Milestone-2 "strategy audit" (kernel lordhansolo/arc-agi-3-milestone-2, public LB 23.84;
source bundle lordhansolo/taaf-kaggle-source, inference/agent/prompts.py STRATEGY_AUDIT_PROMPT and
inference/framework/solver.py _should_audit_strategy). Provenance and the exact text changes: NOTICE.md here.

His mechanism: a per-game timer starts when a level starts; once a quarter of the game's fixed initial runtime
budget has elapsed on the same level, the next analyzer turn carries STRATEGY_AUDIT_PROMPT; the timer restarts
when that turn completes and again when the level is cleared. "Independent of actions, lives and model claims."

Our harness time-shares ~110 games over a few streams through a priority gate, so wall time on a level is
mostly time spent parked. The clock here is the game's own GENERATED TOKENS on the level (the gate's own
measure of work, ToolAgent._session_generated_tokens): the audit is due once AUDIT_TOKENS tokens were generated
on the current level since it started or since the last audit. AUDIT_TOKENS defaults to 56,000 = 25% of a
game's expected token share in the hidden rerun (~200K per game on the incumbent, x1.14 for the turbo-lossless
serving stack).

The text is appended to the turn's opener (the user prompt), never to the system prompt or history, so the
cached prefix is unaffected. A turn that generated nothing (rolled back, failed) does not count as delivered and
the audit is re-sent on the next opener. Nothing else changes; with ARC3_STRATEGY_AUDIT_TOKENS=0 no text is ever
added.

Counters (summary()): audits sent and confirmed, levels audited, how many audited levels were later cleared (the
outcome measure), voluntary RESETs after an audit vs overall (the audit says not to reset), errors.
"""

from __future__ import annotations

import os
import threading

AUDIT_TOKENS_ENV = "ARC3_STRATEGY_AUDIT_TOKENS"
DEFAULT_AUDIT_TOKENS = 56000
MARK = "Strategy audit for this turn"

# lordhansolo's STRATEGY_AUDIT_PROMPT with three harness-specific references adapted (NOTICE.md lists them):
# his python tool takes a `plan` argument and keeps saved modules with tests; ours takes only `code` and keeps
# retained helper functions.
PROMPT = (
    "\nStrategy audit for this turn\n"
    "You may be stuck on this level. Briefly check whether your current approach is making progress.\n"
    "- First check whether recent observations match your predictions and advance a concrete "
    "plan. If so, briefly state the supporting observation and next expected result, then "
    "continue the plan without an extra experiment or the remaining audit steps.\n"
    "- You may also continue a search with a defined scope and stopping condition if its "
    "transition rules have been checked against actual game observations and no unresolved "
    "contradiction undermines it. Briefly state that evidence and the remaining search work. "
    "Confidence, passing tests of your own model or a growing search count alone are not "
    "evidence of progress.\n"
    "- If neither continuation is supported, or observations contradict the approach, use "
    "the audit below. Do not reset the game or abandon a plan solely because of this audit.\n"
    "When an assumption audit is needed\n"
    "- Separate observed facts from assumptions about success, state variables, controls "
    "and how a candidate solution is evaluated. For each assumption, briefly state what "
    "observation supports it. If none does, treat it as untested. Your own notes, retained helper functions "
    "and passing checks of your own code do not establish that the assumed rules are complete.\n"
    "- Choose one uncertain assumption that could explain the contradiction or lack of "
    "confirmed progress and an alternative interpretation. First check existing transitions for evidence that "
    "distinguishes them. If they already settle the question, apply that finding and continue. "
    "Otherwise design one short discriminating experiment that preserves current progress "
    "where possible. A complete test may need both configuration and execution actions.\n"
    "- For a new experiment, write the hypothesis and predicted outcomes as comments at the top of the "
    "python snippet and execute it with action(...). In the same snippet inspect each action's result "
    "before issuing another and print a compact observation of the relevant objects, "
    "status indicators and animation. Stop the sequence if an intermediate result contradicts "
    "the prediction. The model will read this output on the next turn.\n"
    "- After an experiment, on the next turn compare prediction with observation before "
    "further actions and revise any "
    "contradicted rule, state representation or goal. Mark indistinguishable outcomes "
    "inconclusive. Write the finding and supporting observation as a comment in that turn's snippet "
    "so they remain available in your history. Add step numbers when useful. If a retained helper "
    "function encodes a revised rule, fix that function. Use prior audit findings to avoid "
    "repeating an inconclusive experiment unchanged.\n"
)

_LOCK = threading.Lock()
STATS = {"audits_sent": 0, "audits_confirmed": 0, "audits_retracted": 0, "audits_skipped_short_resume": 0,
         "levels_audited": 0, "audited_levels_cleared": 0, "max_audits_one_level": 0,
         "voluntary_resets_after_audit": 0, "voluntary_resets_total": 0, "games": 0,
         "threshold_tokens": 0, "errors": 0, "last_error": ""}


def audit_tokens() -> int:
    """The token clock's period; 0 (or a negative / unreadable value) disables the audit."""
    raw = os.environ.get(AUDIT_TOKENS_ENV, "").strip()
    if not raw:
        return DEFAULT_AUDIT_TOKENS
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return 0


def _bump(key, n=1):
    with _LOCK:
        STATS[key] += n


def _error(exc) -> None:
    with _LOCK:
        STATS["errors"] += 1
        STATS["last_error"] = f"{type(exc).__name__}: {exc}"[:200]


def _opener_replaced_on_resume() -> bool:
    """A resumed turn re-sends the full opener unless ARC3_YIELD_RESUME_PROMPT replaces it (the harness's
    _yield_resume_prompt_mode: 'short' / 'state_only'); then text appended to the opener is never sent."""
    return os.environ.get("ARC3_YIELD_RESUME_PROMPT", "").strip().lower() in ("short", "state_only")


def _voluntary_resets(entries) -> int:
    n = 0
    for e in entries or []:
        if str(getattr(e, "action", "") or "").strip().upper() != "RESET":
            continue
        res = getattr(e, "result", None)
        if not (isinstance(res, dict) and res.get("automatic")):
            n += 1
    return n


def _state(agent, tokens: int) -> dict:
    session = getattr(agent, "_session_runtime_dir", None)
    st = getattr(agent, "_sa_state", None)
    if st is None or st["session"] != session:
        st = {"session": session, "level": None, "anchor": tokens, "audits_level": 0, "pending": None,
              "pending_level": None, "watch_from": None, "seen": 0}
        agent._sa_state = st
        _bump("games")
    return st


def after_user_prompt(agent, out: str, *, current_frame=None, history_entries=None) -> str:
    """Book-keep the token clock and, when due, append the audit to this turn's opener."""
    period = audit_tokens()
    tokens = int(getattr(agent, "_session_generated_tokens", 0) or 0)
    st = _state(agent, tokens)
    level = getattr(current_frame, "level", None)
    entries = list(history_entries or [])

    # 1. the previous delivery: confirmed once the model generated tokens after it, else retracted
    if st["pending"] is not None:
        if tokens > st["pending"]:
            _bump("audits_confirmed")
            st["audits_level"] += 1
            if st["audits_level"] == 1:
                _bump("levels_audited")
            with _LOCK:
                STATS["max_audits_one_level"] = max(STATS["max_audits_one_level"], st["audits_level"])
            st["anchor"] = st["pending"]          # lordhansolo: the timer restarts at the audit turn
        else:
            _bump("audits_retracted")
        st["pending"] = None

    # 2. voluntary resets since the last look (all of them, and those after an audit on this level)
    if st["seen"] > len(entries):                 # a different history (new game / reloaded state)
        st["seen"] = 0
    new = entries[st["seen"]:]
    st["seen"] = len(entries)
    n_new = _voluntary_resets(new)
    if n_new:
        _bump("voluntary_resets_total", n_new)
        if st["audits_level"] > 0:
            _bump("voluntary_resets_after_audit", n_new)

    # 3. a level change restarts the clock; an audited level left upwards was cleared
    if isinstance(level, int) and level != st["level"]:
        if st["level"] is not None and level > st["level"] and st["audits_level"] > 0:
            _bump("audited_levels_cleared")
        st["level"], st["anchor"], st["audits_level"] = level, tokens, 0

    # 4. due?
    if period <= 0 or not isinstance(level, int) or tokens - st["anchor"] < period:
        return out
    if bool(getattr(agent, "_resume_after_yield", False)) and _opener_replaced_on_resume():
        _bump("audits_skipped_short_resume")
        return out
    st["pending"] = tokens
    _bump("audits_sent")
    return out.rstrip("\n") + "\n" + PROMPT


# ----------------------------------------------------------------------------- install
def install(tool_agent_cls) -> bool:
    """Wrap ToolAgent._build_user_prompt (after solved-level memory, whose wrapper it calls). Idempotent."""
    if getattr(tool_agent_cls, "_sa_installed", False):
        return True
    orig_bup = tool_agent_cls._build_user_prompt

    def _build_user_prompt(self, action_num, *args, **kwargs):
        out = orig_bup(self, action_num, *args, **kwargs)
        try:
            return after_user_prompt(self, out, current_frame=kwargs.get("current_frame"),
                                     history_entries=kwargs.get("history_entries"))
        except Exception as exc:  # noqa: BLE001 -- the audit is optional; never break a turn
            _error(exc)
            return out

    for f in dir(orig_bup):
        if f.endswith("_wrapped"):
            setattr(_build_user_prompt, f, getattr(orig_bup, f))
    tool_agent_cls._build_user_prompt = _build_user_prompt
    tool_agent_cls._sa_installed = True
    with _LOCK:
        STATS["threshold_tokens"] = audit_tokens()
    return True


def summary() -> dict:
    with _LOCK:
        out = dict(STATS)
    out["threshold_tokens"] = audit_tokens()
    return out
