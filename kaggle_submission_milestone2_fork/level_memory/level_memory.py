"""Solved-level memory for the milestone-2 (dfranzen) Duck harness.

Ported from sirikilohit's Milestone-2 patch M85 ("the only text addition that
helped", +2.8 on his stack), adapted to this harness's prefix cache.

Problem: the harness trims history from the front once the context passes
~118K tokens, keeping ~59K. Everything the agent learned on earlier levels is
then gone, except the Python helpers it wrote. Levels build on the mechanics of
earlier ones.

What it does (per game, never removes history):
  1. On the first prompt after a level is cleared, record facts the harness
     already has: the level, how many actions it took, its last ACTIONS_KEPT
     actions, and the tail of the model's own reasoning from the clearing turn.
  2. That prompt (and at most ASK_TURNS-1 more) asks for one line starting
     "Rule for level N:"; the rule is read from the reply text or reasoning.
  3. The facts and rules are pinned as one block at the END of the system
     prompt, so no trim can remove them.

Cache-aware deviation from M85: changing the system prompt invalidates the
game's cached prefix, so the block is only (re)applied at the moment history is
evicted -- when the prefix is being rebuilt anyway. Until the first eviction the
same information is still in the conversation, so nothing is lost by waiting.
"""

from __future__ import annotations

import re
import threading

ACTIONS_KEPT = 30
REASONING_CHARS = 900
RULE_CHARS = 600
FULL_LEVELS = 3
ASK_TURNS = 2
HEADER = "=== SOLVED LEVELS (facts from this game + your own conclusions; kept for the whole game) ==="
FOOTER = "=== END SOLVED LEVELS ==="
ASK = ("\n\nLEVEL {n} SOLVED. In your reply this turn, before acting, write one short paragraph that "
       "starts exactly with 'Rule for level {n}:' -- the goal, what each control/action did, and the "
       "plan that won (at most 5 lines). It is pinned for every later level.")

_LOCK = threading.Lock()
STATS = {"levels_recorded": 0, "rules_captured": 0, "rules_missing": 0, "asks": 0,
         "blocks_applied": 0, "block_chars_max": 0, "errors": 0, "last_error": ""}


def _bump(key, n=1):
    with _LOCK:
        STATS[key] += n


def _error(exc) -> None:
    with _LOCK:
        STATS["errors"] += 1
        STATS["last_error"] = f"{type(exc).__name__}: {exc}"[:200]


def message_text(msg) -> str:
    parts = []
    for k in ("content", "reasoning_content", "reasoning"):
        v = msg.get(k) if isinstance(msg, dict) else None
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts += [p.get("text", "") for p in v if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return "\n".join(p for p in parts if p)


def level_actions(history_entries, level: int) -> list[str]:
    """Actions taken while on `level`. history[i].frame is the frame AFTER history[i].action, so an
    action belongs to the level of the frame before it; the seeded first entry has no action."""
    out, prev = [], None
    for e in history_entries or []:
        act = str(getattr(e, "action", "") or "").strip()
        if act and prev == level:
            out.append(act)
        lv = getattr(getattr(e, "frame", None), "level", None)
        if isinstance(lv, int):
            prev = lv
    return out


def find_rule(text: str, level: int):
    m = re.search(r"Rule for level\s*%d\s*:[ \t]*" % int(level), text or "", re.I)
    if not m:
        return None
    lines = text[m.end():].split("\n")
    first = lines[0].strip()
    if len(first) >= 40:
        rule = first
    else:
        out = [first] if first else []
        for ln in lines[1:6]:
            if not ln.strip():
                if out:
                    break
                continue
            out.append(ln.strip())
        rule = " ".join(out)
    rule = " ".join(rule.split())[:RULE_CHARS]
    return rule if len(rule) >= 20 else None


def render_block(levels: list) -> str:
    if not levels:
        return ""
    lines = [HEADER]
    full_from = max(0, len(levels) - FULL_LEVELS)
    for i, rec in enumerate(levels):
        lines.append(f"Level {rec['level']} - cleared after {rec['n_actions']} actions.")
        lines.append(f"  Your rule: {rec['rule'] if rec.get('rule') else 'not stated'}")
        if i >= full_from:
            tail = rec["actions"]
            skipped = rec["n_actions"] - len(tail)
            lines.append("  Last actions before it cleared" + (f" ({skipped} earlier omitted)" if skipped > 0 else "")
                         + ": " + (", ".join(tail) if tail else "none recorded"))
            if rec.get("reasoning"):
                lines.append("  Your reasoning in the turn that cleared it: " + rec["reasoning"])
    lines.append("Reuse these where the new level has the same controls; verify with one probe before relying on them.")
    lines.append(FOOTER)
    return "\n".join(lines)


# ----------------------------------------------------------------------------- per-agent state
def _state(agent) -> dict:
    st = getattr(agent, "_lm_state", None)
    if st is None:
        st = {"levels": [], "base_sp": getattr(agent, "_system_prompt", "") or "", "applied": ""}
        agent._lm_state = st
    return st


def reset(agent) -> None:
    """A new game in this agent: drop the levels and restore the unpinned system prompt."""
    st = getattr(agent, "_lm_state", None)
    if st is not None:
        agent._system_prompt = st["base_sp"]
    agent._lm_state = None


def capture_rules(agent) -> bool:
    st = _state(agent)
    pending = [r for r in st["levels"] if not r.get("rule") and r["asks"] > r["checks"]]
    if not pending:
        return False
    texts = [message_text(m) for m in (getattr(agent, "_history_messages", None) or [])
             if isinstance(m, dict) and m.get("role") == "assistant"]
    changed = False
    for rec in pending:
        rec["checks"] += 1
        for t in reversed(texts):
            rule = find_rule(t, rec["level"])
            if rule:
                rec["rule"] = rule
                changed = True
                _bump("rules_captured")
                break
        if not rec.get("rule") and rec["checks"] >= ASK_TURNS:
            _bump("rules_missing")
    return changed


def after_user_prompt(agent, out: str, *, previous_step_summary=None, current_frame=None, history_entries=None) -> str:
    """Record a just-cleared level, capture pending rules, and append the rule request."""
    st = _state(agent)
    capture_rules(agent)
    summary = previous_step_summary or {}
    level_now = getattr(current_frame, "level", None)
    if summary.get("level_transition") and not summary.get("run_complete") and isinstance(level_now, int):
        solved = level_now - 1
        if solved >= 1 and all(r["level"] != solved for r in st["levels"]):
            acts = level_actions(history_entries or [], solved)
            reasoning = ""
            for m in reversed(getattr(agent, "_history_messages", None) or []):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    t = " ".join(message_text(m).split())
                    if t:
                        reasoning = t[-REASONING_CHARS:]
                        break
            st["levels"].append(dict(level=solved, n_actions=len(acts), actions=acts[-ACTIONS_KEPT:],
                                     reasoning=reasoning, rule=None, asks=0, checks=0))
            _bump("levels_recorded")
    last = st["levels"][-1] if st["levels"] else None
    # A turn resumed after a yield replaces this opener with a short resume prompt, so a request
    # appended here would never be sent; ask on a real opener instead.
    resuming = bool(getattr(agent, "_resume_after_yield", False))
    if last is not None and not last.get("rule") and last["asks"] < ASK_TURNS and not resuming:
        last["asks"] += 1
        _bump("asks")
        out += ASK.format(n=last["level"])
    return out


def apply_on_evict(agent, messages: list) -> list:
    """If history was just evicted, pin the current block into the system prompt (the prefix is
    being rebuilt anyway) and return messages carrying it."""
    if not getattr(agent, "_context_was_trimmed", False):
        return messages
    st = _state(agent)
    block = render_block(st["levels"])
    if block == st["applied"]:
        return messages
    agent._system_prompt = (st["base_sp"].rstrip("\n") + "\n\n" + block) if block else st["base_sp"]
    st["applied"] = block
    with _LOCK:
        STATS["blocks_applied"] += 1
        STATS["block_chars_max"] = max(STATS["block_chars_max"], len(block))
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        return [{**messages[0], "content": agent._system_prompt}, *messages[1:]]
    return messages


# ----------------------------------------------------------------------------- install
def install(tool_agent_cls) -> None:
    if getattr(tool_agent_cls, "_lm_installed", False):
        return
    orig_bup = tool_agent_cls._build_user_prompt
    orig_trim = tool_agent_cls._trim_messages_for_context
    orig_ens = tool_agent_cls._ensure_session

    def _build_user_prompt(self, action_num, *args, **kwargs):
        out = orig_bup(self, action_num, *args, **kwargs)
        try:
            return after_user_prompt(self, out, previous_step_summary=kwargs.get("previous_step_summary"),
                                     current_frame=kwargs.get("current_frame"),
                                     history_entries=kwargs.get("history_entries"))
        except Exception as exc:  # noqa: BLE001 -- memory is optional; never break a turn
            _error(exc)
            return out

    def _trim_messages_for_context(self, messages, *args, **kwargs):
        out = orig_trim(self, messages, *args, **kwargs)
        try:
            return apply_on_evict(self, out)
        except Exception as exc:  # noqa: BLE001
            _error(exc)
            return out

    def _ensure_session(self, state_path):
        before = getattr(self, "_session_runtime_dir", None)
        orig_ens(self, state_path)
        if getattr(self, "_session_runtime_dir", None) != before:
            reset(self)

    for f in dir(orig_bup):
        if f.endswith("_wrapped"):
            setattr(_build_user_prompt, f, getattr(orig_bup, f))
    tool_agent_cls._build_user_prompt = _build_user_prompt
    tool_agent_cls._trim_messages_for_context = _trim_messages_for_context
    tool_agent_cls._ensure_session = _ensure_session
    tool_agent_cls._lm_installed = True


def summary() -> dict:
    with _LOCK:
        return dict(STATS)
