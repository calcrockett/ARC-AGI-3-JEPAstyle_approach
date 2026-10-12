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

Optional TRIED FACTS (env LEVEL_MEMORY_TRIED_FACTS=1, default OFF): at the same eviction point, also pin a
compact facts-only block about the CURRENT unsolved level -- actions spent, game-over count and the action
count at each, the last actions of the most recent fatal runs, and the tail of the model's own reasoning
from the previous turn. Everything is stated as fact, never as advice. Nothing before the eviction point is
touched, and with the flag off none of this code runs.
"""

from __future__ import annotations

import os
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

# ---- tried facts (flag-gated; see module docstring)
TRIED_FACTS_ENV = "LEVEL_MEMORY_TRIED_FACTS"
TRIED_FACTS_MAX_BYTES = 3000
TRIED_RUNS = 3              # most recent fatal runs shown
TRIED_RUN_ACTIONS = 10      # actions shown per fatal run
TRIED_REASONING_CHARS = 600
TRIED_COUNTS_SHOWN = 12     # per-game-over action counts listed
TF_HEADER = "=== CURRENT LEVEL FACTS (recorded by the harness when older history was trimmed) ==="
TF_FOOTER = "=== END CURRENT LEVEL FACTS ==="

_LOCK = threading.Lock()
STATS = {"levels_recorded": 0, "rules_captured": 0, "rules_missing": 0, "asks": 0,
         "blocks_applied": 0, "block_chars_max": 0, "errors": 0, "last_error": ""}


TF_STATS = {"tried_facts_blocks": 0, "tried_facts_bytes_total": 0, "tried_facts_bytes_max": 0,
            "tried_facts_truncations": 0}


def tried_facts_enabled() -> bool:
    return os.environ.get(TRIED_FACTS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


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


# ----------------------------------------------------------------------------- tried facts
def level_life_facts(history_entries, level: int) -> dict:
    """Facts about `level` from the harness history: actions spent (resets included) and, for every game
    over on it, the actions of the life that ended there. history[i].frame is the frame AFTER
    history[i].action, so an action belongs to the level of the frame before it. A game over is the entry
    whose result carries game_over; the RESET that follows it opens the next life and is not part of one."""
    total, life, overs, prev = 0, [], [], None
    for e in history_entries or []:
        act = str(getattr(e, "action", "") or "").strip()
        if act and prev == level:
            total += 1
            res = getattr(e, "result", None)
            res = res if isinstance(res, dict) else {}
            if act.upper() == "RESET":
                life = []
            else:
                life.append(act)
                if res.get("game_over"):
                    overs.append(life)
                    life = []
        lv = getattr(getattr(e, "frame", None), "level", None)
        if isinstance(lv, int):
            prev = lv
    return {"level": level, "actions": total, "game_overs": overs}


def reasoning_tail(messages, chars: int = TRIED_REASONING_CHARS) -> str:
    """Last `chars` characters of the model's own reasoning in its most recent assistant message
    (its visible reply when that message carries no reasoning)."""
    for m in reversed(messages or []):
        if not (isinstance(m, dict) and m.get("role") == "assistant"):
            continue
        parts = []
        for k in ("reasoning_content", "reasoning"):
            v = m.get(k)
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                parts += [p.get("text", "") for p in v if isinstance(p, dict) and isinstance(p.get("text"), str)]
        text = " ".join(" ".join(parts).split())
        if not text:
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and isinstance(p.get("text"), str))
            text = " ".join(c.split()) if isinstance(c, str) else ""
        if text:
            return text[-chars:]
    return ""


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _render_tried(snap: dict, reasoning_chars: int, runs: int, run_actions: int, counts_shown: int) -> str:
    overs = snap["game_overs"]
    lines = [TF_HEADER]
    lines.append(f"Level {snap['level']}: {snap['actions']} actions spent on it so far (resets included); "
                 f"{len(overs)} game over{'' if len(overs) == 1 else 's'} on it.")
    if overs:
        counts = [len(g) for g in overs]
        if len(counts) >= 2 and len(set(counts)) == 1:
            lines.append(f"All {len(counts)} game overs occurred after exactly {counts[0]} actions.")
        else:
            shown = counts[-counts_shown:] if counts_shown > 0 else []
            if shown:
                omitted = len(counts) - len(shown)
                lines.append("Actions taken before each game over, in order"
                             + (f" ({omitted} earlier omitted)" if omitted > 0 else "")
                             + ": " + ", ".join(str(c) for c in shown) + ".")
        recent = overs[-runs:] if runs > 0 else []
        first_no = len(overs) - len(recent) + 1
        for i, g in enumerate(recent):
            tail = g[-run_actions:] if run_actions > 0 else []
            earlier = len(g) - len(tail)
            lines.append(f"Game over #{first_no + i}, last {len(tail)} of {len(g)} actions"
                         + (f" ({earlier} earlier not shown)" if earlier > 0 else "") + ": " + ", ".join(tail))
    if reasoning_chars > 0 and snap.get("reasoning"):
        lines.append("Your reasoning in the turn before older history was trimmed (tail): "
                     + snap["reasoning"][-reasoning_chars:].lstrip())
    lines.append(TF_FOOTER)
    return "\n".join(lines)


def render_tried_facts(snap, max_bytes: int = TRIED_FACTS_MAX_BYTES):
    """The facts-only block, hard-capped at `max_bytes` (utf-8). Returns (text, truncated)."""
    if not snap or not snap.get("actions"):
        return "", False
    full = (TRIED_REASONING_CHARS, TRIED_RUNS, TRIED_RUN_ACTIONS, TRIED_COUNTS_SHOWN)
    text = _render_tried(snap, *full)
    if _utf8_len(text) <= max_bytes:
        return text, False
    # degrade in order: shorter reasoning, fewer fatal runs / actions, fewer listed counts
    for params in ((300, 3, 10, 12), (100, 3, 10, 12), (0, 3, 10, 12), (0, 2, 10, 8), (0, 2, 6, 6),
                   (0, 1, 6, 4), (0, 1, 3, 2), (0, 0, 0, 0)):
        text = _render_tried(snap, *params)
        if _utf8_len(text) <= max_bytes:
            return text, True
    cut = text.encode("utf-8")[: max(0, max_bytes - 4)].decode("utf-8", "ignore")
    return cut + " ...", True


def snapshot_tried_facts(agent, current_frame, history_entries):
    level = getattr(current_frame, "level", None)
    if not isinstance(level, int):
        return None
    snap = level_life_facts(history_entries or [], level)
    snap["reasoning"] = reasoning_tail(getattr(agent, "_history_messages", None))
    return snap


def compose_block(st: dict) -> str:
    """Everything pinned at an eviction: the solved-level block, plus the current-level facts if enabled."""
    block = render_block(st["levels"])
    if not tried_facts_enabled():
        return block
    facts, truncated = render_tried_facts(st.get("facts"))
    st["_facts_text"], st["_facts_truncated"] = facts, truncated
    return "\n\n".join(p for p in (block, facts) if p)


# ----------------------------------------------------------------------------- per-agent state
def _state(agent) -> dict:
    st = getattr(agent, "_lm_state", None)
    if st is None:
        st = {"levels": [], "base_sp": getattr(agent, "_system_prompt", "") or "", "applied": "", "facts": None}
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
    if tried_facts_enabled():
        st["facts"] = snapshot_tried_facts(agent, current_frame, history_entries)
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
    block = compose_block(st)
    if block == st["applied"]:
        return messages
    agent._system_prompt = (st["base_sp"].rstrip("\n") + "\n\n" + block) if block else st["base_sp"]
    st["applied"] = block
    with _LOCK:
        STATS["blocks_applied"] += 1
        STATS["block_chars_max"] = max(STATS["block_chars_max"], len(block))
        if tried_facts_enabled() and st.get("_facts_text"):
            n = _utf8_len(st["_facts_text"])
            TF_STATS["tried_facts_blocks"] += 1
            TF_STATS["tried_facts_bytes_total"] += n
            TF_STATS["tried_facts_bytes_max"] = max(TF_STATS["tried_facts_bytes_max"], n)
            TF_STATS["tried_facts_truncations"] += int(bool(st.get("_facts_truncated")))
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
        out = dict(STATS)
        if tried_facts_enabled():
            out.update(TF_STATS)
            n = TF_STATS["tried_facts_blocks"]
            out["tried_facts_avg_bytes"] = round(TF_STATS["tried_facts_bytes_total"] / n, 1) if n else 0
        return out
