"""Arm C+U (calamitychasm/arc3-jad-exp081-u): install-time checks, markers and run counters for its two changes.

C1  JustAdev742's ours-10 "not yet tried on this level" notice (vendor/kaggle/franzen/patches/ours-10-untried.patch,
    applied as a source patch after the six exp-081 patches), run with OURS_UNTRIED=i1: the I1 line only.
C2  Our stale-state latch fix (patches/calamitychasm-stale-latch-fix.patch, applied after ours-10): a refusal raised
    as StaleStateActionError releases the per-snippet latch, at most STALE_RELEASES times per snippet.

Embedded verbatim in the notebook's install cell (scripts/_build_jad_exp081_u_kernel.py). install() checks that both
changes are live in the imported harness and returns the marker lines; it then wraps three functions to COUNT only:
  - inference.utils.ours_untried.prompt_lines: the I1 lines shown, and for each one whether a click within the next
    RESPONSE_ACTIONS actions hit a named object kind (the definition of JustAdev742's scripts/untried_read.py, on the
    history the prompt saw: the board is the last entry the prompt saw, the next actions are the entries after it);
  - inference.utils.ours_untried.stamp (called by the harness after every action() call that reached the solver):
    stale-state refusals per snippet, and whether a refusal directly followed a released one (must never happen);
  - ToolAgent._run_python_tool: snippet boundaries for the stamp counter.
Each wrapper calls the original first and returns its result unchanged; what the model sees and what the harness
does are untouched. A counter error is counted (``errors``) and never raised into a game.
"""
from __future__ import annotations

import functools
import hashlib
import threading
from pathlib import Path

STALE_MARK = "calamitychasm STALE_LATCH_FIX"
STALE_RELEASES = 3  # the patch's per-snippet cap (`_stale_releases = [3]`)
STALE_FIX_LINES = ("_stale_releases = [3]\n", "_stale_releases[0] -= 1\n")
UNTRIED_MODE = "i1"
I1_PREFIX = "Not yet tried on this level: "
I2_MARK = ", a colour new to this game; no click has hit "
RESPONSE_ACTIONS = 3
FIRINGS_MAX = 1000

_lock = threading.Lock()
_UT = None  # inference.utils.ours_untried, set by install()
_games: dict = {}
_firings: list = []
_U = dict(i1_lines=0, i1_lines_naming_objects=0, i1_named_kinds=0, i1_resolved=0, i1_partial_window=0,
          i1_clicked_named_within_3=0, i1_used_named_action_within_3=0, i1_unmatched=0, i2_lines=0, errors=0,
          last_error="")
_L = dict(snippets=0, action_calls=0, stale_refusals=0, stale_refused_after_released=0, stale_refused_after_cap=0,
          executed_after_stale_refusal=0, snippets_with_stale_refusal=0, max_stale_refusals_in_snippet=0, errors=0,
          last_error="")


class _Game:
    __slots__ = ("name", "snippet", "last", "refusals", "pending", "entries", "lines", "hits", "tracker")

    def __init__(self, name: str):
        self.name = name
        self.snippet = 0
        self.last = None  # "stale" / "exec" / "other": the previous action() result in this snippet
        self.refusals = 0  # stale-state refusals in this snippet
        self.pending: list = []  # I1 firings waiting for their next RESPONSE_ACTIONS history entries
        self.entries = None  # the last history the prompt saw (to close the pending firings at the end)
        self.lines = 0
        self.hits = 0
        self.tracker = None  # ours_untried's Tracker for this game (its .error turns the lines off)


def _game(agent) -> _Game:
    key = getattr(agent, "_session_runtime_dir", None)
    state = getattr(agent, "_cu_game", None)
    if state is None or state[0] != key:
        name = Path(str(key)).name[:48] if key is not None else "?"
        state = (key, _Game(name))
        agent._cu_game = state
        with _lock:
            _games[(id(agent), str(key))] = state[1]
    return state[1]


def _fail(counters: dict, exc: BaseException) -> None:
    with _lock:
        counters["errors"] += 1
        counters["last_error"] = f"{type(exc).__name__}: {exc}"[:200]


# --- install-time checks ---------------------------------------------------------------------------------------------


def _bootstraps(SB) -> dict:
    out = {"plain": SB._SANDBOX_BOOTSTRAP, "search": SB._SANDBOX_BOOTSTRAP_SEARCH}
    for flag, text in sorted(getattr(SB, "_OURS_PERCEPTION_BOOTSTRAPS", {}).items()):
        out[f"perception_search={flag}"] = text
    return out


def check_stale_fix(SB) -> int:
    """Every sandbox bootstrap the harness can launch carries the fix exactly once (they are all derived from the one
    patched source string); returns how many were checked. Raises otherwise."""
    boots = _bootstraps(SB)
    for name, text in boots.items():
        for line in STALE_FIX_LINES:
            if text.count(line) != 1:
                raise RuntimeError(f"STALE_LATCH_FIX missing from the {name} sandbox bootstrap ({line.strip()!r})")
        if STALE_MARK not in text:
            raise RuntimeError(f"STALE_LATCH_FIX marker missing from the {name} sandbox bootstrap")
        compile(text, f"bootstrap {name}", "exec")
    return len(boots)


def check_untried(UT) -> str:
    modes = sorted(UT.mode())
    if modes != [UNTRIED_MODE]:
        raise RuntimeError(f"OURS_UNTRIED gives modes {modes}; this arm runs the I1 line only ({UNTRIED_MODE!r})")
    return hashlib.sha256(Path(UT.__file__).read_bytes()).hexdigest()


# --- C1: the I1 lines and the clicks that follow them ----------------------------------------------------------------


def _named_kinds(record: dict) -> list:
    """The kinds an I1 line names: inference.utils.ours_untried.i1_block's order, first MAX_KINDS."""
    kinds, prev = record.get("kinds") or {}, record.get("prev_kinds")
    order = sorted(kinds.items(), key=lambda kv: (prev is not None and kv[0] in prev, len(kv[1]),
                                                  -kv[1][0].px, kv[1][0].box))
    return [k for k, _ in order[:_UT.MAX_KINDS]]


def _resolve(g: _Game, entries, final: bool) -> None:
    keep = []
    for f in g.pending:
        n = f["step"]
        if entries is None or len(entries) < n:
            with _lock:
                _U["i1_unmatched"] += 1  # the history shrank under the firing: cannot be read
            continue
        if len(entries) < n + RESPONSE_ACTIONS and not final:
            keep.append(f)
            continue
        hit = used = None
        for j in range(n, min(len(entries), n + RESPONSE_ACTIONS)):
            action = _UT._parts(entries[j])[0]
            if f["unused"] and used is None and _UT.action_type(action) in f["unused"]:
                used = j - n + 1
            cell = _UT.mouse_cell(action)
            if cell is None or hit is not None or not f["kinds"] or j < 1:
                continue
            before = _UT._parts(entries[j - 1])[1]
            target = _UT.click_target(_UT.salient(_UT.components(before), _UT._shape(before)), *cell)
            if target is not None and _UT.kind_key(target) in f["kinds"]:
                hit = j - n + 1
        window = min(len(entries), n + RESPONSE_ACTIONS) - n
        with _lock:
            _U["i1_resolved"] += 1
            _U["i1_partial_window"] += window < RESPONSE_ACTIONS
            _U["i1_clicked_named_within_3"] += hit is not None
            _U["i1_used_named_action_within_3"] += used is not None
            g.hits += hit is not None
            if len(_firings) < FIRINGS_MAX:
                _firings.append({"game": g.name, "level": f["level"], "step": n, "reason": f["reason"],
                                 "named": len(f["kinds"]), "unused": sorted(f["unused"]), "hit": hit, "used": used,
                                 "window": window})
    g.pending = keep


def _account(agent, history_entries, out) -> None:
    g = _game(agent)
    entries = history_entries or []
    g.entries = entries
    state = getattr(agent, "_ours_untried", None)
    g.tracker = state[1] if state is not None and state[0] == getattr(agent, "_session_runtime_dir", None) else None
    _resolve(g, entries, final=False)
    for line in out or ():
        if not isinstance(line, str):
            continue
        if line.startswith(I1_PREFIX):
            fired = g.tracker.fired if g.tracker is not None else ()
            rec = next((r for r in reversed(fired) if r.get("line") == "i1" and r.get("text") == line), None)
            if rec is None:
                with _lock:
                    _U["i1_unmatched"] += 1
                continue
            kinds = _named_kinds(rec)
            g.pending.append({"step": len(entries), "kinds": set(kinds), "unused": set(rec.get("unused") or ()),
                              "level": rec.get("level"), "reason": rec.get("reason")})
            g.lines += 1
            with _lock:
                _U["i1_lines"] += 1
                _U["i1_lines_naming_objects"] += bool(kinds)
                _U["i1_named_kinds"] += len(kinds)
        elif I2_MARK in line:
            with _lock:
                _U["i2_lines"] += 1


# --- C2: stale-state refusals per snippet ----------------------------------------------------------------------------


def _snippet(agent) -> None:
    g = _game(agent)
    g.snippet += 1
    g.last = None
    g.refusals = 0
    with _lock:
        _L["snippets"] += 1


def _count_action(agent, raw_payload) -> None:
    if not isinstance(raw_payload, dict):
        return
    g = _game(agent)
    executed = bool(raw_payload.get("executed"))
    stale = not executed and raw_payload.get("stop_reason") == "stale_state"
    with _lock:
        _L["action_calls"] += 1
        if stale:
            g.refusals += 1
            _L["stale_refusals"] += 1
            _L["snippets_with_stale_refusal"] += g.refusals == 1
            _L["max_stale_refusals_in_snippet"] = max(_L["max_stale_refusals_in_snippet"], g.refusals)
            if g.last == "stale":
                # the refusal before this one released the latch unless the snippet had used up its releases
                key = "stale_refused_after_released" if g.refusals - 1 <= STALE_RELEASES else "stale_refused_after_cap"
                _L[key] += 1
        elif executed and g.last == "stale":
            _L["executed_after_stale_refusal"] += 1
    g.last = "stale" if stale else "exec" if executed else "other"


# --- install / summaries ---------------------------------------------------------------------------------------------


def install(SB, TA, UT) -> list:
    """Check both changes in the imported harness, wrap the three functions (count only); the marker lines."""
    global _UT
    boots = check_stale_fix(SB)
    module_sha = check_untried(UT)
    if getattr(UT, "_cu_installed", False):
        return []
    _UT = UT
    lines_orig, stamp_orig, run_orig = UT.prompt_lines, UT.stamp, TA.ToolAgent._run_python_tool

    @functools.wraps(lines_orig)
    def prompt_lines(agent, history_entries, valid_actions):
        out = lines_orig(agent, history_entries, valid_actions)
        try:
            _account(agent, history_entries, out)
        except Exception as exc:  # noqa: BLE001 - a counter never stops a game
            _fail(_U, exc)
        return out

    @functools.wraps(stamp_orig)
    def stamp(agent, raw_payload):
        result = stamp_orig(agent, raw_payload)
        try:
            _count_action(agent, raw_payload)
        except Exception as exc:  # noqa: BLE001
            _fail(_L, exc)
        return result

    @functools.wraps(run_orig)
    def _run_python_tool(self, *args, **kwargs):
        try:
            _snippet(self)
        except Exception as exc:  # noqa: BLE001
            _fail(_L, exc)
        return run_orig(self, *args, **kwargs)

    for fn in (prompt_lines, stamp, _run_python_tool):
        fn.__module__ = "cu_counters"
    UT.prompt_lines, UT.stamp, TA.ToolAgent._run_python_tool = prompt_lines, stamp, _run_python_tool
    UT._cu_installed = True
    return [f"STALE_LATCH_FIX installed bootstraps={boots} releases_per_snippet={STALE_RELEASES}",
            f"UNTRIED_NOTICE installed mode={UNTRIED_MODE} ours_untried.py sha256={module_sha[:16]} counters=on"]


def untried_summary() -> dict:
    with _lock:
        s = dict(_U)
        s["mode"] = UNTRIED_MODE
        s["i1_pending"] = sum(len(g.pending) for g in _games.values())
        s["i1_clicked_named_within_3_share"] = round(s["i1_clicked_named_within_3"] / s["i1_resolved"], 3) \
            if s["i1_resolved"] else None
        s["games_with_i1"] = sum(g.lines > 0 for g in _games.values())
        # games whose ours_untried tracker hit an internal error: it turns the lines off for the rest of the game
        s["tracker_errors"] = sum(bool(getattr(g.tracker, "error", "")) for g in _games.values())
        s["per_game"] = {g.name: [g.lines, g.hits] for g in _games.values() if g.lines}
    return s


def stale_summary() -> dict:
    with _lock:
        s = dict(_L)
        s["releases_per_snippet"] = STALE_RELEASES
    return s


def finish() -> dict:
    """Close every pending I1 firing on the last history its game showed; the files the run cell writes."""
    for g in list(_games.values()):
        try:
            _resolve(g, g.entries, final=True)
        except Exception as exc:  # noqa: BLE001
            _fail(_U, exc)
    with _lock:
        firings = list(_firings)
    return {"untried_summary.json": untried_summary(), "stale_latch_summary.json": stale_summary(),
            "untried_firings.json": firings}
