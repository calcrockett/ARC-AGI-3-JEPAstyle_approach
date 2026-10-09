"""History cache for the milestone-2 harness (variant A').

WHY. Per executed action the harness paid O(history) four times over, all in the one notebook
process every game shares (GIL), while the game holds a GPU stream:
  (a) the solver rewrote the whole game history to tool_runtime_state.json with
      json.dumps(indent=2) after EVERY action of a batch (runtime_state.write_runtime_state);
  (b) every python tool call re-read and re-parsed that file (tool_agent.load_runtime_state);
  (c) after every action() the tool re-read it again, rebuilt ascii + grid payloads for EVERY
      past frame (_ascii_history_view_payload) and shipped the whole history to the sandbox as
      one JSON line;
  (d) the sandbox re-parsed that line and rebuilt every frame view, inside its 30 s budget.
Measured on synthetic 64x64 histories: ~0.3 s per action at 100 history entries, ~1 s at 300,
~4 s at 1000 (18 MB per action).

WHAT (the model and the sandbox see the same information; only the cost changes):
  1. write: each history entry is serialized once; the file is written compactly (same JSON
     content); an unchanged state is not rewritten; inside one action batch (step_env) the
     per-action writes collapse into one write when the batch ends.
  2. read: load_runtime_state returns in-memory objects built by exactly the round trip a file
     read performs (json.loads of the entry's own JSON -> history_entry_from_payload), checked
     against the file's stat; anything unknown or stale falls back to parsing the file.
  3. ascii: each distinct frame is formatted once (ascii, grid JSON) and reused.
  4. sandbox: after an action() the reply carries only the history entries the sandbox does not
     have yet ("history_delta"), whenever the host can prove (identity of every entry already
     sent on that pipe) that the new history extends what it sent; otherwise -- and always for
     a new sandbox process -- the full payload. The sandbox keeps the raw entries it received
     and rebuilds FRESH view objects (fresh grids, fresh result dicts) on every refresh, exactly
     as before, so a snippet that mutates a history frame sees the same thing it saw before.

Ported from sirikilohit's Milestone-2 patch M86 ("game state without the per-action O(history)
rewrite", github.com/LohitSiriki/arc-agi-3-milestone2-solution, cell 38, Apache-2.0) and
adapted to dfranzen's milestone-2 harness; the incremental sandbox protocol (4) and the
instrumentation are new here.

READ: summary(); a "HISTORY_CACHE stats" line is printed every LOG_EVERY python tool calls.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
import uuid
import weakref
from pathlib import Path

LOG_EVERY = 250                      # python tool calls between "HISTORY_CACHE stats" lines
BUCKETS = ((0, 99), (100, 299), (300, 999), (1000, None))
TIMEOUT_PREFIX = "Tool timed out after"
DIED_TEXT = "Sandbox process exited unexpectedly."

_LOCK = threading.RLock()
_TLS = threading.local()
_REG = {}                            # str(state path) -> _Rec
_BY_ENTRY = {}                       # id(loaded HistoryEntry) -> _Item
_BY_VIEW = {}                        # id(cached view dict) -> _Item
_MIRROR = weakref.WeakKeyDictionary()  # sandbox stdin handle -> list[_Item] the sandbox holds
_TOKEN = "hc-history-" + uuid.uuid4().hex
_ORIG = {}

STATS = {
    "installed": False, "errors": 0, "last_error": "",
    "writes": 0, "writes_appended": 0, "writes_skipped_unchanged": 0, "writes_deferred": 0, "write_fallbacks": 0,
    "rebuilds": 0, "write_s_max": 0.0,
    "loads_cached": 0, "loads_file": 0, "loads_stale": 0,
    "view_hits": 0, "view_misses": 0,
    "payload_full": 0, "payload_delta": 0, "payload_plain": 0, "delta_entries": 0,
    "python_calls": 0, "sandbox_timeout_kills": 0, "sandbox_died": 0,
    "max_history": 0, "forgotten": 0,
}
_CALLS = {b: {"calls": 0, "actions": 0, "host_s": 0.0, "host_s_max": 0.0, "cpu_s": 0.0, "wall_s": 0.0}
          for b in BUCKETS}

# ----------------------------------------------------------------------------- sandbox side
# Inserted into the sandbox bootstrap (a separate process) before `def main():`.
SANDBOX_HELPERS = r"""
import gc as _hc_gc

_HC_RAW = []


def _hc_deep(value):
    kind = type(value)
    if kind is list:
        return [_hc_deep(item) for item in value]
    if kind is dict:
        return {key: _hc_deep(item) for key, item in value.items()}
    return value


def _hc_fresh_frame(frame):
    out = {}
    for key, value in frame.items():
        if key == "grid" and type(value) is list and all(type(row) is list for row in value):
            # rows come from the host's normalized int grid: a row copy is a full copy
            out[key] = list(map(list.copy, value))
        else:
            out[key] = _hc_deep(value)
    return out


def _hc_fresh(entry):
    if type(entry) is not dict:
        return entry
    out = {}
    for key, value in entry.items():
        if key == "frame" and type(value) is dict:
            out[key] = _hc_fresh_frame(value)
        else:
            out[key] = _hc_deep(value)
    return out


def _hc_step(entry):
    frame = entry.get("frame") if type(entry) is dict else None
    return frame.get("step") if type(frame) is dict else None


def _hc_history(state_payload):
    delta = state_payload.get("history_delta")
    if isinstance(delta, dict):
        base = delta.get("base_len")
        if base != len(_HC_RAW) or (base and delta.get("base_last_step") != _hc_step(_HC_RAW[base - 1])):
            raise RuntimeError("history delta does not extend the sandbox history")
        _HC_RAW.extend(delta.get("entries") or [])
    else:
        raw = state_payload.get("history")
        if type(raw) is not list:
            del _HC_RAW[:]
            return _history_from_payload(raw)
        _HC_RAW[:] = raw
    # fresh objects on every refresh, as when the whole history was re-sent and re-parsed.
    # The cyclic GC is paused while they are built: tens of thousands of new lists would
    # otherwise trigger full collections over the whole heap (not observable by a snippet,
    # which cannot import gc).
    gc_was_enabled = _hc_gc.isenabled()
    _hc_gc.disable()
    try:
        return _history_from_payload([_hc_fresh(entry) for entry in _HC_RAW])
    finally:
        if gc_was_enabled:
            _hc_gc.enable()

"""
SANDBOX_OLD_LINE = '        history = _history_from_payload(state_payload.get("history"))\n'
SANDBOX_NEW_LINE = '        history = _hc_history(state_payload)\n'
SANDBOX_MAIN = "\ndef main():\n"


def patch_bootstrap(src: str) -> str:
    if src.count(SANDBOX_OLD_LINE) != 1 or src.count(SANDBOX_MAIN) != 1:
        raise RuntimeError("history_cache: sandbox bootstrap anchors not found exactly once")
    src = src.replace(SANDBOX_OLD_LINE, SANDBOX_NEW_LINE)
    return src.replace(SANDBOX_MAIN, "\n" + SANDBOX_HELPERS + SANDBOX_MAIN)


# ----------------------------------------------------------------------------- host side
class _Grid:
    __slots__ = ("ascii", "ascii_json", "shape_json", "grid_json")


class _Item:
    """One loadable history entry: the loaded object, its sandbox view, its view JSON parts."""
    __slots__ = ("entry", "view", "vhead", "grid", "vmid", "vtail", "step")

    def parts(self):
        g = self.grid
        return (self.vhead, g.ascii_json, self.vmid, g.grid_json, self.vtail)


class _Rec:
    def __init__(self):
        self.src_len = 0
        self.src_first = None
        self.src_last = None
        self.fparts = []      # per solver entry: (head, grid_json, tail) or (json,)
        self.items = []       # per loadable entry
        self.entries = []     # items[i].entry, kept as a list for cheap slicing
        self.rows = {}        # interned grid rows
        self.grids = {}       # normalized grid -> _Grid
        self.gjson = {}       # solver grid -> its JSON
        self.snap = None      # (current_frame, n_items, file stat, signature) of the last write
        self.fn = 0           # fparts already in the file
        self.hist_end = 0     # file offset just past the last history entry
        self.appendable = False

    def extended_by(self, hist):
        if self.src_len > len(hist):
            return False
        return self.src_len == 0 or (hist[self.src_len - 1] is self.src_last and hist[0] is self.src_first)


def _err(exc):
    with _LOCK:
        STATS["errors"] += 1
        STATS["last_error"] = f"{type(exc).__name__}: {exc}"[:200]


def _bump(key, n=1):
    with _LOCK:
        STATS[key] += n


def _acc(seconds):
    acc = getattr(_TLS, "acc", None)
    if acc is not None:
        acc["host_s"] += seconds


def _stat_sig(path):
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def forget(path) -> None:
    """Drop everything cached for one state file (end of game)."""
    with _LOCK:
        rec = _REG.pop(str(path), None)
        if rec is None:
            return
        for it in rec.items:
            _BY_ENTRY.pop(id(it.entry), None)
            _BY_VIEW.pop(id(it.view), None)
        STATS["forgotten"] += 1


def _add_entry(rec, e, RS, orig_frame_view):
    p = RS.history_entry_to_payload(e)
    fr = p.get("frame")
    if isinstance(fr, dict) and list(fr) == ["grid", "step", "level"]:
        sgrid = getattr(getattr(e, "frame", None), "grid", None)
        try:
            gj = rec.gjson.get(sgrid)
        except TypeError:          # unhashable grid
            sgrid, gj = None, None
        if gj is None:
            gj = json.dumps(fr["grid"])
            if sgrid is not None:
                rec.gjson[sgrid] = gj
        head = '{"action": ' + json.dumps(p["action"]) + ', "frame": {"grid": '
        tail = (', "step": ' + json.dumps(fr["step"]) + ', "level": ' + json.dumps(fr["level"])
                + '}, "result": ' + json.dumps(p["result"]) + "}")
        part = (head, gj, tail)
    else:
        part = (json.dumps(p),)
    # exactly what a file read builds for this entry
    rt = RS.history_entry_from_payload(json.loads("".join(part)))
    rec.fparts.append(part)
    if rt is None:
        return
    rows = rec.rows
    grid = tuple(rows.setdefault(r, r) for r in rt.frame.grid)
    rt = dataclasses.replace(rt, frame=dataclasses.replace(rt.frame, grid=grid))
    g = rec.grids.get(grid)
    if g is None:
        fp = orig_frame_view(rt.frame)
        g = _Grid()
        g.ascii = fp["ascii"]
        g.ascii_json = json.dumps(fp["ascii"], ensure_ascii=False)
        g.shape_json = json.dumps(fp["shape"])
        same = (len(part) == 3 and fr["grid"] == [list(r) for r in grid]
                and all(type(v) is int for r in fr["grid"] for v in r))
        g.grid_json = gj if same else json.dumps(fp["grid"])
        rec.grids[grid] = g
    result = dict(getattr(rt, "result", {}) or {})
    shape = json.loads(g.shape_json)
    view = {"action": rt.action,
            "frame": {"ascii": g.ascii, "step": rt.frame.step, "level": rt.frame.level,
                      "shape": shape, "grid": grid},
            "result": result}
    it = _Item()
    it.entry, it.view, it.grid, it.step = rt, view, g, rt.frame.step
    it.vhead = '{"action": ' + json.dumps(rt.action, ensure_ascii=False) + ', "frame": {"ascii": '
    it.vmid = (', "step": ' + json.dumps(rt.frame.step) + ', "level": ' + json.dumps(rt.frame.level)
               + ', "shape": ' + g.shape_json + ', "grid": ')
    it.vtail = '}, "result": ' + json.dumps(result, ensure_ascii=False) + "}"
    rec.items.append(it)
    rec.entries.append(rt)
    with _LOCK:
        _BY_ENTRY[id(rt)] = it
        _BY_VIEW[id(view)] = it


def install(*, solver: bool = True) -> bool:
    """Patch the harness modules in place. Returns False if already installed."""
    import inference.agent.python_tool_sandbox as SB
    import inference.agent.runtime_state as RS
    import inference.agent.tool_agent as TA

    if STATS["installed"]:
        return False
    S = None
    if solver:
        import inference.framework.solver as S
    new_bootstrap = patch_bootstrap(SB._SANDBOX_BOOTSTRAP)

    orig_load = TA.load_runtime_state
    orig_hist_view = TA._ascii_history_view_payload
    orig_frame_view = TA._ascii_frame_view_payload
    orig_send = SB._send_json_line
    orig_rsp = TA.run_sandboxed_python
    orig_rpt = TA.ToolAgent._run_python_tool
    orig_write = S.write_runtime_state if S is not None else RS.write_runtime_state

    def write_runtime_state(path, *, current_frame, history):
        t0, c0 = time.perf_counter(), time.thread_time()
        key = str(path)
        try:
            hist = history if isinstance(history, list) else list(history)
            with _LOCK:
                rec = _REG.get(key)
            if rec is not None and not rec.extended_by(hist):
                forget(key)
                rec = None
                _bump("rebuilds")
            fresh = rec is None
            if fresh:
                rec = _Rec()
            for e in hist[rec.src_len:]:
                _add_entry(rec, e, RS, orig_frame_view)
            rec.src_len = len(hist)
            rec.src_first = hist[0] if hist else None
            rec.src_last = hist[-1] if hist else None
            if fresh:
                with _LOCK:
                    _REG[key] = rec
            cf_json = json.dumps(RS.frame_to_payload(current_frame))
            sig = (len(rec.fparts), cf_json)
            p = Path(path)
            snap = rec.snap
            try:
                on_disk = snap is not None and _stat_sig(p) == snap[2]
            except OSError:
                on_disk = False
            if on_disk and snap[3] == sig:
                _bump("writes_skipped_unchanged")
                return None
            closing = '], "current_frame": ' + cf_json + "}"
            if on_disk and rec.appendable:
                # the file still holds exactly what we last wrote: replace only its tail
                body = "".join((", " if (i or rec.fn) else "") + "".join(part)
                               for i, part in enumerate(rec.fparts[rec.fn:]))
                data = (body + closing).encode("ascii")
                with open(p, "r+b") as fh:
                    fh.seek(rec.hist_end)
                    fh.write(data)
                    fh.truncate()
                rec.hist_end += len(body)
                _bump("writes_appended")
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                chunks = ['{"history": [']
                for i, part in enumerate(rec.fparts):
                    if i:
                        chunks.append(", ")
                    chunks.extend(part)
                head = "".join(chunks)
                text = head + closing
                rec.appendable = text.isascii()      # json.dumps escapes non-ASCII: always true
                tmp = p.with_suffix(f"{p.suffix}.tmp")
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(p)
                rec.hist_end = len(head)
            rec.fn = len(rec.fparts)
            cf = RS.frame_from_payload(json.loads(cf_json))
            rec.snap = (cf, len(rec.items), _stat_sig(p), sig)
            dt = time.perf_counter() - t0
            with _LOCK:
                STATS["writes"] += 1
                STATS["write_s_max"] = max(STATS["write_s_max"], round(dt, 4))
                STATS["max_history"] = max(STATS["max_history"], len(rec.items))
            return None
        except Exception as exc:      # never lose the state: drop the cache, use the original writer
            _err(exc)
            forget(key)
            _bump("write_fallbacks")
            return orig_write(path, current_frame=current_frame, history=history)
        finally:
            _acc(time.thread_time() - c0)

    def load_runtime_state(path):
        t0 = time.thread_time()
        try:
            p = Path(path)
            with _LOCK:
                rec = _REG.get(str(p))
                snap = rec.snap if rec is not None else None
            if snap is None:
                _bump("loads_file")
                return orig_load(p)
            try:
                st = _stat_sig(p)
            except FileNotFoundError:
                return None, []
            except OSError:
                st = None
            if st != snap[2]:
                _bump("loads_stale")
                return orig_load(p)
            _bump("loads_cached")
            return snap[0], rec.entries[:snap[1]]
        finally:
            _acc(time.thread_time() - t0)

    def _ascii_history_view_payload(history_entries):
        t0 = time.thread_time()
        out, hits, misses = [], 0, 0
        for e in history_entries:
            it = _BY_ENTRY.get(id(e))
            if it is not None and it.entry is e:
                out.append(it.view)
                hits += 1
                continue
            misses += 1
            out.extend(orig_hist_view([e]))
        with _LOCK:
            STATS["view_hits"] += hits
            STATS["view_misses"] += misses
        _acc(time.thread_time() - t0)
        return out

    def _items_of(hist):
        if type(hist) is not list:
            return None
        out = []
        for d in hist:
            it = _BY_VIEW.get(id(d))
            if it is None or it.view is not d:
                return None
            out.append(it)
        return out

    def _encode(handle, payload):
        """The JSON line for payload, or None to let the original serializer send it."""
        if type(payload) is not dict:
            return None
        state = payload.get("state")
        if type(state) is not dict:
            return None
        items = _items_of(state.get("history"))
        if items is None:
            _MIRROR[handle] = None
            _bump("payload_plain")
            return None
        sent = None if "code" in payload else _MIRROR.get(handle)
        if (sent is not None and len(items) >= len(sent)
                and all(a is b for a, b in zip(sent, items))):
            seq = items[len(sent):]
            new_state = {}
            for k, v in state.items():
                if k == "history":
                    new_state["history_delta"] = {"base_len": len(sent),
                                                  "base_last_step": sent[-1].step if sent else None,
                                                  "entries": _TOKEN}
                else:
                    new_state[k] = v
            kind = "payload_delta"
        else:
            seq = items
            new_state = dict(state)
            new_state["history"] = _TOKEN
            kind = "payload_full"
        text = json.dumps({**payload, "state": new_state}, ensure_ascii=False)
        tok = json.dumps(_TOKEN)
        if text.count(tok) != 1:
            _MIRROR[handle] = None
            return None
        chunks = ["["]
        for i, it in enumerate(seq):
            if i:
                chunks.append(", ")
            chunks.extend(it.parts())
        chunks.append("]")
        line = text.replace(tok, "".join(chunks))
        _MIRROR[handle] = list(items)
        with _LOCK:
            STATS[kind] += 1
            if kind == "payload_delta":
                STATS["delta_entries"] += len(seq)
        acc = getattr(_TLS, "acc", None)
        if acc is not None:
            acc["delta" if kind == "payload_delta" else "full"] += 1
        return line

    def _send_json_line(handle, payload):
        t0 = time.thread_time()
        try:
            line = _encode(handle, payload)
        except Exception as exc:
            _err(exc)
            try:
                _MIRROR[handle] = None
            except TypeError:
                pass
            line = None
        _acc(time.thread_time() - t0)
        if line is None:
            return orig_send(handle, payload)
        handle.write(line + "\n")
        handle.flush()
        return None

    def run_sandboxed_python(*args, **kwargs):
        result = orig_rsp(*args, **kwargs)
        try:
            err = str((result or {}).get("error") or "")
            if err.startswith(TIMEOUT_PREFIX):
                _bump("sandbox_timeout_kills")
            elif err == DIED_TEXT:
                _bump("sandbox_died")
        except Exception as exc:
            _err(exc)
        return result

    def _run_python_tool(self, state_path, arguments):
        acc = {"host_s": 0.0, "full": 0, "delta": 0}
        prev = getattr(_TLS, "acc", None)
        _TLS.acc = acc
        t0, c0 = time.perf_counter(), time.thread_time()
        try:
            return orig_rpt(self, state_path, arguments)
        finally:
            _TLS.acc = prev
            try:
                _record_call(state_path, acc, time.perf_counter() - t0, time.thread_time() - c0)
            except Exception as exc:
                _err(exc)

    _run_python_tool.__wrapped__ = orig_rpt

    _ORIG.update(TA=TA, SB=SB, S=S, load=orig_load, hist_view=orig_hist_view, send=orig_send,
                 rsp=orig_rsp, rpt=orig_rpt, write=orig_write, bootstrap=SB._SANDBOX_BOOTSTRAP)
    SB._SANDBOX_BOOTSTRAP = new_bootstrap
    SB._send_json_line = _send_json_line
    TA.load_runtime_state = load_runtime_state
    TA._ascii_history_view_payload = _ascii_history_view_payload
    TA.run_sandboxed_python = run_sandboxed_python
    TA.ToolAgent._run_python_tool = _run_python_tool
    if S is not None:
        S.write_runtime_state = write_runtime_state
        Sess = S._HarnessGameSession
        _ORIG.update(wrs=Sess.write_runtime_state, step=Sess.step_env, play=Sess.play)
        wrap_session_class(Sess)
    STATS["installed"] = True
    return True


def wrap_session_class(Sess) -> None:
    """Defer the per-action state writes inside one step_env batch to one write at its end, and
    drop the cache when a game's play() returns (the solver unlinks the state file then)."""
    orig_wrs = Sess.write_runtime_state
    orig_step = Sess.step_env
    orig_play = getattr(Sess, "play", None)

    def write_runtime_state(self):
        if getattr(self, "_hc_defer", 0):
            self._hc_dirty = True
            _bump("writes_deferred")
            return None
        return orig_wrs(self)

    def _flush(self):
        if getattr(self, "_hc_dirty", False):
            self._hc_dirty = False
            orig_wrs(self)

    def step_env(self, arguments):
        self._hc_defer = getattr(self, "_hc_defer", 0) + 1
        try:
            result = orig_step(self, arguments)
        except BaseException:
            self._hc_defer -= 1
            if not self._hc_defer:
                try:
                    _flush(self)
                except Exception as exc:   # do not mask the batch's own exception
                    _err(exc)
            raise
        self._hc_defer -= 1
        if not self._hc_defer:
            _flush(self)
        return result

    Sess.write_runtime_state = write_runtime_state
    Sess.step_env = step_env
    if orig_play is not None:
        def play(self):
            try:
                return orig_play(self)
            finally:
                forget(getattr(self, "state_path", None))
        Sess.play = play


def uninstall() -> None:
    """Restore every patched attribute (tests)."""
    if not STATS["installed"]:
        return
    TA, SB, S = _ORIG["TA"], _ORIG["SB"], _ORIG["S"]
    SB._SANDBOX_BOOTSTRAP = _ORIG["bootstrap"]
    SB._send_json_line = _ORIG["send"]
    TA.load_runtime_state = _ORIG["load"]
    TA._ascii_history_view_payload = _ORIG["hist_view"]
    TA.run_sandboxed_python = _ORIG["rsp"]
    TA.ToolAgent._run_python_tool = _ORIG["rpt"]
    if S is not None:
        S.write_runtime_state = _ORIG["write"]
        Sess = S._HarnessGameSession
        Sess.write_runtime_state, Sess.step_env, Sess.play = _ORIG["wrs"], _ORIG["step"], _ORIG["play"]
    with _LOCK:
        for key in list(_REG):
            forget(key)
        STATS["installed"] = False
    _ORIG.clear()


def reset_stats() -> None:
    with _LOCK:
        for k, v in list(STATS.items()):
            if k not in ("installed",):
                STATS[k] = type(v)() if not isinstance(v, bool) else v
        for b in _CALLS.values():
            for k in b:
                b[k] = 0 if k in ("calls", "actions") else 0.0


def _bucket(n):
    for lo, hi in BUCKETS:
        if n >= lo and (hi is None or n <= hi):
            return (lo, hi)
    return BUCKETS[0]


def _record_call(state_path, acc, wall, cpu=0.0):
    with _LOCK:
        rec = _REG.get(str(state_path))
        n = len(rec.items) if rec is not None else 0
        b = _CALLS[_bucket(n)]
        b["calls"] += 1
        b["actions"] += acc["delta"] + max(0, acc["full"] - 1)
        b["host_s"] += acc["host_s"]
        b["host_s_max"] = max(b["host_s_max"], acc["host_s"])
        b["wall_s"] += wall
        b["cpu_s"] += cpu
        STATS["python_calls"] += 1
        due = STATS["python_calls"] % LOG_EVERY == 0
    if due:
        print("HISTORY_CACHE stats", json.dumps(summary()), flush=True)


def summary() -> dict:
    with _LOCK:
        out = dict(STATS)
        calls = {}
        for (lo, hi), b in _CALLS.items():
            if not b["calls"]:
                continue
            per_action = b["host_s"] / max(1, b["calls"] + b["actions"])
            calls[f"{lo}-{hi if hi is not None else ''}"] = {
                "calls": b["calls"], "actions": b["actions"],
                "host_ms_per_call": round(1000 * b["host_s"] / b["calls"], 2),
                "host_ms_per_state": round(1000 * per_action, 2),
                "host_ms_max": round(1000 * b["host_s_max"], 2),
                "cpu_s_per_call": round(b["cpu_s"] / b["calls"], 3),
                "wall_s_per_call": round(b["wall_s"] / b["calls"], 3),
            }
        out["by_history"] = calls
        out["games_cached"] = len(_REG)
    return out
