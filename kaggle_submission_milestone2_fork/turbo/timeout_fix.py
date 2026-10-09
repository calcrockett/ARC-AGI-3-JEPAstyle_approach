"""Sandbox-timeout fix for the milestone-2 (dfranzen) harness, installed at runtime.

Port of JustAdev742's harness patch ``ours-sandbox-timeout-keeps-work.patch`` (Apache-2.0,
github.com/JustAdev742/Arc-Agi-3-Kaggle-comp, kaggle/franzen/patches/; vendored verbatim next to
this file), applied by wrapping instead of by editing the harness tree, so the notebook's
upstream harness patch stays byte-identical.

The bug (verified in da-fr's source): when a python tool call hits the sandbox timeout,
``python_tool_sandbox.run_sandboxed_python`` returns ``{"error": "Tool timed out after Ns",
"stdout": "", "action_results": [...]}`` -- no ``keepable_functions``. ``ToolAgent.
_record_retained_functions`` then rebuilds the retained set from that empty list and drops EVERY
previously retained helper ("Your previous function f is no longer retained"). In dfranzen's demo
r11l lost 13 helpers to one 30 s timeout.

The fix (two parts, as in the patch):
  1. A timed-out call with nothing keepable leaves the previously retained functions in place and
     says so in the tool payload (``function_retention``). Any other result goes to the original
     method unchanged.
  2. ``time`` joins the sandbox's SAFE_MODULES (one snippet in the demo died on ``import time``).
     The 30 s limit itself is untouched.

``install()`` raises if an anchor is missing (the notebook cell then stops before the run).
"""

from __future__ import annotations

import threading

TIMEOUT_PREFIX = "Tool timed out"
SAFE_ANCHOR = "\nSAFE_MODULES = {\n"
SAFE_NEW = "\nSAFE_MODULES = {\n    \"time\",  # [timeout_fix] models time their own snippets\n"
NOTE = ("The call timed out, so nothing it defined was kept; your {n} previously retained "
        "functions are still available.")

_LOCK = threading.Lock()
STATS = {"installed": False, "time_module_allowed": False, "timeouts_seen": 0, "timeouts_kept": 0,
         "functions_kept_max": 0, "errors": 0, "last_error": ""}
_ORIG: dict = {}


def _bump(key, n=1):
    with _LOCK:
        STATS[key] += n


def _error(exc) -> None:
    with _LOCK:
        STATS["errors"] += 1
        STATS["last_error"] = f"{type(exc).__name__}: {exc}"[:200]


def summary() -> dict:
    with _LOCK:
        return dict(STATS)


def reset_stats() -> None:
    with _LOCK:
        for k in ("timeouts_seen", "timeouts_kept", "functions_kept_max", "errors"):
            STATS[k] = 0
        STATS["last_error"] = ""


def is_timeout_without_work(sandbox_result) -> bool:
    return (isinstance(sandbox_result, dict)
            and str(sandbox_result.get("error") or "").startswith(TIMEOUT_PREFIX)
            and not sandbox_result.get("keepable_functions"))


def patch_bootstrap(text: str) -> str:
    if text.count(SAFE_ANCHOR) != 1:
        raise RuntimeError(f"timeout_fix: SAFE_MODULES anchor found {text.count(SAFE_ANCHOR)}x in the sandbox "
                           "bootstrap (expected once); the harness is not the analysed version")
    return text.replace(SAFE_ANCHOR, SAFE_NEW)


def install() -> bool:
    """Patch the harness modules in place. Returns False if already installed."""
    import inference.agent.python_tool_sandbox as SB
    import inference.agent.tool_agent as TA

    if STATS["installed"]:
        return False
    cls = TA.ToolAgent
    orig = cls.__dict__.get("_record_retained_functions")
    if orig is None or not callable(getattr(TA, "_persistent_functions", None)):
        raise RuntimeError("timeout_fix: ToolAgent._record_retained_functions / _persistent_functions not found")
    new_bootstrap = patch_bootstrap(SB._SANDBOX_BOOTSTRAP)

    def _record_retained_functions(self, sandbox_result, payload):
        try:
            if TA._persistent_functions() and is_timeout_without_work(sandbox_result):
                _bump("timeouts_seen")
                previous = getattr(self, "_kept_functions", None) or {}
                if previous:
                    payload["function_retention"] = NOTE.format(n=len(previous))
                    with _LOCK:
                        STATS["timeouts_kept"] += 1
                        STATS["functions_kept_max"] = max(STATS["functions_kept_max"], len(previous))
                    return None
        except Exception as exc:  # noqa: BLE001 -- never break a tool call; fall back to the original
            _error(exc)
        return orig(self, sandbox_result, payload)

    _record_retained_functions.__module__ = __name__
    _record_retained_functions.__qualname__ = "ToolAgent._record_retained_functions"
    _ORIG.update(TA=TA, SB=SB, record=orig, bootstrap=SB._SANDBOX_BOOTSTRAP)
    cls._record_retained_functions = _record_retained_functions
    SB._SANDBOX_BOOTSTRAP = new_bootstrap
    with _LOCK:
        STATS["installed"] = True
        STATS["time_module_allowed"] = '"time",' in SB._SANDBOX_BOOTSTRAP
    return True


def uninstall() -> None:
    """Restore every patched attribute (tests)."""
    if not STATS["installed"]:
        return
    TA, SB = _ORIG["TA"], _ORIG["SB"]
    TA.ToolAgent._record_retained_functions = _ORIG["record"]
    if SB._SANDBOX_BOOTSTRAP.count(SAFE_NEW) == 1:   # leave other installers' later edits in place
        SB._SANDBOX_BOOTSTRAP = SB._SANDBOX_BOOTSTRAP.replace(SAFE_NEW, SAFE_ANCHOR)
    _ORIG.clear()
    with _LOCK:
        STATS["installed"] = False
        STATS["time_module_allowed"] = False
