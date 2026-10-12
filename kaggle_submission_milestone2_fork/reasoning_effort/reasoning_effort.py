"""Static chat-template reasoning effort on every request (variant `re-<level>`).

The served Qwen3.8-Flash-Next chat template reads `reasoning_effort` from the request's chat_template_kwargs and
defaults to "xhigh", which prepends "Reasoning effort is set to xhigh. Please think carefully through the task,
validate key assumptions, consider plausible alternatives, ..." to the system prompt; "medium" adds nothing and
"low" adds "Keep your thinking brief and focused ...". Franzen's harness only sends `reasoning_effort` on its
truncation ladder (ARC3_REASONING_EFFORT_LADDER, off in the incumbent), so every incumbent request runs at xhigh.

`install(ToolAgent, level)` wraps `ToolAgent._harness_template_kwargs` -- the one method the request builder AND the
request log both read, so `<game>_requests.jsonl` records exactly what was sent -- and adds `reasoning_effort=level`
unless the ladder already set a value (a truncation still steps below the static level, as in JustAdev742's
ours-09 patch). Nothing else changes.

Two checks that the knob is real, not a silent no-op:
  * `template_check(model_dir)`: renders the served chat_template.jinja (the launcher passes it to SGLang with
    --chat-template) with jinja2 for the default and for `level`, and reports whether the xhigh text disappears;
  * `server_probe(...)`: two max_tokens=1 requests to the live server, without and with the kwarg; the server
    received and applied it iff the prompt is shorter with it (usage.prompt_tokens).

Idea and evidence: juliancamilovilla's single-knob runs on Franzen's stack and JustAdev742's exp-082 / ours-09
(github.com/JustAdev742/Arc-Agi-3-Kaggle-comp, Apache-2.0). This file is our own implementation (NOTICE.md).
"""

from __future__ import annotations

import functools
import hashlib
import json
import threading
import time
import urllib.request
from pathlib import Path

LEVELS = ("medium", "low")          # the template accepts xhigh (its default), medium and low
XHIGH_TEXT = "Reasoning effort is set to xhigh"
MARK = "REASONING_EFFORT"

_lock = threading.Lock()
_state = {"level": None, "calls": 0, "set": 0, "ladder_kept": 0, "errors": 0,
          "template": None, "server": None}


def summary() -> dict:
    """calls: reads of the kwargs (the request builder and the request log each read them, so ~3 per request with
    request logs on); set: reads that got the static level; ladder_kept: reads where the ladder's value stood."""
    with _lock:
        return json.loads(json.dumps(_state))


def reset() -> None:
    with _lock:
        _state.update(level=None, calls=0, set=0, ladder_kept=0, errors=0, template=None, server=None)


def install(cls, level: str) -> bool:
    if level not in LEVELS:
        raise ValueError(f"reasoning effort {level!r}: allowed {LEVELS} (xhigh is the template default)")
    orig = cls.__dict__.get("_harness_template_kwargs")
    if orig is None:
        raise AttributeError("ToolAgent has no _harness_template_kwargs (harness patch changed?)")
    if getattr(orig, "_re_wrapped", False):
        raise RuntimeError("reasoning effort already installed")

    @functools.wraps(orig)
    def _harness_template_kwargs(self):
        kwargs = orig(self)
        try:
            with _lock:
                _state["calls"] += 1
                if "reasoning_effort" in kwargs:      # the truncation ladder is active: it wins
                    _state["ladder_kept"] += 1
                else:
                    _state["set"] += 1
            if "reasoning_effort" not in kwargs:
                kwargs = dict(kwargs)
                kwargs["reasoning_effort"] = level
        except Exception:  # noqa: BLE001 -- never break a request
            with _lock:
                _state["errors"] += 1
        return kwargs

    _harness_template_kwargs.__module__ = __name__
    _harness_template_kwargs._re_wrapped = True
    cls._harness_template_kwargs = _harness_template_kwargs
    cls._re_installed = True
    with _lock:
        _state["level"] = level
    return True


def _template_text(model_dir) -> tuple[str | None, str]:
    d = Path(str(model_dir))
    f = d / "chat_template.jinja"
    if f.is_file():
        return f.read_text(encoding="utf-8"), f.name
    cfg = d / "tokenizer_config.json"
    if cfg.is_file():
        t = json.loads(cfg.read_text(encoding="utf-8")).get("chat_template")
        if isinstance(t, str):
            return t, cfg.name
    return None, "none"


def render(template: str, **kwargs) -> str:
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def _raise(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = _raise
    messages = [{"role": "system", "content": "You play a game."}, {"role": "user", "content": "Frame 1"}]
    tools = [{"type": "function", "function": {"name": "python", "parameters": {"type": "object"}}}]
    return env.from_string(template).render(messages=messages, tools=tools, add_generation_prompt=True, **kwargs)


def template_check(model_dir, level: str) -> dict:
    """supports: the default render carries the xhigh text and the `level` render does not."""
    out = {"source": None, "sha256": None, "supports": False, "default_has_xhigh": None,
           "level_removes_xhigh": None, "error": None}
    try:
        text, src = _template_text(model_dir)
        out["source"] = src
        if text is None:
            out["error"] = "no chat template found"
        else:
            out["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            default, lev = render(text), render(text, reasoning_effort=level)
            out["default_has_xhigh"] = XHIGH_TEXT in default
            out["level_removes_xhigh"] = XHIGH_TEXT not in lev
            out["supports"] = bool(out["default_has_xhigh"] and out["level_removes_xhigh"])
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    with _lock:
        _state["template"] = out
    return out


def _post(base_url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def server_probe(base_url: str, model: str, level: str, timeout: float = 120.0) -> dict:
    """received: the prompt is shorter with reasoning_effort=level than without (the xhigh text is gone)."""
    out = {"received": False, "default_prompt_tokens": None, "level_prompt_tokens": None, "delta": None,
           "error": None}
    try:
        msgs = [{"role": "system", "content": "You play a game."}, {"role": "user", "content": "Say ok."}]
        tok = {}
        for name, kw in (("default", {"preserve_thinking": True}),
                         ("level", {"preserve_thinking": True, "reasoning_effort": level})):
            r = _post(base_url, {"model": model, "messages": msgs, "max_tokens": 1, "temperature": 0.0,
                                 "chat_template_kwargs": kw}, timeout)
            tok[name] = int((r.get("usage") or {})["prompt_tokens"])
        out.update(default_prompt_tokens=tok["default"], level_prompt_tokens=tok["level"],
                   delta=tok["default"] - tok["level"], received=tok["default"] > tok["level"])
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    with _lock:
        _state["server"] = out
    return out


def server_line(res: dict) -> str:
    return (f"{MARK}_SERVER received={res['received']} default_prompt_tokens={res['default_prompt_tokens']} "
            f"level_prompt_tokens={res['level_prompt_tokens']} delta={res['delta']} error={res['error']}")


def _healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/").removesuffix("/v1") + "/health", timeout=10) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def start_server_probe(base_url: str, model: str, level: str, wait_s: float = 5400.0,
                       poll_s: float = 15.0) -> threading.Thread:
    """Background: wait for /health (the benchmark can be released while the server is still loading), then probe
    once and print the REASONING_EFFORT_SERVER line. Never raises."""
    def run():
        t0 = time.time()
        while time.time() - t0 < wait_s and not _healthy(base_url):
            time.sleep(poll_s)
        res = server_probe(base_url, model, level)
        print(server_line(res), flush=True)

    th = threading.Thread(target=run, name="reasoning-effort-probe", daemon=True)
    th.start()
    return th
