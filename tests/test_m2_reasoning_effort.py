"""Reasoning-effort variant `re-<level>` (arc3-m2-turbo-lossless-tail-re-medium): every request's chat_template_kwargs
carry reasoning_effort=<level> instead of the served template's default xhigh. Unit tests on a fake ToolAgent, the
template check on a synthetic template (and on the real Flash-Next template when JustAdev742's fixture is checked
out), the server probe against a local fake server, integration on the REAL patched ToolAgent (skipped when the
milestone-2 src is unavailable), builder tests on the committed kernel."""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "kaggle_submission_milestone2_fork" / "reasoning_effort"))

import _build_m2_level_memory_kernel as B  # noqa: E402
import reasoning_effort as re_mod  # noqa: E402
from _check_notebook_cell import check_notebook  # noqa: E402

VARIANTS = B.TURBO_LOSSLESS + ("tail", "re-medium")
SLUG, DIRNAME = "arc3-m2-turbo-lossless-tail-re-medium", "kaggle_submission_m2_turbo_lossless_tail_re_medium"
NB = ROOT / DIRNAME / "notebook" / f"{SLUG}.ipynb"
BASE_NB = ROOT / "kaggle_submission_m2_turbo_lossless_tail" / "notebook" / "arc3-m2-turbo-lossless-tail.ipynb"
FLASHNEXT_TEMPLATE = Path("/home/user/ext/JustAdev742_Arc-Agi-3-Kaggle-comp/tests/fixtures/flashnext_chat_template")

# the effort logic of the Flash-Next template, reduced to what the check reads
SYNTH = (
    "{%- set e = reasoning_effort|default('xhigh') %}"
    "{%- if e not in ('xhigh', 'medium', 'low') %}{{ raise_exception('Unexpected reasoning effort ' ~ e) }}{% endif %}"
    "<|im_start|>system\n{{ messages[0].content }}"
    "{%- if e == 'xhigh' %} Reasoning effort is set to xhigh. Please think carefully.{% endif %}"
    "{%- if e == 'low' %} Reasoning effort is set to low. Keep your thinking brief.{% endif %}"
    "<|im_end|>\n{% for m in messages[1:] %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


@pytest.fixture(autouse=True)
def clean():
    re_mod.reset()
    yield
    re_mod.reset()


def fake_tool_agent():
    """A ToolAgent with the upstream _harness_template_kwargs body (preserve_thinking + the ladder)."""
    class ToolAgent:
        def __init__(self, rung=-1):
            self._reasoning_effort_rung = rung

        def _harness_template_kwargs(self):
            kwargs = {"preserve_thinking": True}
            ladder = [x.strip() for x in os.environ.get("ARC3_REASONING_EFFORT_LADDER", "").split(",") if x.strip()]
            rung = getattr(self, "_reasoning_effort_rung", -1)
            if ladder and 0 <= rung < len(ladder):
                kwargs["reasoning_effort"] = ladder[rung]
            return kwargs
    return ToolAgent


# ------------------------------------------------------------------------------------------------- unit

def test_install_adds_the_level_and_the_ladder_still_wins(monkeypatch):
    monkeypatch.delenv("ARC3_REASONING_EFFORT_LADDER", raising=False)
    cls = fake_tool_agent()
    assert re_mod.install(cls, "medium") and cls._harness_template_kwargs.__module__ == "reasoning_effort"
    assert cls()._harness_template_kwargs() == {"preserve_thinking": True, "reasoning_effort": "medium"}
    monkeypatch.setenv("ARC3_REASONING_EFFORT_LADDER", "low")
    assert cls(rung=0)._harness_template_kwargs()["reasoning_effort"] == "low"     # a truncation steps below it
    assert cls(rung=-1)._harness_template_kwargs()["reasoning_effort"] == "medium"
    s = re_mod.summary()
    assert (s["level"], s["calls"], s["set"], s["ladder_kept"], s["errors"]) == ("medium", 3, 2, 1, 0)


def test_install_refuses_bad_levels_twice_and_a_missing_method():
    with pytest.raises(ValueError):
        re_mod.install(fake_tool_agent(), "xhigh")     # the default: nothing to send
    with pytest.raises(ValueError):
        re_mod.install(fake_tool_agent(), "high")      # the template raises on it
    cls = fake_tool_agent()
    re_mod.install(cls, "low")
    with pytest.raises(RuntimeError):
        re_mod.install(cls, "low")
    with pytest.raises(AttributeError):
        re_mod.install(type("ToolAgent", (), {}), "medium")


def test_wrapper_never_breaks_a_request(monkeypatch):
    class Odd(dict):          # a kwargs object whose membership test fails
        def __contains__(self, k):
            raise RuntimeError("boom")

    cls = type("ToolAgent", (), {"_harness_template_kwargs": lambda self: Odd(preserve_thinking=True)})
    re_mod.install(cls, "medium")
    assert cls()._harness_template_kwargs() == {"preserve_thinking": True}
    assert re_mod.summary()["errors"] == 1


def test_template_check_on_a_synthetic_template(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(SYNTH, encoding="utf-8")
    for level in B.RE_LEVELS:
        r = re_mod.template_check(tmp_path, level)
        assert r["supports"] and r["default_has_xhigh"] and r["level_removes_xhigh"] and r["error"] is None
    assert re_mod.summary()["template"]["source"] == "chat_template.jinja"
    # a template without the kwarg: the knob would be a silent no-op
    (tmp_path / "chat_template.jinja").write_text("{{ messages[0].content }}", encoding="utf-8")
    r = re_mod.template_check(tmp_path, "medium")
    assert r["supports"] is False and r["default_has_xhigh"] is False
    # tokenizer_config.json fallback, then nothing at all
    (tmp_path / "chat_template.jinja").unlink()
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": SYNTH}), encoding="utf-8")
    assert re_mod.template_check(tmp_path, "medium")["supports"]
    (tmp_path / "tokenizer_config.json").unlink()
    r = re_mod.template_check(tmp_path, "medium")
    assert r["supports"] is False and r["error"] == "no chat template found"


@pytest.mark.skipif(not (FLASHNEXT_TEMPLATE / "chat_template.jinja").is_file(),
                    reason="JustAdev742's verbatim Flash-Next template fixture is not checked out")
def test_template_check_on_the_real_flash_next_template():
    for level in B.RE_LEVELS:
        r = re_mod.template_check(FLASHNEXT_TEMPLATE, level)
        assert r["supports"], r
        assert r["sha256"].startswith("c3cf9e34")
    text = (FLASHNEXT_TEMPLATE / "chat_template.jinja").read_text()
    assert len(re_mod.render(text, reasoning_effort="medium")) < len(re_mod.render(text))


class _FakeServer:
    """/health and /v1/chat/completions; prompt_tokens grows by 30 when the kwarg is absent (if `honours`)."""

    def __init__(self, honours=True):
        seen = self.seen = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200 if self.path == "/health" else 404)
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append(body)
                kw = body.get("chat_template_kwargs") or {}
                n = 100 + (30 if honours and "reasoning_effort" not in kw else 0)
                out = json.dumps({"usage": {"prompt_tokens": n, "completion_tokens": 1}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def test_server_probe_received_ignored_and_unreachable():
    srv = _FakeServer(honours=True)
    try:
        r = re_mod.server_probe(srv.url, "flashnext", "medium")
        assert r == {"received": True, "default_prompt_tokens": 130, "level_prompt_tokens": 100, "delta": 30,
                     "error": None}
        assert [b["chat_template_kwargs"] for b in srv.seen] == [
            {"preserve_thinking": True}, {"preserve_thinking": True, "reasoning_effort": "medium"}]
        assert all(b["max_tokens"] == 1 and b["model"] == "flashnext" for b in srv.seen)
        assert re_mod.server_line(r).startswith("REASONING_EFFORT_SERVER received=True ")
    finally:
        srv.close()
    srv = _FakeServer(honours=False)
    try:
        r = re_mod.server_probe(srv.url, "flashnext", "medium")
        assert r["received"] is False and r["delta"] == 0
    finally:
        srv.close()
    r = re_mod.server_probe("http://127.0.0.1:9/v1", "flashnext", "medium", timeout=2)
    assert r["received"] is False and r["error"]
    assert re_mod.summary()["server"] == r


def test_background_probe_waits_for_health_then_prints(capsys):
    srv = _FakeServer()
    try:
        th = re_mod.start_server_probe(srv.url, "flashnext", "medium", wait_s=30, poll_s=0.1)
        th.join(30)
        assert not th.is_alive()
    finally:
        srv.close()
    assert "REASONING_EFFORT_SERVER received=True" in capsys.readouterr().out
    assert re_mod.summary()["server"]["received"] is True


# ------------------------------------------------------------------------------------------ real harness
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_harness  # noqa: E402

M2_SRC = m2_harness.m2_src()
real = pytest.mark.skipif(not M2_SRC, reason="milestone-2 src unavailable (ARC3_M2_SRC / ARC3_M2_BASE)")


@real
def test_real_harness_kwargs_and_request_builder_carry_the_level(monkeypatch):
    saved_env = dict(os.environ)
    if "inference.agent.tool_agent" not in sys.modules:
        os.environ.update(m2_harness.notebook_env())
    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")
    os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "flashnext")
    ta, _, _, _ = m2_harness.import_harness(M2_SRC)
    cls = ta.ToolAgent
    orig = cls.__dict__["_harness_template_kwargs"]
    try:
        assert "ARC3_REASONING_EFFORT_LADDER" not in m2_harness.notebook_env()   # the incumbent runs without it
        os.environ.pop("ARC3_REASONING_EFFORT_LADDER", None)
        me = SimpleNamespace(_reasoning_effort_rung=-1)
        assert cls._harness_template_kwargs(me) == {"preserve_thinking": True}     # the incumbent: xhigh by default
        re_mod.install(cls, "medium")
        assert cls._harness_template_kwargs(me) == {"preserve_thinking": True, "reasoning_effort": "medium"}
        os.environ["ARC3_REASONING_EFFORT_LADDER"] = "low"
        assert cls._harness_template_kwargs(SimpleNamespace(_reasoning_effort_rung=0))["reasoning_effort"] == "low"
        os.environ.pop("ARC3_REASONING_EFFORT_LADDER")
        assert cls(model="local")._harness_template_kwargs()["reasoning_effort"] == "medium"   # a real instance
        # the request builder merges these kwargs into the posted payload, and the request log records the same call
        import inspect
        assert "template_kwargs.update(self._harness_template_kwargs())" in inspect.getsource(cls._chat_completion)
        assert inspect.getsource(ta).count("chat_template_kwargs=self._harness_template_kwargs()") == 2
    finally:
        cls._harness_template_kwargs = orig
        if "_re_installed" in cls.__dict__:
            delattr(cls, "_re_installed")
        os.environ.clear()
        os.environ.update(saved_env)


# ---------------------------------------------------------------------------------------------- builder

def test_slug_dir_markers_counters():
    assert B.kernel_slug(VARIANTS) == SLUG and B.kernel_slug(tuple(reversed(VARIANTS))) == SLUG
    assert B.kernel_dir(VARIANTS).relative_to(B.ROOT).as_posix() == f"{DIRNAME}/notebook"
    m = B.kernel_markers(VARIANTS)
    new = ["REASONING_EFFORT medium installed", "REASONING_EFFORT_TEMPLATE supports=True",
           "REASONING_EFFORT_SERVER received=True"]
    assert set(B.kernel_markers(B.TURBO_LOSSLESS + ("tail",))) | set(new) == set(m)
    assert "SPEC_ACCEPT 0.5" not in " ".join(m)
    assert B.kernel_counters(VARIANTS) == ["level_memory_summary.json", "history_cache_summary.json",
                                           "reasoning_effort_summary.json", "timeout_fix_summary.json"]
    assert B.effort_value(VARIANTS) == "medium" and B.effort_value(B.TURBO_LOSSLESS) is None
    # other compositions and the existing slugs
    assert B.kernel_slug(("re-low",)) == "arc3-m2-lm-re-low"
    assert B.kernel_dir(("tail", "re-low")).relative_to(B.ROOT).as_posix() == \
        "kaggle_submission_m2_lm_tail_re_low/notebook"
    assert B.kernel_slug(B.TURBO + ("tail", "re-medium")) == "arc3-m2-turbo-tail-re-medium"
    assert B.kernel_slug(B.TURBO_LOSSLESS + ("tail", "audit", "re-medium")) == \
        "arc3-m2-turbo-lossless-tail-audit-re-medium"
    assert B.kernel_slug(B.TURBO_LOSSLESS + ("tail",)) == "arc3-m2-turbo-lossless-tail"
    assert B.kernel_slug(B.TURBO + ("tail",)) == "arc3-m2-turbo-tail"
    with pytest.raises(SystemExit):
        B.variant_names(("re-medium", "re-low"))
    with pytest.raises(SystemExit):
        B.variant_names(("re-xhigh",))
    with pytest.raises(SystemExit):
        B.effort_token("high")


def test_cli_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    assert B.main(["--turbo-lossless", "--prio-tail", "--reasoning-effort", "medium"]) == 0
    assert (tmp_path / DIRNAME / "notebook" / f"{SLUG}.ipynb").exists()
    with pytest.raises(SystemExit):
        B.main(["--turbo-lossless", "--reasoning-effort", "xhigh"])


def test_committed_kernel_is_current(tmp_path, monkeypatch):
    outs = []
    for i in range(2):
        monkeypatch.setattr(B, "ROOT", tmp_path / str(i))
        outs.append(B.build(VARIANTS).read_text(encoding="utf-8"))
    monkeypatch.undo()
    assert outs[0] == outs[1], "builder is not deterministic"
    assert outs[0] == NB.read_text(encoding="utf-8"), f"{SLUG} is stale: rerun the build"
    assert check_notebook(NB) == []
    meta = json.loads((NB.parent / "kernel-metadata.json").read_text())
    inc = json.loads((ROOT / "kaggle_submission_m2_level_memory" / "notebook" / "kernel-metadata.json").read_text())
    assert meta["id"] == f"calamitychasm/{SLUG}" and meta["code_file"] == NB.name
    for k in ("id", "title", "code_file"):
        meta.pop(k), inc.pop(k)
    assert meta.pop("docker_image_pinning_type") == "original"
    assert meta == inc


def test_kernel_differs_from_arm_b_only_in_header_effort_cell_and_dump():
    a, b = _cells(BASE_NB), _cells(NB)
    assert len(b) == len(a) + 1
    i = next(k for k, c in enumerate(b) if "_RE_SRC" in c)
    rest = b[:i] + b[i + 1:]
    diff = [k for k, (x, y) in enumerate(zip(a, rest)) if x != y]
    assert diff[0] == 0 and len(diff) == 2, diff
    assert rest[diff[1]].replace(B.re_dump("medium"), "") == a[diff[1]]
    lm_cell = next(k for k, c in enumerate(b) if "LEVEL_MEMORY.install(_lm_ta.ToolAgent)" in c)
    hc_cell = next(k for k, c in enumerate(b) if "HISTORY_CACHE installed" in c and "_HC_SRC" in c)
    tf_cell = next(k for k, c in enumerate(b) if "TIMEOUT_FIX installed" in c)
    run_cell = next(k for k, c in enumerate(b) if c.startswith("print('Starting benchmark...')"))
    assert lm_cell < hc_cell < i < tf_cell < run_cell
    assert B.RE_SRC in b[i] and "install(_re_ta.ToolAgent, 'medium')" in b[i]
    assert "REASONING_EFFORT_TEMPLATE supports=" in b[i] and "start_server_probe(SERVER_BASE_URL" in b[i]


def test_effort_cell_and_dump_execute_against_a_stub(monkeypatch, tmp_path, capsys):
    import types
    cells = _cells(NB)
    cell = next(c for c in cells if "_RE_SRC" in c)
    run = next(c for c in cells if c.startswith("print('Starting benchmark...')"))
    dump = run[run.index("try:   # [calamitychasm] reasoning effort counters"):]
    dump = dump[: dump.index("try:   # [calamitychasm] sandbox-timeout fix counters")]
    ToolAgent = fake_tool_agent()
    pkg, sub = types.ModuleType("inference"), types.ModuleType("inference.agent")
    pkg.__path__, sub.__path__ = [], []
    mod = types.ModuleType("inference.agent.tool_agent")
    mod.ToolAgent = ToolAgent
    for n, m in (("inference", pkg), ("inference.agent", sub), ("inference.agent.tool_agent", mod)):
        monkeypatch.setitem(sys.modules, n, m)
    monkeypatch.setitem(sys.modules, "reasoning_effort", None)
    monkeypatch.delenv("ARC3_REASONING_EFFORT_LADDER", raising=False)
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "chat_template.jinja").write_text(SYNTH, encoding="utf-8")
    srv = _FakeServer()
    try:
        ns = {"json": json, "__name__": "cell", "MODEL_DIR": str(tmp_path / "model"), "SERVER_BASE_URL": srv.url,
              "SERVED_MODEL_NAME": "flashnext", "WORKING_DIR": tmp_path}
        exec(compile(cell, "effort_cell", "exec"), ns)
        ns["_re_probe_thread"].join(30)
        exec(compile(dump, "dump", "exec"), ns)
    finally:
        srv.close()
    out = capsys.readouterr().out
    for marker in B.kernel_markers(("re-medium",))[-3:]:
        assert marker in out, marker
    assert ToolAgent()._harness_template_kwargs()["reasoning_effort"] == "medium"
    s = json.loads((tmp_path / "reasoning_effort_summary.json").read_text())
    assert s["level"] == "medium" and s["errors"] == 0 and s["server"]["received"] and s["template"]["supports"]
    assert len(srv.seen) == 2      # the dump does not re-probe after a successful boot-time probe


# ------------------------------------------------------------------------------- request-log digest

def test_request_log_digest_reports_effort_sent_and_the_long_tail(tmp_path):
    import m2_speed_report as rep

    def resp(completion, kw=None):
        rec = {"event": "response", "messages": [{"role": "user", "content": "go"}], "tools": [],
               "usage": {"prompt_tokens": 1000, "completion_tokens": completion}}
        if kw:
            rec["chat_template_kwargs"] = kw
        return json.dumps(rec)

    med = {"preserve_thinking": True, "reasoning_effort": "medium"}
    lines = [json.dumps({"event": "request", "messages": [], "tools": [], "chat_template_kwargs": med})]
    lines += [resp(1000, med) for _ in range(8)] + [resp(7000, med), resp(1000, {"preserve_thinking": True})]
    (tmp_path / "g_p0_requests.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    m = rep.request_log_metrics(tmp_path)
    assert m["reasoning_effort_sent"] == {"default": 1, "medium": 9}     # request records are not counted
    assert m["requests"] == 10 and m["completion_tokens_p90"] == 1000
    assert m["completion_token_share_long"] == round(7000 / 16000, 3)
