"""Haiku relay smoke test for the momentum time policy.

Runs the REAL anim solver locally -- the unpickled HarnessSolver, its semaphore
queue, _HarnessGameSession.play(), the real ToolAgent (system prompt, board
rendering, history, trimming to the served model's 32,768-token window), the
real Python-tool sandbox and the installed MomentumTimePolicy -- with exactly
one substitution: ToolAgent._chat_completion. Each LLM call is written to
RELAY/pending/<id>.md (the exact messages + tool schemas the served model
would receive) and answered by a Claude Haiku subagent, which writes
RELAY/done/<id>.json. Haiku carries its own system prompt, so it is a rough
stand-in for Qwen, not a replica; this tests plumbing, not score.

Time is simulated: the policy's clock reads RELAY/clock.txt, advanced by the
relay operator once per serviced batch, so per-game time is deterministic
regardless of how long a subagent takes. Real-time yielding is disabled
(LOCAL_ANALYZER_YIELD_SECONDS=0) for the same reason; tool steps per turn are
capped instead.

    python scripts/momentum/haiku_relay_smoke.py --anim <anim dir> --games <env dir> --relay <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--anim", required=True, help="anim bundle dir (has src/ and benchmark_initial.pkl)")
ap.add_argument("--games", required=True, help="environment_files dir")
ap.add_argument("--relay", required=True)
ap.add_argument("--game-ids", default="ft09-0d8bbf25,vc33-5430563c,tu93-0768757b,sk48-d8078629")
ap.add_argument("--concurrency", type=int, default=2)
ap.add_argument("--tool-steps", type=int, default=4)
ap.add_argument("--base", type=float, default=900.0)
ap.add_argument("--momentum", type=float, default=450.0)
ap.add_argument("--stall", type=float, default=630.0)
ap.add_argument("--hard-mult", type=float, default=1.5)
args = ap.parse_args()

ANIM = Path(args.anim)
RELAY = Path(args.relay)
(RELAY / "pending").mkdir(parents=True, exist_ok=True)
(RELAY / "done").mkdir(parents=True, exist_ok=True)
CLOCK_FILE = RELAY / "clock.txt"
CLOCK_FILE.write_text("0")
OUT = RELAY / "run"
OUT.mkdir(exist_ok=True)

# The served model's harness settings (taaf_setup_env.json of a real anim run),
# set BEFORE importing inference: tool_agent reads them at import time.
os.environ.update({
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_SEED": "20260825",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    # relay-specific
    "LOCAL_ANALYZER_TOOL_STEPS": str(args.tool_steps),
    "LOCAL_ANALYZER_YIELD_SECONDS": "0",
    "MULTIMODAL_CONTEXT": "none",          # a subagent reading a file cannot see an inline PNG
    "LOCAL_ANALYZER_PROVIDER": "anthropic",
    "LOCAL_ANALYZER_BASE_URL": "http://relay.invalid/v1",
    "LOCAL_ANALYZER_MODEL_ID": "haiku-relay",
    "RECORDINGS_DIR": str(OUT / "server_recording"),
})
sys.path[:0] = [str(ANIM / "src" / "ARC3-Inference"), str(ANIM / "src" / "tufa-arc-agi-framework" / "src")]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaggle_submission_duck_nvfp4_anim_momentum"))

import arc_agi  # noqa: E402
import taaf.game_api  # noqa: E402
import inference.agent.python_tool_sandbox as sandbox  # noqa: E402
import inference.agent.tool_agent as tool_agent  # noqa: E402
import inference.framework.solver as solver_mod  # noqa: E402
import momentum_time  # noqa: E402

assert tool_agent._LOCAL_ANALYZER_CONTEXT_WINDOW == 32768

# Windows: the sandbox's kill path uses os.killpg (POSIX only).
if not hasattr(os, "killpg"):
    sandbox._kill_process_group = lambda process: process.kill()

# ---- the relay: the one substitution ----------------------------------------
_lock = threading.Lock()
_counter = {"n": 0}
STATS = {"calls": 0, "tool_calls": 0, "text_replies": 0, "bad_replies": 0}


def _render(messages, tools) -> str:
    out = ["# LLM call inside the ARC-AGI-3 game harness", "",
           "Below is the EXACT input the harness sends to its model: the system prompt, the",
           "conversation so far, and the tools it may call. You are that model. Produce the",
           "model's next assistant turn.", ""]
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                else:
                    parts.append(f"[{p.get('type')} omitted]")
            content = "\n".join(parts)
        out.append(f"===== {role.upper()}" + (f" (tool_call_id={m.get('tool_call_id')})" if m.get("tool_call_id") else ""))
        if content:
            out.append(str(content))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            out.append(f"[assistant called tool {fn.get('name')} with {fn.get('arguments')}]")
        out.append("")
    out.append("===== TOOLS YOU MAY CALL (OpenAI function schemas)")
    out.append(json.dumps([t.get("function", t) for t in (tools or [])], indent=1))
    out.append("")
    out.append("===== HOW TO ANSWER")
    out.append('Write ONE JSON object: {"content": "<optional short reasoning>", '
               '"tool_calls": [{"name": "<tool name>", "arguments": {...}}]}.')
    out.append("Call at most one tool. To act in the game, call the action tool; to inspect the grid, "
               "call the python tool. Use \"tool_calls\": [] only if you want to reply with text alone.")
    return "\n".join(out)


def _lenient_json(text: str):
    """Haiku's hand-written JSON has formatting slips the served model's
    structured tool-call output does not (a `\\'` escape; a missing closing
    brace). Repair only those, then give up."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    for candidate in (text, text.replace("\\'", "'")):
        for extra in ("", "}", "}}"):
            try:
                return json.loads(candidate + extra, strict=False)
            except Exception:
                continue
    return None


def _relay_chat_completion(self, messages, *, tools, request_timeout_seconds=None):
    with _lock:
        _counter["n"] += 1
        rid = f"{_counter['n']:04d}_{threading.current_thread().name.replace(' ', '_')}"
        STATS["calls"] += 1
    (RELAY / "pending" / f"{rid}.md").write_text(_render(messages, tools), encoding="utf-8")
    done = RELAY / "done" / f"{rid}.json"
    while not done.exists():
        time.sleep(1.0)
    time.sleep(0.2)
    reply = _lenient_json(done.read_text(encoding="utf-8"))
    if reply is None:
        reply = {"content": "(unparseable reply)", "tool_calls": []}
        STATS["bad_replies"] += 1
    (RELAY / "pending" / f"{rid}.md").rename(RELAY / "pending" / f"{rid}.served")
    calls = []
    for i, tc in enumerate(reply.get("tool_calls") or []):
        calls.append({"id": f"call_{rid}_{i}", "type": "function",
                      "function": {"name": tc.get("name"), "arguments": json.dumps(tc.get("arguments") or {})}})
    msg = {"role": "assistant", "content": reply.get("content") or ""}
    if calls:
        msg["tool_calls"] = calls
        STATS["tool_calls"] += 1
    else:
        STATS["text_replies"] += 1
    return tool_agent._ChatCompletionResult(message=msg, finish_reason="tool_calls" if calls else "stop",
                                            usage={"prompt_tokens": 0, "completion_tokens": 0})


tool_agent.ToolAgent._chat_completion = _relay_chat_completion


# ---- the policy, on the simulated clock ---------------------------------------
def sim_clock() -> float:
    try:
        return float(CLOCK_FILE.read_text() or 0)
    except Exception:
        return 0.0


class RelayPolicy(momentum_time.MomentumTimePolicy):
    """session.started_at is real monotonic time; on the simulated clock a
    game starts when the policy first sees it (play() consults should_stop()
    before its first turn)."""

    def _state(self, session):
        key = id(session)
        with self._lock:
            st = self._games.get(key)
            if st is None:
                now = self.clock()
                st = momentum_time._GameClock(now, momentum_time._levels(session), now)
                self._games[key] = st
            return st


POLICY = RelayPolicy(base_s=args.base, momentum_s=args.momentum, stall_s=args.stall,
                     hard_mult=args.hard_mult, clock=sim_clock)
momentum_time.install(solver_mod._HarnessGameSession, POLICY)

# ---- the benchmark: the pickled anim one, local games, no vLLM ------------------
class _PortableUnpickler(pickle.Unpickler):
    """The benchmark was pickled on Linux; its PosixPath objects cannot be
    instantiated on Windows. Load them as this platform's Path."""

    def find_class(self, module, name):
        if module.startswith("pathlib") and name in ("PosixPath", "PurePosixPath"):
            return Path
        return super().find_class(module, name)


with open(ANIM / "benchmark_initial.pkl", "rb") as fh:
    bm = _PortableUnpickler(fh).load()
assert bm.label == "anim-20260807-anim", bm.label
bm.job_dir = OUT
spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=args.games)
wanted = [g.strip() for g in args.game_ids.split(",")]
bm.games = [taaf.game_api.GameAPI(env_name=g, arcade_spec=spec) for g in wanted]
bm.n_passes = 1
bm.game_weights = None
s = bm.solver
s.max_runtime_s_per_game = args.base
s.analyzer_timeout = 300.0
s.concurrency = args.concurrency
s.max_actions_per_game = None
s.save_request_logs = False
s.start_local_server = False
s.model = "local"
print(f"RELAY_SMOKE games={wanted} concurrency={s.concurrency} policy={POLICY.summary()['params']}", flush=True)


async def main():
    await bm.run(soft_end_time=datetime.now() + timedelta(hours=6), runtime_environment=None,
                 minimal_diagnostics=True)


t0 = time.time()
error = None
try:
    asyncio.run(main())
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    error = repr(exc)

runs = list(bm.game_runs)
rows = [{"game": r.game_id, "state": r.state, "levels": r.levels_completed, "actions": len(r.history),
         "final_score": r.final_score, "note": r.solver_note} for r in runs]
report = {"error": error, "wall_s": round(time.time() - t0, 1), "sim_clock_end": sim_clock(),
          "runs": rows, "policy": POLICY.summary(), "relay": STATS}
(RELAY / "report.json").write_text(json.dumps(report, indent=2, default=str))
print("RELAY_SMOKE_REPORT", json.dumps(report, default=str), flush=True)
(RELAY / "FINISHED").write_text("1")
