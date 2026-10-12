"""Per-action / per-call cost of the milestone-2 history plumbing, with and without the
history cache, on the REAL patched ToolAgent + sandbox (synthetic 64x64 game, no model).

    python scripts/bench_m2_history_cache.py [--ns 100,300,1000] [--actions 12] [--no-baseline]

Per history length N it plays one python tool call that issues K back-to-back action() calls
and reports:
  wall/action   median interval between consecutive step_env calls (host + sandbox round trip)
  host cpu/act  host thread CPU per action, from the difference between a K-action call and a
                2-action call (removes the per-call constant: sandbox start, first payload)
  call wall     wall time of the K-action call (includes the sandbox start + initial payload)
  MB/action     bytes written to the sandbox's stdin per action() reply (K-call minus 2-call)
The actions cycle ACTION1..4, each of which changes the board (no frame is a repeat).
Needs the reconstructed milestone-2 src (see tests/m2_harness.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "kaggle_submission_milestone2_fork" / "history_cache"))

import m2_harness as h  # noqa: E402


class _Counting:
    """Stand-in for the sandbox's stdin that counts the characters written to it."""
    n = 0

    def __init__(self, real):
        self.real = real

    def write(self, s):
        _Counting.n += len(s)
        return self.real.write(s)

    def flush(self):
        return self.real.flush()


def count_payload_bytes(SB):
    """Wrap SB._send_json_line (outermost, after any install) to count payload characters. One
    proxy per real handle, so the history cache still recognises the pipe between sends."""
    import weakref
    inner, proxies = SB._send_json_line, weakref.WeakKeyDictionary()

    def send(handle, payload):
        proxy = proxies.get(handle)
        if proxy is None:
            proxy = proxies[handle] = _Counting(handle)
        return inner(proxy, payload)

    SB._send_json_line = send
    return inner


def measure(TA, RS, SB, S, hc, n, k, cached):
    d = Path(tempfile.mkdtemp(prefix="hcbench_"))
    out = {}
    for kk in (2, k):
        sp = d / f"k{kk}_tool_runtime_state.json"
        if cached:
            hc.install()
        try:
            g = h.FakeGame(RS, S, sp, level_every=10**9, die_every=10**9)
            g.seed()
            g.advance(max(0, n - 1), write_each=False)
            agent = h.make_agent(TA, sp, g)
            code = (f'for _i in range({kk}):\n'
                    '    action(["ACTION1", "ACTION2", "ACTION3", "ACTION4"][_i % 4])\n')
            inner = count_payload_bytes(SB)
            _Counting.n = 0
            try:
                t0, c0 = time.perf_counter(), time.thread_time()
                res = agent._run_python_tool(sp, {"code": code})
                wall, cpu = time.perf_counter() - t0, time.thread_time() - c0
            finally:
                SB._send_json_line = inner
            assert '"error"' not in res.content, res.content[:400]
            iv = sorted(b - a for a, b in zip(g.calls, g.calls[1:]))
            out[kk] = dict(wall=wall, cpu=cpu, iv=iv[len(iv) // 2] if iv else float("nan"), chars=_Counting.n)
        finally:
            if cached:
                hc.uninstall()
    return dict(wall_per_action=out[k]["iv"], host_cpu_per_action=(out[k]["cpu"] - out[2]["cpu"]) / (k - 2),
                call_wall=out[k]["wall"], mb_per_action=(out[k]["chars"] - out[2]["chars"]) / (k - 2) / 1e6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="100,300,1000")
    ap.add_argument("--actions", type=int, default=12)
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()
    src = h.m2_src()
    if src is None:
        print("no milestone-2 src (set ARC3_M2_SRC or ARC3_M2_BASE)")
        return 1
    os.environ.update(h.notebook_env())
    TA, RS, SB, S = h.import_harness(src)
    import history_cache as hc
    print(f"{'N':>6} {'arm':>9} {'wall/action':>12} {'host cpu/act':>13} {'call wall':>10} {'MB/action':>10}")
    for n in [int(x) for x in args.ns.split(",")]:
        for cached in ([False] if not args.no_baseline else []) + [True]:
            r = measure(TA, RS, SB, S, hc, n, args.actions, cached)
            print(f"{n:>6} {'cache' if cached else 'baseline':>9} {r['wall_per_action']:>11.3f}s "
                  f"{r['host_cpu_per_action']:>12.4f}s {r['call_wall']:>9.2f}s {r['mb_per_action']:>10.3f}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
