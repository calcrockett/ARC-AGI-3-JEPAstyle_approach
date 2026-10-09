"""Per-action / per-call cost of the milestone-2 history plumbing, with and without the
history cache, on the REAL patched ToolAgent + sandbox (synthetic 64x64 game, no model).

    python scripts/bench_m2_history_cache.py [--ns 100,300,1000] [--actions 12] [--no-baseline]

Per history length N it plays one python tool call that issues K back-to-back action() calls
and reports:
  wall/action   median interval between consecutive step_env calls (host + sandbox round trip)
  host cpu/act  host thread CPU per action, from the difference between a K-action call and a
                2-action call (removes the per-call constant: sandbox start, first payload)
  call wall     wall time of the K-action call (includes the sandbox start + initial payload)
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


def measure(TA, RS, S, hc, n, k, cached):
    d = Path(tempfile.mkdtemp(prefix="hcbench_"))
    out = {}
    for kk in (2, k):
        sp = d / f"k{kk}_tool_runtime_state.json"
        if cached:
            hc.install()
        try:
            g = h.FakeGame(RS, S, sp, level_every=10**9, die_every=10**9)
            g.seed()
            g.advance(max(0, n - 1))
            agent = h.make_agent(TA, sp, g)
            code = f'for _a in ["ACTION5"] * {kk}:\n    action(_a)\n'
            t0, c0 = time.perf_counter(), time.thread_time()
            agent._run_python_tool(sp, {"code": code})
            wall, cpu = time.perf_counter() - t0, time.thread_time() - c0
            iv = sorted(b - a for a, b in zip(g.calls, g.calls[1:]))
            out[kk] = dict(wall=wall, cpu=cpu, iv=iv[len(iv) // 2] if iv else float("nan"))
        finally:
            if cached:
                hc.uninstall()
    return dict(wall_per_action=out[k]["iv"], host_cpu_per_action=(out[k]["cpu"] - out[2]["cpu"]) / (k - 2),
                call_wall=out[k]["wall"])


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
    print(f"{'N':>6} {'arm':>9} {'wall/action':>12} {'host cpu/act':>13} {'call wall':>10}")
    for n in [int(x) for x in args.ns.split(",")]:
        for cached in ([False] if not args.no_baseline else []) + [True]:
            r = measure(TA, RS, S, hc, n, args.actions, cached)
            print(f"{n:>6} {'cache' if cached else 'baseline':>9} {r['wall_per_action']:>11.3f}s "
                  f"{r['host_cpu_per_action']:>12.4f}s {r['call_wall']:>9.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
