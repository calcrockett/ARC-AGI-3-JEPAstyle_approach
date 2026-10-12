"""Fixed per-python-call cost of the milestone-2 sandbox, on the REAL patched ToolAgent + sandbox.

    python scripts/bench_m2_sandbox_call.py [--ns 1,100,300,1000] [--reps 6] [--no-baseline]

Every python tool call starts a NEW sandbox process (`python -I -S -c <bootstrap>`) and sends it one
full history payload.  Per call this reports, with no action() in the snippet:

  (a) spawn      `python -I -S -c pass` wall; compile() of the sandbox bootstrap source (parent-side
                 proxy for the child's compile-on-start); the call at N=1 (everything but history)
  (b) per N      call wall; host thread CPU (the GIL part); sandbox process CPU (getrusage CHILDREN);
                 initial payload size; host share = time inside _send_json_line (json.dumps + write)
                 -- first call after the history was built (cold: frame views not cached yet) and the
                 median of the following warm calls (what a real run sees: each frame is formatted
                 once, when it is first sent)
  (c) segmentation  in-process segment_layer() on synthetic ARC-like boards with C components, and
                 in the sandbox as a snippet (`current_frame.segmentation`)

Needs the reconstructed milestone-2 src (see tests/m2_harness.py).  Numbers are machine-specific.
"""

from __future__ import annotations

import argparse
import os
import random
import resource
import statistics as st
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "kaggle_submission_milestone2_fork" / "history_cache"))

import m2_harness as h  # noqa: E402


def med(xs):
    return st.median(xs) if xs else float("nan")


def child_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


class Probe:
    """Wraps SB._send_json_line: counts characters and times json.dumps + write + flush."""

    def __init__(self, SB):
        self.SB, self.inner = SB, SB._send_json_line
        self.sends = []             # (seconds, chars) per send within the current call

    def __enter__(self):
        import weakref
        proxies = weakref.WeakKeyDictionary()

        class Counting:
            n = 0

            def __init__(s, real):
                s.real = real

            def write(s, t):
                Counting.n += len(t)
                return s.real.write(t)

            def flush(s):
                return s.real.flush()

        self.Counting = Counting

        def send(handle, payload):
            proxy = proxies.get(handle)
            if proxy is None:
                proxy = proxies[handle] = Counting(handle)
            before = Counting.n
            t0 = time.perf_counter()
            try:
                return self.inner(proxy, payload)
            finally:
                self.sends.append((time.perf_counter() - t0, Counting.n - before))

        self.SB._send_json_line = send
        return self

    def __exit__(self, *a):
        self.SB._send_json_line = self.inner


def one_call(agent, sp, probe, code):
    probe.sends.clear()
    t0, c0, k0 = time.perf_counter(), time.thread_time(), child_cpu()
    res = agent._run_python_tool(sp, {"code": code})
    wall, cpu, kid = time.perf_counter() - t0, time.thread_time() - c0, child_cpu() - k0
    assert '"error"' not in res.content, res.content[:300]
    first = probe.sends[0]                      # the initial payload (sandbox start)
    return dict(wall=wall, cpu=cpu, kid=kid, send_s=first[0], chars=first[1])


def measure_n(TA, RS, SB, S, hc, n, reps, cached, code):
    d = Path(tempfile.mkdtemp(prefix="sbbench_"))
    sp = d / "tool_runtime_state.json"
    if cached:
        hc.install()
    try:
        g = h.FakeGame(RS, S, sp, level_every=10**9, die_every=10**9)
        g.seed()
        g.advance(max(0, n - 1), write_each=False)
        agent = h.make_agent(TA, sp, g)
        with Probe(SB) as probe:
            cold = one_call(agent, sp, probe, code)
            warm = [one_call(agent, sp, probe, code) for _ in range(reps)]
    finally:
        if cached:
            hc.uninstall()
    w = {k: med([x[k] for x in warm]) for k in warm[0]}
    return cold, w


def spawn_times(SB, reps=30):
    def run(args, **kw):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
            ts.append(time.perf_counter() - t0)
        return med(ts)

    env = SB._sandbox_env()
    null = run([sys.executable, "-I", "-S", "-c", "pass"], env=env)
    # the bootstrap source, run to the point it blocks on stdin and then sees EOF (-> exits): this is
    # interpreter start + compile of the bootstrap + definitions, with no payload
    boot = run([sys.executable, "-I", "-S", "-c", SB._SANDBOX_BOOTSTRAP], env=env, stdin=subprocess.DEVNULL)
    comp = []
    for _ in range(reps):
        t0 = time.perf_counter()
        compile(SB._SANDBOX_BOOTSTRAP, "<bootstrap>", "exec")
        comp.append(time.perf_counter() - t0)
    return null, boot, med(comp), len(SB._SANDBOX_BOOTSTRAP)


def board(n_components, seed=0, size=64):
    """ARC-like board: background + n solid rectangles (4-connected components ~ n+1, some merge)."""
    r = random.Random(seed)
    g = [[0] * size for _ in range(size)]
    for _ in range(n_components):
        c = r.randrange(1, 16)
        y, x = r.randrange(size - 2), r.randrange(size - 2)
        for i in range(y, min(size, y + r.randrange(1, 6))):
            for j in range(x, min(size, x + r.randrange(1, 6))):
                g[i][j] = c
    return g


def seg_inprocess(S_mod, grids, reps=5):
    from inference.utils import segmentation as seg
    out = []
    for g in grids:
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            res = seg.segment_layer(g, "".join(h_chars()))
            ts.append(time.perf_counter() - t0)
        out.append((len(res["nodes"]), med(ts)))
    return out


def h_chars():
    from inference.utils.grid_utils import ARC_COLOR_CHARS
    return ARC_COLOR_CHARS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="1,100,300,1000")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()
    src = h.m2_src()
    if src is None:
        print("no milestone-2 src (set ARC3_M2_SRC or ARC3_M2_BASE)")
        return 1
    os.environ.update(h.notebook_env())
    TA, RS, SB, S = h.import_harness(src)
    import history_cache as hc

    print(f"host: nproc={os.cpu_count()}  python={sys.version.split()[0]}")
    null, boot, comp, blen = spawn_times(SB)
    print(f"\n(a) spawn   python -I -S -c pass            {null*1e3:7.1f} ms")
    print(f"            bootstrap run to EOF ({blen/1e3:.0f} kB src) {boot*1e3:7.1f} ms   (compile alone {comp*1e3:.1f} ms)")

    code = "result = len(history)\n"
    print(f"\n(b) per python call, snippet `{code.strip()}` (no action), unit = ms")
    print(f"{'N':>5} {'arm':>9} {'phase':>5} {'call wall':>10} {'host cpu':>9} {'sbx cpu':>8} {'send':>7} {'payload MB':>11}")
    for n in [int(x) for x in args.ns.split(",")]:
        for cached in ([False] if not args.no_baseline else []) + [True]:
            cold, warm = measure_n(TA, RS, SB, S, hc, n, args.reps, cached, code)
            for tag, r in (("cold", cold), ("warm", warm)):
                print(f"{n:>5} {'cache' if cached else 'baseline':>9} {tag:>5} {r['wall']*1e3:>10.1f} {r['cpu']*1e3:>9.1f} "
                      f"{r['kid']*1e3:>8.1f} {r['send_s']*1e3:>7.1f} {r['chars']/1e6:>11.3f}", flush=True)

    print("\n(c) segmentation (pure Python, runs in the sandbox on first `.segmentation` access per frame)")
    grids = [board(c, seed=c) for c in (5, 20, 60, 150, 400)]
    for ncomp, t in seg_inprocess(S, grids):
        print(f"    synthetic board, {ncomp:>4} components: segment_layer {t*1e3:8.1f} ms")
    # in the sandbox: trivial call vs a call that touches .segmentation (N=100 -> FakeGame board)
    d = Path(tempfile.mkdtemp(prefix="sbseg_"))
    sp = d / "tool_runtime_state.json"
    hc.install()
    try:
        g = h.FakeGame(RS, S, sp, level_every=10**9, die_every=10**9)
        g.seed()
        g.advance(99, write_each=False)
        agent = h.make_agent(TA, sp, g)
        with Probe(SB) as probe:
            one_call(agent, sp, probe, code)
            base = med([one_call(agent, sp, probe, code)["wall"] for _ in range(4)])
            segc = "s = current_frame.segmentation\nresult = len(s['nodes'])\n"
            ws = [one_call(agent, sp, probe, segc) for _ in range(4)]
            two = "a = current_frame.segmentation; b = history[-2].frame.segmentation\nresult = 1\n"
            w2 = [one_call(agent, sp, probe, two) for _ in range(4)]
    finally:
        hc.uninstall()
    print(f"    in sandbox (FakeGame N=100): trivial call {base*1e3:.0f} ms; +1 segmentation {med([x['wall'] for x in ws])*1e3:.0f} ms; "
          f"+2 segmentations {med([x['wall'] for x in w2])*1e3:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
