"""Token-clock replay of dfranzen's milestone-2 priority gate, for scheduler variants (CPU only).

Write-up: experiments/stage7_milestone2_improvements.md section 4. The priority function is NOT re-implemented:
priority_value() is executed from the harness patch in the upstream notebook (the file the rerun runs), with the
incumbent's settings (tail lookup "remaining", A-only score normalization, fade over the final 20%).

Gate mechanics (tool_agent.py _PriorityGate / _maybe_handover, read from the patch):
  * 110 games start together; one common deadline (532 min); S slots; aggregate rate S x tok_per_slot.
  * never-started games queue at 2,000,000 - dispatch index, above every started game (fresh first).
  * a slot changes hands only at a context trim (first after N(62k, 6k) generated tokens, then every
    N(37k, 7.5k); JustAdev742's measured quanta on this config) or when a game wins all its levels.
  * at a trim the game re-queues with P(level, actions on level, tokens on level, N); every waiter is
    re-scored at each pump (only the fade changes); ties go to the earliest arrival.
Game model (per draw, shared by all variants = common random numbers): each of the 110 games is a public game's
level list (human action counts below); a level costs median x hard x game factor x level noise tokens (level 1
x0.5), is unsolvable with probability p_unsolv, and is solved with h x ratio actions (accrued linearly in its
tokens). Scored with RHAE (w_l = l, S = min(1.15, h/a)^2, E = min(completion, efficiency)), x100.
Calibration: hidden-like worlds score ~28-32 (our LB 30.5); draw-to-draw sd ~2.5-3.9 (copies: 3.93).

    python scripts/sim_m2_priority_gate.py --draws 150 --slots 10,14
    python scripts/sim_m2_priority_gate.py --draws 120 --slots 10:70,14:47 --variants tail,dprime,dprime+final5

D' (section 8 of the write-up): shiiin9's slot priority A*M*C + B*phi, executed from the vendored module
kaggle_submission_milestone2_fork/dprime/ours_form_priority.py (verbatim from their notebook), with its two other
changes: fade over the final 40%, and never-started games priced by the formula (l = 1, a = t = 0) instead of
queued first. Its pace M is the game's mean generated tokens per cleared level (FormPace).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
import types
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "kaggle_submission_milestone2_fork" / "upstream" / "arc-agi-3-milestone-2-solution.ipynb"
DPRIME_SRC = ROOT / "kaggle_submission_milestone2_fork" / "dprime" / "ours_form_priority.py"

# base_actions_per_level of the 25 public games (benchmark.json of a milestone-2 run)
PUBLIC = {"ar25": [32, 50, 75, 37, 89, 159, 233, 73], "bp35": [21, 48, 44, 38, 33, 87, 86, 131, 163],
          "cd82": [55, 8, 41, 21, 23, 23], "cn04": [29, 54, 85, 300, 208, 113], "dc22": [59, 102, 67, 98, 324, 578],
          "ft09": [43, 12, 23, 28, 65, 37], "g50t": [78, 175, 179, 230, 96, 54, 67],
          "ka59": [28, 109, 51, 51, 33, 132, 326], "lf52": [32, 81, 60, 71, 205, 148, 244, 109, 164, 225],
          "lp85": [17, 38, 31, 16, 41, 60, 26, 159], "ls20": [22, 123, 73, 84, 96, 192, 186],
          "m0r0": [30, 111, 203, 26, 500, 237], "r11l": [22, 33, 51, 26, 52, 49],
          "re86": [26, 42, 86, 108, 189, 139, 424, 241], "s5i5": [20, 89, 106, 54, 162, 38, 86, 83],
          "sb26": [18, 28, 18, 19, 31, 23, 58, 18], "sc25": [36, 6, 32, 83, 143, 50],
          "sk48": [61, 177, 101, 103, 230, 181, 125, 92], "sp80": [39, 58, 25, 148, 96, 152],
          "su15": [22, 42, 26, 115, 36, 31, 8, 40, 41], "tn36": [32, 72, 26, 40, 30, 55, 62],
          "tr87": [54, 58, 40, 45, 71, 146], "tu93": [19, 16, 34, 42, 123, 80, 14, 23, 111],
          "vc33": [7, 18, 44, 61, 131, 34, 152], "wa30": [71, 119, 183, 98, 368, 68, 79, 442, 415]}
GAMES = sorted(PUBLIC.items())

WORLDS = {   # name -> World kwargs; all but public_like are calibrated to score ~28-32 (the hidden LB level)
    "public_like": dict(hard=1.0, p_unsolv=0.07),
    "tok2.5": dict(hard=2.5, p_unsolv=0.07),
    "tok2_u.12": dict(hard=2.0, p_unsolv=0.12),
    "unsolv.20": dict(hard=1.0, p_unsolv=0.20),
    "depth1.15": dict(hard=1.6, p_unsolv=0.07, depth_growth=1.15),
    "finalU3x": dict(hard=2.0, p_unsolv=0.10, final_unsolv_mult=3.0),
    "siggame0.3": dict(hard=2.1, p_unsolv=0.07, sig_game=0.3, sig_level=0.95),
    "coupled0.7": dict(hard=2.5, p_unsolv=0.07, couple=0.7, eff_med=1.0),
    "eff1.5": dict(hard=1.5, p_unsolv=0.07, eff_med=1.5, eff_sig=0.6),
    "coupled0.7_eff1.5": dict(hard=2.2, p_unsolv=0.07, couple=0.7, eff_med=1.5, eff_sig=0.6),
}
VARIANTS = {   # name -> run() kwargs; "base" is the incumbent (h 25, last-level B 0)
    "base": {},
    "final5": dict(final_b=5.0),
    "final8": dict(final_b=8.0),
    "h40": dict(human=40.0),
    "h60": dict(human=60.0),
    "h100": dict(human=100.0),
    "h60+final5 (shipped: tail)": dict(human=60.0, final_b=5.0),
    "fade0.4": dict(fade=0.4),
    "lookup80k": dict(lookup=True),
    # D' (shiiin9): its priority, fade 0.4, fresh games priced by the formula; + its parts and + our tail idea
    "dprime": dict(prio="dprime"),
    "dprime (fresh first)": dict(prio="dprime", fresh_first=True),
    "dprime+final5": dict(prio="dprime", final_b=5.0),
    "dprime+final10": dict(prio="dprime", final_b=10.0),
}
DEFAULT_VARIANTS = tuple(v for v in VARIANTS if not v.startswith("dprime"))
SLOT_RATE = {10: 70.0, 14: 55.0}   # generated tok/s per held slot (~700 aggregate at 10; ~770 at 14)


def load_priority_scheduler() -> types.ModuleType:
    """priority_scheduler.py exactly as the upstream harness patch creates it."""
    nb = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    cell = next("".join(c["source"]) for c in nb["cells"]
                if "b/ARC3-Inference/inference/agent/priority_scheduler.py" in "".join(c["source"]))
    lines = cell[cell.index("+++ b/ARC3-Inference/inference/agent/priority_scheduler.py"):].split("\n")[2:]
    body = []
    for line in lines:
        if line.startswith("diff --git"):
            break
        body.append(line[1:])
    mod = types.ModuleType("priority_scheduler_upstream")
    sys.modules[mod.__name__] = mod
    exec(compile("\n".join(body), "priority_scheduler.py", "exec"), mod.__dict__)
    return mod


PS = load_priority_scheduler()
BASE_80K = dict(PS.TAIL_LOOKUP_80K)


def load_dprime() -> types.ModuleType:
    """shiiin9's D' module, verbatim (the file their notebook exec's as `ours_form_priority`)."""
    mod = types.ModuleType("ours_form_priority")
    sys.modules[mod.__name__] = mod
    exec(compile(DPRIME_SRC.read_text(encoding="utf-8"), "ours_form_priority.py", "exec"), mod.__dict__)
    return mod


DP = load_dprime()
DP_BASE = dict(DP.D_PRIME)


class World:
    def __init__(self, rng: random.Random, n_games=110, hard=1.0, p_unsolv=0.07, sig_game=0.6, sig_level=0.8,
                 median=22000.0, lvl1=0.5, eff_med=0.8, eff_sig=0.5, depth_growth=1.0, final_unsolv_mult=1.0,
                 couple=0.0):
        self.games = []
        for _ in range(n_games):
            _, human = GAMES[rng.randrange(len(GAMES))]
            dg = math.exp(rng.gauss(0, sig_game))
            levels = []
            for i, h in enumerate(human):
                c = median * hard * dg * math.exp(rng.gauss(0, sig_level)) * (lvl1 if i == 0 else 1.0)
                c *= depth_growth ** i
                pu = p_unsolv * (final_unsolv_mult if i == len(human) - 1 else 1.0)
                unsolv = rng.random() < pu
                ratio = math.exp(rng.gauss(math.log(eff_med), eff_sig)) * (c / (median * hard)) ** couple
                levels.append(dict(h=h, cost=c, unsolv=unsolv, rate=max(1.0, h * ratio) / c))
            self.games.append(levels)
        self.order = list(range(n_games))
        rng.shuffle(self.order)
        self.trim_seed = rng.random()


def score_game(levels, solved_actions) -> float:
    wsum = len(levels) * (len(levels) + 1) / 2
    comp = sum(i + 1 for i in range(len(solved_actions))) / wsum
    eff = sum((i + 1) * min(1.15, levels[i]["h"] / max(1.0, a)) ** 2 for i, a in enumerate(solved_actions)) / wsum
    return min(comp, eff)


def run(world: World, *, slots=10, minutes=532.0, tok_per_slot_s=70.0, fade=None, final_b=0.0,
        lookup="remaining", human=25.0, prio="franzen", fresh_first=None) -> dict:
    """prio "franzen": upstream priority_value (fade default 0.2, fresh games first); "dprime": D' (fade default 0.4,
    fresh games priced by the formula unless fresh_first=True; final_b is D''s B with 0 levels remaining)."""
    PS.TAIL_LOOKUP_REMAINING = {n: (8.0,) * (n - 3) + (7.0, 5.0, final_b) for n in range(6, 11)}
    PS.TAIL_LOOKUP_80K = {n: v[:-1] + (final_b / 0.8,) for n, v in BASE_80K.items()}
    dprime = prio == "dprime"
    fade = (0.4 if dprime else 0.2) if fade is None else fade
    fresh_first = (not dprime) if fresh_first is None else fresh_first
    rng = random.Random(world.trim_seed)
    T, G = minutes * 60.0, len(world.games)
    window = fade * T
    st_ = [dict(level=0, a=0.0, t=0.0, solved=[], next_trim=rng.gauss(62000, 6000), parked_last=0.0,
                active_last=0.0, pace_tok=0.0, pace_n=0, played=0.0) for _ in range(G)]
    trims = [random.Random(rng.random()) for _ in range(G)]
    if dprime:
        dp_params = dict(DP_BASE)

    def prio_value(gi, now):
        s = st_[gi]
        frac = max(0.0, T - now) / window
        if dprime:   # D''s FormPace: mean tokens per cleared level, 0 before the first clear
            pace = s["pace_tok"] / s["pace_n"] if s["pace_n"] else 0.0
            snap = PS.PrioritySnapshot(s["level"] + 1, int(s["a"]), s["t"], pace, len(world.games[gi]))
            phi = min(1.0, max(0.0, frac))
            u, v = DP.d_parts(snap.level, snap.actions, snap.tokens, pace, snap.total_levels, dp_params)
            if snap.level >= snap.total_levels:      # 0 levels remaining: D''s B is 0; the variant's final_b
                v = final_b
            return max(1, int((u + v * phi) * DP.SCALE))
        snap = PS.PrioritySnapshot(s["level"] + 1, int(s["a"]), s["t"], 1.0, len(world.games[gi]))
        return PS.priority_value(snap, tail_fraction=frac, normalize_score=True,
                                 tail_lookup=lookup, human_actions=human)

    fresh = list(world.order)
    active, fresh, waiting = fresh[:slots], fresh[slots:], []
    if not fresh_first:     # D': never-started games wait in the same queue, priced like everyone else
        waiting, fresh = fresh, []

    def admit(now):
        while len(active) < slots and (fresh or waiting):
            if fresh:
                active.append(fresh.pop(0))
                continue
            best = max(range(len(waiting)), key=lambda k: (prio_value(waiting[k], now), -k))
            active.append(waiting.pop(best))

    now = 0.0
    while now < T and active:
        best = None
        for gi in active:
            s, lv = st_[gi], world.games[gi][st_[gi]["level"]]
            to_solve = math.inf if lv["unsolv"] else max(0.0, lv["cost"] - s["t"])
            ev = (to_solve, gi, "solve") if to_solve <= s["next_trim"] else (max(0.0, s["next_trim"]), gi, "trim")
            if best is None or ev[0] < best[0]:
                best = ev
        dt = min(best[0] / tok_per_slot_s, T - now)
        for gi in active:
            s = st_[gi]
            s["played"] += dt
            s["t"] += dt * tok_per_slot_s
            s["a"] += dt * tok_per_slot_s * world.games[gi][s["level"]]["rate"]
            s["next_trim"] -= dt * tok_per_slot_s
            if s["level"] == len(world.games[gi]) - 1:
                s["active_last"] += dt
        for gi in waiting:
            if st_[gi]["level"] == len(world.games[gi]) - 1:
                st_[gi]["parked_last"] += dt
        now += dt
        if now >= T:
            break
        _, gi, kind = best
        s = st_[gi]
        if kind == "solve":
            s["solved"].append(s["a"])
            s["pace_tok"] += s["t"]
            s["pace_n"] += 1
            s["level"] += 1
            s["a"] = s["t"] = 0.0
            if s["level"] >= len(world.games[gi]):
                active.remove(gi)
                admit(now)
        else:
            s["next_trim"] = trims[gi].gauss(37000, 7500)
            active.remove(gi)
            waiting.append(gi)
            admit(now)
    parked = sum(s["parked_last"] for s in st_)
    return dict(score=100.0 * sum(score_game(world.games[g], st_[g]["solved"]) for g in range(G)) / G,
                levels=sum(len(s["solved"]) for s in st_),
                starved=sum(1 for s in st_ if s["played"] <= 0.0),
                parked_last_share=parked / max(1e-9, parked + sum(s["active_last"] for s in st_)))


def job(args):
    wname, seed, slots, rate, variants, n_games, minutes = args
    world = World(random.Random(f"{wname}/{seed}"), n_games=n_games, **WORLDS[wname])
    return wname, slots, {v: run(world, slots=slots, tok_per_slot_s=rate, minutes=minutes, **VARIANTS[v])
                          for v in ("base",) + tuple(x for x in variants if x != "base")}


def _parse_slots(spec: str) -> list[tuple[int, float]]:
    """"10,14" (default per-slot rates) or "10:70,14:47" (slots:generated tok/s per held slot)."""
    out = []
    for item in spec.split(","):
        n, _, r = item.partition(":")
        out.append((int(n), float(r) if r else SLOT_RATE[int(n)]))
    return out


ALIASES = {"tail": "h60+final5 (shipped: tail)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--slots", default="10", help='"10,14" or "10:70,14:47" (slots:tok/s per slot)')
    ap.add_argument("--worlds", default=",".join(WORLDS))
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS[1:]),
                    help="comma list of VARIANTS keys (alias: tail); base is always run")
    ap.add_argument("--vs", default="", help="also report each variant against this one (e.g. tail)")
    ap.add_argument("--games", type=int, default=110)
    ap.add_argument("--minutes", type=float, default=532.0)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(argv)
    slots = _parse_slots(a.slots)
    worlds = a.worlds.split(",")
    variants = tuple(ALIASES.get(v, v) for v in a.variants.split(",") if v)
    vs = ALIASES.get(a.vs, a.vs)
    for v in variants + ((vs,) if vs else ()):
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; known: {list(VARIANTS)}")
    if vs and vs not in variants:
        variants += (vs,)
    tasks = [(w, s, sl, rate, variants, a.games, a.minutes) for sl, rate in slots for w in worlds
             for s in range(a.draws)]
    with Pool(a.procs) as p:
        res = p.map(job, tasks, chunksize=4)
    boot = random.Random(0)
    for sl, rate in slots:
        for w in worlds:
            rows = [o for wn, s2, o in res if wn == w and s2 == sl]
            base = [r["base"]["score"] for r in rows]
            print(f"\n== {w}  slots={sl} x {rate:g} tok/s  draws={len(rows)}  base {st.mean(base):.2f} "
                  f"(sd {st.stdev(base):.2f})  last-level parked share "
                  f"{st.mean(r['base']['parked_last_share'] for r in rows):.2f}  "
                  f"starved {st.mean(r['base']['starved'] for r in rows):.2f}")
            for ref in ("base",) + ((vs,) if vs else ()):
                for v in variants:
                    if v == ref:
                        continue
                    d = [r[v]["score"] - r[ref]["score"] for r in rows]
                    bs = sorted(st.mean(boot.choices(d, k=len(d))) for _ in range(2000))
                    tag = v if ref == "base" else f"{v} - {ref}"
                    print(f"  {tag:40s} {st.mean(d):+.3f} [{bs[100]:+.3f}, {bs[1900]:+.3f}]  "
                          f"{100 * st.mean(d) / st.mean(base):+.2f}%  P(draw>0) {sum(x > 0 for x in d) / len(d):.2f}  "
                          f"levels {st.mean(r[v]['levels'] - r[ref]['levels'] for r in rows):+.2f}  "
                          f"parked share {st.mean(r[v]['parked_last_share'] for r in rows):.2f}  "
                          f"starved {st.mean(r[v]['starved'] for r in rows):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
