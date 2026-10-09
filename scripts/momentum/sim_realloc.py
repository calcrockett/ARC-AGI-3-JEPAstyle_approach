"""Simulate time reallocation on a 110-game hidden run from real trajectories.
Stall-stops are scored exactly (the observed trajectory's later level-ups are
forfeited); extension past the observed clock uses the measured gap hazard."""
import heapq, json, os, random, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
games = json.load(open(os.path.join(ROOT, "experiments", "stage7_momentum_time_level_timing.json")))
traj = [(sorted(g["level_up_turns"]), g["final_turn"], g["number_of_levels"]) for g in games if g["final_turn"] > 0]
HAZ = [(2, .044), (5, .096), (9, .072), (14, .036), (10**9, .033)]
def hz(gap):
    for hi, h in HAZ:
        if gap <= hi: return h
BASE, BUDGET, SLOTS, N = 7920.0, 32400 - 600 - 1200, 28, 110
def pts(levels, n): return 100.0 * sum(range(1, min(levels, n) + 1)) / (n * (n + 1) / 2)

def run(sample, stall_min, extend, rng):
    stall_s = None if stall_min is None else stall_min * 60
    queue = list(range(len(sample))); t = 0.0; slots = []; total = 0.0
    stops = collections.Counter(); ext_levels = 0
    def play(i, t0):
        nonlocal ext_levels
        ups, T, n = sample[i]; dur = BASE / T
        levels, last = 0, 0
        # observed part
        for turn in range(1, T + 1):
            now = t0 + turn * dur
            if now > t0 + BASE or now > BUDGET: return levels, min(now, BUDGET), "cap"
            if turn in ups: levels += ups.count(turn); last = turn
            if levels >= n: return n, now, "won"
            if stall_s is not None and (turn - last) * dur >= stall_s: return levels, now, "stall"
        return levels, t0 + T * dur, "observed_end"
    ends = []
    for i in queue[:SLOTS]:
        lv, te, why = play(i, 0.0); heapq.heappush(ends, (te, i, lv, why))
    nxt = SLOTS
    while ends:
        te, i, lv, why = heapq.heappop(ends)
        ups, T, n = sample[i]
        if why == "observed_end" and extend and nxt >= len(sample):
            # queue empty and still progressing: extend on the gap hazard
            dur = BASE / T; last = max([u for u in ups] + [0]); turn = T
            while te + dur <= BUDGET and lv < n:
                turn += 1; te += dur
                if rng.random() < hz(turn - 1 - last): lv += 1; last = turn; ext_levels += 1
                if stall_s is not None and (turn - last) * dur >= stall_s: break
        stops[why] += 1; total += pts(lv, n)
        if nxt < len(sample):
            if te < BUDGET:
                lv2, te2, why2 = play(nxt, te); heapq.heappush(ends, (te2, nxt, lv2, why2))
            else:
                stops["never_started"] += 1
            nxt += 1
    return total / len(sample), stops, ext_levels

rng = random.Random(0)
samples = [[rng.choice(traj) for _ in range(N)] for _ in range(300)]
base = [run(s, None, False, random.Random(k))[0] for k, s in enumerate(samples)]
b = sum(base) / len(base)
print(f"baseline (fixed 7920 s, 4 waves, last truncated): {b:.2f} completion-pts/game")
for stall in (None, 40, 55, 70, 85, 100):
    for extend in (False, True):
        if stall is None and not extend: continue
        res = [run(s, stall, extend, random.Random(k)) for k, s in enumerate(samples)]
        m = sum(r[0] for r in res) / len(res)
        d = [r[0] - bb for r, bb in zip(res, base)]
        d.sort()
        st = collections.Counter(); [st.update(r[1]) for r in res]
        print(f"stall={str(stall)+' min' if stall else 'none':8s} extend={str(extend):5s}: {m:.2f} pts  "
              f"({100*(m-b)/b:+5.1f}%, 90% band {100*d[15]/b:+.1f}..{100*d[-16]/b:+.1f}%)  "
              f"stops/run: stall={st['stall']/len(res):.1f} never_started={st['never_started']/len(res):.1f}")
