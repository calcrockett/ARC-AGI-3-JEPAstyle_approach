import heapq, json, os, random, sys, collections
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_realloc.py")).read().split("def run(")[0])   # data + hazard + constants

def run(sample, stall_min, mom_min, hard_mult, rng):
    stall_s = None if stall_min is None else stall_min * 60
    mom_s = None if mom_min is None else mom_min * 60
    ends, total, stops, nxt = [], 0.0, collections.Counter(), SLOTS
    def play(i, t0):
        ups, T, n = sample[i]; dur = BASE / T
        levels, last, turn, t = 0, 0, 0, t0
        # enough budget reserved for games still queued? (start guard below)
        while True:
            turn += 1; t = t0 + turn * dur
            if t > BUDGET: return levels, BUDGET, "deadline"
            if turn <= T:
                if turn in ups: levels += ups.count(turn); last = turn
            else:
                if rng.random() < hz(turn - 1 - last): levels += 1; last = turn
            if levels >= n: return n, t, "won"
            gap_s = (turn - last) * dur
            if stall_s is not None and gap_s >= stall_s: return levels, t, "stall"
            if t - t0 >= BASE:
                progressing = mom_s is not None and gap_s < mom_s
                if not progressing or t - t0 >= hard_mult * BASE: return levels, t, "cap"
    for i in range(SLOTS):
        lv, te, why = play(i, 0.0); heapq.heappush(ends, (te, i, lv, why))
    while ends:
        te, i, lv, why = heapq.heappop(ends)
        _, _, n = sample[i]; stops[why] += 1; total += pts(lv, n)
        if nxt < len(sample):
            if te < BUDGET - 600:
                lv2, te2, why2 = play(nxt, te); heapq.heappush(ends, (te2, nxt, lv2, why2))
            else:
                stops["never_started"] += 1
            nxt += 1
    return total / len(sample), stops

rng = random.Random(0)
samples = [[rng.choice(traj) for _ in range(N)] for _ in range(300)]
base = [run(s, None, None, 1.0, random.Random(k))[0] for k, s in enumerate(samples)]
b = sum(base) / len(base)
print(f"baseline: {b:.2f} pts/game")
for stall in (None, 85, 100, 120):
    for mom in (20, 30, 45):
        for hard in (1.5, 2.0):
            res = [run(s, stall, mom, hard, random.Random(k)) for k, s in enumerate(samples)]
            m = sum(r[0] for r in res) / len(res)
            d = sorted(r[0] - bb for r, bb in zip(res, base))
            st = collections.Counter(); [st.update(r[1]) for r in res]
            print(f"stall={str(stall):4s} momentum<{mom}min hard={hard}x: {100*(m-b)/b:+5.1f}% "
                  f"(90% band {100*d[15]/b:+.1f}..{100*d[-16]/b:+.1f})  never_started/run={st['never_started']/len(res):.1f} deadline-cut/run={st['deadline']/len(res):.1f}")
