# Stage 7 — what to improve on the milestone-2 fork

Baseline: verbatim fork of `dfranzen/arc-agi-3-milestone-2-solution`, our
draw **24.99** (rank 249).

## 0. The noise floor decides the method [DATA]

Forum analysis of the hundreds of one-shot copies submitted in the first
24 h (discussion 745062): **mean 25.77, sd 3.93, median 26.19**. Our 24.99
and the author's 27.89 / 31.47 are ordinary draws. One draw resolves only
~8-point effects (90%); a +10% (~2.6 pt) change needs ~9 draws per arm --
more than a month of slots. **So prefer changes whose effect can be measured
without a submission** (throughput, startup), and treat prompt/harness
changes as bundles judged on mechanism.

## 1. The run is decode-bound and compute-starved [MEASURED, our check run]

10 public games, 25 min each, 10 streams (`m2fork_v1`): 544 requests,
**94.4% of prompt tokens served from cache**, 0.9M tokens generated. Decode
throughput by concurrent requests (SGLang log, 1,034 batches):

| running | 1 | 4 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| tok/s (median) | 213 | 492 | 664 | 707 | 757 | 709 |

KV pool 1,011,264 tokens (FP8), peak usage **0.91** (p90 0.76), Mamba
usage peak 0.67; `max_running_requests=10`; ~4.25 GB GPU memory left free.
Hidden-run arithmetic: ~700 tok/s x 532 min = ~22M tokens / ~110 games =
**~200K tokens per game**, against the author's per-level token reference of
27-67K. Author: scores were still rising at the end of runs.

## 2. What the other two Milestone-2 winners did differently [READ, discussion 744792]

| | dfranzen (ours) | lordhansolo | sirikilohit |
|---|---|---|---|
| KV pool | 1.01M (mem 0.96) | **1.42M** (vLLM, util 0.98) | 1.00M + **48 GB host tier** |
| streams | 10 | **14** | **16** |
| peak decode | 946 | **1,135** | **1,159** |
| level memory | functions kept | modules kept | **rules + last 30 winning actions of each solved level pinned in the system prompt** |
| extra prompt | level transfer | **strategy-audit prompt at 25% of a game's time** | Wang/Ludvig addenda |

Also from that thread: one team reports **NVFP4 KV** costs very little NLL
versus FP8 (halves KV memory again); another moved from 14.5 to 22.5 mainly by
spending freed KV memory on longer retained history.

## 3. Why 4.58 -> 24.99 [MEASURED, both check runs' server metrics]

| | old stack (sheetu12b + momentum) | milestone-2 fork |
|---|---|---|
| model | Qwen3.8-Flash-Next NVFP4, vLLM | Qwen3.8-Flash-Next W4A16 + drafter, SGLang |
| context window | 32K | 128K (trim to ~59K in one block) |
| prefix-cache hits | **0** (caching disabled) | **94.4%** of prompt tokens |
| prompt tokens re-processed | 22.8M in 2 h (all of them) | 1.9M of 33.9M in 25 min |
| generated tokens / s | **~250** | **~590** |
| preemptions | 112 | n/a (bounded admission, 10 streams) |

Same base model; ~2.4x the generated tokens per second while carrying 4x
the context. Consistent with other teams' published ablations on this stack:
FP8 KV spent on longer history took one team 14.5 -> 22.5, and
sirikilohit's biggest single step was the same change (14.49 -> 22.53);
dfranzen credits model, context, animations, 10x images and UNDO. Our own
switch changed all of these at once, so the split between serving efficiency
and harness/perception changes is not measured here.

## 4. The priority gate parks final levels -- and the cause is A, not only B [CODE + SIM, 2026-10-09]

**Variant built: `--prio-tail` (token `tail`)**: `ARC3_PRIORITY_HUMAN_ACTIONS=60` and last-level B = 5, both set
at runtime in the install cell. Kernels `calamitychasm/arc3-m2-lm-tail` (incumbent + this),
`arc3-m2-turbo-tail` and `arc3-m2-turbo-lossless-tail` (`kaggle_submission_m2_*tail*/`). Simulated effect on the
hidden-like rerun: **+0.4% to +3.9% RHAE, median ~+2.2% (~+0.7 LB points), positive on average in all 20 world x
slot cells**; not measurable by draws (sd 2.2-3.9), so it rides on mechanism + the replay below.

### 4.1 Mechanism (read from the harness patch; `priority_scheduler.py`, `tool_agent.py` `_PriorityGate`)

`P = (A + B·φ) · C`, with our notebook's `ARC3_PRIORITY_TAIL_LOOKUP=remaining`, `SCORE_NORMALIZATION=1`, fade 0.2:

- **A = ℓ · (h/(h+a))² · 55/(N(N+1)/2)**, h = `ARC3_PRIORITY_HUMAN_ACTIONS` = **25**, a = actions on the level.
  A *does* carry w_l = ℓ and the game's total weight N(N+1)/2 exactly: it is the level's true RHAE share (x55) times
  an efficiency proxy. A fresh final level is not disadvantaged: N = 7, P at a = 0 is 996 / 1192 / 1389 / 1585 /
  1682 / 1678 / **1375** for levels 1..7.
- **B = 8/7/5/0** by levels remaining after the current one; not normalized, not efficiency-discounted. B = 0 on the
  last level is intended (no later levels exist).
- **C** = 0.25·2^-(a/115)² + 0.75·2^-(t/62000)² (same for every level).
- Slots change hands **only at a context trim** (first after ~62k generated tokens, then ~37k) or a win.
  Never-started games queue at 2,000,000 − index (fresh first). A waiter keeps its snapshot (ℓ, a, t), so a parked
  game's P changes only through the fade φ (last 106 of 532 minutes).

**The cliff is A's efficiency proxy.** h = 25 against real human counts (public games: median **60** per level,
**113** on final levels). A falls to 11% after 50 actions, where RHAE still pays S = 1 (S = min(1.15, h/a)² is
capped until a > h/1.15). Mid-game levels hide this behind B; on the last level A is all there is. N = 7, a waiting
game at 50 actions / 40k tokens on its level: **final level P = 119 vs 632-693 for levels 1-5**; at 100 actions /
60k: 29 vs 399-448. A game reaches its first trim on a level with dozens of actions spent, so it is parked until the
fade. The simulated rerun parks last levels **86-92% of their game time** (JustAdev742 measured 70% on D′, whose A
decays as (300/(300+a))^2.5 and so only has the B cliff; tu93 71 min, sc25 50 min, several wins just in time).

### 4.2 Replay (`scripts/sim_m2_priority_gate.py`; CPU, ~8 min for 120 draws x 10 worlds x 2 slot counts)

Token-clock model of the gate: `priority_value` executed from the upstream patch itself, trims N(62k,6k) then
N(37k,7.5k) (JustAdev742's measured quanta), fresh-first, fade, 110 games drawn from the 25 public games' level
lists and human counts, 532 min, 10 slots x 70 tok/s or 14 x 55. Level cost lognormal (median 22k tokens, JustAdev742's
solved-level median 21.8k; game factor sd 0.6, level sd 0.8; level 1 x0.5), a share of levels unsolvable, actions
at solve = h x ratio (median 0.8). Scored with real RHAE. **Calibration**: hidden-like worlds score 26.5-33 (our LB
30.5) with draw sd 2.3-3.7 (the unmodified copies' sd: 3.93). Paired over common random numbers.

RHAE change vs the incumbent gate, % of base (10 slots / 14 slots; 120 draws each; 90% CIs of the mean are
~±0.05-0.15 pt; for the built column P(single draw > 0) is 0.58-0.95 in the hidden-like worlds):

| world (base score) | final B=5 | final B=8 | h=60 | h=100 | **h=60 + final 5 (built)** | fade 0.4 | 80k lookup |
|---|---|---|---|---|---|---|---|
| tokens x2.5 (28.5) | +0.8 / +1.1 | +1.6 / +1.7 | +3.6 / +3.2 | +6.2 / +5.2 | **+3.9 / +3.5** | −0.2 / −0.3 | +4.0 / +3.1 |
| tokens x2, 12% unsolvable (29.9) | +0.9 / +1.1 | +1.4 / +1.3 | +2.4 / +2.2 | +4.0 / +3.3 | **+2.7 / +2.6** | −0.5 / 0.0 | +2.2 / +1.9 |
| 20% of levels unsolvable (31.0) | +0.4 / +0.3 | +0.4 / +0.3 | +0.5 / +0.4 | +0.8 / +0.6 | **+0.6 / +0.5** | +0.2 / +0.2 | +0.4 / +0.3 |
| cost x1.15 per level of depth (30.0) | +0.8 / +0.8 | +1.6 / +1.6 | +3.1 / +2.2 | +5.0 / +4.1 | **+3.4 / +2.6** | −0.1 / −0.1 | +2.9 / +2.2 |
| final level 3x as often unsolvable (30.7) | +0.5 / +0.6 | +0.5 / +0.4 | +2.3 / +2.0 | +3.8 / +2.9 | **+2.4 / +2.1** | −0.1 / −0.2 | +2.5 / +2.0 |
| weak game-difficulty correlation (28.5) | +0.6 / +0.8 | +1.2 / +1.4 | +2.9 / +2.7 | +5.2 / +4.8 | **+3.0 / +3.0** | −0.5 / −0.3 | +2.6 / +1.9 |
| slow levels cost actions too (29.0) | +0.6 / +0.8 | +0.9 / +1.0 | +1.4 / +1.4 | +2.4 / +2.0 | **+1.5 / +1.5** | −0.1 / 0.0 | +2.7 / +2.4 |
| agent at 1.5x human actions (28.2) | +0.8 / +0.9 | +1.0 / +1.0 | +1.4 / +1.1 | +2.1 / +1.7 | **+1.6 / +1.3** | 0.0 / +0.3 | +1.5 / +1.2 |
| both of the last two (26.5) | +0.6 / +0.5 | +0.5 / +0.4 | +0.7 / +0.2 | +1.3 / +0.5 | **+1.0 / +0.4** | −0.1 / 0.0 | +2.3 / +1.4 |
| public-like (60.2) | +0.6 / +0.2 | +0.6 / +0.2 | +0.6 / +0.3 | +1.0 / +0.5 | +0.7 / +0.3 | +0.2 / +0.1 | +0.5 / +0.3 |

Reading it:
- **The tail tuple alone (the question asked) is worth ~+0.3% to +1.7%**: (…,7,5,5) +0.3..+1.1%, B = 8 a little
  more, but no better than B = 5 where final levels are often unsolvable. Positive, but at or below the ~1% bar.
- **h is the bigger lever** because it removes the cause (A's collapse) for every level, and it is a plain env knob
  upstream already reads (`ARC3_PRIORITY_HUMAN_ACTIONS`). The gain grows with h up to 100 in every world, but how
  much depends on the assumption that solved levels cost about the human count regardless of how long they took;
  when slow levels also cost actions and the agent is 1.5x human, h = 60 shrinks to +0.2..+0.7% and h = 100 to
  +0.5..+1.3%. **h = 60 was chosen as the calibrated value (the public median), not the simulated optimum**, and
  B = 5 on the last level adds only +0.0..+0.1 pt on top once h is fixed (kept: it is the direct fix for the
  question asked and never hurt).
- Our own env alternatives: a longer fade (0.4, D′'s) is ~0 or negative; the empirical **80k lookup** (`=1`) is
  +0.3..+4.0% here, but dfranzen's own runs did not show it (with B also normalized) and it re-weights every level
  by depth, so it is a lead, not built.
- **Agreement with JustAdev742**: their replay of D′ found last-level B 0→10 worth +0.26 [−0.35, +1.09] public-25
  points and ≈0 in hard worlds. Our final-B-only columns are the same size (+0.1..+0.5 pt); the larger effect here
  comes from A, which their gate does not collapse.

### 4.3 What could make it wrong

- The replay is a model: per-level costs are iid within a game (plus a game factor) and actions-at-solve do not depend
  on how long the level took except in the "slow levels cost actions" worlds -- the two assumptions that drive the
  size of the h effect. A public-25 run cannot check it: JustAdev742 showed 25-game x 121-min replays get the sign
  of scheduler changes wrong (their §4.5), and one draw resolves ~8 points.
- Who pays: un-parked final levels take slot time from games with levels left; in the worst world (both pessimistic
  assumptions, 14 slots) the net is +0.1 pt with P(draw > 0) 0.58. No world was negative on average.
- It does not change anything the model sees, cache bands, or the slot count; a kernel with it runs exactly like its
  base except for the order in which waiting games are re-admitted.

### 4.4 Check-run and adoption

- Markers: the base kernel's plus `PRIORITY_TAIL installed` (the line prints `human_actions 60.0` and the B tables
  ending `(…, 7.0, 5.0, 5.0)`); the install cell asserts the upstream table ends `(…, 5.0, 0.0)` before patching and
  that `tool_agent` reads the same `priority_value` and h. Build: `python scripts/_build_m2_level_memory_kernel.py
  [--turbo | --turbo-lossless] --prio-tail`; `tests/test_m2_prio_tail.py` pins the kernels and executes the install
  lines on the upstream scheduler.
- **Recommendation: bundle it with whichever serving kernel is adopted next** (turbo-tail / turbo-lossless-tail):
  zero throughput cost, expected ~+2% (range +0.4..+3.9%), never negative in the replay. Alone on the incumbent
  (`arc3-m2-lm-tail`) it would need ~100 draws per arm to see, so spend slots on it only as part of a bundle.
