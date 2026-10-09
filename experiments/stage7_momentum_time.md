# Stage 7 — Momentum time allocation for the anim solver

**Question.** The competition run is 100% time-bound: every game plays until its
fixed 7,920 s clock runs out (all 25 of 25 public games hit the wall in every
recorded run). Is that time spent where it earns the most score, and can it be
re-allocated -- without more draws on an unchanged configuration?

**Status:** built and unit-tested; Haiku smoke test and real submissions
pending. Incumbent to beat: **anim, n=6, mean 3.258** (3.43, 3.79, 3.37, 3.02,
2.66, 3.28) vs NVFP4 baseline n=5 mean 2.760 -- exact permutation p = 0.024.

Code: `kaggle_submission_duck_nvfp4_anim_momentum/momentum_time.py`
(policy), `scripts/_build_momentum_kernels.py` (both kernels),
`tests/test_momentum_time.py` (12 tests), `scripts/momentum/` (analysis),
`experiments/stage7_momentum_time_level_timing.json` (data).

## 1. Where the turns go [MEASURED]

Level-up turns (`analysis_step` of every `level_completed` event) for 225
game-runs across 9 distinct public-25 runs of this solver family (duplicates
removed by content). Turns are ~330 s each at production settings; median 24
per game.

| state | turns | level-ups / turn | completion pts / turn | pts per level-up |
|---|---:|---:|---:|---:|
| progressing (<= 9 turns since last level-up) | 3,718 | 0.068 | **0.423** | 6.2 |
| stuck (>= 10 turns, ~55 min) | 1,995 (**35%**) | 0.034 | **0.155** | 4.5 |

A stuck turn earns 2.7x less, and stuck games are not sitting on deep,
valuable levels. But stuck games are not hopeless either:

| stuck for | games that ever level again |
|---|---:|
| 5 turns (~27 min) | 53.8% |
| 10 turns (~55 min) | 39.3% |
| 15 turns (~82 min) | 33.1% |
| 20 turns (~110 min) | 21.9% |

48% of games were still progressing (a level-up within 10 turns) when the clock
cut them off. The per-turn level-up rate peaks 3-5 turns *after* a level-up
(0.096/turn; 0.044 in the first 2) -- momentum is real. And momentum does not
fade with depth: with a recent level-up the rate is 0.065 / 0.080 / 0.082 per
turn at levels 0-1 / 2-3 / 4+ (n = 3,023 / 598 / 97 turns). "Levels get
harder" shows up only in *stuck* games (0.036 shallow, 0.013 at levels 2-3).

## 2. The obvious policy is wrong [SIMULATED]

A 110-game hidden run (28 slots; 32,400 s budget minus setup and soft-stop
margins), resampled 300x from the 225 real trajectories. Stopping a stuck
game is scored **exactly** (the trajectory's later level-ups are forfeited);
time past the observed clock uses the measured gap-based level-up rates.

Stopping stuck games at 40 / 55 / 70 minutes: **-38% / -17% / -2%**. Every game
is started anyway in the baseline, so freed time cannot buy a new game; it can
only extend games at the very end. My first estimate (+10-15%) came from the
2.7x value gap without modelling the queue, and was wrong.

## 3. The momentum policy [SIMULATED]

Suggested in review: once a game has found the rule, do not kill its
momentum. So: **at 7,920 s, a game that leveled up within the last 45 minutes
keeps going** (to at most 1.5x); games stuck 100 minutes stop to pay for it.

| stall stop | momentum window | cap | projected | 90% band | never started |
|---|---|---|---:|---:|---:|
| none | 45 min | 2.0x | +11.2% | +0.3 .. +23.6 | 0.7 / run |
| 85 min | 45 min | 1.5x | +9.7% | +0.4 .. +22.4 | 0 |
| **100 min** | **45 min** | **1.5x** | **+9.9%** | **+1.0 .. +21.8** | **0** |
| 120 min | 45 min | 2.0x | +12.5% | +1.5 .. +25.1 | 0.5 / run |

Chosen: 100 / 45 / 1.5x -- positive across the band, no starved games. The
long stall stop keeps the queue flowing (games cut by the deadline 16 / run vs
26 without it).

**The whole gain lives past 7,920 s, which no recorded run has observed.** It
assumes a game with momentum keeps leveling at the measured rates. That is
the untested assumption, and the real submissions are what test it.

## 4. Implementation

`MomentumTimePolicy` replaces `_HarnessGameSession.runtime_limit_reached` and
`timing_payload` from a notebook cell; no bundle file is edited. The solver
polls the stop rule before every turn and from inside long LLM calls; its
28-slot semaphore queue starts the next game the moment one stops; the
framework's own soft deadline still cancels everything at the notebook
budget. The model is shown `time_remaining_seconds`: until 7,920 s it sees
exactly what the unmodified solver shows (play before then is unchanged);
an extended game sees its extension, and request timeouts follow it.

The real-submission kernel `arc3-duck-nvfp4-anim-momentum` is the anim
notebook -- verified identical in code to the live kernel that produced all six
anim draws -- plus one install cell and a 5-line decision dump.

## 5. Pre-registration (before any run)

**Haiku smoke test** (`arc3-momentum-haiku-smoke`, CPU + internet, not scored).
The anim notebook with the vLLM boot removed and Claude Haiku as the analyzer,
under the served model's harness settings (32,768-token context, same tool,
yield and temperature knobs), so it sees only what the harness shows. 6 games
(3 that usually level early, 3 that usually stall) on 3 slots -- the queue is
exercised, which a public-25 GPU run cannot do (25 games < 28 slots). Clocks
scaled: base 600 s, momentum 240 s, stall 420 s, cap 1.5x. Pass = every game
finalizes, none crash, the policy sees every game, and its decisions are
recorded. It tests plumbing, not score.

**Real submissions**, alternating days with plain anim so both arms are drawn
in the same period.
- *Outcome* -- score vs the anim distribution. Projection: +~10% (~+0.3 on
  3.26). One draw has ~10% relative noise, so n >= 3 per arm before any claim.
- *Mechanism* -- not observable on the hidden run (no logs come back). The
  Haiku run and, if needed, a free public-25 run are where the decisions are
  counted.
- *Falsifier* -- anim-momentum below the anim mean after 3 draws each.

## 6. Haiku relay smoke test (2026-09-28): PASSED on plumbing

`scripts/momentum/haiku_relay_smoke.py`, run locally. The REAL anim solver --
the unpickled `HarnessSolver`, its semaphore queue, `_HarnessGameSession.play()`,
the real `ToolAgent` (system prompt, board rendering, history, trimming to the
served model's 32,768-token window) and the real Python-tool sandbox -- with the
policy installed and one substitution: `ToolAgent._chat_completion`. Each LLM
call was written to a file holding exactly the messages and tool schemas the
served model would receive, and answered by a fresh Claude Haiku subagent that
read only that file. Haiku carries its own system prompt, so it is a rough
stand-in for Qwen, not a replica. Time was simulated (150 s per batch) so the
policy's clock did not depend on subagent latency.

4 games (ft09, vc33, tu93, sk48) on 2 slots; base 900 s, momentum 450 s, stall
630 s, cap 1.5x. Result (`experiments/stage7_momentum_time_haiku_smoke.json`):

- no error; all 4 games finalized as `gave_up` (none crashed), scored;
- the policy saw all 4 and recorded 4 stall stops (at 750 s, limit 630 s);
- **the queue refilled**: tu93 and sk48 started in the slots ft09 and vc33
  vacated -- the path a public-25 GPU run cannot exercise;
- 24 relayed calls: 20 tool calls, 4 no-action replies, 0 unparseable
  (Haiku's hand-written JSON needed a repair for `\'` escapes and a missing
  brace -- a relay artifact, not a harness one).

**Not exercised: extension.** Haiku did not level up within ~5 turns per game,
so no game had momentum at the base time. That path is covered by unit tests
and will be counted in the real-kernel GPU run on the 25 public games.

Two relay replies per pair were written by the operator, not Haiku: requests
created a moment before the clock passed the stall limit, which the harness
stops at its next check whatever the reply. They carried no action.

## 7. Real kernel, public-25 check run (2026-09-28 02:56-06:17 UTC)

`arc3-duck-nvfp4-anim-momentum` v1 on the production stack. A catastrophe and
mechanism check, not a ranking (public-25 SE +/-2.46): 25/25 games finalized
(`gave_up`), none crashed; audit passed; public-25 score 11.72 (prior anim runs
9.97, 8.71). The vLLM teardown traceback in the log is pre-existing -- the
unmodified anim run logs it too. Decisions
(`experiments/stage7_momentum_time_public25_decisions.json`):

- **11 stall stops** at 6,032-7,662 s. On public-25 there is no queue (25
  games < 28 slots), so this time goes unused here; on the hidden run it
  starts queued games sooner.
- **6 extensions** past 7,920 s: tr87 ran to 10,810 s and went **4 -> 5
  levels**; ft09 lost momentum within a minute; s5i5, sc25, r11l, sp80 ran
  to 8,571-10,424 s with no further level. One level from six extensions.

**Submitted: ref 56631362, 2026-09-28 06:19 UTC** -- draw 1 of the momentum
arm, to be compared with anim (n=6, mean 3.258, sd 0.385). Next draws
alternate with plain anim so both arms are measured in the same period.

## 8. Result, draw 1 (2026-09-28): **4.37**

Ref 56631362, COMPLETE, public score **4.37** -- the project's best real score
(previous best 3.79, an anim draw).

| arm | n | scores | mean | sd |
|---|---:|---|---:|---:|
| anim | 6 | 3.43, 3.79, 3.37, 3.02, 2.66, 3.28 | 3.258 | 0.385 |
| **anim + momentum** | **1** | **4.37** | -- | -- |

+1.11 over the anim mean (2.9 anim-sd), and above anim's best draw. **Not yet
evidence at the standard this project holds itself to**: with n=1 vs 6, the
exact permutation p cannot go below 1/7 = 0.14 however high the draw, and
three earlier effects here shrank as n grew (anim's own first two draws were
3.43 and 3.79). It is also a larger gain than the simulation projected
(+9.9%, band to +21.8%; this is +34%) -- consistent with a favourable draw on
top of a real effect, or with the simulation understating the value of time
past 7,920 s. Only more draws, interleaved with anim, separate those.
