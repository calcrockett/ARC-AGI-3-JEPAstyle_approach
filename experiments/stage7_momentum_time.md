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
