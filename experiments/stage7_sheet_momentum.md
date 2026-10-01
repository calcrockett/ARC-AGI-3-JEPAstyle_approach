# Stage 7 — sheetu12b + momentum time

**Question.** A public notebook, `scottlegrand/taaf-flashnext-sheetu12b-0922`
(Apache-2.0), scored **5.19** on the hidden set (n=1, its only version) --
above every score this project has (anim n=6 mean 3.258; anim + momentum
n=1 4.37). Its changes are agent-side; ours (momentum time allocation) is
scheduling-side. Do they stack?

Kernel: `calamitychasm/arc3-sheetu12b-momentum`, built by
`scripts/_build_sheet_momentum_kernel.py` from the vendored upstream notebook
(`kaggle_submission_sheetu12b_momentum/upstream/`). Diff vs upstream, checked
cell by cell: a credit cell, the momentum install cell (production settings:
stall 100 min, momentum 45 min, cap 1.5x) and the decisions dump after `bm.run`.
Nothing else.

## 1. What the upstream notebook is [READ, 2026-09-28]

Same serving stack, model, and run settings as ours (7,920 s/game, 28 slots,
analyzer timeout 900 s). The solver is the plain keithtyser Duck bundle (not
anim), with a ~2,700-line runtime patch cell. Switches ON in the 5.19 version:

| fix | effect | in anim? |
|---|---|---|
| F13 anim frames | intermediate boards exposed in the python sandbox | yes, own form (`animation()`) |
| F19 frame sheet | all frames of the last action as ONE labelled image beside the board | no (anim: text + sandbox) |
| ARM P upscale 12 | board image 768 px instead of 256 px | no (4) |
| F1 images | only 2 newest board images kept in history; flat 120-token image estimate | no |
| F3 memory | tolerant label parser; game_over does not wipe the world model | no |

Its own notes (unverified, their numbers): F13 moved them 3.20 -> 3.71; a
harness-written TEXT narration of the frames fell to 2.57 ("frames yes,
narration no"); several other fixes (F2, F4, F5, F7, F10) were turned off after
regressions. Its public-25 run scored 10.02 (our anim runs 9.97, 8.71; our
anim + momentum 11.72) -- public-25 cannot rank these.

## 2. The same defects measured in anim [MEASURED, anim public-25 transcripts]

- **World-model updates dropped: 367 / 583 (63%)** of assistant replies that
  write a world model label it `World model (revised):` etc.; anim's exact-prefix
  parser drops them and the stale model stays in the prompt. 18 game-overs in
  the run each wiped the world model.
- **Image over-charge ~5x:** anim's estimator charges a 256 px board PNG
  ~480 tokens (len(b64)//3, median over 8 games) vs ~100 real vision tokens.

These are reasons the upstream agent may be stronger than anim; they are not
tested here.

## 3. Pre-registration

- *Check run* (kernel v1 push, public-25, capped at 7,200 s by upstream's
  validation cap, so no extension can fire): catastrophe check only. Pass =
  run completes, `AGENTFIX LIVE` and `ARM P installed` and `MOMENTUM_TIME
  installed` in the log, no crashed games.
- *Submission*: authorised by the user on 2026-09-28 ("submit it when it is
  built"); first slot after the check run is 2026-09-29 UTC.
- *Comparison*: vs upstream's 5.19 (n=1, someone else's draw, possibly
  selected) and vs anim + momentum 4.37 (n=1). One draw ranks nothing; the
  falsifier for "this is our new incumbent" is a draw below 4.37.
- The momentum gain was simulated for the anim solver's level-up timing; this
  agent's timing is unmeasured, so the policy may help less (or more) here.

## 4. Check run (kernel v1, 2026-09-28 16:26-18:56 UTC): PASSED

Log shows `AGENTFIX LIVE` (F1, F3, F13, F19 on; F2/F4/F5/F7/F10/F11 off --
identical to upstream), `ARM P installed: MULTIMODAL_UPSCALE 4 -> 12`,
`MOMENTUM_TIME installed` at production settings, 457 frame-sheet events.
25/25 games ended `gave_up` (9) or `cancelled` (16, upstream's 7,200 s
validation cap) -- none crashed. Public-25 9.86 (upstream's own capped run:
10.02). The only tracebacks are the keithtyser `serving_teardown.py` gate after
the run, which the anim runs log too. Momentum: 8 stall stops (6,016-7,191 s);
no extension could fire under the 7,200 s cap
(`experiments/stage7_sheet_momentum_public25_decisions.json`).

Submission scheduled for 2026-09-29 00:02 UTC.

## 5. Result, draw 1 (2026-09-29): **3.89**

Ref 56655708, submitted 00:02 UTC, COMPLETE 09:02 UTC, public score **3.89**.

| arm | n | scores | mean |
|---|---:|---|---:|
| anim | 6 | 3.43, 3.79, 3.37, 3.02, 2.66, 3.28 | 3.258 |
| anim + momentum | 1 | 4.37 | -- |
| **sheetu12b + momentum** | **1** | **3.89** | -- |
| sheetu12b (upstream, not ours) | 1 | 5.19 | -- |

**Pre-registered falsifier met: 3.89 < 4.37, so this is not the new incumbent.**
It is above the anim mean (+0.63, 1.6 anim-sd) and 1.30 below upstream's own
5.19. At n=1 per arm none of these gaps is separable from draw-to-draw noise
(anim's own range is 2.66-3.79). Three readings, not distinguishable yet:
upstream's 5.19 was a high or selected draw; momentum helps this agent less
than anim, or hurts it (its level-up timing was never measured); or this draw
landed low. A plain-upstream draw of our own is what would separate the
first two.

## 6. Result, draw 2 (2026-10-01): **4.38**

Ref 56713526, kernel v1 byte-identical resubmit, submitted 2026-09-30 16:42 UTC,
COMPLETE 01:44 UTC, public score **4.38** -- tied with the project's best.

| arm | n | scores | mean |
|---|---:|---|---:|
| anim | 6 | 3.43, 3.79, 3.37, 3.02, 2.66, 3.28 | 3.258 |
| anim + momentum | 1 | 4.37 | 4.37 |
| **sheetu12b + momentum** | **2** | **3.89, 4.38** | **4.135** |

Draw-to-draw spread on identical code: 0.49 (anim's own range is 1.13).
sheetu12b + momentum vs anim: both draws above every anim draw, exact
permutation p = 1/28 = 0.036 (one-sided). This is evidence the arm beats plain
anim; it does NOT separate the two parts (their agent vs our momentum), and it
does NOT rank it against anim + momentum (4.135 vs 4.37, n=2 vs n=1). Every
momentum-bearing draw so far (3 of 3) beats every plain-anim draw (p = 1/84),
but the two momentum arms use different agents, so that pooled figure is a
pattern, not a test of momentum.

## 7. Result, draw 3 (2026-10-01): **4.58** -- the project's best

Ref 56750615, kernel v1 byte-identical resubmit, submitted 12:04 UTC, COMPLETE
21:06 UTC, public score **4.58** (previous best 4.38 / 4.37).

| arm | n | scores | mean | sd |
|---|---:|---|---:|---:|
| anim | 6 | 3.43, 3.79, 3.37, 3.02, 2.66, 3.28 | 3.258 | 0.385 |
| anim + momentum | 1 | 4.37 | 4.37 | -- |
| **sheetu12b + momentum** | **3** | **3.89, 4.38, 4.58** | **4.283** | **0.355** |

vs anim: all three draws above every anim draw, exact permutation p = 1/84 =
0.012 (one-sided); mean +1.03 (+31%). Spread on identical code is similar to
anim's (sd 0.355 vs 0.385). This arm is now the incumbent on evidence (n=3).
Still open: whether momentum or the upstream agent carries the gain, and
where it stands against anim + momentum (n=1).
