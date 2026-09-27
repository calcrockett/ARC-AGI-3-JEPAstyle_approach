# Stage 7 — CodeWorldAgent prototype on the served model

**Kernel:** `calamitychasm/arc3-cwm-prototype` (free; no submission quota)
**Date:** 2026-09-23
**Verdict:** the 2026-09-09 verdict is **overturned**. The model *can* write
replay-passing world models. It still does not convert them into levels.

---

## 1. The claim being re-tested

Commit `0603d60`, merged to `master`, states:

> **VERDICT: the coder model cannot write a replay-passing world model.**
> 25 LLM completions -> 14 compiled -> 0 passed replay. Best match ever:
> 2 of 9 transitions.
> *Recommendation: stop fixing CodeWorldAgent in its current form.*

Three things were different there, and all three mattered:

1. It used **Qwen3-Coder-30B-A3B** over `transformers`, not the
   **Qwen3.8-Flash-Next-NVFP4** we actually serve.
2. The transcript was never reset per level, so clearing one level made
   the replay gate permanently unsatisfiable.
3. **Its prompts were over the context window.** That run recorded
   "prompts growing to 142 KB" as a *throughput* note. At ~47k tokens
   against 32,768, those prompts were being rejected, not answered.

Point 3 means the original verdict was measuring rejection in part, not
only capability.

## 2. What was fixed first

| fix | evidence it was needed |
|---|---|
| per-level transcript reset | boundary steps rewrite 693-1054 of 4096 cells; no rule reproduces a fresh layout |
| reasoning-field fallback in the client | backtest saw `{'reasoning': 46}` with `content` never populated |
| `enable_thinking=False` | with thinking on, 27/27 replies truncated, **zero** contained a `class WorldModel` |
| **compact transcript rendering** | **93 of 109 calls rejected in ~0.1s; prompts to 281,603 chars (~94k tokens)** |

The last one is the decisive fix, and it was **my own known defect**: the
backtest write-up recorded this exact renderer problem as finding #4 on
2026-09-21, and the fix was not carried into the engine the live agent
uses.

## 3. Result — the renderer fix changes everything

Identical kernel, identical games, identical model. Only the prompt
encoding differs.

| | v1 (per-step grids) | v2 (one grid + diffs) |
|---|---:|---:|
| prompt chars, median | 160,900 | **21,446** |
| prompt chars, max | 281,603 | **53,469** |
| prompts over context | **76 / 109** | **0 / 308** |
| calls returning text | 16 / 109 (15%) | **284 / 308 (92%)** |
| candidates loaded | 13 | **227** |
| **replay passes** | **0** | **16** |
| games with a passing model | 0 | **4 of 12** |
| real model installed | **no** | **yes, 3 games** |
| wall clock | 140 s | 1,650 s |

**Longest passing replay: 28 of 28 transitions** (`lp85-305b61c3`). That is
not a lucky one-liner on a 3-step transcript; it is a model that
reproduces 28 consecutive observed steps exactly.

Graded signal across all 227 replays: median prefix **28%**, mean
**40.2%**, and **86 of 227** reproduced at least half the transcript.

Games with at least one passing model: `dc22`, `lp85`, `ls20`, `m0r0`.

## 4. What this does NOT show

**Zero levels were completed, in any of the 12 games.** A correct model of
the first 9-28 transitions did not convert into progress within 120
actions. The capability question is answered; the *usefulness* question is
not, and nothing here says the planner turns a good model into score.

Remaining generation waste is also real: **77 of 304 candidates failed to
load** — 30 missing `predict`/`goal_hint`, 16 unclosed parenthesis, 11
indentation errors. The unclosed-paren and indentation cases are
truncation, so some reply budget is still being lost.

## 5. Honest status of the design

- **Settled:** the served model writes replay-passing world models for
  real 64x64 ARC-AGI-3 games. The merged verdict saying otherwise was
  confounded by context overflow and should not be cited.
- **Not settled:** whether that converts to RHAE. 0 levels in 12 games is
  not encouraging, but 120 actions with a model installed for only 3 of
  them is a weak test of conversion.
- **Not a submission candidate.** With no model installed the agent plays
  random legal actions, which is the ~0.06 floor. 3 of 12 games installing
  a model does not change that arithmetic.

## 6. Next, in order

1. **Raise the install rate.** 4/12 games produce a passing model; the
   binding losses are truncation (still) and missing-method candidates.
2. **Then test conversion**, with a longer action budget on the games that
   *do* install a model — the only place a planner can show anything.
3. Only after both: consider a scored submission.

## 7. Caveats

- One free run, 12 games, 120 actions. No repeats.
- Public-25 games, which the whole community iterates against.
- `enable_thinking=False` throughout; the thinking arm was not re-tested
  after the renderer fix and may now behave differently, since its
  truncation was measured under the oversized prompts.
- Replay here is teacher-forced, as in the backtest: each step is scored
  from its own real `frame_before`, so errors cannot compound. A planner
  needs closed-loop rollout, which is strictly harder.

---

## 8. Coder-model arm (2026-09-25): a coder-specialist is NOT better

Kernel `calamitychasm/arc3-cwm-backtest-coder`, `COMPLETE`, 70 min.
**Qwen3-Coder-30B-A3B-Instruct** via `transformers`, on the same 12
segments, same prompt, same calibration as the Flash-Next arms. Only the
model differs.

### Result: 0 / 12

| arm | passed | reached the sandbox and ran |
|---|---:|---:|
| Flash-Next, thinking, 16k | 1/12 | 0 (27/27 truncated) |
| Flash-Next, no thinking, 8k | 0/12 | **4** |
| **Qwen3-Coder-30B** | **0/12** | **1** |

Attempt-level taxonomy across all 36 attempts:

| outcome | count |
|---|---:|
| `bad_shape` (malformed return triple) | **16** |
| `load_error` | 10 |
| `predict_raised` | 9 |
| `grid_mismatch` (ran, wrong) | 1 |
| `no_code` / `no_response` | **0 / 0** |

### What this actually says

**Compliance is perfect and capability is not.** The coder produced a
`class WorldModel` on every one of 36 attempts -- zero `no_code`, zero
`no_response`, unlike Flash-Next which frequently emitted nothing usable.
The preflight confirmed it too (179-char reply, contained the class).

But **35 of 36 candidates never produced a usable prediction**. The
dominant failure is `bad_shape` (16): `predict()` returns something that
is not a well-formed `(next_state, levels_delta, done)` triple -- the
exact slip `world_model.py` documents, returning one layer instead of the
list of layers. Nine more crashed at predict time.

So on the measure that matters -- candidates that load, run, and produce a
gradeable prediction -- the coder scores **1**, against Flash-Next's **4**.
It writes more code and less *working* code.

### Strategic consequence

The appliance is sealed; we serve Flash-Next and cannot swap it. A coder
win here would have been an awkward result -- a ceiling we could measure
but not act on. **This removes that awkwardness: the model we are allowed
to serve is not the bottleneck relative to a coder-specialist.**

### The one concrete lead

`bad_shape` at 16/36 is a **contract** failure, not a reasoning failure --
the model understands the task and botches the return type. That is the
kind of thing a sharper prompt (an explicit worked example of the exact
return shape, rather than a prose description) could plausibly fix, and it
would apply to *both* models. Untested; recorded as a lead, not a plan.

### Caveats

- One run, 12 segments, no repeats.
- `transformers`, so ~95 s/attempt and a 607 s model load. Slow, but the
  backtest makes only 36 calls so throughput is not a confound here.
- `max_tokens=8192`; responses ran to 29 KB, so truncation is not the
  story this time the way it was for Flash-Next's thinking arm.

---

## 9. Prompt fix (2026-09-25): the contract failure is solved, and a
## capability wall is what was underneath it

Single-variable A/B. Only `SYSTEM_PROMPT` differs -- same model
(Qwen3-Coder-30B-A3B), same 12 segments, same everything else. The old
prompt showed an elliptical skeleton and described the return in prose;
the new one states the contract explicitly and carries a **complete,
runnable** worked example.

### The fix worked, and by a lot

| attempt-level, 36 each | old | new | delta |
|---|---:|---:|---:|
| `bad_shape` | 16 | **0** | **-16** |
| `predict_raised` | 9 | **0** | **-9** |
| `load_error` | 10 | 8 | -2 |
| `grid_mismatch` (loaded, ran, gradeable) | 1 | **28** | **+27** |

**Attempts producing a usable prediction went from 1/36 to 28/36.** The
contract failure is not merely reduced, it is *gone*: zero malformed
triples, zero predict-time crashes. Responses also got shorter (median
10,601 -> 7,938 chars), consistent with less flailing.

That confirms the 2026-09-25 diagnosis exactly: `bad_shape` was a
**contract** failure, not a reasoning one, and a prompt that *shows* the
shape rather than describing it fixes it.

### And the pass rate did not move at all

**Still 0/12.** Best prefix across all 12 segments:

    old  [0,0,0,0,0,0,0,0,0,0,0,0]
    new  [0,0,0,0,0,0,0,0,0,0,0,1]

Eleven of twelve segments are wrong at **step 0**. The one exception
reproduced a single step.

So removing the confound did not reveal a near-miss underneath. It
revealed a **capability wall**: the coder now reliably writes valid,
loadable, runnable world models that are wrong about the rule
immediately and essentially always.

### Coder vs the served model, now on equal footing

| | reached the sandbox | best prefix |
|---|---:|---:|
| Flash-Next, no thinking | 4/12 segments | **9/40** |
| Qwen3-Coder-30B, new prompt | **9/12 segments** | 1/27 |

The coder produces **more than twice as many runnable candidates** and
gets **nowhere near as close to the rule**. Flash-Next reached 9
consecutive correct steps once; the coder never exceeds 1. And the live
prototype on Flash-Next produced **16 replay passes**, which neither
coder run came close to.

**The coder hypothesis is now firmly dead** -- tested twice, the second
time with the interface confound removed, and it lost on the measure that
matters both times.

### What to keep

The prompt change is a real, isolated, large improvement in code validity
and should be kept for **both** models -- Flash-Next's own runs lost
candidates to `load_error` too. It just is not what stands between this
design and a working world model.

`load_error` at 8/36 is the remaining validity loss and was barely touched
by the contract block (10 -> 8), so it is a different problem (syntax /
truncation), not the same one.

---

## 10. The prompt fix on the served model (2026-09-25): replicated on code
## validity, null on rule inference

Same single-variable A/B, now on **Qwen3.8-Flash-Next-NVFP4**. Kernel
byte-identical to the previous run; only the prompt inside the dataset
changed.

| | think-16k | | nothink-8k | |
|---|---:|---:|---:|---:|
| | old | new | old | new |
| **passed** | 1 | 0 | 0 | **1** |
| `bad_shape` (attempts) | 0 | 0 | **5** | **0** |
| `grid_mismatch` (attempts) | 0 | 0 | 7 | 4 |
| `load_error` (attempts) | 1 | 3 | 8 | 10 |
| `no_response` (segments) | 3 | 4 | 4 | 7 |
| best prefix | 6 | 0 | 9 | 6 |
| finish reasons | 27 `length` | 26 `length` | 16L/12S | 13L/7S |

### What replicates

**`bad_shape` 5 -> 0 on nothink**, exactly as it went 16 -> 0 on the coder.
Two different models, same direction, effect size large in both. The
contract block does what it was designed to do, and it is the one finding
here strong enough to keep.

### What does not move

**Passes are a wash: think 1 -> 0, nothink 0 -> 1.** At n=12 with about
one pass either way, that is noise and should not be read as either an
improvement or a regression. Best prefixes stay in the same 0-9 band they
were in before.

**think-16k is still entirely truncation-bound**: 26 of 26 replies hit
`length`, unchanged from 27 of 27. The new prompt was never going to fix
that -- it is a budget problem, and this arm still has not produced a
measurement of anything else.

### A hypothesis of mine that was wrong, tested rather than assumed

`no_response` rose 4 -> 7 on nothink and total replies fell 28 -> 20. My
first explanation was that the longer system prompt (1,357 -> 2,524 chars)
had pushed requests over the context window. **Measured: it had not.**
Across all 61 segments the largest total prompt is 38,508 chars (~12,836
tokens) against 16,384+ tokens of headroom in both arms -- **0 of 61**
exceed it in either. The response-rate difference is unexplained and, at
these counts, most likely variance. Recorded as unexplained rather than
given a mechanism it does not have.

### Where this leaves the design

Across two models and four arms the picture is now consistent:

* **Interface compliance is solved.** The contract block eliminates
  malformed returns on both models.
* **Rule inference is the wall.** Valid, loadable, runnable world models
  that are wrong -- usually at step 0 -- are now the dominant outcome.
* The best prefix anyone has reached on these segments is **9 of 40**, by
  Flash-Next, under the old prompt.

The remaining honest caveat is that the *live prototype* on Flash-Next
produced **16 replay passes** against the backtest arms' 0-1. The likely
difference is transcript length: the prototype drafts early, against 9-28
step transcripts, while these segments run to 40. That suggests **shorter
transcripts are the lever**, not a better prompt and not a better model --
untested, and the next thing worth testing.


## 11. goal_hint (2026-09-25): the half of the model that plays was never checked

Sections 1-10 measured `predict()`. **Action selection does not run on
`predict()`.** `planner.plan()` ranks rollouts by `(levels predicted,
goal_hint)`, and a transcript with no level-up gives `predict()` no basis
to ever predict one -- so in practice `goal_hint` is the whole objective.
Three things were wrong with it, and none was visible to any instrument in
this project, because nothing in `llm_engine/replay.py` or `arc3_cwm`
ever called `goal_hint`:

1. **The stall test compared its spread against an absolute 0.05.** Both
   real goal_hints on record (`experiments/stage7_codeworld_live_test_artifacts/`,
   ar25 and bp35) are cell-count *ratios* over the 4096-cell board, which
   move by ~0.0012 per five cells changed. Clearing 0.05 needs ~205 cells
   to differ between the best and worst candidate, so such a model
   **stalled on every step and was never consulted**.
2. **On a stall the agent played the planner's pick anyway** whenever the
   action head had nothing to offer -- always, in this kernel, which runs
   with `ACTION_LLM_CALL_BUDGET=0`. With a fixed candidate order and a
   stable sort, a tie went to the first candidate: `ACTION1`, every step.
3. **Both prompts invited a constant.** The engine skeleton said "return
   0.0 if you have no idea yet"; the backtest's contract prompt (section
   9) showed `return 0.0` as the complete valid answer and said to replace
   only the `predict` line.

Together these are a complete mechanistic account of section 8's headline:
**16 replay passes, 4 of 12 games with an installed model, 0 levels.**
The models were installed and then not used.

Also found on the way: a predicted terminal state `break`-ed out of the
planner's candidate loop, hiding every later candidate.

### What changed (branch `stage7-goal-hint`)

| | before | after |
|---|---|---|
| stall test | spread < 0.05 (absolute) | hints tied relative to their own magnitude |
| stalled pick | played if no action-head answer | discarded; random fallback |
| ties | first candidate (ACTION1) | random (candidates shuffled) |
| terminal state | `break`s the candidate loop | ends that rollout only |
| replay gate | predict only | predict **and** goal_hint: runs, finite, not constant across >=2 observed boards |
| prompts | "return 0.0 if you have no idea" | goal_hint is what the agent plays with; must not be constant; only order matters |

The backtest (`stage7-codeworld-backtest`, `1c77acb`) now reads
`predict_passed` explicitly: its oracle positive control has a stub
goal_hint by design, and under the gated engine the raw `.passed` would
have collapsed the 58/61 ceiling (verified: oracle `.passed` False,
predict half True, on a changing-board segment). Every backtest number
keeps its meaning.

### Pre-registered before the run (free kernel `arc3-cwm-prototype` v3)

Same configuration as section 8 (Flash-Next, thinking off, 12 games, 120
actions, coder budget 8), same games. Measures, all counted:

- **mechanism** -- of planner calls in games with an installed model, the
  fraction that decide the action (`planned / calls`). Each call also
  records whether the OLD rule would have stalled on that same call, so the
  counterfactual comes from the same run with no noise.
  *Expectation:* the old rule stalls on a large majority of calls; the new
  rule on few. If the old rule would NOT have stalled either, defect 1 was
  not binding and this account is wrong.
- **gate** -- candidates passing `predict` but rejected on `goal_hint`, and
  whether a retry recovers them. *Expectation:* few rejections, because
  the prompt now asks for a non-constant objective; a large number would
  mean the prompt change did not land.
- **outcome** -- levels completed (section 8: 0). One draw of a noisy
  quantity; a nonzero count is weak positive evidence, zero is not
  evidence the mechanism failed -- the mechanism measure decides that.
  A model that is used but plays badly is the expected next wall: the gate
  establishes that goal_hint distinguishes boards, not that it measures
  progress.

### Result (kernel v3, 2026-09-26 03:18-04:08 UTC): the falsifier fired

Ran clean: all stale-engine checks OK, 0 errors, 1,634 s wall clock, 296
LLM completions returning text, 12 games x 121 actions.

| pre-registered measure | value |
|---|---:|
| planner calls (3 games with an installed model) | 320 |
| stalled under the NEW rule | 299 |
| the OLD 0.05 rule would have stalled on | 299 |
| **rescued by the new rule** | **0** |
| candidates passing predict | 15 |
| of those, rejected on goal_hint | 1 |
| levels completed | **0** |

**The scale explanation is refuted for this run.** Every spread was either
exactly 0 (299 calls) or >= 1.0 (21): nothing in between, so the absolute
threshold never decided anything. Caveat that limits what this refutes:
the new prompt says "no need to normalise", and every goal_hint in this run
is a raw count. So this run cannot say whether the absolute threshold was
binding in section 8's run (whose two recorded hints were ratios). What it
does say: **fixing the scale does not fix stalling**, because the stalls
are exact ties.

### Why the hints tie exactly: they measure the HUD

| game | installed goal_hint |
|---|---|
| dc22 | count of color 3 in **row 63** |
| ls20 | minus count of color 11 in **rows 61-62** |
| lp85 | count of non-3/4 cells; accepted **vacuously** (1 distinct board) |

The bottom strip is the one thing that changes on every step of every
transcript, so a goal_hint reading it passes the "not constant" gate --
and is exactly the kind of objective that gate was always going to admit
(this failure was named as a risk in the design and not guarded against).
Every candidate advances the strip identically, so the hints tie.

Then the decisive check -- is it the objective or the simulator? Replaying
each installed model over every 4th recorded board, all 13 candidates:

| game | frames | goal_hint tied across candidates | predicted playfield (rows < 60) identical across candidates | some candidate changes the playfield |
|---|---:|---:|---:|---:|
| dc22 v7 | 31 | 18 | **0** | **31** |
| ls20 v6 | 31 | 20 | **0** | **31** |
| lp85 v1 | 31 | 31 | 31 | 0 |

For dc22 and ls20, **predict() distinguishes the actions on the playfield
on every frame, and goal_hint throws that away.** The simulator is doing
its job; the objective is looking at the wrong part of the board. lp85 is
a different failure: its model predicts that nothing ever changes, so no
objective could plan with it -- and it was admitted because its probe
transcript contained one distinct board, where the gate passes vacuously.

### What this changes

- The gate tested the wrong property. The planner needs goal_hint to
  **distinguish the boards predict() says different actions lead to**, not
  to vary across the transcript's history (which a step counter satisfies
  trivially). That property is directly checkable at gate time: from each
  observed board, predict every candidate's successor and require
  goal_hint to separate them somewhere.
- A single-distinct-board transcript gives no evidence about goal_hint and
  should not be a pass; nor, arguably, should a predict() that is the
  identity everywhere.
- Steps 1-3 stay: they are correct and were necessary (a constant or
  unvalidated objective would stall regardless), but they were not
  sufficient, and the 0-levels outcome is unchanged.


## 12. A gate that tests what the planner needs (2026-09-26)

Section 11 asked the wrong question of goal_hint ("does it vary across the
transcript?"). Two stronger versions were measured against the live run's
real models and boards before one was built:

| criterion | dc22 (HUD objective) | ls20 (HUD objective) |
|---|---:|---:|
| varies across history (section 11's gate) | passes | passes |
| separates the successors of different actions | separates 13/31 boards | 11/31 |
| ...among actions that change the board | 11/31 | 2/31 |
| ...ignoring a 3-cell edge band | 9/31 | 2/31 |
| **counterfactual: same edge band, different interior** | **0/31** | **0/31** |

The separation variants are all gamed by the strip: an action that only
ticks the counter scores differently from one that moves something. On
dc22's 13 separating boards, **the preferred action left the playfield
unchanged 13 times out of 13** -- the objective's only signal was a
preference for doing nothing.

The counterfactual check gives two actions' predicted boards the same edge
band and lets them differ only in the interior; goal_hint must respond to
that somewhere. Controls on the same boards: a position-weighted sum over
the whole board responds on **31/31** (both games); the same sum restricted
to rows >= 58 on **0/31**. Colour-count controls also scored 0/31 -- a moved
object has the same colour counts -- which is itself a finding: counting
colours, the models' habit, is blind to movement. The prompt now says so.

Also: the agent no longer drafts before its transcript shows two distinct
boards (lp85's model was admitted on one and stalled 112/112), and a
simulator whose predict() never changes the playfield differently is not
blamed on goal_hint.

Engine `39842e4`; 255 tests; the engine's own check, on the real models:
dc22 and ls20 REJECTED, position control ACCEPTED, lp85 not blamed.

### Pre-registered before kernel v4

Same configuration as v3 (Flash-Next, thinking off, 12 games, 120 actions,
coder budget 8). v3 for reference: 3 games with a model, 320 plan calls,
**21 planned (6.6%)**, 0 levels.

- **mechanism** -- fraction of plan calls that decide the action, in games
  with an installed model. *Prediction:* a majority. **Falsifier:** below
  25% means the gate did not change what gets installed in a way the
  planner feels, and the account in this section is wrong or incomplete.
- **gate** -- candidates rejected as "cannot tell actions apart", and
  whether a retry in the same game then passes.
- **installs** -- the honest risk. A stricter gate can mean fewer models
  (v3: 3/12). If installs fall to 0, the model cannot write a playfield
  objective under these prompts, and that is the result.
- **outcome** -- levels completed (v3: 0). One draw; the mechanism measure
  decides the question, not this.

### Result (kernel v4, 2026-09-26 12:36-13:27 UTC): the stall is gone; a new wall behind it

Clean run: all engine checks OK, 0 errors, 300 LLM completions with text.

| | v3 (history gate) | v4 (counterfactual gate) |
|---|---:|---:|
| games with an installed model | 3 | 4 |
| plan calls | 320 | 324 |
| **stalled** | **299 (93%)** | **31 (9.6%)** |
| **planner's pick actually played** | **21 (6.6%)** | **129 (39.8%)** |
| predict passes rejected on goal_hint | 1 | 4 ("cannot tell actions apart") |
| accepted with a simulator that never distinguishes actions | -- | 2 |
| levels completed | 0 | 0 |

Per game: dc22 80/104 played, 0 stalled; ls20 49/96 played, 3 stalled;
lp85 28/28 stalled (its simulator never distinguishes actions, as
predicted); **ft09 96 calls, 0 stalled, 0 played.**

**Against the pre-registration:** the falsifier (< 25% played) did not
fire -- 39.8%. The prediction (a majority) did not come true either. The
stall -- the thing this section set out to remove -- fell from 93% to 9.6%.

**Where the other half went: the planner never receives
`available_actions`.** It searches all seven actions in every game. Of its
non-stalled picks, 63 were ACTION5 or ACTION7, which none of the four games
allow (dc22 [1,2,3,4,6], ls20 [1,2,3,4], ft09 and lp85 [6] only). In ft09,
click-only, it chose a non-click action on every one of 96 calls. Every
such pick is discarded by the agent and replaced with a random action: 164
of 324 calls (51%). The objective is now working and the search is
spending it on moves the game will not accept.

Next: restrict the planner's candidates (and the gate's) to the actions the
game currently allows. 0 levels stands; it is one draw, and until the
planner can only choose legal moves it is not yet a test of the plan.

## 13. Legal moves only (2026-09-26)

Planner and gate now search only the frame's `available_actions`;
inapplicable opening probes are skipped (`f1b57e6`, 261 tests).
Counterfactual on v4's real boards with its installed models, before any
run: legal decisions dc22 26 -> **31/31**, ls20 22 -> **31/31**; ft09 and
lp85 **0/31** either way -- their simulators predict that no action changes
anything (ft09: none of 256 click points), which the gate records as
`predict_distinguishes_actions=False` rather than blaming goal_hint.

### Pre-registered before kernel v5

Same configuration as v3/v4.

- **mechanism** -- in games whose installed simulator distinguishes
  actions, the fraction of plan calls whose decision is played.
  *Prediction:* >= 90% (the counterfactual above says ~100%).
  **Falsifier:** below 60%.
- **click-only games** -- expected 0 decisions wherever the simulator
  predicts no click effect; that is the predict-side wall, not this fix.
- **outcome** -- levels completed. This is the first run in which the
  model's own plan actually chooses the moves, so it is the first real test
  of whether these objectives measure progress. Honest prior: probably
  still 0 -- the gate establishes that goal_hint responds to the playfield,
  not that it points toward a win.

### Kernel v5 (2026-09-26 21:35 -> 09-27 09:40 UTC): hung, no result

Not a measurement. The driver froze 24 minutes in (log t = 1,475 s,
~440 s into play) and the kernel sat silent until Kaggle's 12-hour session
limit cancelled it -- **~11.5 GPU-hours burned for nothing.** Evidence
captured to the freeze: 85 replays, 2 plan calls, no per-game records.

What is established: all 12 game threads stopped within milliseconds,
between two adjacent `print`s in one thread; vLLM had already answered
every outstanding request and went idle at 22:01:15; the 900 s client
timeout never fired. So some thread held the GIL, or a lock every thread
needs, for 11 hours. **The culprit is unknown** -- nothing recorded thread
stacks. Ruled out: the vLLM server; loading response #90's code
(reproduced locally, 1 ms); a full stdout pipe (v4's capture was truncated
at 0.39 MB yet its driver finished; v3 captured 3.96 MB, v5 froze at 2.38).

Fixes to the harness, not the agent: `faulthandler` dumps every thread's
stack to `driver_stacks.txt` every 10 minutes (a C watchdog; needs no
GIL); the driver writes to a file rather than the notebook's stdout; the
notebook heartbeats every minute and kills the driver at 200 minutes
(clean runs take ~27), so a hang costs at most ~3.5 GPU-hours and the
evidence file flushed up to the kill is still collected. Both paths
tested locally: a sleeping driver was killed at its limit; the real
selftest completed normally with the stacks file written.

The pre-registration above stands unchanged for the rerun (v6).

### Result (kernel v6, rerun of v5, 2026-09-27): the planner works; the goal is a guess

Clean: 1,927 s, no hang, all engine checks OK, 0 errors, 326 LLM
completions. Games with a model: dc22, ls20, lp85.

| pre-registered measure | value |
|---|---:|
| **decisions played, games whose simulator distinguishes actions** | **194 / 196 (99%)** -- dc22 114/114, ls20 80/82 |
| lp85 (simulator never distinguishes actions) | 45 / 45 stalled, as predicted |
| predict passes rejected on goal_hint | 1 of 8 |
| levels completed | **0** |

Prediction (>= 90%) met; falsifier (< 60%) not triggered. For the first
time in this arm, the model's own plan chose the moves -- ~100 of 121 in
two games -- and both games' objectives read the playfield (the gate and
the prompt landed).

**And the agent achieved what its model told it to.** ls20's goal_hint
climbed -11.3 -> -6.7 -> -0.2 -> 0.0 (its maximum) over the four quarters
of the game, and stayed there. Its code states the target outright:
*"Target is the 5x5 area at x=34, y=40 (where the object started)"*.
With no win anywhere in its transcript, the model guessed that the goal
is to return the object to its start; the planner did exactly that, and
nothing happened. dc22's objective is `1000 * counter - distance to the
nearest item of colour 8/9/11/13` -- the counter term is action-invariant,
so it walks the player toward items; no level either.

**The wall is now win-condition inference, and it is a clean one**: search,
legality, and objective plumbing all work, so what remains is that the
model has never seen a win and is guessing. That also suggests the next
lever, which uses information the agent already produces: **reaching the
maximum of goal_hint without a level-up falsifies that goal.** The agent
can detect it (the objective plateaus at a value no move improves, with
no levels_delta) and trigger a redraft that says so -- "you drove the
object to (34,40) and nothing happened; the goal is something else" --
turning each wrong guess into evidence rather than a place to park.

Watchdog: `driver_stacks.txt` was written; not needed this time.

## 14. Goal falsification (2026-09-27)

v6 showed the agent reaching its model's goal and nothing happening. So:
reaching a goal's peak without a level-up falsifies it (engine
`stage7-goal-hint`, 274 tests).

A move is "at peak" when no legal move improves goal_hint over staying put
**with the edge band held equal** -- without that, dc22's
`1000 * counter - distance` objective looks like it improves forever (996
-> 60998 over v6), which hid that its player parked beside an item from
frame ~7 (peak on 115/121 frames). Peaks on 3 of the last 6 consulted
moves falsify the goal; the board goes on the transcript, and the replay
gate then rejects any goal_hint under which it is still a peak. The agent
asks for a revision with the evidence; if that fails it explores instead
of following the falsified plan; at most 3 revisions per level.

Replayed on v6's real frames (final model versions, so approximate for
early frames): fires at frame 9 on dc22, frame 35 on ls20 -- the latter at
a local peak (-15, held 9 frames) before the true peak (0) was reached.

### Pre-registered before kernel v7

Same configuration as v6.

- **mechanism** -- in games with a model that distinguishes actions: goals
  falsified, revisions accepted, and whether a revised goal leads the
  agent somewhere new (the falsified boards are not revisited as peaks).
  *Prediction:* >= 1 falsification in each of dc22 and ls20 if they get
  models again. **Falsifier for the detector:** a game with a model that
  sits at a peak on most consulted moves (as both did in v6) and is never
  falsified.
- **revision quality** -- how many revisions pass the gate (a different
  goal that also responds to the playfield), and how many fail.
- **outcome** -- levels completed. Honest prior: still likely 0 --
  falsification says the goal is wrong, not what the right goal is; the
  model still has never seen a win. Any level at all would be the first in
  this arm.

### Result (kernel v7, 2026-09-27 18:07-18:34 UTC): the detector works; revision is the bottleneck; and a long-standing state bug

Clean: 1,636 s, all engine checks OK, 0 errors, 283 LLM completions.

| | v6 | v7 |
|---|---:|---:|
| games with a model | 3 | 4 (dc22, ls20, lp85, m0r0) |
| goals falsified | -- | **4** (dc22 x2, ls20, m0r0) |
| revisions accepted / failed | -- | **1 / 3** |
| predict passes rejected for re-proposing a falsified goal | -- | 8 (all dc22) |
| planner decisions played | 196 | **27** |
| levels completed | 0 | **0** |

**Detector: prediction met** -- falsified in both dc22 and ls20 (and m0r0);
the falsifier (a game parked at a peak and never falsified) did not fire.

**Revision is the bottleneck.** Asked for a different goal, the model
mostly rewrote the one it was told had failed: 8 dc22 candidates were
rejected by the gate for making the falsified board a peak again (the
guard worked). The one accepted revision is a genuinely new hypothesis --
*"Target: the 8x8 area of 8s in the right panel"*. When revision fails the
agent explores at random instead of following a falsified plan, so planner
decisions played fell from 196 to 27: most of v7 was random play, as the
design implies. That was predictable and I did not predict it.

**ls20 never got to revise: its coder budget was gone** -- 7 "repairs",
every one returning `ok` with **0 LLM attempts**, each charged a budget
unit. A repair returns with 0 attempts only when the current source,
freshly loaded, already reproduces the whole transcript: the divergence
that triggered it was not real. Cause, verified: ls20's model keeps hidden
state (`self.counter_x += 1` in `predict`), and **the planner searches on
the installed instance itself** -- one `plan()` call moved `counter_x` from
13 to 34, writing 21 imagined moves into the real simulator. Whenever the
counter matters, the installed instance mispredicts the real move, the
agent "repairs", and a fresh instance replays perfectly. This has been
live in every run of this arm; the replay gate's hypothetical probes and
`peak_escape` do the same to the instance they return.

Fixes indicated, not yet made: plan, probe and peak-test on copies of the
model (`copy.deepcopy`) so only real transitions advance its state; do not
charge coder budget for a repair that made no LLM call; instrument
`revise_goal_hint` rounds in the driver (they are invisible in `rounds`,
which wraps only draft/repair).
