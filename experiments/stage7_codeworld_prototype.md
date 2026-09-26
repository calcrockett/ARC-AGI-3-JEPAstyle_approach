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
