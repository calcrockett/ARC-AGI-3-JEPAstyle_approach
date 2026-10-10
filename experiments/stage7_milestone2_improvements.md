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

## 5. Next harness challenger: lordhansolo's strategy audit [READ + BUILT, 2026-10-09; never run]

**Check run done 2026-10-09: 663.4 tok/s, 1547 tok/req, 39 levels / 9.25 public-25, 0 errors, but only 1 audit fired in 25 games (threshold 56K tokens), 0 audited levels cleared; not scheduled.**

**Variant built: `--strategy-audit` (token `audit`)** on `--turbo-lossless --prio-tail` ->
`calamitychasm/arc3-m2-turbo-lossless-tail-audit` (`kaggle_submission_m2_turbo_lossless_tail_audit/`). Module
`kaggle_submission_milestone2_fork/strategy_audit/strategy_audit.py`, provenance and the exact text diffs in its
`NOTICE.md`; tests `tests/test_m2_strategy_audit.py` (unit, the real patched ToolAgent, the committed kernel).

### 5.1 What lordhansolo's audit is (recovered verbatim)

Source: `lordhansolo/taaf-kaggle-source` (mirror github.com/tonghuikang/daniel-franzen-arc-agi-3 @ f472820),
`inference/agent/prompts.py` `STRATEGY_AUDIT_PROMPT` (~2.6 KB, ~600 tokens) and `inference/framework/solver.py`
`_should_audit_strategy`. A per-game timer starts at each level start; once 25% of the game's initial runtime limit
(his cap 3,918 s -> ~980 s) has passed on the same level, the next analyzer turn carries the prompt; the timer
restarts at that turn and at every level clear. The text tells the model to first check whether recent observations
support its plan and, if so, to continue without an experiment; only if not, to separate observed facts from
assumptions, pick one uncertain assumption, check existing transitions, and run ONE short discriminating experiment
with predictions written down and each action's result inspected before the next. It says explicitly: do not reset
or abandon a plan because of the audit.

Our port changes the clock to **generated tokens on the level** (the gate time-shares ~110 games, so wall time on a
level is mostly time parked): due every 56,000 tokens on one level (25% of ~228K expected tokens per game on the
turbo-lossless stack). Appended to the turn opener only (no system-prompt / history edit, so no cache cost); a turn
that generated nothing does not count and the audit is re-sent. Three references to his harness (the python tool's
`plan` argument, saved modules with tests) are reworded for ours; everything else is verbatim.

### 5.2 Evidence (ranked; none of it is a hidden-set ablation)

| candidate | evidence | type | verdict |
|---|---|---|---|
| strategy audit (lordhansolo) | part of a Milestone-2 winner (public LB 23.84, one draw, below Franzen's 27.89 base, different serving stack); never ablated publicly | bundle component, hidden set, n=1 | **built** (default per brief; clear mechanism, low disruption) |
| added advice text hurts (rellik13 / sirikilohit WRITEUP) | per-prompt text 6.42 -> 4.16; level-start advice block 13.40 -> 9.98 | hidden set, one draw each, at a ~37K context | the main risk: our context is 128K (crowding ~0.5%), the audit fires only on long levels, but "advice made it doubt rules it had right" applies |
| remove misleading text (rellik13: UNDO name, "Level restarted", budget-bar line) | 13.40 -> 14.49 bundled with a scheduler change | hidden set, n=1 | **already in our base** (Franzen renamed UNDO, rewrote the game-over line, describes the bar as budget) |
| solved-level memory (rellik13 M85) | +2.8 hidden on his stack; +4.8 on ours (n=4 vs n=6 control) | hidden set | **already the incumbent** |
| longer retained history (rellik13 14.49 -> 22.53) | hidden set, n=1, KV-bound | serving | our KV lead (`stage7_m2_speed.md`), not a harness change |
| Franzen's own negatives | stronger step-verification hint, summaries, resume prompts, extra guards: no clear gain | public-25 repeats | weakens the prior for any prompt nudge |
| Hivemind "unanchor" level-start prompt | +3% levels, p = 0.82 | public-25, 4 passes | null |
| JustAdev742 mid-level context reset (ours-07) | stuck-level solve rate 32% vs ~61% expected; bundle 39.7/45.0 vs 49.4 | public-25, mechanism count | a stuck-level intervention that REPLACES context hurts; the audit adds a view and keeps the context |

Other surveyed repos (adsdemaybe = Franzen's write-up, lokeshjasrotia, ppx16, Beiciccc, AREx) carry no hidden-set
evidence for a harness change beyond the rows above. Prior on the audit's effect: small, sign unknown (perhaps
-1..+2 points); Hivemind's decomposition (97% of lost points are unfinished levels; unfinished levels end after a
median 14 turns, i.e. they run out of time) is where it would act. It is **not resolvable by draws** (sd 2.2), so it
gets no slots of its own; it only rides along if its check run is clean and the bundle's arm is chosen.

### 5.3 Check run: pass / kill (25 public games x 25 min, the turbo-lossless-tail shape)

Pass, all of:
- markers: turbo-lossless-tail's plus `STRATEGY_AUDIT installed` (prints `threshold_tokens 56000`); the cell asserts
  it wrapped `_build_user_prompt` after level memory and that `ARC3_YIELD_RESUME_PROMPT` is unset.
- `strategy_audit_summary.json`: `errors` 0; `audits_confirmed` >= 1 (the clock fires: a check-run game generates
  ~50-90K tokens, so expect a handful); `audits_retracted` <= `audits_sent` / 4; `max_audits_one_level` consistent
  with tokens / 56K.
- outcome guards (registered, not only mechanism): `voluntary_resets_after_audit` <= 1 per 4 confirmed audits
  (the text says not to reset; JustAdev742's fresh-start finding); `audited_levels_cleared` reported (outcome measure:
  read it against the 25-min base, where ~1/3 of levels still open at that depth clear later, without a pass bar at
  this n).
- throughput and length vs turbo-lossless-tail's check run (657.4 gen tok/s, 1579 tokens/request): gen tok/s within -5%, output tokens/request within
  +-10%, 0 exact repeated assistant turns -- the audit adds ~600 prompt tokens once per 56K generated.
- level_memory / history_cache / timeout_fix counters as turbo-lossless-tail.

Kill: any of `errors` > 0, `audits_confirmed` == 0 with a level past 56K tokens, resets after audits above the bar,
a level-memory or histcache regression, or any guard above tripping. Kill date for a slot decision: the audit only
competes for slots after arm B (`arc3-m2-turbo-lossless-tail`) has n >= 3; until then it is check-run only.

## 6. Static reasoning effort "medium" [READ + BUILT, 2026-10-10; check run done, KILL]

**Variant built: `--reasoning-effort medium` (token `re-medium`)** on `--turbo-lossless --prio-tail` (arm B) ->
`calamitychasm/arc3-m2-turbo-lossless-tail-re-medium` (`kaggle_submission_m2_turbo_lossless_tail_re_medium/`).
Module `kaggle_submission_milestone2_fork/reasoning_effort/reasoning_effort.py` (ours; provenance in its `NOTICE.md`),
tests `tests/test_m2_reasoning_effort.py`. Arm B is lossless, so B's check run is the control for exactly one change.

### 6.1 Mechanism (read in the code and the template)

- The served Qwen3.8-Flash-Next `chat_template.jinja` resolves `reasoning_effort|default('xhigh')` (accepted: xhigh,
  medium, low; anything else raises). At xhigh it prepends *"Reasoning effort is set to xhigh. Please think carefully
  through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness,
  consistency, and clarity in the final answer."* to the system prompt; medium adds nothing; low adds "Keep your
  thinking brief and focused ...". Source: JustAdev742's verbatim fixture (sha256 `c3cf9e34...`; their
  `mtp-drafter-finetune.md` records the same hash for the template served on Franzen's Intel W4A16 stack) and their
  lesson 0021. Our launcher passes `--chat-template MODEL_DIR/chat_template.jinja` to SGLang.
- Franzen's harness (`tool_agent.py` `ToolAgent._harness_template_kwargs`, from the upstream harness patch) sends only
  `preserve_thinking`, plus `reasoning_effort` from the truncation ladder `ARC3_REASONING_EFFORT_LADDER`, which the
  incumbent leaves unset. **So every incumbent request runs at xhigh.**
- The variant wraps `_harness_template_kwargs` at runtime (own cell after the install cell) and adds
  `reasoning_effort="medium"` unless the ladder set a value. That method feeds both the posted payload
  (`_chat_completion` merges it into `chat_template_kwargs`) and the request log, so `<game>_requests.jsonl` records
  what was sent. Same behaviour as JustAdev742's `ours-09-reasoning-effort.patch` (exp-082) and juliancamilovilla's
  E1 (`ARC3_STATIC_REASONING_EFFORT`); our own code.
- Three checks that it is not a silent no-op, each a gate marker: `REASONING_EFFORT medium installed` (prints the
  kwargs dict the harness now builds), `REASONING_EFFORT_TEMPLATE supports=True` (renders the served template with
  jinja2 at default and at medium; the xhigh line must disappear; logs its sha256), `REASONING_EFFORT_SERVER
  received=True` (a daemon thread waits for /health, then sends two `max_tokens=1` requests without / with the kwarg;
  the server applied it iff `usage.prompt_tokens` is lower with it, expected delta ~40 tokens; re-probed after the run
  if the boot probe failed).

### 6.2 Evidence (none of it a hidden-set ablation)

| source | conditions | result | weight |
|---|---|---|---|
| juliancamilovilla E1 (`arc-agi3-e1-med-g15`), tabulated in JustAdev742 `docs/research/beat-tufa/intel-oct10.md` s2.3 | Franzen's unchanged stack (Intel W4A16, 10 streams), hard 15 at 37.5 min on 10 slots (~25 slot-min/game), one run per arm, runs 6.7 h apart | levels 17 -> 32 (score 5.37 -> 16.66); 9 games up, 1 down (sign test p ~0.02); tokens/action 1,154 -> 876; gen tok/s 541 -> 593 | strongest, but n=1 per arm, short budget |
| same, easy 10 (`arc-agi3-e1-med`) | 25 min/game, 10 slots | levels 47.5 +/- 1.0 (4 base runs, 2 authors) -> 41 (score 49.1 -> 38.01); tr87 lost level 1 | a real cost on games that need long planning |
| request logs ka59, tn36 (same runs) | | mean completion -25..-32%; p90 5.4k -> 3.4k and 4.8k -> 3.4k; median unchanged; share of tokens in completions >= 6k 28% -> 17%, 41% -> 0% | mechanism: medium cuts the long-deliberation tail |
| JustAdev742 exp-037 (Sep 23) | older TAAF anim harness, full length | 6.70 / 37 levels vs 7.86 / 38; +35% requests, -35% tokens/request | null at full length on a weaker harness |
| JustAdev742 exp-082 (pushed 2026-10-10 01:43 UTC) | their D' + REAP-448 + 14 streams + acceptance 0.5, 25 games x 121 min | **pending** at the time of writing; their rule: hard-15 >= 60 and score >= 47.3 -> repeat; <= 50 -> drop | the full-length test on our serving stack; read it before spending a slot |
| leaderboard | | no hidden-set draw of any medium-effort notebook yet | |

Also learned in this read: JustAdev742's exp-074t (REAP-448 + acceptance 0.5, submission 56980485) drew **27.97**
on the hidden set (their status.md, 2026-10-10 00:15 UTC). By CLAUDE.md's rule (<= ~28) that is a reason to prefer
arm B over arm A; n=1, so weak.

Prior: the hidden set probably behaves like the hard games (most hidden games stay near level 1), which is why the
sign could be positive. But julian's runs gave each game ~25 slot-minutes, while the hidden rerun gives ~70
(532 min x 14 slots / ~110 games), the regime of a full-length public-25 run, where exp-037 saw nothing and exp-082
is the pending test. Our check run (25 games x 25 min on 14 slots) is again the short regime, so a check-run gain
over-predicts the hidden-set gain. Expected effect: -1..+3 points (JustAdev742's estimate), sign unknown; not
resolvable by one draw.

### 6.3 Check run: pass / kill (25 public games x 25 min on 14 streams, arm B's shape)

Control: arm B's check run (2026-10-09): **657.4 gen tok/s, 1,579 output tokens/request, 36 levels, mean_score
8.17**; read B's per-game levels (hard 15 vs easy 10) and its `request_log` long-tail share from B's artifact before
judging (not recorded in this repo).

Mechanism (each must hold, or the run is a no-op / broken -- kill):
- markers: B's plus the three above; `REASONING_EFFORT_TEMPLATE` sha256 logged (expect `c3cf9e34...`);
  `REASONING_EFFORT_SERVER` delta > 0.
- `reasoning_effort_summary.json`: `errors` 0, `ladder_kept` 0, `set` == `calls`.
- `request_log.reasoning_effort_sent`: every response "medium" (no "default").
- **output tokens/request falls**: <= 1,500 (B -5%; expected about -25%, i.e. ~1,200), `completion_tokens_p90` and
  `completion_token_share_long` (tokens in completions >= 6k) below B's.
- serving unchanged: gen tok/s >= 624.5 (B -5%); retractions <= 2x B's; 0 long repeated assistant turns;
  level_memory / history_cache / timeout_fix counters as B.

Outcome (registered now, before the data; one public-25 run has SE ~2.5 points, so the bar is set at the size of
julian's effect, not at "any gain"):
- **Advance** (candidate for slots, see below): total levels >= 42 (B +6) **and** hard-15 levels >= B's hard-15 + 6
  **and** mean_score >= 8.17.
- **Kill**: total levels <= 33, or hard-15 levels not above B's, or mean_score < 5.7 (B - 2.5).
- Otherwise: one repeat check run before deciding.
- Also read, not gated: actions per solved level (RHAE pays for efficiency; medium acts more and thinks less),
  tokens per action, and which easy games lose levels (tr87 lost level 1 in julian's run).
- External kill: if JustAdev742's exp-082 (same mechanism, our serving stack, full length) comes back <= 50 hard-15
  levels (their own drop rule), do not give this variant slots even if the check run advances.

Slots: not in the 10-10..10-17 explore schedule (A/B only). If it advances and exp-082 is not a drop, it competes in
the exploit phase as a third arm against B (it is B plus one change), alternating with B until n >= 3; the drop rule
of the schedule (mean more than 2.2 below the incumbent after 4 draws) applies.

### 6.4 Check-run result (2026-10-10, kaggle-ops `vllm-trem-read-1`, run 38027035237): KILL

| | arm B (control) | re-medium |
|---|---|---|
| gen tok/s | 657.4 | **624.49** (bar 624.5: missed by 0.01) |
| completion tokens/request | 1579 | **1469** (-7%; bar <= 1500; expected -25%) |
| p90 completion / share >= 6k | not recorded | 3500 / 0.112 |
| total levels | 36 | **34** (advance >= 42; kill <= 33) |
| hard-15 / easy-10 levels | 15 / 21 | **14 / 20** |
| mean_score | 8.17 | **7.23** (kill < 5.7) |

Mechanism was real, not a no-op: all three markers, template sha `c3cf9e34...`, server prompt delta 38 tokens,
`set`==`calls`==2356, `errors` 0, every logged request `medium`, 0 tracebacks, 25/25 gave_up, 2 retractions, 0 repeated
turns. Outcome by the registered rule: **KILL (hard-15 14 not above B's 15)**, and advance is far out of reach (34 < 42).
The expected token cut did not appear (-7% not -25%), consistent with the 25-min short regime and one-run SE ~2.5; this
is not evidence against medium at full length (exp-082 pending), but no slots. Watch games: vc33 3, tn36 2, tr87 0.

## 7. Artificial Agency Lab route (2026-10-10) [READ (second-hand) + SIM; nothing built]

Question: Artificial Agency Lab (team `artificialagencylab.com`, Kaggle `richardcsaky` + `cmechevalier`; LB 38.62,
Oct 7 22:01; rank 9 on Oct 9) reached ~38.6 with a documented serving-heavy route. What did their scored setup do, and
what of it ports to arm B (`arc3-m2-turbo-lossless-tail`)?

### 7.1 Sources and what could be read

- **Primary artifacts were not readable from here.** The cloud box's egress proxy rejects kaggle.com (CONNECT 403;
  WebFetch: DNS failure), artificialagencylab.com is rejected too, no public GitHub repo was found (github.com/richardcsaky
  404; web search finds nothing), and kaggle-ops `pull_kernel` was not used (request.json left untouched).
- **Every configuration claim below is second-hand**, from JustAdev742 `docs/research/beat-tufa/intel-oct8.md` §1.3
  (marked there "verified: I read the configs; I did not run anything"), which read `configs/comparison.json` of the
  public dataset `richardcsaky/arc-arena-hero-v2-blocks2-p50-int3-runtime` (variant "hero-v2", dated 29 Sep), plus the
  sibling datasets `arc-expert-gptq-arc-v1` (64 GB) and `arc-expert-int3-g16-v1` (49 GB). Licence field: "other".
- **Mapping to a score is inference.** The config predates their climb 20.17 (10-02) -> 28.12 (10-03) -> 37.94 (10-04)
  -> 38.62 (10-07) (intel-oct8 §1.1, monitor history); no submission id is tied to it, and the team's 48 submissions
  could have used later variants. Earlier public kernels exist and were not read: `richardcsaky/arc-agi-3-milestone-2-submission`
  (v20) and `cmechevalier/face-of-agi-arc-agi-3-rtx6000`, the team then at 20.00 (ppx16 ledger leaderboard CSVs 09-29..10-01).
- intel-oct8 is internally inconsistent on the model: §1.3 lists int3 g16 routed experts "re-rounded from GPTQ-ARC",
  §2.3 says they "serve the base NVFP4 model" (Swift profile kept, unused). Likely an NVFP4 base with the int3 expert
  overlay; unresolved.

### 7.2 Their configuration vs arm B

| item | Artificial Agency Lab (hero-v2; evidence: doc claim of read config) | arm B (read in our notebook/builder) | portable? |
|---|---|---|---|
| serving | custom "arena" runtime (vLLM-style, `max_num_seqs` 80), 257 MB | Pennyroyal SGLang fork | no (whole-stack swap; our vLLM attempt vllm-s14 was killed by its prefix-cache probe) |
| streams / window | **76 games at 98,304 tokens** | 14 at `CTX` 136K, analyzer window 128K, drain 58K (contexts ~58-118K; KV peak 0.92 of 1.48M) | width only via the host tier (below) |
| KV dtype | **int4** | fp8_e4m3 | **no**: our QSA backend has no FP4/int4 read path (CLAUDE.md, 2026-10-09 source audit) |
| experts | **50% pruned** (`prune50-layer`) + **int3 g16 GPTQ calibrated on ARC traffic** | W4A16 AutoRound, REAP-448 (12.5% pruned) | no: our 448 list is a router-row match to a third-party pruned checkpoint (`turbo/reap448_kept_experts.meta.json`), there is no ranking for 256/384; their overlay is "other"-licensed (rules need a permissive licence) |
| MTP | **off** (sensible when compute-bound at 76 wide) | 3-step MTP, accept median 2.66 (the source of B's speed) | no reason to |
| other memory | PLE + input embedding in host memory (UVA); int8 dense kernels (lm_head, GDN in-proj, QSA qkv/out) | none | no (kernel work in their runtime) |
| admission | "exact KV gate", never preempts | bounded admission, 2 retractions per check run | already equivalent |
| schedule | two wall-clock blocks of 15,250 s: 76 games then 34 | priority gate, fresh-first, slots change hands at trims/wins, tail fade | no (the gate is the stronger form) |
| **retire rule** | a game with no level cleared retires at **90,000 completion tokens**; allowance **90,000 x (levels cleared + 1)** | none (`max_generated_tokens_per_game` exists upstream, unset); the gate's C factor rations stuck games instead | **yes**, runtime patch (7.3) |
| harness | Duck port ("duck-transfer"), reasoning kept verbatim, one model-written summary at the context cap, nine fixes, `run_plan` helper | milestone-2 harness + level memory + tail | no (different harness; upstream has unused `ARC3_SUMMARY_*` knobs, unmeasured) |
| unscored code | Gittins / hazard-index / Thompson schedulers, adaptive thinking, trained draft head, Holo4-27B probe (2.10M KV) | -- | nothing scored to port |

Two readings of the allowance cannot be told apart second-hand: (a) cumulative game tokens >= 90K x (L+1), with L =
levels cleared; (b) tokens on the current level >= 90K x (L+1). Both coincide at L = 0. With ~190K generated tokens
per game for B in the rerun (657 tok/s x 532 min / 110), (a) binds in practice only at L = 0 and L = 1.

### 7.3 Is the retire rule worth porting? Two models disagree

Mechanism on our stack: override `_HarnessGameSession.token_limit_reached` (upstream already calls it from `should_stop`
at every loop boundary, reading `_analyzer_reported_tokens(self.analyzer) - self.game_token_baseline`) with a
level-scaled limit from `self.game.current_state.levels_completed`; `play()`'s `finally` releases the gate slot. About 15
lines in an install cell, the `--prio-tail` pattern; no serving or prompt change.

**Our replay** (scratch extension of `scripts/sim_m2_priority_gate.py`, not committed: base = the shipped tail gate,
14 slots x 47 tok/s = B's 657; a game that hits its limit leaves every queue for good; 120 draws per world). RHAE
change vs B's gate, LB points:

| rule | 8 hidden-like worlds (26-31) | `unsolv.20` world | public-like (59) |
|---|---|---|---|
| (a) cumulative 90K x (L+1) | **+0.25..+0.79** (+0.9..+3.0%) | **-0.41** | +0.12 |
| (b) per-level 90K x (L+1) | +0.13..+0.57 | -0.12 | +0.25 |
| L = 0 only, 90K | +0.10..+0.48 | -0.09 | +0.16 |
| (a) at 60K / 75K / 110K / 120K | 60K -3.2..+2.4 (unstable); 75K up to +2.3 but -1.3 in unsolv.20; 110K +0.04..+0.28; 120K <= +0.14 | | |
| flat 300K per game (upstream knob) | -0.2..-0.9 | -1.5 | -2.1 |

At arm A's 772 tok/s, (a) is +0.84..+1.54 in 6 worlds and -1.07 in unsolv.20 (60 draws). The gain sits on a cliff in
the threshold relative to the assumed per-level cost (median 22K x 2-2.5 here), and it turns negative where compute is
not the constraint (unsolv.20: cheap levels, so retired games forfeit tokens nobody needs).

**JustAdev742's hazard-calibrated replay says the opposite for time-based give-ups** (`docs/research/beat-tufa/time-allocation.md`
§3-4, fitted on three full-length D′ runs, 14 slots): measured P(solved) 0.85 by 30 active minutes, hazard still
0.027/min at 30-45 min; "give up after X active minutes on a level" is -6.9 [-13.5, -1.7] at 30 min, -1.7 at 40, never
positive at <= 40 min in any of 11 variants (harder worlds included), ~0 at 50-60 min. `hopeless-signal.md`: level 1
is hopeless (> 60 active min) only 2.3% of the time. 90K tokens is ~32 active minutes at B's 47 tok/s per slot. Their
rule ends every level, AAL's binds mainly on levels 1-2, so neither model tests AAL's rule exactly; ours is calibrated to
the LB level, theirs to measured hazards, and theirs is the better evidence on the question that decides the sign
(how often a level unsolved at ~30 active minutes is still solved).

Why AAL needed it and we may not: at 76 wide with a two-block wall-clock schedule and no priority gate, a token cap is
their only rationing of stuck games. Our gate already rations them (C halves after ~62K tokens on a level; per
time-allocation §4.3, a D′ game stuck 50 active minutes gets only ~19K more tokens).

Not measurable either way: a 25-game x 25-min check run generates ~40K tokens per game (657 tok/s x ~25 min / 25), so a
90K limit never fires, and JustAdev742 §4.5 shows public-25 replays flip the sign of give-up rules. A hidden draw (sd
2.2-3.9) cannot resolve +-0.5.

### 7.4 Recommendation

**Nothing in AAL's width/compression stack ports**: int4 KV (no QSA read path), 50% pruning (no expert ranking; needs a
GPU calibration pass, JustAdev742's plan item 3), the int3 GPTQ-ARC overlay and the runtime (licence "other", custom
kernels), MTP-off (only pays at their width). The only lever on our stack that buys AAL-style width is the host KV tier.

1. **Build: `arc3-m2-turbo-lossless-tail-hic16`** (`python scripts/_build_m2_level_memory_kernel.py --turbo-lossless
   --prio-tail --hicache-gb --streams 16`): arm B + 32 GB system-RAM KV tier + 16 streams (Mamba 96). Mechanism and
   evidence: on arm A the same change measured 831.6 vs 772.4 gen tok/s (+7.7%), KV peak 0.99, MemAvailable min 29.9 GiB,
   2 retractions, tokens/request -2.4%; B is now the preferred arm (JustAdev742's hidden draw 27.97). Expected: ~+7-8%
   generated tokens, all else unchanged; the score effect is not measurable by draws, so adoption rides on throughput
   (CLAUDE.md rule: same mechanism, better measured throughput may take an arm's slots).
   - Pass: gen tok/s >= **710** (+8% over B's 657.4); `HICACHE_TIER attached hicache_attached=True` (post-run);
     `STREAMS max_running_requests=16`; MemAvailable >= 10 GiB in every `[sys] RAM` line; no `Not enough host memory`;
     retractions <= 4; tokens/request 1421-1737 (B 1579 +-10%); 0 repeated assistant turns; 0 tracebacks; 25/25
     gave_up; histcache / timeout_fix / level_memory counters clean.
   - Kill: gen tok/s < 657.4, `hicache_attached` missing, a host-memory error, or catastrophe (total levels <= 29 or
     hard-15 <= 9; B 36 / 15).
   - 657.4-710: near-pass, rerun once before any adoption (as hic16 on A).
2. **Optional, no slots: B + retire at L = 0** (`RETIRE0 installed R=90000`; the least aggressive reading, ~+0.1..+0.5
   in our replay, likely <= 0 by JustAdev742's hazards). Build only if a hazard-calibrated rerun model with an "L = 0
   only, 90K cumulative" row shows a positive point estimate. Check run is mechanism-only, with R scaled to the check
   run's budget (R_check = 20K) so it fires: marker present, `retire_summary.json` errors 0 and retired >= 1, every
   retired game ends `gave_up` with 0 levels and its slot is re-admitted, 0 tracebacks, gen tok/s within -5% of 657.4,
   tokens/request within +-10% of 1579; the check run says nothing about the score.

Do not bundle the retire rule into arms A/B: its sign is unknown and it would contaminate the arm comparison.
