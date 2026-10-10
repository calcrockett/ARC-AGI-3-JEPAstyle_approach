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

### 6.5 exp-082 result (2026-10-10): the reasoning-effort line is closed

JustAdev742's exp-082 (reasoning effort medium on every request, full length, our serving stack: D' + REAP-448 + 14
streams + acceptance 0.5, 25 games x 121 min; their `docs/research_log.md` 2026-10-10 04:05 UTC, lesson 0040):
**hard-15 29 levels** against 39-58 in their other runs, **easy-10 55** against 56-66, **84 levels in total, the
worst of their 9 runs**. Read second-hand; n=1. Section 6.3's external kill rule (exp-082 hard-15 <= 50) fires, on
top of our own check-run KILL (6.4). **The reasoning-effort line is closed**, including a stuck-level-only variant
(xhigh until a level has used T generated tokens, then medium; proposed 2026-10-10, not built for this reason): the
full-length test shows medium hurting the hard games most, which is where that variant would apply it.

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

## 8. D′ slot priority (shiiin9 / AFF AI CLUB) on arm B [READ + SIM + BUILT, 2026-10-10; never run]

Public notebook `shiiin9/affectify-arc-31-54-in-a-single-sub` = Franzen's Milestone-2 notebook with one change: the
scheduler's slot priority "D′", patched in the cell before `await bm.run(...)`. Claimed LB 31.54 (n=1).

### 8.1 Code and licence

- **Exact code obtained** without a kaggle-ops pull: JustAdev742 vendored the notebook unmodified
  (`kaggle/dprime/affectify-arc-31-54-in-a-single-sub.ipynb`, pulled 2026-10-07, sha256 `f649d005…`). The module is the
  string `_FP_SOURCE` in cell 21; we extracted it verbatim to `kaggle_submission_milestone2_fork/dprime/ours_form_priority.py`
  (sha256 `6456efdd…`, pinned by `tests/test_m2_dprime.py`, which re-extracts it from JustAdev742's copy when present).
  Today's kaggle-ops pull (`pk-1010a`) shows the same 26-cell notebook and the same description.
- **Licence**: public Kaggle notebooks are Apache-2.0 (Kaggle's default for public code; the kernel metadata has no other
  licence field). Franzen's harness is Apache-2.0. Compatible with our rules. Provenance in `dprime/NOTICE.md`.
- **What `install_d(tool_agent, solver)` does** (read in the module; the published formula matches the code, test
  `test_formula_matches_the_published_description`):
  - `priority = A·M·C + B·φ`, ×1000 to an int. A = (1 + 0.5(ℓ−1))·norm(N)·(300/(300+a))^2.5, norm = 55/(N(N+1)/2) with
    N clamped to 6..10; M = clip((30,000/p)^0.4, 0.25, 4), p = the game's mean generated tokens per cleared level (1
    before the first clear); C = 0.1·max(0.1, 1−a/115) + 0.9·max(0.1, 1−t/T), T = 225,000·clip(p/30,000, 0.5, 2)^0.5;
    B = 16 / 14 / 10 / 0 for ≥3 / 2 / 1 / 0 levels remaining; φ = tail fade over the last **40%** of the run.
  - Replaces `tool_agent.priority_value`, `tool_agent.ProgressPace` (by `FormPace`), sets `ARC3_PRIORITY_PACE=1`,
    `..._REFRESH_QUEUE=1`, `..._TAIL_FADE=1`, `..._TAIL_FADE_FRACTION=0.4` (upstream 0.2).
  - Removes fresh-first: `_PriorityGate.acquire` prices a never-started game by the formula (ℓ = 1, a = t = 0, its
    level count taken in a wrapped `_HarnessGameSession.play`) instead of the 2,000,000 band.
  - Compared with our `tail` fix: D′ has no h-collapse (its efficiency factor halves at ~83 actions, upstream's at h = 25),
    keeps last-level B = 0, and B dominates the value (16 vs A·M·C ≈ 1-5), so the order is "levels remaining" first.

### 8.2 Hidden-set evidence

| config | draws (hidden, public LB) | n | mean |
|---|---|---|---|
| Franzen, unmodified, unselected | jvilladuque 25.01, 25.17, 23.39, 27.15, 27.63, 25.97; ours 24.99; D′ authors' resubmission 27.62 | 8 | **25.87** (sd 1.51) |
| Franzen, selected bests (not used) | 27.89 (Franzen's team best of 90), 31.47 / 34.30 (notebook bests), spark328 30.75, vinicius 28.85 | -- | -- |
| **D′ unchanged** | shiiin9 31.54 (published because of it), JustAdev742 copy **28.87** (pre-registered, unselected) | 2 | **30.20** |
| D′ + REAP-448 + 14 streams + acceptance 0.5 (JustAdev742 exp-074t) | 27.97 (unselected) | 1 | -- |
| D′ forks, notebook bests (not used) | amatlas 29.27 → 30.82, lwq255 29.15; AFF AI CLUB team 33.92 (config unknown) | -- | -- |
| ours: Franzen + level memory (incumbent) | 33.29, 28.18, 31.03, 29.38, 25.21 | 5 | 29.42 |

- D′ (n=2) vs Franzen (n=8): **+4.3**; known-σ test with the copies' σ = 3.93: z = 1.40, one-sided **p = 0.08**; Welch
  t = 3.02, df 1.3, p ≈ 0.08. Adding exp-074t (D′ + serving, n=3): +3.6, z 1.35, p 0.09 (Welch t 3.0, df 3.1, p 0.03).
- Without the published 31.54 (selection): 28.87 vs 25.87 = +3.0, z = 0.72. One unselected draw.
- Same period (Oct 3-9) as the jvilladuque control, so no period confound. **Reading: suggestive, not established**;
  the size is about our own level-memory effect (+3.6 vs the same control), so D′ alone ≈ level memory alone.
- Priors: D′'s authors estimate +1.5 over their own earlier formula (not over Franzen); JustAdev742's hazard-calibrated
  replay puts D′ +1.2 [0.1, 2.6] (14 slots) / +2.2 [0.5, 4.2] (10 slots) public-25 points above Franzen's gate, i.e.
  roughly +0.7 / +1.3 LB points (`docs/research/beat-tufa/time-allocation.md`).

### 8.3 Replay (`scripts/sim_m2_priority_gate.py`, variants `dprime*`)

D′ is executed from the vendored module (its `d_parts`), with its pace tracker, fade 0.4 and fresh-game pricing; the
sim tracks games that never got a slot (`starved`). 120 draws per cell, common random numbers; LB points vs upstream
(base) and vs our shipped `tail` gate. 10 slots × 70 tok/s (upstream) and 14 × 47 (arm B's 657 tok/s):

| world (base 10 / 14) | tail | **D′** | D′ fresh-first | D′ + final B 5 | **D′ − tail** | starved games (D′) |
|---|---|---|---|---|---|---|
| tok2.5 (28.5 / 26.7) | +1.11 / +1.00 | **+2.83 / +2.71** | +2.74 / +2.56 | +2.91 / +2.72 | +1.72 / +1.71 | 3.2 / 4.1 |
| tok2_u.12 (29.9 / 28.2) | +0.79 / +0.77 | +1.75 / +1.72 | +1.70 / +1.68 | +1.75 / +1.73 | +0.96 / +0.95 | 0.9 / 2.0 |
| unsolv.20 (31.0 / 30.6) | +0.20 / +0.22 | +0.54 / +0.49 | +0.55 / +0.50 | +0.55 / +0.50 | +0.34 / +0.26 | 0 / 0 |
| depth1.15 (30.0 / 28.4) | +1.03 / +1.06 | +2.62 / +2.59 | +2.47 / +2.35 | +2.71 / +2.74 | +1.59 / +1.53 | 5.9 / 8.4 |
| finalU3x (30.7 / 29.0) | +0.73 / +0.71 | +1.81 / +1.81 | +1.77 / +1.76 | +1.85 / +1.80 | +1.08 / +1.10 | 2.1 / 3.0 |
| siggame0.3 (28.5 / 26.4) | +0.85 / +1.01 | +2.98 / +3.12 | +2.73 / +2.74 | +2.99 / +3.18 | +2.13 / +2.11 | 4.1 / 5.7 |
| coupled0.7 (29.0 / 27.2) | +0.44 / +0.48 | +1.38 / +1.63 | +1.37 / +1.59 | +1.41 / +1.62 | +0.94 / +1.15 | 0.1 / 0.3 |
| eff1.5 (28.2 / 26.8) | +0.45 / +0.43 | +0.44 / +0.46 | +0.45 / +0.46 | +0.46 / +0.50 | −0.01 / +0.03 | 0 / 0 |
| coupled0.7_eff1.5 (26.5 / 25.4) | +0.25 / +0.14 | +0.22 / +0.16 | +0.23 / +0.17 | +0.25 / +0.18 | −0.03 / +0.03 | 0 / 0 |
| public_like (60.2 / 58.1) | +0.45 / +0.69 | +0.91 / +1.03 | +0.90 / +0.99 | +0.87 / +1.02 | +0.46 / +0.34 | 0 / 0 |

Reading:
- **D′ ≥ our tail fix in every world** (+0.0..+2.1 LB, P(draw > 0) 0.47-0.94), and ≥ upstream everywhere (+0.2..+3.1, i.e.
  +0.6% to +11%). Its edge over tail is largest where tokens are scarce (tok2.5, depth1.15, siggame0.3) and vanishes when
  slow levels also cost actions (eff1.5 worlds), the same pattern as section 4's h effect. The 14-slot / 47 tok/s
  (arm B) numbers match the 10-slot ones: D′'s constants are not tuned to 10 streams in any way the model sees
  (B and the fade are relative, the pace reference is per level, not per slot).
- **Our tail tweak adds nothing on top of D′**: last-level B 0 → 5 or 10 is +0.0..+0.25 (cf. JustAdev742's +0.26 for B = 10),
  inside noise, so the build keeps D′ verbatim (one fewer unmeasured change; the scored 31.54 / 28.87 are verbatim D′).
- **Where the gain comes from**: most of it is the formula (fresh-first D′ keeps ~90% of it). Pricing fresh games buys the
  rest by **never starting 2-9 of 110 games** in token-scarce worlds (their slot time goes to deeper levels of fast games).
  That is a real hidden-set risk if the model's level-value assumptions are wrong; fresh-first D′ is the hedge (zero
  starvation, ~−0.1..−0.4 LB vs D′), not built.
- **Relation to the hidden evidence**: the sim's D′ − upstream (median ~+1.8 at 10 slots in hidden-like worlds) and
  JustAdev742's (+0.7..+1.3 LB) sit below the draws' +3.0..+4.3 but well inside their noise (SE ≈ 3); the draws cannot
  falsify the sim, and the sim says the draw gap is at least half luck. As with tail, the replay is a model, and public-25
  replays get the sign of scheduler changes wrong (JustAdev742 §4.5).
- **Check-run shape** (25 games × 25 min, 14 × 47 tok/s): D′ starves ~5-6 of the 11 waiting games in the sim (upstream
  0.05), because the first trims come at ~22 min and D′ re-admits trimmed games over fresh ones. The check run's public-25
  numbers are therefore NOT comparable to B's (fewer games played); only mechanism and throughput are read from it.

### 8.4 Compatibility

- **Level memory, history cache, timeout fix**: orthogonal. D′ touches `tool_agent.priority_value`, `ProgressPace`,
  `_PriorityGate.acquire`, `_HarnessGameSession.play` and four `ARC3_PRIORITY_*` env vars (read at call time); level
  memory wraps `ToolAgent._build_user_prompt / _trim_messages_for_context / _ensure_session`; the history cache wraps
  `play` again (composes: it calls the wrapped original). Tested against the real patched harness
  (`test_install_replaces_priority_pace_and_fresh_queueing`): the gate's `_snapshot_priority` returns D′'s value, a fresh
  `acquire` is priced, pace records mean tokens per level.
- **14 streams**: no slot-count dependence in the formula (above); JustAdev742 ran D′ at 14 streams for three 121-min
  public-25 runs and the exp-074t hidden draw without a mechanism problem.
- **Not with `tail`**: both set the priority; the builder refuses the combination.

### 8.5 Built: `calamitychasm/arc3-m2-turbo-lossless-dprime` (never pushed)

`python scripts/_build_m2_level_memory_kernel.py --turbo-lossless --dprime` -> `kaggle_submission_m2_turbo_lossless_dprime/`.
Arm B's stack (turbo-lossless: histcache + timeout fix + REAP-448 + ARC hotmap + 14 streams + level memory) with D′ in place
of PRIORITY_TAIL, plus the 2026-10-10 standing rule's boot grace (`ARC3_HTTP_RETRY_INITIAL_SECONDS` 900 -> 2400; only this new
kernel). Versus `arc3-m2-turbo-lossless` it differs in exactly four cells (header, setup grace line, install cell + D′ lines,
run-cell counter dump; `test_differs_from_turbo_lossless_only_where_intended`). Our only addition to D′ is a call/error counter
around `d_priority` (falls back to upstream's value on an exception) -> `dprime_summary.json` {calls, errors, last_error,
fresh_replaced}. Markers: arm B's minus `PRIORITY_TAIL installed`, plus `PRIORITY_DPRIME installed` (the line carries D′'s own
`#OURS_FORM ok version=d_prime ...`); the run cell also prints `#OURS_FRESH replaced=N`.

**Check run (pre-registered; 25 public games × 25 min, arm B's shape). Mechanism + catastrophe only:**
- Pass: every marker incl. `PRIORITY_DPRIME installed` and `INPUT_RESOLVED`; `dprime_summary.json` errors 0, calls > 0,
  `fresh_replaced` = 25 (every game's first acquire priced); 0 tracebacks; histcache / timeout_fix / level_memory counters
  clean; gen tok/s >= **624.5** (−5% of B's 657.4); retractions <= 4; tokens/request 1421-1737 (B 1579 ±10%).
- Starvation: games with 0 actions <= **8** (sim expects ~5-6 at this shape; B ~0), each a never-admitted game with no
  error, and every other game ends gave_up/won. Kill if > 11 (more than the 11 that wait at start: admission broken) or any
  game thread dies.
- Outcome (catastrophe only; public-25 is not comparable to B here): kill if total levels <= 26 or hard-15 <= 8 (B 36 / 15).
- Also grep `DEADLINE at`, `READY after`, `analyzer failed` (standing rule 2).

### 8.6 Recommendation

D′ is the best-evidenced scheduler change available: two hidden draws +3.0..+4.3 over the same-period Franzen control (p ~0.08),
a positive sign in both independent replays (ours: ≥ tail in every world), and zero throughput cost. It replaces our tail fix
rather than adding to it. Proposed: after its check run passes, **give D′ arm A's remaining explore slots** (A is the
lossy-acceptance arm we already deprioritised after exp-074t's 27.97; B stays as the tail control so B vs B-D′ is a same-stack,
same-period comparison of the scheduler alone). Do not take slots from B. Ranking still needs n >= 3 per arm.

### 8.7 Fresh-first hedge: `calamitychasm/arc3-m2-turbo-lossless-dprime-ff` [BUILT + SIM, 2026-10-10; never pushed]

Ready if the D′ check run trips its starvation criterion (> 8 games with 0 actions). `python scripts/_build_m2_level_memory_kernel.py
--turbo-lossless --dprime-fresh-first` -> `kaggle_submission_m2_turbo_lossless_dprime_ff/notebook/` (token `ff`, implies `dprime`,
refused with `tail`). Differs from the dprime kernel in exactly two cells (header blurb; install cell, +`ff` lines after D′'s).

- **Mechanism**: D′'s module is installed verbatim by its own `install_d` (sha256 `6456efdd…` still pinned; the notebook embeds it
  unchanged). The `ff` lines then wrap `_PriorityGate.acquire` once more: a priority at or above `_PRIORITY_UNTRIMMED_BASE -
  _PRIORITY_BAND` (a never-started game; the harness gives it `2,000,000 - dispatch_index`) takes upstream's own `acquire` body
  (`_enqueue_and_wait(priority)` without a snapshot), so it queues in the band above every priced game, in dispatch order, and
  nothing re-ranks it; any other priority goes to D′'s acquire. D′'s formula, pace tracker, fade 0.4 and handover pricing of started
  games are untouched. Because D′'s own fresh pricing is bypassed, `fresh_replaced` stays 0.
- **Markers**: `PRIORITY_DPRIME installed` and `DPRIME_FRESH_FIRST installed`. `dprime_summary.json` = D′'s {calls, errors,
  last_error, fresh_replaced} plus `fresh_first_admissions` (every game's first acquire: expect = number of game threads, 25 in the
  check run, ~110 in a submission).
- **Replay** (same sim, 120 draws, common random numbers; LB points vs upstream; ff = `dprime-ff`, alias of `dprime (fresh first)`).
  Full-run shape, 110 games × 532 min, 10 × 70 / 14 × 47 tok/s:

  | world | tail 10 / 14 | D′ 10 / 14 | **D′-ff 10 / 14** | ff − D′ 10 / 14 | ff − tail 10 / 14 | D′ starved 10 / 14 (ff 0) |
  |---|---|---|---|---|---|---|
  | public_like | +0.45 / +0.69 | +0.91 / +1.03 | +0.90 / +0.99 | −0.01 / −0.04 | +0.45 / +0.30 | 0 / 0 |
  | tok2.5 | +1.11 / +1.00 | +2.83 / +2.71 | +2.74 / +2.56 | −0.09 / −0.16 | +1.62 / +1.55 | 3.2 / 4.1 |
  | tok2_u.12 | +0.79 / +0.77 | +1.75 / +1.72 | +1.70 / +1.68 | −0.05 / −0.04 | +0.91 / +0.91 | 0.9 / 2.0 |
  | unsolv.20 | +0.20 / +0.22 | +0.54 / +0.49 | +0.55 / +0.50 | +0.01 / +0.01 | +0.35 / +0.27 | 0 / 0 |
  | depth1.15 | +1.03 / +1.06 | +2.62 / +2.58 | +2.47 / +2.35 | −0.15 / −0.23 | +1.44 / +1.30 | 5.9 / 8.4 |
  | finalU3x | +0.73 / +0.71 | +1.81 / +1.81 | +1.77 / +1.76 | −0.04 / −0.05 | +1.04 / +1.05 | 2.1 / 3.0 |
  | siggame0.3 | +0.85 / +1.01 | +2.98 / +3.12 | +2.73 / +2.74 | −0.25 / −0.38 | +1.88 / +1.73 | 4.1 / 5.7 |
  | coupled0.7 | +0.44 / +0.48 | +1.38 / +1.63 | +1.37 / +1.58 | −0.01 / −0.04 | +0.93 / +1.11 | 0.1 / 0.3 |
  | eff1.5 | +0.45 / +0.43 | +0.44 / +0.46 | +0.45 / +0.46 | +0.01 / 0.00 | 0.00 / +0.03 | 0 / 0 |
  | coupled0.7_eff1.5 | +0.25 / +0.14 | +0.22 / +0.16 | +0.23 / +0.17 | +0.01 / +0.01 | −0.02 / +0.03 | 0 / 0 |

  Reading: ff gives up 0.00-0.38 LB points against D′ (median −0.04; worst in the worlds where D′ starves most, which is the
  cost of playing the 2-9 games D′ never starts), keeps 85-100% of D′'s gain, and still beats the shipped tail fix by +0.3..+1.9
  wherever D′ does (eff1.5 worlds: tie). Starvation is 0 in every cell (D′: up to 8.4 of 110 games). Check-run shape (25 games ×
  25 min, 14 × 47 tok/s): D′ starves **6.0 / 5.3** games (public_like / tok2.5), ff **0.05 / 0.03** (upstream 0.05).
- **Check run**: D′'s criteria with the starvation line replaced by **0 games with 0 actions** (any game with 0 actions and no error
  is a defect of the wrapper), `fresh_first_admissions` == 25 and `fresh_replaced` == 0 instead of `fresh_replaced` == 25.
  Throughput/token guards, kills and the grep list are D′'s. Its public-25 numbers are comparable to arm B's (all 25 games play).
- **Tests**: `tests/test_m2_dprime_ff.py` runs the wrapper on the real patched harness (fresh games admitted first in dispatch
  order, then started games in D′ order; control build prices a fresh game at the formula value; ff queues it at
  `2,000,000 - 1`), pins the notebook, its two-cell diff against the dprime kernel and the verbatim module.

## 9. JustAdev742 exp-081 (2026-10-10) [READ; nothing built]

JustAdev742's exp-081 drew **34.04** on the hidden set (submission 57026621, rank 25 of 4,089), against 27.97 for their
exp-074t and 28.87 for their unchanged D′ copy. Source: their repo (github.com/JustAdev742/Arc-Agi-3-Kaggle-comp, Apache-2.0),
read at commit `c7a462b`. Main records: `docs/research_log.md` entries 2026-10-08 17:05..2026-10-10 10:19, `docs/status.md`,
`docs/SUBMITTING.md`, `scripts/build_candidates.sh`, `scripts/build_franzen_nb.py`, `kaggle/franzen/patches/`,
`docs/research/beat-tufa/patch-*.md`.

### 9.1 What exp-081 is

Notebook `scottmahony/arc3-dprime-r14a05-harness4-percept-full` (script version 356590073; private). Built with their builder
`scripts/build_franzen_nb.py` on `--base dprime`, the public D′ notebook in `kaggle/dprime/`. That notebook is Franzen's with only
the slot priority replaced; it is the same D′ we vendored in `kaggle_submission_milestone2_fork/dprime/`. Their exp-084 entry
(research log 2026-10-10 04:10) says exp-084 is "exp-081's exact build (reproduced byte-identically apart from today's 55-min
fail-fast limit)" plus the fine-tuned draft and the 2400 s grace. So exp-081's flags are `build_candidates.sh`'s exp084 command
without `--draft/--draft-manifest` and without `--env ARC3_HTTP_RETRY_INITIAL_SECONDS=2400`:

```
build_franzen_nb.py --base dprime --full25 121 --input-fallback --wait-inputs 120 \
  --env ARC3_MAX_ACTIVE_STREAMS=14 \
  --cfg MAXREQ=14 --cfg CUDAGRAPH_MAXBS=14 --cfg MAMBA_CACHE=84 --cfg SPEC_ACCEPT_SINGLE=0.5 --cfg SPEC_ACCEPT_ACC=0.5 \
  --reap-kept kaggle/franzen/reap448_kept_experts.json --hot-tokens kaggle/franzen/hot_tokens_64k_arc.pt --fail-fast \
  --env-add OURS_BUDGET_METER=1 --env-add OURS_WIN_LEDGER=1 --env-add OURS_SEARCH_HELPER=1 \
  --env-add OURS_LEVEL_MEM=1 --env-add OURS_PERCEPTION=1 \
  --patch ours-sandbox-timeout-keeps-work.patch --patch ours-02-budget-meter.patch --patch ours-04-search-helper.patch \
  --patch ours-03b-win-ledger-on-02-04.patch --patch ours-05-level-mem.patch \
  --patch ours-08b-perception-on-01-02-04-03b-05.patch --compact
```

`--full25` and `--fail-fast` change only the Save & Run. The competition rerun is D′'s own path.

Changes vs Franzen's milestone-2 notebook:

| layer | exp-081 | our arm A (turbo-tail) | our arm B |
|---|---|---|---|
| model / draft | Intel W4A16 + albucino MTP (unchanged) | same | same |
| REAP-448 at load | yes | yes (ported from them) | yes |
| MTP acceptance | 0.5 / 0.5 (lossy) | 0.5 / 0.5 | 1.0 |
| ARC FR-Spec map | yes | yes | yes |
| streams / Mamba / mem | 14 / 84 / 0.96 | same | same |
| first-request grace | 900 s (D′ default) | 900 s | 900 s |
| scheduler | **D′ (verbatim)** | PRIORITY_TAIL (Franzen's gate + our fix) | PRIORITY_TAIL |
| sandbox-timeout fix | source patch ours-01 | runtime `timeout_fix.py` (same fix) | same as A |
| history cache | no | yes | yes |
| our level memory (M85) | no | yes | yes |
| EXPOSE_RESET | off (default) | off | off |
| harness bundle (below) | **02 / 04 / 03b / 05 / 08b** | no | no |

The bundle is five source patches applied with `git apply` right after Franzen's harness patch in cell 4. Each patch is behind an
env flag. With its flag off the harness is byte-identical.

- **ours-02 budget meter** (`OURS_BUDGET_METER`; new `inference/utils/budget_bar.py`, hooks in `tool_agent.py` and
  `python_tool_sandbox.py`). It reads the edge "budget bar" from the frames and adds one user-prompt line ("about N more actions
  before it is empty"), a "budget death" verdict after a game over, and a `budget` dict in the sandbox.
- **ours-04 search helper** (`OURS_SEARCH_HELPER`; `search_helper.py`). It adds `search()` (BFS / A* / beam over a step function
  the model writes) and `run_plan()` (one real action at a time, stopping at the first surprise) to the sandbox. It costs 10 prompt
  lines (+461 cached tokens), and a call that uses `search(` gets +15 s on its time limit. It uses its own sandbox bootstrap
  (`_SANDBOX_BOOTSTRAP_SEARCH`).
- **ours-03b win ledger** (`OURS_WIN_LEDGER`). At each level-up the opener gets an exact record of the win (about 280 tokens:
  actions, game overs, object changes). At every context trim, a ledger of all win records, the current level's game overs and
  the retained function names is re-pinned right after the system prompt (about 400 tokens). The sandbox gets `level_wins`.
- **ours-05 level mem** (`OURS_LEVEL_MEM`). This is a sandbox dict `mem` that persists across python calls on one level (JSON
  only, up to 200 KB, emptied at a level change). It costs 3 system-prompt lines and one `mem` field per tool result. It is
  **not** our level memory: it is scratch storage for the model's own data. It does not pin anything across levels.
- **ours-08b perception helpers** (`OURS_PERCEPTION`; host `ours_perception.py`, sandbox `ours_perception_sandbox.py` spliced
  into the bootstrap, 4 system-prompt lines, a 10-line segmentation change). These are the "perception helpers":
  1. A whole-view scroll estimate. It keeps a running `view_offset` on every frame and adds a `[view] scrolled at step N` line
     to the action echo.
  2. `left_view`: objects a scroll carried out of view, such as lf52's cart.
  3. `logical_grid()`: the frame's cell lattice, one character per cell.
  4. `.segmentation8`: 8-connected segmentation.

  Their validation used 39 recorded runs (107,083 frames). All 285 scrolls were measured exactly, and none was claimed on
  103,477 still steps. The cost is 1.7 ms per action.

What the bundle is not: no fresh start (07 rejected: 39.73 / 45.02), no effect table (06b dropped), no RESET exposure (it hurt
in exp-077/078), no reasoning-effort change.

### 9.2 Evidence

Their hidden draws, all on the Franzen M2 family:

| draw | config | score |
|---|---|---:|
| 2026-10-07 (56922501) | exp-070d: D′ copy, unchanged | 28.87 |
| 2026-10-09 (56980485) | exp-074t: D′ + REAP-448 + 14 streams + acceptance 0.5 + sandbox fix | 27.97 |
| 2026-10-10 (57026621) | **exp-081**: exp-074t + ARC map + bundle 02/04/03b/05/08b | **34.04** |

The two D′ configurations above are the closest control for exp-081. Other draws of the same family:
- D′'s author: 31.54.
- Franzen-family copies: 25.8 ± 3.9 per draw.
- Our arm A (same serving as exp-081, our harness extras and tail instead of the bundle and D′): **26.59** (n=1).
- Our incumbent: 29.42 (n=5).

**34.04 is n=1.** The per-draw sd is ~2-4, and the sd of a two-draw difference is ~5.5. Against exp-074t the +6.1 is about
1.1 sd. Against the mean of the three same-serving draws without the bundle (27.97, our 26.59, and D′ 28.87 at lower
throughput) it is about +6.3. That is suggestive, not established.

Their public-25 runs, at full length (25 games × 121 min/game; not comparable to our 25 × 25 min check runs):
- exp-081: **50.00, 113 levels**, 5,690 actions, 803 output tok/s, accept 3.14. The `[view]` notes fired in exactly the two
  scrolling games: bp35 (9) and lf52 (6). The model read `view_offset` 46× / 26×, `left_view` 10× (lf52) and `logical_grid()`
  28× in 6 games. lf52 cleared 3/10 levels (base 1-3). NameErrors were 35 per ~2,770 tool results (earlier bundles 53-64).
- Base config without the bundle: 56.00 / 42.89, plus 49.45 lossless.
- Bundle without perception: 48.96 (RESET on) and 46.48.

So on public-25 no bundle differs from the base beyond noise (one run's sd ~4.5). The public-25 to hidden-set ratio is 0.68 for
exp-081 against 0.57 for exp-074t. That gap is either a hidden-set effect of the bundle or the high draw itself.

Their next candidates do not carry the same draft as exp-081:
- exp-084 (exp-081 + a fine-tuned MTP draft + 2400 s grace; 48.83 / 114, accept 3.33) replaces exp-081 in their rotation from
  Oct 12.
- exp-085 (exp-084 + "untried objects" notices, ours-10) is running.

The draft mounts the private kernel output `scottmahony/arc3-mtp-session-a`, so exp-084 cannot be forked unless they publish it.
Their exp-084 draws are still evidence on the bundle, at no cost to us.

### 9.3 Mapping onto our stack

- **Serving**: identical to arm A, no work. Arm B differs only by acceptance 1.0. exp-081's 34.04 with acceptance 0.5 weakens
  the "0.5 is suspect" reading we took from exp-074t's 27.97.
- **D′**: we have it verbatim in `kaggle_submission_m2_turbo_lossless_dprime*/` (built, never pushed).
- **Timeout fix**: we have it as a runtime install. Their bundle patches carry ours-01's context lines, so the bundle needs the
  **source** patch ours-01 applied first. A port would drop our runtime `--timeout-fix`.
- **Bundle into our builder** (arm B + D′ + 01/02/04/03b/05/08b): this has a real conflict with our **history cache**.
  `history_cache.install` rewrites only `SB._SANDBOX_BOOTSTRAP`. It anchors on
  `history = _history_from_payload(state_payload.get("history"))` and sends `history` as a token plus deltas. With
  `OURS_SEARCH_HELPER` or `OURS_PERCEPTION` on, the sandbox runs `_SANDBOX_BOOTSTRAP_SEARCH` or `_OURS_PERCEPTION_BOOTSTRAPS[...]`.
  Those strings are built at import time from the unpatched text, so they would receive the token and fail. A port needs
  `patch_bootstrap` applied to all four bootstrap strings, plus a check that perception's `ours_view` state rides along in the
  delta path. We would also have to add `perception=` and `search_helper=` to every wrapped signature (`run_sandboxed_python` is
  wrapped `*args, **kwargs`, which is fine).
  - Effort: about a day, including a CPU bed run of the combined tree.
  - Risk: medium-high. The composite has never run anywhere, and a silent sandbox-state mismatch is exactly the failure that
    looks like a weak result.
- **Our level memory vs their bundle**: the overlap is with **03b, not 05**. Both act at the context trim:
  - ours appends facts + "Rule for level N:" lines at the end of the system prompt;
  - theirs re-pins exact win records right after it.

  Mechanically they compose: our runtime wrappers on `_build_user_prompt` and `_trim_messages_for_context` would wrap their
  patched methods. Both fire when the prefix is already broken, so there is no extra cache cost. The content is partly
  redundant: ~400 ledger tokens plus our block, and two level-up insertions in one opener. Our +4.8 was measured without a
  ledger, so its marginal value on top of 03b is unknown, and probably smaller.
- **Forking exp-081 verbatim**: cleanly reproducible. Every input is in their repo at `c7a462b`:
  - the builder, `franzen_tree.py` and its `bundle/` delta + manifest;
  - the D′ notebook (sha-pinned in the builder);
  - the six patches;
  - the REAP list + meta, the ARC map and `sglang_reap_patch.py`.

  The apply check needs Franzen's public repo (`da-fr/arc-agi-3-solution` @ 10882e3). The only edits needed are the builder's
  hard-coded `scottmahony/` kernel id (line ~1034) and the notes cell. The inputs are the same public datasets and models our
  incumbent mounts.
  - Effort: ~2 h. Vendor the files with their LICENSE + NOTICE (as for `turbo/`), add a wrapper script and a test pinning the
    built notebook.
  - Risk: low. It is the exact artifact that drew 34.04 and ran a clean full-length public-25.
  - It brings no history cache and no level memory.

### 9.4 Recommendation (pre-registered)

1. **Build arm C = exp-081 verbatim**: `calamitychasm/arc3-jad-exp081`, with their flags above and their builder at `c7a462b`.
   - The single deviation is `--env ARC3_HTTP_RETRY_INITIAL_SECONDS=2400`. That is our 2026-10-10 standing rule, and their own
     exp-083/084 do the same. It changes nothing unless the server boots late.
   - Keep it verbatim otherwise, so that our draws pool with JustAdev742's exp-081 draw. This mirrors the Franzen fork that gave
     us +5.
   - Do not add our level memory or the history cache to arm C.
2. **Check run** (one push, full length so it compares with their nine full-length runs: `--full25 121`, ~2.4 GPU-h; push only
   into a free GPU slot). Their notebook prints none of our markers, so the gate markers are theirs.
   - **Pass, mechanism**:
     - `harness patch applied successfully`;
     - `our harness patches applied successfully: 6`;
     - D′'s `#OURS_FORM ok version=d_prime`;
     - `priority gate active: 14 concurrent streams`;
     - serve.log `ARC3 REAP: kept 448 of 512 ... router sha256 verified`;
     - `ours: FR-Spec map hot_tokens_64k_arc.pt written`;
     - 0 tracebacks outside serving teardown, and every game terminal;
     - `[view] scrolled` lines present in bp35 and lf52;
     - `view_offset` or `logical_grid()` used at least once.
   - **Pass, serving**:
     - output tok/s ≥ 763 (−5% of their 803);
     - accept 3.0-3.3;
     - exact repeated assistant turns ≤ 1;
     - retractions ≤ 4.
   - **Reproduction**: score ≥ 39.7 and levels ≥ 95 (the lowest of their nine full-length runs).
   - **Kill**:
     - any patch-apply, REAP or FR-map assert failure;
     - tracebacks raised from `ours_*` modules;
     - tok/s < 723;
     - more than 2 repeated turns;
     - score < 35 or levels < 90.
   - **Near-pass** (tok/s 723-763 or levels 90-95): rerun once before taking any slots.
3. **Slots**: if C passes, it takes **arm A's remaining explore slots** (10-12, 10-14, 10-16).
   - Arm A (n=1, 26.59) has the same serving as C, so A vs C is the bundle + D′ against our harness extras + tail.
   - B keeps its slots as the lossless control.
   - C supersedes the D′ kernel's claim on A's slots (§8.6), because C already contains D′ and has a hidden draw. The D′ kernel
     stays built and unpushed.
   - The drop rule and the n ≥ 3 rule apply unchanged. Re-read JustAdev742's exp-084 draws (Oct 12 on), and pool them as
     bundle evidence with a draft caveat.
4. **Later, only if C's mean leads after n ≥ 3**: arm C+LM, which is C plus our level-memory install cell. It is a runtime wrapper,
   so the harness patches are untouched. Check-run criteria: C's, plus `LEVEL_MEMORY installed`, LM errors 0 and blocks applied
   > 0, and tokens/request within +10% of C's. Do not port the bundle into arm B's histcache stack unless the history cache is
   first fixed to patch every bootstrap variant (§9.3).
