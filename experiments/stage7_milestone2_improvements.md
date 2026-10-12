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

### 9.5 JustAdev742 update (read 2026-10-12)

Source: their `docs/status.md` and `docs/research_log.md` at commit `ccb7214` (2026-10-11). Second-hand; nothing built from it.

- **Hidden draws** (one draw each unless stated):

  | config | score | note |
  |---|---:|---|
  | exp-074t (base D′ + REAP + 14 streams + acceptance 0.5) | 27.97 | |
  | unchanged D′ | 28.87 | |
  | exp-083 (base D′ + fine-tuned drafter + ARC map + 40-min first-request grace) | 26.04 | ref 57056251 |
  | exp-081 (harness bundle + ARC map + albucino drafter) | 34.04 | ref 57026621, n=1; retired for exp-085 without a second draw |

  Base family mean 27.0 (n=2 by their count) vs bundle 34.04 (n=1). They call the gap suggestive only (z ~1.4). Our arm C is a
  verbatim exp-081 fork, so its draws are the second sample of the bundle.
- **exp-082** (reasoning effort medium): full-length public-25 32.53 / 84 levels, the lowest of their 9 full runs; never
  submitted. This confirms our closed line (§6.5).
- **Fine-tuned MTP drafter**: held-out accept +0.24..+0.31 (~+9-11%); +3.5-8% tok/s at equal batch. The weights are private
  (kernel output of `scottmahony/arc3-mtp-session-a`, unpublished), so we cannot adopt it.
- **Untried-objects notices** (their `ours-10` patch, `OURS_UNTRIED=1`): lists objects not yet clicked on the level.
  - Click-on-named-object within 3 actions: 21.5% vs 10.1% baseline.
  - exp-085 (bundle + drafter + notices) public-25: 53.77 / 119 levels, and 47.51 on a slow-storage boot.
  - Their own analysis calls the gain modest (87% of named entries are inert). On the base config (exp-086) it failed their
    adoption rule.
  - Candidate add-on to our arm C only if exp-085 lands a strong hidden draw. Not scheduled.
- **Infrastructure**: slow-storage boots recur. The 2,400 s first-request grace held on its first live use, which supports our
  standing rule. Their fix for restarts: re-read the model shards sequentially before any server restart.
- **Their rotation**: exp-083 / exp-085 alternating from 10-12.

## 10. Arm C+LM: our solved-level memory on exp-081 [CODE + BUILT, 2026-10-12; never pushed]

Kernel `calamitychasm/arc3-jad-exp081-lm`, `kaggle_submission_jad_exp081_lm/notebook/`, built by
`scripts/_build_jad_exp081_lm_kernel.py`, tests `tests/test_jad_exp081_lm.py` (+ probe `tests/jad_lm_probe.py`). This is §9.4
item 4, built early so that its check run can go into a free GPU slot. Whether it gets slots is still decided by §10.4.

### 10.1 Mechanism

Arm C v2's notebook (sha256 `d108fa76…`, the version 2 that the scheduled C submits pin; read, never rewritten) plus two edits:

1. One code cell, inserted right before the run cell (after D′'s install cell), installs
   `kaggle_submission_milestone2_fork/level_memory/level_memory.py` on the patched `ToolAgent`. It is the incumbent's install
   cell, generated by the same function (`_build_m2_level_memory_kernel.install_cell`). It wraps `_build_user_prompt`,
   `_trim_messages_for_context` and `_ensure_session` at runtime and prints `LEVEL_MEMORY installed`. Tried facts stay off.
2. The run cell writes `level_memory_summary.json` and prints `LEVEL_MEMORY summary` after `bm.run` (the incumbent's dump, same
   text).

Nothing else changes. The six source patches, D′, REAP-448, acceptance 0.5, 14 streams, the 2400 s first-request grace and the
25 × 121 min Save & Run shape are byte-identical. `kernel-metadata.json` differs from arm C only in `id`, `title` and
`code_file`. The notebook is 821,723 bytes (Kaggle's limit is near 1 MB).

### 10.2 Overlap analysis: complementary, not redundant

What each mechanism carries across a context trim (read in the vendored patches and in `level_memory.py`):

| content | ours-05 level mem | ours-03b win ledger | our level memory |
|---|---|---|---|
| when | every python call on one level | opener at each level-up; one ledger message at each trim | rule request at each level-up; block at each eviction |
| where | sandbox dict `mem` + a key list in tool results | user message right after the system prompt (re-pinned when the history head moves) | end of the system prompt |
| survives a level change | **no** (emptied at every level change) | yes | yes |
| actions of a won level | — | run-length encoded, ≤ 100 chars in the ledger, of the **winning attempt** | last 30 actions (all 3 most recent levels) |
| action / game-over counts | — | yes, with the step of every game over; current level's resets too | action count only |
| board/object changes | — | harness-computed, ≤ 4 lines per level | — |
| retained function names | — | yes | — |
| the model's own conclusion | — | **none** ("facts … never written by the model and never interpreted") | **"Rule for level N:" in the model's words: goal, what each control did, the plan that won** (≤ 600 chars) |
| reasoning from the clearing turn | — | — | last 900 chars |

- **ours-05 is not memory across levels.** It is scratch storage that is emptied at every level change, so it does not overlap.
- **03b overlaps on the facts** (counts, the winning actions), not on the content M85 was built for. sirikilohit reported
  M85's +2.8 as "the only text addition that helped". What it adds is the stated rule, a distilled what-the-controls-do / what-the-goal-is
  line that the model writes once and that later levels read. 03b deliberately states only harness facts.
- After an eviction, exp-081 keeps that knowledge only in the retained Python helpers and in whatever the model re-derives. The
  base harness's own knowledge/summary machinery is not touched by the six patches, so it was present in the incumbent too,
  where the memory still measured +4.8.
- Expected effect: smaller than the incumbent's +4.8 (the facts half is now redundant; the rule half is not). It cannot be
  measured on public-25 (one full-length run's sd is ~4.5) and would need interleaved hidden draws against C.

### 10.3 Mechanical compatibility (checked on the real tree)

`tests/jad_lm_probe.py` builds `/kaggle/taaf-kaggle-source-share` exactly as the C+LM notebook builds it. It uses Franzen's
patch plus the six patches extracted from the notebook's own `--compact` cells, through JustAdev742's vendored
`franzen_tree.py`, with `da-fr/arc-agi-3-solution` @ 10882e3. It then turns on the five `OURS_*` flags and executes the
notebook's install cell verbatim against that tree.

- **Hooks present.** All three wrapped methods exist with the same signatures. `_context_was_trimmed` is set by
  `_note_history_evicted` inside the trimmer, and D′ keeps Franzen's `_maybe_handover`, which consumes it after the trim, as in
  the incumbent. `_session_runtime_dir` and `_resume_after_yield` exist.
- **Patched code is not touched.** 03b patches the bodies of `_build_user_prompt` (`_ours_ledger_observe`, opener lines) and
  `_trim_messages_for_context` (ledger strip/reserve/pin). Our wrappers call those patched methods and then act on the result:
  they append the rule request to the opener and replace `messages[0]`, the system message, at an eviction. 03b only ever
  inserts its ledger at index 1, and its head-identity check looks at `history[0]`, which our edit does not change. No patch
  edits are needed.
- **Probe result**:
  - a level-up opener carries their "Level 1 record (exact facts …)" and then our "Rule for level 1:" request;
  - the rule is captured from the reply;
  - the first eviction pins our block at the end of the system prompt;
  - a later trim (with the head moved) puts their ledger right after the same system prompt;
  - LM summary `levels_recorded 1, rules_captured 1, blocks_applied 1, errors 0`.
- **Cost.** Both mechanisms rewrite the prompt only where the trim has already broken the prefix cache, so the combination adds
  no extra prefill. Per game, the system prompt grows by ≤ ~6.4 KB (the incumbent check run's `block_chars_max` was 6,413),
  plus ≤ 1.5 KB of ledger. The rule paragraph adds a few hundred output tokens per cleared level, in the turn after the
  level-up.

### 10.4 Check run and slots (pre-registered)

**Shape.** Arm C's full-length shape: all 25 public games × 121 min/game, about 2.0-2.3 GPU-h (C v2 ran for 2 h 01 m 44 s).
Push only into a free GPU slot, with `jadlm-push-1` (CLAUDE.md).

The comparison baseline is arm C's two full-length runs:

| run | gen tok/s | levels | score | tokens/request |
|---|---:|---:|---:|---:|
| v1 | 728.7 | 107 | 49.21 | — |
| v2 | 789.9 | 111 | 48.03 | 1,989 |

**Pass, mechanism**:
- all six arm C submit markers, plus `LEVEL_MEMORY installed`;
- `level_memory_summary.json` with `errors` 0, `levels_recorded` > 0, `blocks_applied` > 0, and `rules_captured` ≥ 0.5 ×
  `levels_recorded` (incumbent: 32 / 47);
- 0 tracebacks outside serving teardown; 25/25 games terminal.

**Pass, serving**:
- gen tok/s ≥ 750 (−5% of C v2's 789.9);
- accept 3.0-3.3;
- repeated assistant turns ≤ 1;
- retractions ≤ 4;
- tokens/request ≤ 2,190 (C v2 + 10%; the rule paragraph should add only a few percent).

**Pass, outcome** (catastrophe guard only; public-25 cannot rank C+LM against C): levels ≥ 103 (C v2 − 8) and score ≥ 44.

**Kill** (any one):
- gen tok/s < 723;
- levels < 95 or score < 40;
- any traceback raised from the `level_memory` module;
- LM `errors` > 0;
- `blocks_applied` = 0 (the hook never fired, so the run cannot test anything);
- tokens/request > 2,390 (+20%).

**Near-pass** (tok/s 723-750, levels 95-103, or score 40-44): rerun once before any slot.

**Slots.** A pass makes C+LM eligible; it does not by itself earn slots.
- Per §9.4 item 4, C+LM is drawn only if C's mean leads after n ≥ 3.
- It is then interleaved with C (alternating days in the exploit phase) so that the two are compared in the same period.
- The drop rule and the n ≥ 3 final-selection rule apply unchanged.
- Expected effect is below +4.8 (§10.2), so expect several draws per arm before the two separate.

## 11. Where arm C stalls on the weak public games, and what the harness could fix [READ + CODE, 2026-10-12; nothing built]

Brief: find fixable failure modes on the games where arm C (`calamitychasm/arc3-jad-exp081`, the exp-081 fork) stalls, and
propose 1-3 harness or prompt changes with mechanism-level evidence. **Short answer:** two small changes have evidence at
the mechanism level, an instrument change comes first, and part of the gap is not harness-fixable.
- The two changes: the untried-objects notice I1 (JustAdev742's ours-10, already measured on this stack), and a
  three-line fix to a stale-state latch bug confirmed in the source.
- The instrument change: per-game transcript excerpts in `kernel_output`.
- Not harness-fixable: wa30 level 2 and sk48 look like planning or understanding speed, not missing harness facts.

### 11.0 Data, and what could not be read

- **Arm C's own transcripts could not be read.**
  - v2's full output is artifact 11720435997 of kaggle-ops run 38186946527 (451 MB, expires 2026-10-19).
  - `gh run download`, `gh api …/logs` and the MCP `download_workflow_run_artifact` all redirect to
    `productionresultssa16.blob.core.windows.net`. The session's egress proxy refuses that host: 403 at CONNECT, logged
    in `recentRelayFailures` on 2026-10-12 00:33 UTC. That is an organisation policy denial, so it was not retried.
  - The MCP `get_job_logs` tool does work. So this section uses the two `kernel_output` digests:
    - v1: request `jad81-out-1`, run 38072197706;
    - v2: request `daily-1012-s`, run 38186946527.
  - The digests give levels per game, speed and request-log counts. They contain no reasoning.
- **Behaviour on these games is taken from transcripts of the same harness family**, all treated as data:
  - JustAdev742's exp-073b failure analysis (`docs/research/beat-tufa/exp073b-failure-analysis.md`). This is
    exp-081's base without the bundle: D′, REAP-448, 14 streams, acceptance 0.5. It used whole-run benchmark.json plus
    each game's last-call window.
  - Their mechanism reads of exp-077, exp-081 and exp-085/085r/086 (`docs/research_log.md`, and
    `docs/research/beat-tufa/notice-response-2026-10-10.md`).
  - Their older-harness studies `docs/research/hard-games-level1.md` and `levels2plus-exp054.md`. These are the Duck base
    with source-traced mechanics. Use them for the mechanics, not the rates.
  - Son Pham's stuck-level tally of 11 Franzen-line runs, and the rebuilt Franzen-harness transcripts for wa30, bp35,
    lf52 and sp80 (`sonpham-org_arc-3/docs/2026-10-06-stuck-levels-tally.md`, `datasets/spark-runner-snapshots/*.json`).
- **Game source.** `ARC-AGI-3-Agents/environment_files/` is absent in this checkout. The public games' source is in
  `sonpham-org_arc-3/docs/static/games/src/<game>-<ver>/`. Only wa30's level table was read directly here (step budgets);
  the other mechanics are as JustAdev742 traced them.
- **skarin's audit** (Kaggle kernel `skarin/arc-agi-3-27-80-lb-100-trajectory-audit`) could not be opened from this box,
  so its figures (~50 points from unfinished levels, ~2.6 from efficiency) are taken from the brief.
  - exp-073b's own decomposition has the same shape: 29.8 points in levels never reached, 11.4 in the levels being
    played at the end, and 2.9 in efficiency.
  - jvilladuque's G15 measurement (DESIGN §8.87) makes the same point from another side. Franzen's harness scores 46.3 on
    the 10 easy public games and 5.4 on the 15 hard ones. Level 1 takes a median 2.6 min against 11.6, and tokens per
    action are ~600 against ~2,200. The gap is in how fast the model understands the mechanics.

### 11.1 Arm C's weak games (two full-length runs, 25 games × 121 min)

| game | levels | v1 levels / score | v2 levels / score | Franzen line: reliably cleared through / first stuck level (11 runs) | value of the next level, mean points on public-25 |
|---|---:|---|---|---|---:|
| bp35 | 9 | 2 / 6.49 | 1 / 1.45 | L1 / L2 (1 of 11 clear it) | 0.18 (L2) |
| wa30 | 9 | 1 / 1.32 | 1 / 2.22 | L1 / L2 | 0.18 (L2) |
| sk48 | 8 | 0 / 0.00 | 3 / 16.67 | – / L1 | 0.44 (L4) |
| g50t | 7 | 2 / 10.71 | 3 / 21.43 | – / L1 | 0.57 (L4) |
| lf52 | 10 | 1 / 1.82 | 4 / 12.61 | L2 / L3 | 0.36 (L5) |
| sp80 | 6 | 1 / 4.76 | 4 / 47.62 | L1 / L2 | 0.95 (L5) |
| also low: tn36, ka59, su15, vc33 | 7 / 7 / 9 / 7 | 6 / 2 / 3 / 4 levels | 1 / 2 / 3 / 3 levels | L1 / L2 (soft) / L3 / none | – |

Notes on the table:
- **Value column.** It is the level's weight share × 100 / 25, the upper bound if the level is solved at human efficiency.
  On the hidden set, a mechanism matters through how often its failure mode recurs, not through these six games.
- **Size of the loss.** In v2 the six weak games lost 498 of 2,500 game points, which is 19.9 of the 52 points arm C
  missed. tn36, ka59, su15 and vc33 lost another 14.2.
- **Same build, different outcomes.** v1 and v2 are the same notebook, yet sk48 went 0→3, sp80 1→4, lf52 1→4 and tn36
  6→1. Single runs swing by whole levels per game, so one check run cannot show a per-game effect. Read every change
  below by its mechanism counts.

### 11.2 Per game: what the level needs, where the reasoning goes wrong, and whether the harness can help

- **sp80, level 2: harness-fixable, with the best evidence of any game here.**
  - **What it needs.** A click selects which bar moves, and the liquid must reach all 3 cups.
  - **What goes wrong.** The losing runs never click the red bars, which they assume are inert.
    - In JustAdev742's bimodal post-mortem, 7 of 8 losing game-runs over tn36, vc33, sp80 and cn04 never clicked the
      object that mattered. Every winning run clicked it first.
    - In exp-073b, sp80 ended on level 5 with "CONCLUSION: The level as I understand it is unsolvable", after 39.8 min
      parked and an 18.8-min tail with no action.
  - **Harness fit.** The I1 line is the matching fix: sp80 L2 was won in all 3 runs that carried it, and the winning
    click cited the line (§11.4 C1).
- **g50t, levels 1-3: partly harness-fixable.**
  - **What it needs.** SPACE rewinds the player to the start. From then on, a grey clone replays the recorded path one
    step per action, and the clone standing on the emitter holds the barrier open. The effect of SPACE is delayed and
    spread over later actions.
  - **What goes wrong.** This is effect attribution, not perception:
    - SPACE was first tried late, or its effect was misread ("SPACE spent one [life], so avoid SPACE"; older harness,
      engine-checked);
    - in exp-073b, the model counted ghosts per run by hand from `history` (7 of its last 15 calls). Its level started
      24.6 min before the end.
  - **Harness fit.** Partial. I1 names an unused action type (SPACE) after 10 minutes. An exact per-action effect table
    (ours-06) would state "after SPACE, a grey 5×5 appeared and moved" (C3). The cut-off at the end is a time problem.
- **sk48, levels 1-4: mostly not harness-fixable.**
  - **What it needs.** A head on a rail extends, retracts and slides an arm, and pushes cascade. After an arrow move, the
    arm's blocks read from the head must match a reference arm. The budget is 196 moves.
  - **What goes wrong.**
    - On the older harness, the goal was found late in all runs, and 15-41-action open-loop batches ran on a rule model
      that each batch contradicted.
    - In exp-073b the transition model checked 111/111 and A* returned a 42-move plan, but the run's own simulation said
      `sim win False`. The goal predicate was never verified.
    - The last 20 actions, sent at 120.1 min, were silently not executed.
  - **Harness fit.** Little. `run_plan`/`search` are already in arm C. The binding constraint is how fast the model works
    out the rules, which is the G15 point.
  - **Small follow-up.** Check what `action()` does near the per-game deadline, so a ready plan is not dropped. It is
    worth at most part of one level, and only when a plan is ready at the deadline.
- **bp35, level 2: only partly harness-fixable.**
  - **What it needs.** Gravity pulls upward, the camera scrolls, and the goal gem starts off-screen. Clicks break or
    toggle terrain, and spikes kill. The character must reach the gem.
  - **What goes wrong.**
    - On the older harness, the model took the green blocks for collectibles; both solves were partly accidental.
    - In exp-073b, the model did not know what killed the player ("Hmm, so what killed the player?"), lost the level-1/2
      goal after a trim ("I don't have that context in this conversation"), and misread a transition record.
  - **What arm C already covers.** The perception helpers report the scroll exactly (exp-081: `view_offset` read 46×
    in bp35), yet bp35 stayed at 1/9 in exp-081 and at 2 and 1 in our two runs. So scroll perception was not the
    binding constraint.
  - **Harness fit.** What remains is the cause of death and the goal's identity. ours-06's game-over line ("what the
    changed objects overlapped in the fatal frame") targets the first; our level memory (C+LM) targets the second.
    Evidence for either on bp35 is thin.
- **lf52, levels 2-5: partly harness-fixable.**
  - **What it needs.** Peg solitaire over a pipe network: a lock or cart carries pegs along the pipe, the view scrolls,
    and the goal is to reduce to one peg. The goal kind stays the same in every level (lesson 0016).
  - **What goes wrong.**
    - In Son's Franzen-line transcript, level 2 re-derived a goal that level 1 had already established ("re-examine the
      level-1 initial frame for … a lock/key analog"; "I keep going in circles"). The win came at analysis step 47
      ("the level's goal = 1 peg total").
    - In exp-073b, the cart left the view and was lost for about 8 calls. The stale-state guard then stopped 8 batched
      probes ("blocks batched actions after a no-op, even inside try/except").
  - **What arm C already covers.** Perception 08b fixes the off-screen cart (`left_view` was read 10× in exp-081).
  - **Harness fit.** Two pieces remain fixable:
    - the guard bug (C2; confirmed in the source, §11.3);
    - carrying the goal across levels, which is what our level memory's "Rule for level N" states. C+LM is already
      built (§10).
- **wa30, level 2: not harness-fixable.**
  - **What it needs** (read in `wa30.py`). Level 2 has 5 boxes and an orange helper that carries boxes on its own but
    too slowly. Its `StepCounter` is **70**, against a human baseline of 119, which counts failed attempts.
  - **What goes wrong.**
    - The level needs a near-optimal plan that also uses the player as a carrier inside a 70-step bar.
    - Arm C spent 229 and 251 actions per run, nearly all on level 2. That is about three bar-outs per run.
    - In exp-073b the model burned its bar on purpose to restart ("since the level is unwinnable now, let me just trigger
      the death").
  - **Harness fit.**
    - The budget meter (ours-02) and `search()` are already in arm C. Exposing RESET hurt elsewhere (exp-077/078), and a
      mid-level context reset halves solves (lesson 0037).
    - This is budget-limited multi-agent planning: a model capability and speed problem.
    - A level-2 win at about 2× the baseline actions would still score about 0.25 of the level.

### 11.3 Cross-cutting failure modes, and what arm C already covers

| mode (share among stalled levels in the transcript studies) | in arm C already | gap left |
|---|---|---|
| Mechanic never decoded, or the key object never acted on (8 of 17 in exp-073b; 15 of 48 in exp-054; 7 of 8 losing bimodal runs) | none directly | **I1 notice (C1)**; effect table (C3) |
| Out of time while progressing, or the level started too late (8 of 17) | D′ gate, 14 streams, budget meter | scheduling; closed for us (lesson 0039, §8) |
| Data lost between calls (NameErrors: 3.5% of calls in exp-073b) | ours-05 `mem` (NameErrors 35 per ~2,770 results, from 53-64) | none |
| Earlier levels lost after a trim; goal re-derived (bp35, ar25, lf52) | ours-03b ledger (facts only) | the goal and rule in words: **C+LM (§10, built)** |
| Scroll, off-screen and lattice perception (lf52, bp35, ls20) | ours-08b (exact on 285 scrolls) | none |
| Effects credited to the wrong action; delayed effects (dc22, g50t, ka59) | none | effect table (C3) |
| The stale-state guard refuses every later action even when caught (lf52 8 stops, m0r0 33 refusals in exp-073b) | **bug present in arm C** | **C2** |
| Runaway turns with no tool call (ar25 8, sp80 5 in exp-073b's windows) | none | possible; lesson 0022 risk, not proposed |

**The C2 bug, read in the source.**
- Location: Franzen's `python_tool_sandbox.py` as patched into arm C (`_stale_state` appears 6× in the notebook), with
  `solver.py` `_stale_detail`.
- **How it fires.** Arm C sets `ARC3_STALE_STATE_BLOCK=1` with `ARC3_GUARDS_FROM_LEVEL=2`.
  - When an action changes nothing, `action()` arms a per-snippet latch, and the next `action()` is refused with
    `StaleStateActionError`.
  - That error is a plain `Exception`. Its message says it "can be caught if a snippet has good reason to keep acting
    through no-ops".
  - But the latch is cleared only by a later result with `gameplay_changed is True`. A refused action never executes, so
    nothing can clear it: once the latch is set, **every** later `action()` in that snippet is refused, caught or not.
- **Interactions.** `run_plan` (ours-04) already treats the error as a stop, so the fix does not change it.
  `RepeatedActionInStateError` still blocks exact repeats from the same board, so the blind-loop protection stays.

### 11.4 Candidate changes, ranked

"Levels per run" means per 25-game × 121-min public run, about 50 game-hours of play. Per game-hour the gains are
small: +1 level per run is 0.02 levels per game-hour. On the hidden set they show up as about +0.5-1 point per level per
run-equivalent, which is below one draw's sd (~3). None of these is separable by leaderboard draws at our remaining n;
each is judged by mechanism counts.

| # | change | failure mode → weak games | evidence | expected levels per run | build / test cost | risk |
|---|---|---|---|---:|---|---|
| **C0** | **Instrument**: per-game transcript excerpts in `kernel_output` (spec below) | needed to check C1-C3 on arm C's own runs | – | 0 (enables the reads) | ~2 h incl. a unit test on a synthetic `*_requests.jsonl`; no GPU (re-read v2's output) | none (read-only op) |
| **C1** | **I1 "not yet tried on this level" notice**: JustAdev742's `ours-10-untried.patch` (sha256 `a3a7d89f…`, Apache-2.0, applies on exp-084's stack = exp-081's patches + a draft change) with `OURS_UNTRIED=i1` (I2 dropped: 2 clicks in 24 lines) | never acted on → sp80 L2 directly; g50t (unused SPACE), tn36, vc33, cn04 | Same stack, 3 full-length runs: the named object was clicked within 3 actions 21.5% (43/200) vs 10.1% (32/318) without the line (z ~3.6); critical levels won 9 of 12 with the line vs 9 of 16 (base) and 5 of 8 (bundle without it); no-harm line held (53.77 / 119, 47.51 / 111, 49.35 / 116). Only ~2 useful clicks per run; 87% of the named entries are inert | +1 to +2 | ~3 h: vendor the patch with its NOTICE, add a builder flag to `_build_jad_exp081_kernel.py`, test pin. ~2.3 GPU-h full-length check | low-medium: the code ran in 3 runs; exp-086 failed its critical-level rule (2 of 4); ~9 inert clicks per run cost actions on the levels where it fires. Their hidden draw of exp-085 (from Oct 12) is free extra evidence |
| **C2** | **Stale-latch fix**: clear `_stale_state` before raising `StaleStateActionError`, so a caught refusal lets the next action run, as the message promises. About 3 lines; a new patch after the six | guard deadlock → lf52 (the cart probes), m0r0 | Bug confirmed in the source (§11.3). Behaviour seen in 2 games of 1 run (exp-073b); never measured on arm C | 0 to +1 | ~2 h: patch, apply check with `franzen_tree.py check` after the six, a bed test (catch → next action runs; uncaught → snippet stops as now) | low: restores documented behaviour; exact-repeat and known-no-op guards unchanged |
| C3 | **Effect table** `ours-06b` in `note` mode (an exact per-level action→effect table, `effects()` in the sandbox, one line per new outcome class) | effects misattributed or delayed → g50t (ghost replay), bp35 (cause of death), dc22, ka59 | Replay-validated: 4,819 table lines and 448 notes with 0 errors. Only GPU-run inside the 7-patch bundle with fresh start (07), where both runs lost (39.73, 45.02) for 07's reasons; `effects()` was called 17-27× per run; no separate reading. Franzen saw no gain from automatic animation timelines | −1 to +3 (unknown) | ~4 h: 06b was cut against 02/04/03b/05, so it needs a rebase check on top of 08b (and on top of C1 if both go in). A separate check run read by mechanism | medium: prompt lines are added every turn in `note` mode; no clean ablation exists anywhere |
| – | C+LM (§10, built, check run pushed) | goal or rule lost after a trim → lf52 L2, bp35 | the incumbent's +4.8 on another stack; overlap analysis §10.2 | see §10 | – | – |

**Not proposed:**
- fresh start / mid-level reset (07: halved the solve rate, lesson 0037);
- exposing RESET (sc25 and others, exp-077/078);
- hopeless-level ending or demotion (lesson 0039);
- reasoning effort medium (closed, §6.5);
- an idle nudge (idle stretches are a symptom; no clean arm);
- a forced "try every action" at level start (costs actions on every level).

**Recommendation.**
- Do C0 first. It is cheap, and it turns every later check run into per-game evidence instead of a level count.
- Then build **C+U = arm C + C1 (i1) + C2** as one kernel, and read it by mechanism against v1/v2:
  - I1 click-within-3 ≥ 15%;
  - stale-state refusals after a catch = 0, and no snippet with more than 1 refusal in a row;
  - critical levels (sp80 L2, tn36 L2, vc33 L4, cn04 L2) ≥ 3 of 4;
  - no harm: levels ≥ 103 and score ≥ 44, the §10.4 lines.
- Give it slots only if C's mean holds up and exp-085's hidden draws are not below exp-081's. Keep C3 for a later
  isolated arm.

### 11.5 Instrument C0: transcript excerpts from `kernel_output` (proposal; built and run, see 11.7)

The check-run output holds one `<game>…_requests.jsonl` per game: 25 in v2, 2,952 responses. Each `"event": "response"`
record carries the request `messages`, so the merged assistant sequence covers the whole game, not just the last window.
`m2_speed_report.merge_assistant_turns` already does this merge, on hashes. The proposed op option:

```json
{"op": "kernel_output", "kernel": "calamitychasm/arc3-jad-exp081",
 "transcripts": {"games": ["sk48", "wa30", "bp35", "lf52", "g50t", "sp80"], "last_turns": 10, "first_turns_of_last_level": 4,
                 "reasoning_chars": 900, "code_chars": 500, "result_chars": 400, "max_bytes_per_game": 30000,
                 "count": ["StaleStateActionError", "NameError", "GAME OVER", "[view] scrolled", "unsolvable",
                           "Not yet tried", "budget"]}}
```

For each listed game, the op would print to the job log, which `get_job_logs` can read:
1. **A per-level timeline.** Built from the user openers' "Current state: step N, level L": requests, completion tokens,
   tool calls containing `action(`, game overs, and the first and last request index per level.
2. **Counts of the `count` strings in tool results**, per level. The `[view] scrolled` lines and the guard refusals live
   there, not in the notebook log, which is why the digest's marker for them read False in both runs.
3. **The first K turns and the last N turns of the level the game ended on.** Each turn shows the reasoning tail, the
   head of the python code and the head of the tool result.

It would be about 80 lines in `scripts/kaggle_ops.py` (`transcript_digest(out, cfg)`, called from `digest()` when the
option is set), plus a test on a synthetic two-level `*_requests.jsonl`. The size is bounded by `max_bytes_per_game`:
six games × 30 KB stays well inside a readable job log. Run it once on v2 (artifact until 2026-10-19; `kernel_output`
re-downloads from Kaggle in any case) to check the per-game claims of §11.2 on arm C's own play. That check needs no GPU.

### 11.6 What this cannot decide

- **None of arm C's own reasoning was read** (§11.0). The per-game diagnoses come from other runs of the same harness
  family, plus one older harness. Arm C adds 08b and the bundle to exp-073b, so some exp-073b failures may already be
  gone (NameErrors, the off-screen cart). C0 is how to check.
- **Every gain estimate is a judgement.** Each rests on 1-3 public runs whose per-game outcomes swing by whole levels
  between identical runs (§11.1). C1's mechanism effect is the only measured one (z ~3.6 on clicks). Its effect on
  levels (critical levels 9 of 12 against 9 of 16) is suggestive, not established.
- **The weak public games stand for a hidden-set population that is about 60% "hard"** (G15). Fixing sk48- or
  wa30-type stalls would need faster rule induction or planning by the model, which no harness line here supplies.

### 11.7 Arm C v2 transcript check (C0 built and run, 2026-10-12)

**What ran.** C0 is built as the `transcripts` option of the `kernel_output` op (`scripts/kaggle_ops.py`, tests in
`tests/test_kaggle_ops_transcripts.py`, usage in CLAUDE.md). It read arm C v2's own `*_requests.jsonl` through three kaggle-ops
requests: `jad81-tx-1` (run 38189671297), `jad81-tx-2` (38192378153) and `jad81-tx-3` (38192686222). Each re-downloaded
v2's output (212 files, 26 `*_requests.jsonl`; the layout is flat, `<game>-<hash>_p0_requests.jsonl`, one per game, 34-123 MB
each). Nine games were read: sp80, lf52, g50t, bp35, sk48, wa30, tn36, vc33, cn04.
- **Record layout** (probed, then fixed in the reader): keys `action, analysis_step, chat_template_kwargs, event, finish_reason,
  messages, request_index_within_turn, tool_choice, tools, usage`. `messages` holds the whole request including the reply, and
  `Current state: step N, level L.` is in the turn opener.
- **No wall-clock time anywhere** in the records or the opener. The timeline therefore reports requests, completion tokens and the
  action counter per level, not minutes.
- **Counting rule.** One refusal prints `StaleStateActionError` twice (the traceback line and the message), so divide the mention
  count by 2 for refusals; a caught-and-printed refusal prints it once more.
- The diagnoses below come from one run per game. They say what happened in v2's play, not how often it happens.

**Per-game readings (levels in the table are v2's: sp80 4, lf52 4, g50t 3, bp35 1, sk48 3, wa30 1, tn36 1, vc33 3, cn04 6).**

| game, level | requests / tokens / action() | what the excerpts show | diagnosis in 11.2 |
|---|---|---|---|
| sp80 L2 | 58 req, 175k tok, 22 `action()` (counter 12-44), cleared | The model clicked: `MOUSE(25,33)`, `(21,30)`, `(17,12)`, `(17,36)`. Its notes on the winning plan say "the red should be at cols 4-15 and 20-31" and "swap worked exactly as modeled", so it was moving the red bars with clicks and won in 33 actions. `(17,12)` lies in the red columns. The "red" greps only matched L1 (the blue bar turning red under SPACE), so the colour of each click is inferred from the coordinates, not read. | **Not reproduced.** "Red bars never clicked" does not hold for this run. It is one win; the failure may still occur in other runs (v1 cleared only 1 level of sp80). |
| sp80 L5 (ended) | 19 req, 51k tok, 4 `action()` | Level 5 had a new purple and a yellow object. The model spent its last turns on `logical_grid()` (returned a non-list object), then on writing a search for a bar layout. It was still searching when time ran out. | Time and planning speed, not a missing fact. |
| lf52 L2 | 24 req, 27k tok, 20 `action()`, cleared | Goal carried over. The first L2 turn reads: "we reduced 5 pegs to 1 and it completed. So the goal is 'reduce to 1 peg'", then jumps. L3 turn 89 states "Goal: 1 peg" again with no re-derivation. | **Refuted for L2** (no goal re-derivation). L3 is slow for another reason: 79 req, 156k tok, a long search over block and peg states. |
| lf52 stale refusals | 9 refusals in 9 results (L2 1, L3 4, L4 4) | Every sample is a single refusal that stops the snippet (the intended guard). The one caught case (L3 rec 44, a `try/except` around a 4-key loop) ended its loop at the refusal, so no second refusal is visible. | **Not confirmed for lf52.** No snippet in lf52 shows a refused action after a caught refusal. |
| lf52 scroll | `[view] scrolled` 38 (L3 10, L4 28) | The perception patch is live; the view moved with the pegs. L5 notes "mem was emptied (level change)" (the ours-05 scratch, not the goal). | Matches 11.2 (08b covers the off-screen cart). |
| g50t L1 | 24 req, 57k tok, 15 `action()` (counter 1-41), cleared | SPACE was tried at action 3, so it is not unused. It was **misread**: after SPACE moved the player back up, the model concluded "SPACE = UNDO" (turns 7-8, step 6-7). By turn 15 it doubts that ("if it were a replay ghost... unless the replay only starts on some trigger (like SPACE?). But SPACE = undo... Hmm"), and by turn 24 it uses SPACE as the restart that makes the ghost replay. The correction took about 16 requests. | **SPACE misread: confirmed (transient, self-corrected in L1). SPACE unused: refuted.** |
| g50t L4 (ended) | 37 req, 63k tok, 6 `action()` | Mostly deliberation: `logical_grid` found no lattice and the model fell back to segmentation reads and probing. Stale refusals in g50t: 7 mentions, each a single stop. | Partly harness: the lattice helper returned nothing on this level. |
| bp35 L2 (ended) | 58 req, 174k tok, 10 `action()`, 6 GAME_OVER mentions (4 results) | The model repeats "Hmm, so the goal appears when something happens. Hmm, so..." for a whole turn (rec 96) and then fell back to row-by-row ASCII scans. It never states what killed the ball. `[view] scrolled` 9 (L1 5, L2 4), so the scroll is perceived. | Goal identity unknown (the "lost goal after a trim" reading fits better than "cause of death"). Scroll perception is not the limit. |
| sk48 L4 (ended) | 13 req, 22k tok, 1 `action()` | L1 took 70 req / 152k tok, L2-L3 about 20-30 each. L4 started late and the last turns are model-building (cable extent, "the tip at 34"). | Time and rule-induction speed; no harness fact is missing. |
| wa30 L2 (ended) | 37 req, 120k tok, 25 `action()` (counter 61-252), 4 GAME_OVER mentions | Bar-outs, then at the end: "if the orange can't place E, the level fails ... observe first. 3 SPACEs". | Budget-limited multi-agent planning, as in 11.2. |
| tn36 L2 (ended) | 58 req, 163k tok, 29 `action()`, 3 RESET mentions | A switch-panel puzzle; the model keeps clicking the ball (58,46) and tests switch layouts in batches of 6. Refusals: see below. | Stale-latch: confirmed here. |
| vc33 L4 (ended) | 63 req, 205k tok, 24 `action()` | Four refusals, all single uncaught stops (clicking the same bar twice). 2 GAME_OVER mentions. | Time spent on pour-level modelling. |
| cn04 L6 (ended) | 27 req, 23k tok, 18 `action()` | L2-L5 cleared with only 12-62 requests each. L6 ended at the first shape-placement attempts. | Time. |

**The stale-latch bug (C2): confirmed in the transcript of one game.** tn36 L2, request 24 (step 27), a snippet wrapped in
`try/except`: its stdout reads `retry StaleStateActionError` three times in a row, then `bot cap clicks -1`, and the snippet ended
in an uncaught `StaleStateActionError: MOUSE(row=58, col=46) was not executed. The action immediately before it,
MOUSE(row=58, col=46), was a no-op`. That is three caught refusals each followed by a retry that was refused again, which is the
deadlock 11.3 derived from the source. It is the only snippet of that shape in the nine games; the other ~20 refusals in the nine games are
single stops. So the bug is real on arm C, but in this run it cost a few actions in one level.

**Verdicts for the three checks you asked about.**
- **sp80 L2, red bars never clicked: refuted for this run.** L2 was cleared in 33 actions with at least four clicks, and the
  model's own plan text moves the red bars. Unchanged: v1 cleared only one sp80 level, and the I1 notice's case rests on other runs.
- **lf52, goal re-derived on L2 and stale-latch refusals after a caught error: both not confirmed.** The goal carries over from L1
  in the first L2 turn. lf52 has 9 refusals, none after a caught one.
- **g50t, SPACE misread: confirmed, transient.** The "SPACE = UNDO" reading appears at step 6-7 and is corrected by step 36; L1
  was still cleared in 41 actions. The "SPACE unused" form of the claim is refuted (first used at action 3).

**What this changes in 11.4.** C1's sp80 case loses its best example in v2 (it cleared L2 without the notice), though the
bimodal post-mortem still holds. C2 is confirmed once and is worth its three lines only as hygiene (expected effect near zero on
this evidence). For lf52 the carried-goal part of C+LM has no failure to fix at L2. The largest visible loss across the nine
games is time spent in model-building on the last level (sk48 L4, sp80 L5, cn04 L6, g50t L4), which no harness line here targets.

## 12. Arm C+U: the I1 notice and the stale-latch fix on exp-081 [CODE + BUILT, 2026-10-12; never pushed]

Kernel `calamitychasm/arc3-jad-exp081-u`, `kaggle_submission_jad_exp081_u/notebook/`, built by
`scripts/_build_jad_exp081_u_kernel.py`; tests `tests/test_jad_exp081_u.py` (+ probe `tests/jad_u_probe.py`). This is
§11.4's recommendation **C+U = arm C + C1 (i1) + C2**, built as one kernel.

### 12.1 What changes against arm C v2

Arm C v2's committed notebook (sha256 `d108fa76…`; read, never rewritten, because the scheduled C submits pin version
2) plus exactly these edits; the diff is pinned by a test:

1. **Two patch cells** after arm C's last patch cell:
   - `/kaggle/ours-07-ours-10-untried.patch`: JustAdev742's `ours-10-untried.patch`, vendored verbatim in
     `kaggle_submission_jad_exp081_u/vendor/` with their `LICENSE` (Apache-2.0) and the Franzen `NOTICE.md`.
     - sha256 `a3a7d89fbb2cb253afcfc8eb9b4d43f763c45ea8b8624139db54949e7bc6a52e`.
     - Source commit `c7a462b` (added in their `85bc492`), byte-identical at `ccb7214`.
     - The cell is made by their own builder's `_file_cell` in `--compact` form. A test runs their builder with
       exp-081's flags + `--env-add OURS_UNTRIED=i1` + `--patch ours-10-untried.patch`: it produces the same cell
       and the same setup cell, except for our 8th patch.
     - Nothing from their newer builder is needed: the patch adds one stdlib module and two 3-line call sites,
       and it applies on top of the six exp-081 patches with line offsets only.
   - `/kaggle/ours-08-calamitychasm-stale-latch-fix.patch`: our C2 patch, in a plain `%%writefile` cell so it reads
     in the notebook (a 3-line preamble names it; `git apply` skips the preamble).
2. **Setup cell, three lines**:
   - `'OURS_UNTRIED': 'i1',  # ours (--env-add)`;
   - the notebook's own apply loop now runs over all eight patches (their `_apply_code`: the same `git apply`, the
     same check that every named file applied);
   - its success line becomes `our harness patches applied successfully: 8`.
3. **One install cell** before the run cell (after D′'s install cell). It embeds
   `kaggle_submission_jad_exp081_u/cu_counters.py` and does three things:
   - it checks that the fix is in all four sandbox bootstraps and that `OURS_UNTRIED` resolves to `{i1}`, and raises
     otherwise;
   - it prints `STALE_LATCH_FIX installed bootstraps=4 releases_per_snippet=3` and
     `UNTRIED_NOTICE installed mode=i1 ours_untried.py sha256=… counters=on`;
   - it wraps `ours_untried.prompt_lines`, `ours_untried.stamp` and `ToolAgent._run_python_tool` to count only.
4. **The run cell** writes `untried_summary.json`, `stale_latch_summary.json` and `untried_firings.json` after
   `bm.run` and prints `UNTRIED_NOTICE summary` / `STALE_LATCH_FIX summary`.

Unchanged and byte-identical: cell 0, D′, REAP-448, acceptance 0.5, 14 streams / Mamba 84, the ARC map, the six
patches, the 2400 s first-request grace and the 25 × 121 min Save & Run. `kernel-metadata.json` differs only in
`id`, `title` and `code_file`. The notebook is 831,697 bytes.

### 12.2 C1 configuration: `OURS_UNTRIED=i1` (the I1 line only)

- exp-085 ran `OURS_UNTRIED=1`, which is I1 + I2. Their own reads after exp-085 / exp-085r ("a later build should
  drop I2 (`OURS_UNTRIED=i1`) since it draws no clicks"; I2 drew 2 clicks in 24 lines) and §11.4 both say to keep
  I1 only.
- `i1` is a native mode of the verbatim patch, so no code changes. The measured mechanism (I1 click-within-3 21.5%
  vs 10.1%) belongs to the I1 line alone, so dropping I2 does not touch it. I2 cost ~12 lines × ~40 tokens per run.
- Their "notice v2" (cap 3 kinds, first line at 5 min when MOUSE is unused, level-1 clock from the first answered
  request) exists only as a recommendation in their `docs/status.md`; no code at `ccb7214`. Not used.
- **Caveat**: the i1-only mode has never run on a GPU anywhere. It differs from exp-085 only by the I2 lines.

### 12.3 C2: the latch fix and its semantics

**The bug** (§11.3, confirmed again on the built tree). `action()` in the sandbox arms `_stale_state` after a no-op.
While the latch is set, every later `action()` call sends `stale_after`, and the solver refuses it with
`stop_reason: stale_state`, which the sandbox raises as the catchable `StaleStateActionError`. Only a later executed
result with `gameplay_changed is True` clears the latch, and a refused action never executes. So once the latch is
armed, every later action in the snippet is refused, caught or not. The probe reproduces this on the unfixed tree:
`SECOND refused / THIRD refused`, and a caught 10-call loop executes 1 action and is refused 9 times.

**The fix** (`patches/calamitychasm-stale-latch-fix.patch`, 4 code lines in `python_tool_sandbox.py`'s bootstrap
source, so all four bootstrap variants inherit it):

```python
_stale_releases = [3]          # per snippet (the closure is rebuilt per execution)
...
if action_result.get("stop_reason") == "stale_state":
    if _stale_releases[0] > 0:
        _stale_releases[0] -= 1
        del _stale_state[:]
    raise StaleStateActionError(...)
```

**Semantics: a one-shot refusal per no-op, released at the raise, at most 3 releases per snippet.**
- **Uncaught** (the common case): the snippet ends at the refusal, exactly as before.
- **Caught**: the next `action()` executes, as the error message has always promised ("can be caught if a snippet has
  good reason to keep acting through no-ops"). If that action is again a no-op, the latch re-arms and the following
  call is refused once more.
- Releasing at the raise is the same thing as releasing "when caught": a refusal is always raised, and when the
  error is not caught, the snippet ends anyway.
- **Why the cap.** §11.3 assumed that `RepeatedActionInStateError` would still block blind repeats. **In arm C it does
  not.** `ARC3_NOOP_REPEAT_GUARD` and `ARC3_REPEAT_STATE_GUARD` default to off and arm C's `setup_env` does not set
  them; only the stale latch, the batch no-op stop and the terminal latch are active. Without a cap, a caught loop
  such as `while ...: try: action(UP) except StaleStateActionError: pass` through no-ops would execute every other
  call until the sandbox timeout. On a budget-bar game that could spend hundreds of actions, or end the attempt.
  With the cap, such a loop gets at most 3 extra executed actions per snippet, and after that the original
  behaviour holds (every later call is refused).
- **Unchanged**: `run_plan` (ours-04) still stops at the first stale refusal; the terminal, known-death and
  known-no-op paths are untouched; the solver is untouched.

**Checked on the real tree** (probe, real `ToolAgent._run_python_tool` + sandbox, with a fake game that applies the
solver's stale rule):
- no-op → refused-and-caught → a state-changing action: `FIRST True False / SECOND refused / THIRD True True`, 2
  actions executed (unfixed tree: 1);
- caught loop of 10 no-ops: `LOOP 4 6` (executed 4, refused 6: 3 releases, then the latch holds; unfixed tree:
  `LOOP 1 9`);
- the install cell refuses to install on the unfixed tree (`STALE_LATCH_FIX missing from the plain sandbox
  bootstrap`).

### 12.4 Instruments

All of these are counted in-process; the counters never raise into a game.

- **`untried_summary.json`** counts the I1 lines the model was shown:
  - `i1_lines`, `i1_lines_naming_objects`, `i1_named_kinds`;
  - `i1_clicked_named_within_3` and `i1_clicked_named_within_3_share`;
  - `i1_used_named_action_within_3`;
  - `i2_lines` (must be 0);
  - `tracker_errors` (games where ours_untried turned itself off);
  - `errors`, and a per-game count `[lines, hits]`.

  The click-within-3 definition is JustAdev742's `scripts/untried_read.py`, so the share compares with their 21.5%
  and 10.1% baseline:
  - the board is the last history entry the prompt saw;
  - the named kinds are the ≤ 6 listed in the line, in `i1_block`'s order;
  - a hit is a click among the next 3 history entries whose `click_target` is of a named kind;
  - the denominator is every I1 line shown.

  `untried_firings.json` holds one record per line: game, level, step, reason, hit, used, window.
- **`stale_latch_summary.json`** records:
  - `snippets`, `action_calls`, `stale_refusals`;
  - **`stale_refused_after_released`**: a refusal that directly follows a refusal in the same snippet while the
    snippet still had releases. It is 0 by construction with the fix, and > 0 is the bug;
  - `stale_refused_after_cap`;
  - `executed_after_stale_refusal`: the fix doing work, which happens only if the model catches the error;
  - `snippets_with_stale_refusal`, `max_stale_refusals_in_snippet`, `errors`.

  It is counted host-side from the payload of every `action()` call that reaches the solver, with snippet
  boundaries taken from `_run_python_tool`.
- **Fallback if a counter file is missing**: `kernel_output` with `transcripts`. Use
  `grep_text {"Not yet tried on this level": 8}` and `samples {"StaleStateActionError": 5}` on the four critical
  games. For each I1 line, read the next three `action(` arguments in the following turns and match the clicked
  cell to a named box (grown by one cell). For the latch, check that no tool result shows two consecutive stale
  refusals from one snippet before its 4th refusal.

### 12.5 Check run (pre-registered; full-length shape, 25 games × 121 min, ~2.0-2.3 GPU-h)

Baseline: arm C v1 / v2 (728.7 / 789.9 gen tok/s, 107 / 111 levels, 49.21 / 48.03). Push only into a free GPU slot
with `jadu-push-1` (CLAUDE.md).

**Pass, mechanism**:
- markers present:
  - `harness patch applied successfully`;
  - `our harness patches applied successfully: 8`;
  - `#OURS_FORM ok version=d_prime`;
  - `priority gate active: 14 concurrent streams`;
  - `48 layers x 448 experts, router sha256 for every layer`;
  - `ours: FR-Spec map hot_tokens_64k_arc.pt written`;
  - `STALE_LATCH_FIX installed`;
  - `UNTRIED_NOTICE installed mode=i1`;
- `untried_summary.json`: `errors` 0, `tracker_errors` 0, `i2_lines` 0, `i1_lines` ≥ 20;
- **I1 click-within-3 ≥ 15%** (`i1_clicked_named_within_3_share`; their bar for the base stack in exp-086, 1.5× the
  10.1% baseline);
- `stale_latch_summary.json`: `errors` 0 and **`stale_refused_after_released` = 0**;
- **0 tracebacks** outside serving teardown; 25/25 games terminal.

**Pass, outcome**:
- **≥ 3 of the 4 key levels won**: sp80 L2, tn36 L2, vc33 L4, cn04 L2, i.e. levels ≥ 2 / 2 / 4 / 2 in the per-game
  digest. Arm C by the same reading: v1 won tn36 (6) and vc33 (4) but not sp80 (1); v2 won sp80 (4) but not tn36
  (1) or vc33 (3); cn04 is not in the §11.1 digests, so read it from the artifact for v2 as the comparison;
- **levels ≥ 103 and score ≥ 44**.

**Pass, serving**: gen tok/s ≥ 750; accept 3.0-3.3; repeated assistant turns ≤ 1; retractions ≤ 4.

**Kill** (any one):
- gen tok/s < 723;
- levels < 95 or score < 40;
- a traceback raised from the new code (`ours_untried`, `cu_counters`, the install cell, or the patched
  stale-refusal lines);
- also, as mechanism kills: a missing install marker, `stale_refused_after_released` > 0, or `i1_lines` = 0 (the
  notice never fired, so the run tests nothing).

**Near-pass** (tok/s 723-750, levels 95-103, score 40-44, click share 10-15%, or 2 of 4 key levels): rerun once
before any slot.

**Also read**: tokens/request against C v2's 1,989 (each I1 line adds ~100 prompt tokens, at ~100 lines per run);
`executed_after_stale_refusal`; `untried_firings.json` on the four key games.

### 12.6 Slots

A pass makes C+U eligible; it does not earn slots by itself.
- Per §11.4 it is drawn only if C's mean holds up and JustAdev742's exp-085 hidden draws (from Oct 12) are not
  below exp-081's.
- It is then interleaved with C, as for C+LM (§10.4), and the drop rule and the n ≥ 3 final-selection rule apply.
- C+U and C+LM are separate arms on C. A combined C+U+LM is not built.

### 12.7 Risks

- **Level-1 clock on slow boots.** The I1 clock for level 1 starts at the game's first prompt. In exp-085r a slow
  boot made 13 lines fire at steps 1-2; our 2400 s grace allows the same. Those lines name objects nobody could have
  clicked yet. They lower the click share and add tokens. This is the reason the share bar is 15% and not 20%.
- **The latch fix changes behaviour only when the model catches the error.** Its effect is bounded by how often
  snippets catch, which arm C's own transcripts have not shown. `executed_after_stale_refusal` measures it. Expect a
  small effect (§11.4: 0 to +1 level per run).
- **The counters' hit rule needs the solver's history to be append-only.** That is ours_untried's own assumption. A
  shrinking history is counted as `i1_unmatched`, not as a miss. A prompt that is built and then discarded would be
  counted as a line shown.
- **Score effects are not separable by draws** (§11.4). Adoption rides on the mechanism counts, the key levels and
  the no-harm lines.

## 13. Arm C's token split between cleared and final levels, and per-level levers [MEASURED + SIM, 2026-10-12; nothing built]

Brief: arm C v2's transcripts (§11.7) showed that the largest visible loss is model-building on the last level a game
reaches, which then ends unsolved. How are tokens split between cleared levels and that final level, and is there a
harness-side lever (no model change) that would turn more of that budget into levels? The candidates were:
- a per-level token cap followed by a fresh-context restart of the level;
- earlier D′ demotion of a game stuck on one level;
- a forced "abandon this hypothesis, re-probe" prompt after N tokens without progress;
- anything else the data suggests.

**Short answer: close the line.**
- About a third of arm C's tokens (32.1%) go to the final, uncleared levels.
- Nothing separates those levels from the expensive levels that do get cleared. Eleven cleared levels cost more than
  the p90 (83k tokens), and those games went on to clear 7 more levels after them.
- In the rerun-shaped model, every cap, park and earlier-demotion variant loses or ties against D′. That includes a cap
  at the p90 with a fresh restart, unless one assumes that a restart is a free re-roll. Lesson 0037 measured the
  opposite for a real mid-level reset.
- This agrees with JustAdev742's two studies, which reached the same conclusion on eight runs and a separate simulator.

### 13.0 Data

- **Source.** kaggle-ops request `jad81-tx-alloc-1` (run 38193211301): `kernel_output` on `calamitychasm/arc3-jad-exp081`
  v2 (status COMPLETE, version 2) with `transcripts {"games": "all", "first_turns": 0, "last_turns": 1, "turn_chars": 300}`.
  All 25 `*_requests.jsonl` files were read: 2,952 responses and **5,871,402 completion tokens**. That matches the
  request-log digest (2,952 × 1,989).
  - A 26th file holds a single stray response and is ignored.
- **Unit.** The unit is completion tokens per level. Prompt tokens are ~94% prefix-cached, and the run is decode-bound,
  so completion tokens are the resource that D′ allocates.
- **How a request is assigned to a level.** Each request goes to the last `Current state: step N, level L` marker in
  its messages. A request that straddles a level-up can shift by one level, which is a small effect.
- **No wall clock.** The records carry no time (§11.7), so everything here is in tokens.
- **Converting tokens to time.** At 790 tok/s over 14 slots (56.4 per held slot), 100k tokens is about 30 active
  slot-minutes.
- **Level counts per game** were read from the public game sources (`sonpham-org_arc-3/docs/static/games/src`). They
  match §11.1.
- **Six games won every level**: ar25, cd82, cn04, lp85, sb26, tr87. Their last level counts as cleared.
  - **Correction to §11.7:** cn04's level 6 was **cleared**, because the game was won (6/6, score 100). It was not a
    level the game "ended" on.
  - So 19 games ended on an uncleared level, and the run cleared 111 levels.
- **One run.** v2 is one public-25 run at the full 25 × 121 min shape. Per-game outcomes swing by whole levels between
  identical runs (§11.1).

### 13.1 Split

- **Final uncleared levels: 1,883,902 tokens = 32.1%** of the run, on 19 levels.
- **Cleared levels: 67.9%**, on 111 levels.
- JustAdev742 measured 36.4% in their three plain D′ runs (range 25.8-43.4%, `time-allocation.md` §3.4). Arm C sits
  inside that range, so the bundle did not visibly change the split.

| final level (L / N) | tokens | requests | `action()` calls | tokens per call |
|---|---:|---:|---:|---:|
| vc33 4/7 | 205,223 | 63 | 24 | 8.6k |
| bp35 2/9 | 174,393 | 58 | 10 | 17.4k |
| su15 4/9 | 168,521 | 45 | 14 | 12.0k |
| tn36 2/7 | 163,135 | 58 | 29 | 5.6k |
| r11l 5/6 | 159,523 | 67 | 21 | 7.6k |
| ft09 5/6 | 133,758 | 55 | 13 | 10.3k |
| sc25 5/6 | 128,312 | 46 | 29 | 4.4k |
| ka59 3/7 | 121,693 | 57 | 32 | 3.8k |
| wa30 2/9 | 119,639 | 37 | 25 | 4.8k |
| dc22 5/6 | 103,741 | 84 | 34 | 3.1k |
| ls20 5/7 | 87,624 | 73 | 23 | 3.8k |
| m0r0 6/6 | 66,090 | 38 | 31 | 2.1k |
| g50t 4/7 | 62,873 | 37 | 6 | 10.5k |
| tu93 9/9 | 55,128 | 33 | 2 | 27.6k |
| sp80 5/6 | 50,835 | 19 | 4 | 12.7k |
| s5i5 6/8 | 35,946 | 17 | 8 | 4.5k |
| sk48 4/8 | 21,593 | 13 | 1 | 21.6k |
| lf52 5/10 | 15,782 | 9 | 2 | 7.9k |
| re86 7/8 | 10,093 | 5 | 3 | 3.4k |

- Most final levels are deep: 11 of the 19 sit at level 5 or later, and 8 are the last or second-to-last level of
  their game.
- Per game, total spend has a median of 251k tokens (range 74-365k).

### 13.2 Tokens to clear a level (111 cleared levels)

| level | n | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|
| all | 111 | 26.4k | 45.1k | **83.3k** | 175.1k |
| 1 | 25 | 21.2k | 35.6k | 56.8k | 152.1k |
| 2 | 22 | 26.9k | 52.5k | 120.0k | 175.1k |
| 3 | 21 | 30.0k | 46.8k | 69.0k | 156.1k |
| 4-5 | 27 | 26.4k | 47.1k | 93.8k | 108.2k |
| 6+ | 16 | 23.4k | 31.4k | 54.8k | 83.3k |
| 2+ | 86 | 26.4k | 46.5k | 84.8k | 175.1k |

- **Cost does not grow with depth.** The median is 21-30k at every level index, so the level index carries no cost
  signal. JustAdev742 found the same for hopelessness (AUC 0.49 on level index alone).
- **The JustAdev742 equivalent.** Their solved levels had a median of 21.8k and a p90 of 71.1k in the plain runs.
  Arm C's levels cost slightly more. That is consistent with more deliberation per level under the bundle, but one run
  cannot separate it from noise.

### 13.3 Spend on final levels when the game ended, and what lies past the p90

- **Final levels.** At the end of the run they had spent a median of 104k tokens (p25 53k, p75 147k, range 10-205k).
  - **11 of 19 were past the cleared-level p90 (83k).** Their spend beyond 83k is 0.65M tokens, **11.1% of the run**.
    That 11.1% is the most a p90 cap could take from final levels.
  - The other 8 were levels started late that were still short: re86 L7 10k, lf52 L5 16k, sk48 L4 22k, s5i5 L6 36k,
    sp80 L5 51k, tu93 L9 55k, g50t L4 63k, m0r0 L6 66k. A cap cannot touch them. They are the "started too late"
    share of the loss.
- **Past the p90 the solve rate falls but stays well above zero.** Solves per 100k tokens at risk (all 130 level
  attempts, final levels censored):

  | tokens on the level | 0-25k | 25-50k | 50-83k | 83-120k | 120-180k |
  |---|---:|---:|---:|---:|---:|
  | solves / exposure | 52 / 2.52M | 35 / 1.30M | 12 / 0.93M | 6 / 0.65M | 6 / 0.45M |
  | per 100k | 2.1 | 2.7 | 1.3 | 0.9 | 1.3 |

  - A level past 83k is solved at about half the rate of a fresh one. That is the same shape as JustAdev742's 0.027
    against 0.08 per active minute.
  - 12 of the 23 levels still open at 83k were solved later. Lesson 0039's arithmetic applies directly: a flag must be
    right about two times in three.
- **Expensive solved levels look like stuck ones.** These 11 cleared levels cost more than 83k:
  - sp80 L2 175k, ka59 L2 169k, lf52 L3 156k, sk48 L1 152k, cn04 L2 126k, g50t L3 121k, dc22 L4 108k, lf52 L4 105k,
    s5i5 L5 105k, sp80 L4 86k and bp35 L1 84k (plus re86 L6 at exactly the p90).
  - The same games then cleared 7 more levels after them.
  - **Deliberation per action does not separate them either.** Tokens per `action()` call have a median of 3.7k on
    cleared levels and 7.6k on final levels. But the expensive cleared levels run at 3.0-8.0k per call, and the
    expensive final levels at 3.1-17.4k.
  - Only bp35 L2 (17.4k), su15 L4 (12.0k) and ft09 L5 (10.3k) exceed every expensive cleared level. That is 3 of 11,
    in-sample, in one run, and measured over the whole level rather than at the moment the level crosses 83k.
- **Cap and end, applied to this run with no reallocation.** At K = 83k, 17 games are capped (8 of them on their
  final level).
  - The run loses 18 cleared levels: the capped levels plus every later level of the same games. That is 211 game
    points, **8.45 mean-score points**.
  - It frees 1.97M tokens (33.6%).
  - At K = 120k / 150k / 175k: 15 / 9 / 3 levels lost (7.15 / 2.92 / 1.71 points), and 20.1% / 12.0% / 3.5% of tokens
    freed.

### 13.4 Simulation: caps, demotion and restarts under D′ (rerun-shaped)

**Model.** This is the gate replay in `scripts/sim_m2_priority_gate.py` (§4.2, §8.3), extended by a scratch script that
adds cap events. The script is not committed; its logic is below.
- **Rerun shape**: 110 games, 532 min, 14 slots at 56.4 tok/s per held slot (790 tok/s, 25.2M tokens per rerun).
- **Handover**: only at context trims, the first after N(62k, 6k) generated tokens and then every N(37k, 7.5k), or
  when a game wins.
- **Priority**: D′ executed from the vendored module, including the 0.4 fade and never-started games priced by the
  formula.
- **Level costs**: lognormal with median 22k × the world's hardness, level σ 0.8, game σ 0.6, and level 1 at × 0.5.
  Some levels are unsolvable, as set by the world. Scoring is RHAE with the public games' human baselines.
- **Worlds**: `public_like` plus five hidden-like worlds that score 32-35. In the check-run shape (25 × 121 min),
  `public_like` gives 124 levels / 58.1 and `tok2.5` 79 / 30.7. Arm C v2's 111 / 48.0 lies between them.
- **Draws**: 60 paired draws per world (common random numbers); intervals are 90% bootstrap.
- **Policies.** Each acts on t, the tokens on the current level. The harness already counts it as
  `_session_generated_tokens − _tokens_at_level_start`.
  - **T0.5 / T0.67**: D′'s `token_scale` cut from 225k to 112k / 150k. This is "earlier demotion of a stuck game".
  - **end K**: end the game at t = K.
  - **park K**: re-queue the game at priority 1 at t = K, so it runs only when nobody else waits.
  - **rr K**: a fresh-context restart at t = K, at most 2 per level. The remaining cost is re-drawn from the level's
    own distribution. This is the optimistic case, where "stuck" was the run's bad luck; an unsolvable level stays
    unsolvable. D′'s t keeps counting.
  - **rr2 K**: the same, with the re-drawn cost doubled. This is lesson 0037: ours-07's real mid-level reset left 14 of
    44 touched levels solved against ~61% expected.
  - K is 83k (arm C's p90), 120k or 150k.

**Result** (Δ mean score against D′; the base row is D′'s own score):

| policy | public_like | tok2.5 | tok2_u.12 | unsolv.20 | siggame0.3 | finalU3x |
|---|---:|---:|---:|---:|---:|---:|
| D′ (score) | 63.5 | 34.1 | 34.2 | 31.7 | 34.9 | 35.2 |
| T0.5 | −1.05 | −0.78 | −0.68 | −0.45 | −1.02 | −0.67 |
| T0.67 | −0.43 | +0.01 | −0.02 | −0.23 | −0.47 | −0.43 |
| end 83k | −14.1 | −11.3 | −11.0 | −7.0 | −13.0 | −10.9 |
| end 120k | −6.5 | −1.7 | −2.6 | −3.4 | −1.8 | −2.5 |
| end 150k | −3.2 | −0.3 | −0.1 | −2.0 | −0.4 | −0.4 |
| park 83k | −5.2 | −4.0 | −4.3 | −3.2 | −4.8 | −4.7 |
| park 120k | −3.7 | −1.1 | −1.6 | −2.2 | −1.4 | −1.8 |
| park 150k | −2.4 | −0.3 | −0.1 | −1.6 | −0.4 | −0.4 |
| rr 83k (optimistic) | −0.45 | −0.81 | −1.09 | −0.27 | **+1.84** | −0.75 |
| rr 120k (optimistic) | −0.06 | −0.36 | −0.10 | −0.02 | **+1.60** | −0.38 |
| rr 150k (optimistic) | +0.03 | +0.05 | +0.04 | −0.00 | **+1.09** | −0.11 |
| rr2 83k (lesson 0037) | −3.96 | −5.56 | −5.07 | −1.89 | −4.18 | −4.96 |
| rr2 120k | −1.70 | −2.68 | −2.08 | −0.81 | −1.65 | −2.39 |
| rr2 150k | −0.90 | −1.31 | −1.02 | −0.54 | −0.83 | −1.26 |

Intervals on these differences are ±0.1-0.5. Signs outside ±0.3 are stable across draws.

- **Answer to "how many extra levels would a p90 cap buy".**
  - Cap and end at 83k changes the level count by −54 to −101 per 110 games (≈ −12 to −23 per 25 games).
  - Cap and park at 83k: −1 to −33.
  - Cap and restart at 83k: −2 to −8 on the optimistic re-roll (+8 only in `siggame0.3`), and −14 to −36 under
    lesson 0037.
  - At 120-150k, end and park *raise* the level count in several hard worlds (+5 to +12) but lower the score. The freed
    time buys shallow levels on games that start late, and the capped levels were deep (w_l = l).
- **Why.**
  - **D′ already rations stuck games.** Its patience factor C is ≈0.65 at 83k tokens on the level and reaches its floor
    near 200k. A game on its last level (B = 0) also waits behind every game with levels left until the fade.
  - **The tokens a stuck game still gets are worth at least as much as the receivers would make of them.** In arm C
    v2 the p90 cap costs 8.45 points for 1.97M tokens. At equal tokens per game, that is ≈1.0 mean-score point per
    1M tokens on the 110-game scale. Extra capacity is worth 0.16-1.15 points per 1M (+5% tokens = +0.21 / +0.56 /
    +1.45 in `unsolv.20` / `public_like` / `tok2.5`).
- **The one positive cell is an assumption, not a finding.** Restart pays only if level cost is mostly per-run luck
  (`siggame0.3`, level σ 0.95) *and* a fresh context loses nothing. The only measurement of a real restart (ours-07,
  lesson 0037) halved the solve rate, and that is the rr2 row.
- **A check run would mislead.** In the check-run shape (25 games × 121 min, 100 draws), park 150k is +0.38 / +0.28
  and park 120k is −0.03 / +0.84 in `public_like` / `tok2.5`, against −2.4 / −0.3 and −3.7 / −1.1 in the rerun.
  This is lesson 0036 again: freed time goes to parked last levels in the check run, and to early levels of late
  games in the rerun.
- **What a flag at 83k would have to achieve.** "Hopeless" means the level needs more than 250k more tokens, or is
  unsolvable. The flagged game is parked, and fp is the share of other levels flagged.

  | detector at 83k | public_like | tok2.5 | unsolv.20 | finalU3x |
  |---|---:|---:|---:|---:|
  | oracle (recall 1, fp 0) | +1.68 | +2.46 | +0.62 | +3.05 |
  | recall 0.7, fp 0.05 | +0.02 | +1.39 | −0.04 | +1.60 |
  | recall 0.5, fp 0.05 | −0.11 | +0.92 | −0.07 | +0.93 |
  | recall 0.3, fp 0.05 | −0.48 | +0.39 | −0.17 | +0.36 |
  | recall 0.5, fp 0.15 | −1.95 | +0.54 | −0.97 | +0.38 |
  | recall 0.5, fp 0.30 | −4.41 | −0.48 | −2.42 | −0.74 |

  - Even a perfect flag is worth +0.6 to +3.1. A useful one needs recall ≥ 0.5 at ≤ 5% false positives, and even then
    it is ~0 in two worlds.
  - The only candidate in this run (tokens per `action()` call above ~9k) is 3 of 11 final levels against 0 of 11
    expensive cleared levels. That is in-sample, in one run, and measured over the whole level, not at the 83k
    crossing.
  - JustAdev742's on-level features (actions, rate, tokens per action, stretches without an action) carried no
    signal at 5-15 minutes over six runs.

### 13.5 The "abandon this hypothesis, re-probe" prompt

A token-clock model cannot price it, because it changes what the model does, not who gets the GPU. The evidence
against it:
- **Our strategy audit (§5).** The check run fired once in 25 games at a 56K-token threshold, and the audited level
  was not cleared.
- **Similar public variants.** sujanmajhisuzan's "strategy audit after 12 prompts, telling the model to UNDO/RESET" has
  no draw outside copy noise (JustAdev742 `intel-oct10.md`).
- **Lesson 0037.** Discarding the working hypotheses at 20 minutes halved the solve rate.
- **It would fire as often on levels about to be solved as on stuck ones.** At 83k the open levels split 12 solved
  later against 11 not. The levels it would hit are the expensive-but-solving ones (sp80 L2, ka59 L2, lf52 L3, sk48 L1
  ...) as often as the stuck ones.
- **What the end-of-game excerpts show.** In §11.7 (sk48 L4, sp80 L5, g50t L4), the model was still building a model
  or searching when the clock ran out. Its hypotheses were not spent.

The one prompt-side change with a measured mechanism is a *fact* notice (I1, click-within-3 21.5% vs 10.1%). That is
already in arm C+U (§12). An "abandon" instruction carries no fact.

### 13.6 What else the data says

- **The loss is depth, not waste on the stuck level.** Per-level cost is flat in depth (§13.2), and final levels sit at
  level 4-6 of 6-10.
  - The time that would buy those levels is spread over everything the game did before it reached them.
  - The lever that is positive in every world is more tokens per game: +5% capacity is worth +0.2 to +1.5 points.
  - That is a serving lever (throughput, §9.5's drafter, which is private), not a harness one.
- **About 8 of the 19 final levels lost to lateness, not to being stuck.** These are the levels started late (§13.3,
  10-66k tokens when the run ended).
  - In the rerun, D′'s fade already decides these trade-offs.
  - JustAdev742's one positive scheduler tweak, last-level B = 10, was +0.26 [−0.35, +1.09] in their main model and ≈0
    in the hard worlds. That is not worth a slot.

### 13.7 Recommendation: close the line (pre-registered)

- **Do not build** any of the following:
  - a per-level token cap, with or without a fresh restart;
  - a park or end rule;
  - a smaller D′ `token_scale`;
  - an "abandon / re-probe" prompt.
- **Each is negative or a tie in the rerun model** under every world except one optimistic restart assumption, which
  lesson 0037 contradicts.
  - A public-25 check run would show the park variants as small gains (lesson 0036), so a check run cannot rescue them.
- **No GPU, no kernel, no slots.**
- **Reopen only if both of these hold:**
  1. A per-level signal, computed at the moment a level crosses 83k tokens, reaches recall ≥ 0.5 for "not cleared by
     the end of the run" at ≤ 5% false positives among levels that were cleared. It must be measured out of sample on
     at least three full-length arm-C-family runs (v2 plus the C+U and C+LM check runs, §10, §12) and on more than one
     game.
  2. The rerun model then shows ≥ +1 point in at least four of the six worlds.
- **What reopening would cost.** Testing condition 1 needs a per-request token series per level (the `transcripts`
  digest currently gives only level totals): a ~30-line `kaggle_ops.py` change and no GPU. It is worth doing only if
  those check runs land anyway.

### 13.8 Reconciliation with JustAdev742 (lessons 0036, 0037, 0039; `time-allocation.md`, `hopeless-signal.md`)

Their two studies used eight D′ runs and a simulator validated on three of them. They found:
- no give-up or cap that beats D′ (cut-offs ≤ 40 active minutes lose 1.7-29 points; 50-60 minutes fall between −0.9
  and +0.3);
- an entry signal of AUC 0.68-0.70, too weak to use, and no on-level signal;
- every one of 48 end / park / demote policies losing.

This section reaches the same answer on arm C's own run with an independent replay, and adds three things:
1. **The split holds with the bundle.** 32.1% of arm C's tokens are on final levels, against 25.8-43.4% for their plain
   runs. The bundle did not visibly change where tokens go.
2. **Restart and earlier demotion lose too.** A fresh-context restart at the p90 loses under the measured reset penalty
   and ties under a free re-roll. Earlier D′ demotion (T0.5, T0.67) is between +0.01 and −1.05 in every world. Their study did
   not test either directly; it tested the C floor (0 vs 0.1, ≈0).
3. **The p90 cap would cut solves.** Arm C's p90 (83k ≈ 25 active minutes) sits where their hazard is still
   0.046-0.027 per minute. In arm C v2, 12 of the 23 levels open at that point were solved later.

The two results differ in one respect only: their tokens-per-action feature was measured at 5-15 minutes, while the
in-sample hint in §13.3 is at the end of a level. That leaves the narrow reopen condition in §13.7, not a build.

## 14. Arm C+hic16: the system-RAM KV tier and 16 streams on exp-081 [CODE + BUILT, 2026-10-12; never pushed]

Kernel `calamitychasm/arc3-jad-exp081-hic16`, `kaggle_submission_jad_exp081_hic16/notebook/`, built by
`scripts/_build_jad_exp081_hic16_kernel.py`; tests `tests/test_jad_exp081_hic16.py`. Motivation: §13.6 found that the
only lever positive in every simulated world is more tokens per game (+5% capacity = +0.2 to +1.5 points), and that it
is a serving lever. On arm A the host tier + 16 streams measured +7.7% gen tok/s (`stage7_m2_speed.md`,
"Turbo-tail-hic16"); arm C has arm A's serving (below), so this is the cheapest throughput test available on C.

### 14.1 What changes against arm C v2

Arm C v2's committed notebook (sha256 `d108fa76…`; read, never rewritten, because the scheduled C submits pin version
2) plus exactly these edits; a test pins the diff line by line:

| cell | arm C v2 | C+hic16 |
|---|---|---|
| setup (`setup_env`) | `'ARC3_MAX_ACTIVE_STREAMS': 14` | **16**. The only stream knob of the harness: Franzen's patch builds `_PriorityGate(slots)` from it and logs `priority gate active: 16 concurrent streams`. D′ (`install_d`) replaces the priority value, the pace tracker and the fresh-game queueing; it never reads or sets the slot count. |
| launcher `CFG` | `MAXREQ=14`, `CUDAGRAPH_MAXBS=14`, `MAMBA_CACHE=84` | **16 / 16 / 96**. Arm C's own Mamba formula is 6 per stream (84 = 6 × 14, the same rule as our builder's `mamba_cache`), so 96; 96 // 5 = 19 ≥ 16 (SGLang caps running requests at `max_mamba_cache_size // 5`). The graph-batch list is the launcher's own formula, `{1, 2, 4, 7, 8, 9, 10, 16}`, as in turbo-tail-hic16. |
| launcher args | — | `--enable-hierarchical-cache --hicache-size 32 --hicache-write-policy write_through --hicache-io-backend kernel`, after the prefetch flag: the hic16 kernels' line (`_build_m2_speed_kernels.hicache_line`), reused. |
| launcher, before the launch | — | `STREAMS max_running_requests=16 cuda_graph_bs=[…] mamba_cache=96` and `HICACHE requested --hicache-size 32 GB (write_through, kernel io)` (with an assert that the flags are in `args`): the hic16 kernels' print lines. |
| launcher, after the health loop | — | the hic16 kernels' serve.log check (`_build_m2_level_memory_kernel.HICACHE_CHECK`): `HICACHE_TIER attached hicache_attached=True | …` only if SGLang logged `hicache_attached=True`; `HICACHE_TIER NOT ATTACHED` / `NOT CONFIRMED` otherwise. Never raises. |
| run cell, after `bm.run` | — | the same check again (`HIC_DUMP`, `… (post-run)`), because the launcher's check runs before a slow boot finishes (it did on arm B's hic16 run). |

Unchanged and byte-identical: cell 0 (its fork note still lists arm C's `MAXREQ=14` etc.; it is description only, left
as is like the other arm-C derivatives), D′, REAP-448, acceptance 0.5, the ARC map, the six patches, `MEMFRAC` 0.96,
`SERVER_STARTUP_TIMEOUT` 12 min, the 2400 s first-request grace and the 25 × 121 min Save & Run. The `[sys] RAM
used/total` census (every 2 min outside a rerun) is arm C's own cell, the same census our hic16 kernels print and
`kaggle_ops.sys_ram` parses, so it is reused unchanged. `kernel-metadata.json` differs from arm C only in `id`, `title`
and `code_file`. The notebook is 800,239 bytes. Flags SGLang rejects with the tier (`--disable-radix-cache`,
`--enable-int8-mamba-checkpoint`) are absent (tested). The launcher is otherwise identical to turbo-tail-hic16's (the
two launcher cells differ only in marker prints and comments), so the tier's flag set has run on this exact serving
config once.

### 14.2 Why the tier helped A (+7.7%) and not B (+1.6%), and which one C resembles

The data (one check run each, 25 games × 25 min):

| | A, 14 streams | A + tier, 16 | B, 14 streams | B + tier, 16 |
|---|---:|---:|---:|---:|
| gen tok/s | 772.4 | 831.6 (+7.7%) | 657.4 | 667.97 (+1.6%) |
| accept median | 3.02 | 3.06 | 2.66 | 2.62 |
| KV peak | 0.93 | 0.99 | 0.92 | 0.99 |
| retractions | 2 | 2 | 2 | 2 |
| tokens/request | 1,779 | 1,736 | 1,579 | 1,770 (+12%) |
| boot | not recorded | not recorded (`DEADLINE at` was not grepped then) | not recorded | **`DEADLINE at 721s`**; after-boot tier check `NOT CONFIRMED` (server still loading at the 12-min release) |

The tier itself adds no device KV. It makes evictions cheap (write-through pages reload over PCIe instead of
re-prefilling), so the 16th stream can run on an oversubscribed pool without retractions. The gain is the extra
concurrency minus the per-step cost of a bigger batch. Both arms reached a 0.99 peak with 2 retractions, so the tier
worked the same way in both.

Three explanations, in order of weight; none can be settled from the digests we have:

1. **Boot time in the denominator (most likely the largest part).** `gen tok/s` is the harness's `generated
   tokens/sec` in `summary.txt`, and the game clocks start at the release (`bm.run`), which is clamped at 12 min after
   the notebook started, not at server-ready (`stage7_m2_speed.md`, boot-time audit, point 2). Every minute the server
   boots past the release is idle time in a 25-minute run: about 4% per minute. B + tier printed `DEADLINE at 721s`
   and its post-boot check found no tier line, so its server was not ready at the release. A + tier's boot was never
   recorded. The same pattern shows on arm C itself, on an identical notebook: v1 printed `DEADLINE at 722s` and ran
   728.7 tok/s; v2 printed `READY after` and ran 789.9 (−7.7% for v1). A 32 GB pinned host pool could also lengthen
   the boot by itself. That is unmeasured.
2. **Single-run noise.** Arm C v1 and v2 (identical code) differ by 8%. The difference between two single-run deltas
   (+7.7 vs +1.6) is therefore within about one sd of zero, before any boot correction.
3. **Acceptance (weak).** To first order, more streams raise throughput by the same factor whatever the acceptance:
   the draft (3 MTP steps) and verify (4 tokens per request) cost the same in both arms, and acceptance only scales
   tokens per step. Two second-order effects favour A: lossy acceptance makes each step yield ~16% more tokens
   (3.06 vs 2.62), so a larger share of wall time is decode rather than host-side work; and B + tier's requests got
   12% longer, which keeps more context per stream on the pool. Neither predicts a 6-point gap.

**Arm C resembles A.** Its serving is A's: REAP-448, acceptance 0.5 / 0.5 (accept median 3.12-3.14), the ARC map, 14
streams / Mamba 84, `MEMFRAC` 0.96; the launcher cells differ only in prints. C differs from A in the harness (D′
instead of PRIORITY_TAIL, the six-patch bundle, no history cache) and in the check shape:

- **The full-length shape dilutes boot loss.** One late minute is 0.8% of a 121-minute run against 4% of a 25-minute
  run, so this check run measures the tier more cleanly than either hic16 check did.
- **Held slots that are not decoding.** C v2's running median was 13 at 14 slots. A slot is held while the game runs
  Python, plays actions or waits on a handover, and in v1 the samples at 2-5 running requests were ~19% of the total.
  Two more slots raise the running count in proportion only while ≥ 15 games want to run. With 25 games ticking for
  121 minutes and 6-7 won early, that is most of the run.
- **KV.** C v2 peaked at 0.91 of 1,477,888 tokens at 14 streams. Scaled to 16 streams that is ~1.54M tokens against a
  pool of ~1.43M (Mamba 96 costs ~50k tokens; A + tier's pool was 1,427,968), about 108% of the pool. A was at about
  110%, and its tier absorbed that with 2 retractions.
- **Host RAM.** C v2's `MemAvailable` minimum was 61.6 GiB. The 32 GB tier is 29.8 GiB, so expect a minimum near
  31-32 GiB (A + tier measured 29.9). That leaves 3× the 10 GiB bar.

Expectation: if C gains as A did, 789.9 × 1.077 ≈ 851 tok/s, just over the 845 pass bar; if it gains as B did, ≈ 802,
a near-pass. The pass bar is about the size of the identical-run spread (C v1/v2), so one run can land on either side
of it. That is the reason for the near-pass rerun rule.

### 14.3 Check run (pre-registered; full-length shape, 25 games × 121 min, ~2.0-2.3 GPU-h)

Baseline: arm C v2 (789.9 gen tok/s, accept 3.12, KV peak 0.91, retractions 2, tokens/request 1,989, MemAvailable min
61.6 GiB, 111 levels, 48.03). Push only into a free GPU slot, with `jadh16-push-1` (CLAUDE.md).

**Pass** (all of):
- gen tok/s **≥ 845** (+7% over C v2's 789.9);
- `HICACHE_TIER attached hicache_attached=True` (the post-run line counts);
- 16 streams: `priority gate active: 16 concurrent streams` and `STREAMS max_running_requests=16 cuda_graph_bs=` in
  the notebook log;
- `MemAvailable` ≥ 10 GiB in every `[sys] RAM` census line (`sys_ram.all_ge_10`);
- retractions ≤ 4;
- accept 3.0-3.3;
- tokens/request within ±10% of 1,989 (1,790-2,188);
- 0 tracebacks outside serving teardown; 25/25 games terminal;
- levels ≥ 103 and score ≥ 44;
- arm C's other submit markers present (`harness patch applied successfully`, `our harness patches applied
  successfully: 6`, `#OURS_FORM ok version=d_prime`, REAP `48 layers x 448 experts`, `FR-Spec map
  hot_tokens_64k_arc.pt written`).

**Kill** (any one): gen tok/s < 790 (no better than C v2); the tier not attached (`HICACHE_TIER NOT ATTACHED`, or no
attached line even after the run); `Not enough host memory` anywhere; levels < 95; score < 40.

**Near-pass**: 790-845 tok/s with every other pass line met. Rerun once before any adoption.

**Also read** (not pass lines): `DEADLINE at` / `READY after` (a late boot costs ~0.8% per minute here; record it
next to the tok/s); the speed digest's `by_running` (decode tok/s and sample share at 15-16 running requests against
13-14) and `running_med` (C v2: 13); KV pool and peak (a peak pinned at 1.0 with retractions means the tier is not
absorbing the overflow); the `oom` hits (the benign SGLang lazy-Triton advisory in hic16 runs; read them with
`show_lines`, and check `killed` 0); `#OURS_FRESH replaced=` (25 in C v2).

### 14.4 Slots

A pass makes C+hic16 eligible; it does not by itself earn slots.
- It is the same mechanism as arm C with better measured throughput, so under the CURRENT STATUS rule ("a variant
  that passes its check run may take over an arm's remaining slots only if it is the same mechanism with better
  measured throughput") it may take over arm C's remaining slots. Decide at the 10-18 exploit decision, not before.
- Cost of switching: arm C's draws pool with JustAdev742's exp-081 draw (34.04) as a verbatim fork. C+hic16 draws do
  not. Report them as a separate arm, or pool them with C only as "same harness, faster serving".
- A near-pass or a pass that rests on a late-boot C v2 comparison is not adopted on one run.
- C+hic16 is not combined with C+LM or C+U. Each of the three is one change against C.

### 14.5 Risks

- **Boot time.** Allocating and pinning a 32 GB host pool may add to the boot. The release stays at 12 min, so in the
  check run a later boot costs ~0.8% of tok/s per minute. In a rerun, the 2400 s grace covers a boot up to ~57 min
  after the notebook starts.
- **Host RAM in a rerun is unmeasured.** The census runs only outside a rerun (`if not TRUE_SUBMISSION`). A rerun has
  ~110 game sessions where the check run has 25. SGLang checks `MemAvailable − 10 GiB` only when it allocates (a
  too-large tier fails loudly at boot). A host OOM after boot would kill the server mid-run. Neither hic16 kernel has
  ever run in a rerun. The check run's minimum `MemAvailable` (expected ~31 GiB) is the margin for the 85 extra
  sessions.
- **Thin device headroom.** A + tier logged 0.84 GiB free when SGLang lazily loaded Triton kernels (an advisory, not an
  allocation failure). Mamba 96 takes ~0.67 GB more than 84. `MEMFRAC` stays at 0.96, because 0.98 OOMs.
- **Graph padding.** With 11-15 running requests the decode graph pads to 16, as in turbo-tail-hic16. A + tier showed
  no per-request gain at 15-16 over 12-14 running (`stage7_m2_speed.md`), so the gain, if any, comes from the share of
  time at higher concurrency, not from faster steps.
- **Score.** The score effect of +7% tokens is +0.2 to +1.5 points in the §13 model. That cannot be measured by hidden
  draws. Adoption rides on the throughput line and the no-harm lines.

## 15. Reproducing JustAdev742's MTP drafter fine-tune for arm C [READ + BUILT (L0), 2026-10-12; never pushed, no GPU]

**Question.** JustAdev742 fine-tuned the dense weights of the MTP draft head; the experts stayed frozen. Their output
weights are private (the output of kernel `scottmahony/arc3-mtp-session-a`). Can we rerun their published pipeline to
make our own private draft and mount it in an arm-C-plus-drafter kernel ("C+D")?

**Source.** Their repo at HEAD `ccb7214` (fetched 2026-10-12; no newer commit). The repo is Apache-2.0.
- Design doc: `docs/research/beat-tufa/mtp-drafter-finetune.md` (sections 0-12).
- Run records: `docs/research_log.md` 2026-10-10 00:50 to 13:28, and `docs/SUBMITTING.md`.

**Answer.** Yes. The code is complete and has already produced a working draft on our exact serving stack. But it
cannot run unchanged for us, because three of its inputs are their private Kaggle artifacts. The measured gain on an
arm identical to ours is about **+3% run-level output tok/s**. That is roughly +0.1 to +0.9 LB points: real but
unmeasurable by draws. The recommendation is a **conditional GO on a lean path** of about 3.5-4 GPU-h (15.6).

### 15.1 Code

Everything below is in `scripts/` and is Apache-2.0 (their `LICENSE`).

| Part | File | Lines |
|---|---|---|
| Session-A notebook builder | `build_mtp_session.py` | 436 |
| Session orchestration, steps A0-A11 | `mtp_session_a.py` | 1,388 |
| Hidden-state dump patch | `sglang_hc_dump_patch.py` | 1,006 |
| Snapshot replay driver (dump) | `hc_dump_driver.py` | 548 |
| Probe-request dump | `mtp_probe_dump.py` | 245 |
| Replica of the draft block | `mtp_replica.py` | 1,069 |
| Trainer | `mtp_train.py` | 812 |
| Exporter | `mtp_write_draft.py` | 389 |
| Probe notebooks (`--probe`) and the gate | `fidelity_sample.py`, `fidelity_probe.py`, `probe_accept_gate.py` | – |

Details:
- **Patch file.** The dump patch is an anchored, sha-checked edit to the wheel's model file, in the style of their
  REAP patch.
- **Notebook base.** The notebook is D′ up to its launcher, plus the nine scripts written as compact cells, plus REAP-448
  and the ARC FR-Spec map.
- **Builder version.** Their `build_franzen_nb.py` at `ccb7214` is byte-identical to the copy vendored for arm C at
  `c7a462b`. It already has `--draft` and `--probe`.
- **Tests.** The repo has about 30 tests for the session, including whole GO / NO-GO sessions against stub servers.
  It also has torch tests for the replica, the trainer and the exporter.

**What stops it running as-is.** These are build-time constants:
- The kernel id is hard-coded as `scottmahony/<slug>`.
- **Training logs** are the kernel output of their exp-073, `scottmahony/arc3-dprime-reap448-r14-full`, which is
  private. The 25 file names and byte sizes are pinned in `kaggle/mtp/exp073-request-logs.json`.
- **Probe prompts** come from `scottmahony/arc3-fidelity-prompts`, a private dataset. It holds 154 requests from
  11 held-out games, sampled from Franzen's public run.
- **Replica-check reference.** This is `fidelity.json` from `scottmahony/arc3-fidelity-base`, a private kernel, pinned by
  sha256. That probe run used D′ with no REAP, the generic map, the stock draft, greedy decoding and lossless acceptance.

Our own build can replace all of these from a wrapper that monkeypatches the constants, as
`scripts/_build_jad_exp081_kernel.py` already does for arm C. Their files would be vendored unmodified.

**Known gap.** The research log at 13:28 asks for a pre-read of the model shards before a *second* server start. It was
never built (15.3).

### 15.2 Training data

**What it trains on:**
- Franzen-harness request logs (`<game>-<hash>_p0_requests.jsonl`: exact messages, inline images, template kwargs).
- These are replayed as maximal snapshots through a dump server: no speculation, no CUDA graph, radix cache off,
  REAP-448 on.
- The server writes the target's final hyper-connection hidden states for assistant spans, plus 256 context rows
  before each span, in FP8.
- The loss is forward KL to the target's own logits. Sampled tokens therefore choose only the contexts, never the
  labels.

**Hygiene.** The pipeline drops assistant turns that exactly repeat an earlier turn, and turns that contain "stuck in a
loop". It holds out 11 games (ar25 ft09 lp85 r11l re86 sb26 sc25 tn36 tr87 tu93 vc33).

**Their run:**
- 53 of 75 planned snapshots, stopped at the 33.5 GB dump cap.
- 2.75M training rows, from 10 games. The data source was not the binding limit.
- The source run, exp-073, used lossless acceptance.

**Our equivalent: arm C v2's own output.** It is the best fit available, because it is exactly the traffic C+D will
serve: the bundle harness, REAP-448, the ARC map and 14 streams.
- 25 per-game logs, plus one stray file with a single response, to be ignored.
- 2,952 responses and 5.87M completion tokens.
- 34-123 MB per file, flat layout (§11.7, §13.0).
- 0 repeated assistant turns.

How to use it:
- Mount it as a kernel source: `calamitychasm/arc3-jad-exp081` mounts its latest version, which is v2. **Do not push
  a v3 before session A has run.**
- Pin the file names and sizes from a kaggle-ops listing.

Two caveats:
- **Relaxed acceptance.** Arm C ran 0.5/0.5, so its contexts are slightly draft-biased. Their doc §12.5 option 3
  judges this acceptable, because the KL target is the target's logits.
- **Probe prompts would need our own dataset.** Sampling 154 prompts from the same logs with `fidelity_sample.py`
  needs a dataset upload: about 40 MB, and kaggle-ops has no dataset-create op.

The C+LM and C+U logs (today's runs) could be added for variety. But the dump cap, not supply, binds the v1 recipe.

### 15.3 Compute and quota

| Run | Wall / GPU-h | Notes |
|---|---|---|
| Session A v1 | 0.4 | Stopped at A2: storage 4-5x slow; the fixed 20-min boot wait was hit. Fixed (boot wait up to 50 min while the log grows) |
| **Session A v2** | **88 min, ~1.5** | GO. 2 server boots (709 s and 1,601 s), dump about 25 min, A9 training 356 steps in 30 min |
| Probe B0 (stock draft) / B1 (new draft) | ~0.5 / ~0.6 | B1 − B0 = +0.26 seq, +0.31 conc |
| Session A2 (v2 recipe: 50 GB, 3 epochs) | 79 min run, after **~5 h held in the queue** | **FAILED** (infra). The second server load (A6) hit a cold page cache, read ~8-9 MB/s, and had 22 of 38 shards after the 50-min boot cap. Whether the 5 h queue hold counted against quota is unknown |
| A full-length production check (exp-083/084) | ~2.4-2.5 each | |

**Quota.** We have used about 11-13 of 30 h this week (resets Sat 2026-10-17 00:00 UTC), so about 17-19 h are left
before any other queued run. The session's own deadline must be capped to bound the burn. The builder refuses a value
under its steps' worst case (325 min with all steps), so the lean path below must drop steps for a cap of about 3-4 h
to be accepted.

**Failure modes and mitigations:**
1. **Slow storage** (their lesson 0038), mostly near the quota reset. Do not push late Friday or early Saturday.
2. **The cold-cache second boot.** The lean path below has *one* server start, the training-dump server, which loads
   while the precache warms the page cache, as production boots do. That removes this failure mode outright.
   - The full path needs the pre-read (about 4 min sequential) or `--set budgets_min.boot_max=100`.
3. **The 2-GPU-session limit** and the first-push-rejected mount bug: push only into a free slot.

### 15.4 Plumbing

**How the draft is mounted.** Session A's A11 writes `/kaggle/working/mtp-draft/`: the stock draft's 12 files with the
26 dense tensors replaced (4.16 GB), plus `arc3-draft-manifest.json`. The frozen tensors are checked byte-identical by
sha256. Their submission notebooks mount it as a **kernel source**.
- **Proven in a scored run.** exp-083 drew 26.04 on the hidden set while mounting `scottmahony/arc3-mtp-session-a`,
  so the competition rerun does mount another private kernel's output.
- **The one hazard: a kernel source mounts the latest version.** Their rule is: never push a new version of the session
  kernel, and train later drafts under a new slug.
- **Private model (optional).** They advise converting the draft to a private Kaggle model before a final selection
  depends on it. That means downloading 4.16 GB (a GitHub Actions runner has room) and running
  `kaggle models instances create`. Neither op exists in kaggle-ops.
- **Our recommendation.** Keep the kernel source and freeze the slug.

**Rules.**
- `rules.md` limits *external* data and pretrained models to publicly available ones. Weights we train ourselves, from
  our own runs of the public games, are our own solution's artifact, like its code.
- `docs/SUBMITTING.md` and the scored exp-083 show the mechanism works.
- One condition is prize-only: a winning entry would have to publish the training code and the weights (the CC-BY /
  OSI clause). Their code is Apache-2.0, so this is unproblematic.
- The weights are a derivative of the stock draft head and carry its licence status.
- Neither `rules.md` nor their docs forbid mounting a private kernel output. Nothing found says otherwise.

**Arm C's notebook change.** Rebuild with the vendored builder plus
`--draft calamitychasm/<session-slug> --draft-manifest <pulled arc3-draft-manifest.json>`, under a **new kernel
slug**, so arm C v2 stays as submitted. The builder then:
- drops the stock-draft model source and adds the kernel source;
- replaces cell 4's `DRAFT_MODEL_DIR` with a wait of up to 600 s in either mount layout;
- checks the sha256 of the manifest and of the dense shard, and prints
  `ours: MTP draft <path> after N s (manifest <sha12>, dense <sha12>)`. Add that line as a submit marker.

Everything else is byte-identical to arm C v2: the six patches, D′, REAP-448, acceptance 0.5/0.5, the ARC map,
14 streams and the 2400 s grace. This is exactly their exp-084 recipe with our weights.

### 15.5 Risks to output quality

1. **Sharpening under 0.5/0.5** (their §6.4). A better draft proposes the target's mode more often, so more tokens are
   snapped or boosted, and output moves toward greedy decoding. Measured on the bundle arm, exp-084 against exp-081:
   - exact repeated turns 0 / 3,022;
   - output tokens per request 1,995 against 2,015;
   - NameErrors 66 against 35-60, and tracebacks 305 against 224-252 per ~3,000 tool results: the high end of the
     spread;
   - score 48.83 / 114 against 50.00 / 113.

   On the base arm (exp-083), tokens per request fell 12%: shorter, more frequent turns, with no repeats.
2. **Hidden-set evidence is null at n = 1.** exp-083 (base + draft) drew 26.04, against 27.97 for the same base
   without it.
3. **Run-level gain is smaller than the accept gain.**
   - Acceptance length rose 3.14 → 3.33 (+6%).
   - Decode tok/s at equal batch rose +3.5% to +8%.
   - Output tok/s over the whole run rose only 803 → 827 (+3.0%) on exp-084, and was flat (804) on exp-083. With the
     draft, the run spends less time at a full batch.
4. **Self-reinforcing data.** Training on relaxed traffic can teach copy continuations. Mitigations: the hygiene filter
   (0 repeats in arm C v2) and KL to the target's logits.
5. **Replica drift.** Their replica check passed twice on this exact wheel and model (mean diff −0.039, Pearson
   0.95). The lean path skips it, so a mismatch would surface only in the C+D check run.

### 15.6 Recommendation (pre-registered)

**Conditional GO, lean path (L).** Run it this week only on spare quota, and never ahead of a scheduled arm's check
run. Adoption rides on throughput alone, because draws cannot measure +3%. This follows CLAUDE.md's rule that a variant
may take over an arm's slots if it is the same mechanism with better measured throughput.

| Step | Work | GPU-h |
|---|---|---|
| L0 (CPU, ~1 agent-day) | Vendor their 9 session scripts, `build_mtp_session.py` and tests at `ccb7214`, unmodified. Write a wrapper that sets the owner and pins arm C v2's 25 log names and sizes. For the lean shape, the wrapper drops A2, A3, A5 and A10 and the prompts and reference inputs. Settings: `session_hours` about 4; `train.minutes` 75 (their v1 recipe: 34 GB, 2 epochs); hold-out games as theirs. Run their CPU stub-session tests against the trimmed notebook | 0 |
| L1 | Session A (lean): cell 12 (5 min), one boot (10-27 min), A7 dump (25-45 min), A9 training (30 min), A11 export (5 min) | **~1.2** (≤ 2.5 on slow storage) |
| L2 | C+D check run, full length (25 × 121 min) | **~2.3** |
| Contingency | One retrain, or the full path's probes: R + B0 run in parallel (~1.0) and B1 (~0.6), which needs the prompts dataset | ≤ 2.5 |
| **Total** | | **~3.5 nominal, ≤ 6.5 worst** |

The full path (their exact pipeline) adds the prompts dataset and the R, B0 and B1 probes, about +1.6 GPU-h. It is
worth taking only if L1's offline gain is borderline.

**Drafter gate.** Read from session A's A9 held-out evaluation on the 11 held-out games.
- **Pass:** realized accept gain ≥ **+0.20**, about +7% over about 2.75 (theirs +0.24); greedy gain ≥ +0.20; step-1 KL
  down by at least 40%; top-1 agreement up at all three steps; A11 manifest written with the frozen tensors
  byte-identical.
- **Kill:** gain < +0.10.
- **Between +0.10 and +0.20:** no C+D run. Run the full path's B0/B1 probe instead (pass mark +0.08, with the CI
  above 0 in both passes).

**C+D check run gate.** Baselines are arm C v2's: accept 3.12, 789.9 gen tok/s, decode median 920, 1,989
tokens/request, 0 repeated turns, 2 retractions, 111 levels, 48.03.

| Outcome | Criteria |
|---|---|
| **Pass** | All of the following: (1) all six of arm C's submit markers plus the `ours: MTP draft` line with the expected sha prefixes; (2) accept median ≥ **3.27**; (3) gen tok/s ≥ **805** (+2%) or decode median ≥ **948** (+3%); (4) exact repeated assistant turns ≤ 1, and "stuck in a loop" loop reports ≤ 2; (5) tokens/request 1,790-2,190; (6) tracebacks and NameErrors per tool result ≤ 1.3× arm C v2's; (7) retractions ≤ 4, 0 tracebacks outside teardown, 25/25 terminal; (8) levels ≥ 100 and score ≥ 42 (catastrophe check only) |
| **Kill** | Any one of: draft mount or sha failure; accept < 3.17; both gen tok/s < 790 and decode median < 920; repeated turns > 2; levels < 95 or score < 40 |
| **Otherwise** | Rerun once |

**What a pass buys.**
- C+D takes arm C's remaining slots.
- If C+LM or C+U is adopted instead, rebuild it with `--draft` (about 2.3 GPU-h check each).
- The weights stay valid only for this exact target and REAP list (their safeguard S6).

**No-GO triggers:**
- the week's quota is needed for a scheduled check run;
- L0 shows A9 depends on the dropped steps;
- arm C's own draws by 10-17 make another family the exploit arm.

The no-GPU alternative is to ask JustAdev742 to publish their draft as a public Kaggle model.

### 15.7 L0 built (2026-10-12) [BUILT, CPU only; never pushed]

**What exists.**

| Piece | Where | What it does |
|---|---|---|
| Their code, unmodified | `kaggle_submission_mtp_session/vendor/` | `build_mtp_session.py`, `mtp_session_a.py`, `sglang_hc_dump_patch.py`, `hc_dump_driver.py` (+ `fidelity_probe.py`, `fidelity_sample.py`, which it imports), `mtp_probe_dump.py`, `mtp_replica.py` (imported by the trainer), `mtp_train.py`, `mtp_write_draft.py`, their builder deps (`build_franzen_nb.py`, byte-identical to arm C's c7a462b copy; `franzen_tree.py`; `sglang_reap_patch.py`), `kaggle/` data their builder reads, and their CPU tests. `LICENSE` (Apache-2.0), `NOTICE`, `SHA256SUMS` (64 files, all equal to `ccb7214`) |
| Session kernel | `calamitychasm/arc3-mtp-session-a`, `kaggle_submission_mtp_session/notebook/` (882 KB, pinned) | Built by `scripts/_build_mtp_session_kernel.py` |
| Training-log manifest | `kaggle_submission_mtp_session/arc3-jad-exp081-v2-request-logs.json` | Arm C v2's 25 log names (the public set's game ids, identical to exp-073's listing and our own recorded runs); sizes `null` (not listed yet); response count 2,952 |
| C+D builder | `scripts/_build_jad_exp081_draft_kernel.py` -> `calamitychasm/arc3-jad-exp081-draft` | Arm C rebuilt through their vendored builder with `--draft calamitychasm/arc3-mtp-session-a --draft-manifest FILE`. **Refuses to build without a real session manifest; no notebook is committed** |
| kaggle-ops | `scripts/kaggle_ops.py` | `arc3-mtp-` added to `GPU_KERNEL_PREFIXES`; `kernel_output` gains read-only `list_files` (name, bytes, sha256 of a glob; one `LIST_FILES {json}` line) and `print_files` (small text files in full, with sha256; JSON also as one `PRINT_FILE_JSON` line) |
| Tests | `tests/test_mtp_session_lean.py` (20), `tests/test_jad_exp081_draft.py` (11), `tests/test_kaggle_ops_output_files.py` (2) | Below |

**How the wrapper overrides their builder without editing it.** `_build_mtp_session_kernel.py` imports their
`build_mtp_session` in a fresh interpreter and replaces five module attributes for that run only:
- `LOGS_MANIFEST`: our manifest, so the logs input is the kernel source `calamitychasm/arc3-jad-exp081` (mounts its
  latest version, v2) and the build record says so.
- `STEP_CELLS`: A2, A3, A4, A5 and A10 dropped; A1, A6, A7, A8, A9, A11 and the summary kept.
- `_setup_cell`: the session's inputs are the logs only (their private probe-prompts dataset and replica reference
  are not inputs and not in the metadata); the build record names the wrapper and the dropped steps.
- `worst_case_minutes`: the lean steps (one boot): 265 min, accepted by `session_hours` 4.5.
- `_steps_markdown`: the step list's header.

With their own `--set`: `session_hours=4.5`, `train.reserve_minutes=20` (only A11 follows A9), `storage.probe_gb=0.0`
(no probe dump). Unchanged from their v1 recipe: REAP-448 on the dump server, the 11 hold-out games (ar25 ft09 lp85
r11l re86 sb26 sc25 tn36 tr87 tu93 vc33; 14 train games), the repeat / "stuck in a loop" / covered-turn hygiene of
`hc_dump_driver.py`, the 34 GB dump cap (held-out part 6 GB), 30 + 20 min dump budgets, the 75-min training cap with
their restart watch, the ARC FR-Spec map A9 trains with, the exporter and its check. Metadata: ours, private, GPU,
no internet, `NvidiaRtxPro6000`, their pinned image, datasets/models/competition as arm C, kernel source arm C.

**Our four cells** (marked `# >>> calamitychasm (lean path)`):
1. After their setup cell (inputs + A0): the data check. Waits up to `wait_inputs_s` for the 25 names, counts
   responses with completion tokens over all `*_requests.jsonl` (the digest's rule) and **stops the session unless the
   count is 2,952** (arm C v2's; another version of the kernel output would differ). Prints
   `ours (lean): data loaded N responses (T completion tokens) from F request logs, X GB` and one line per split
   (train 14 games, holdout 11 games). While the sizes are unlisted this is the version pin.
2. Before A6: the A5 stand-in. Records `replica_check: skipped` in `session-a.json` and replaces the session's
   `require_go` (which reads A5's verdict) so A6-A11 run; `go()` still stops after a failed step.
3. After A9: the drafter gate of 15.6 from `mtp-train/train-report.json`: `ours (lean): held-out accept_realized
   before X after Y delta Z` (also greedy, expected), per-step KL / top-1 lines, and `ours (lean): drafter gate
   PASS|BETWEEN|KILL|UNREADABLE`. A11 runs either way.
4. After A11: `ours (lean): MTP draft manifest sha256 <64 hex> (<n> bytes); mtp-dense.safetensors sha256 <..>;
   <k> dense tensors replaced; frozen files byte-identical to the stock draft a/b` (each frozen file's draft sha256 =
   its manifest entry = the stock draft's; the dense shard's frozen tensors are their exporter's own check), then
   `MTP_DRAFT_MANIFEST_JSON {...}`; a frozen-file mismatch stops the notebook.

**Tests (all CPU).** The session notebook and metadata are pinned and the rebuild is byte-identical; every vendored
file matches `SHA256SUMS` and the upstream clone at `ccb7214`; their own 68 CPU tests pass unmodified inside
`vendor/`; the overrides are checked (inputs, dropped steps, deadline vs worst case, recipe, no private input or
`scottmahony` id, machine shape); the shipped scripts round-trip byte for byte; and **the notebook's own lean cells
run in order against their real session module with their stub scripts and stub server** (their `Bed`): one server
(the REAP training dump), no probe dump or replica call, every command line uses only options the real scripts
have, the data/gate/export markers print, and the manifest rebuilt from the `MTP_DRAFT_MANIFEST_JSON` line has the
file's sha256. Failure paths: a wrong response count or a missing log stops before the server; a frozen file that
differs from the stock draft stops at the export; the gate's PASS / BETWEEN / KILL boundaries. The C+D builder
refuses no manifest, a missing file, a wrong `--expect-sha`, a placeholder (`not_for_push` / `placeholder`), a foreign
format, an empty `replaced`, a missing dense sha256; with a fake manifest (temp dir) its notebook differs from arm C
v2 in exactly two cells: markdown cell 0 and cell 4, where the stock-draft `DRAFT_MODEL_DIR` line becomes their
600 s sha-checked draft wait; metadata differs only in the slug, the dropped stock-draft model source and the session
kernel source. Full suite: 750 passed.

**What L0 found (no blocker).** A9 does not depend on the dropped steps: the trainer reads the training dump, the
snapshot files and the ARC map only; the replica checks feed nothing downstream (their module only gates on A5's
verdict, which our stand-in replaces). No private input remains. One gap: **the 25 log byte sizes are not listed**
(the GitHub artifact of `jad81-tx-1` is on blob storage this box cannot reach), so the session pins names + the
response count instead; `mtps-ls-1` below lists them, and a rebuild with sizes then uses their own size pins.

**Estimated cost of L1 (session A).** Notebook start to A0 ~5 min, data check ~1-3, install + precache ~5-10, one
boot 10-27 (cap 50), A7 dump ~25-50 (held-out 20 + train 30 budgets), A8 ~3, A9 ~30 (their v2: 356 steps in 30 min;
cap 75 + 10), A11 + export check ~7: **~1.5 GPU-h nominal (1.3-1.9), <= 2.5 on slow storage, hard cap 4.5 h** (the
session's own deadline).

**kaggle-ops requests** (one per commit; change `id` every time; push only into a free GPU slot, not late Friday or
early Saturday UTC, and never ahead of a scheduled arm's check run):

1. (Optional, recommended) list arm C v2's logs, then fill the sizes in the manifest and rebuild + repin:
   ```json
   {"id": "mtps-ls-1", "ops": [{"op": "kernel_output", "kernel": "calamitychasm/arc3-jad-exp081",
     "list_files": "*_requests.jsonl", "artifact": false}]}
   ```
2. Push session A (first push of this slug; only into a free slot, so its kernel source mounts):
   ```json
   {"id": "mtps-push-1", "ops": [{"op": "push_kernel", "dir": "kaggle_submission_mtp_session/notebook"}]}
   ```
3. Read it (the output holds the 4.2 GB draft, so no artifact; the manifest, the session record and the training
   report are printed in full):
   ```json
   {"id": "mtps-out-1", "ops": [{"op": "kernel_output", "kernel": "calamitychasm/arc3-mtp-session-a",
     "grep": ["ours: inputs mounted after", "[session A inputs] logs:", "ours (lean): data loaded",
              "ours (lean): train split", "ours (lean): holdout split", "[session A A0]",
              "ours: FR-Spec map hot_tokens_64k_arc.pt written", "layers x 448 experts",
              "ours (session A): the launcher arguments and environment are ready", "arc3 HC dump:",
              "ours (lean): A2-A5 and A10 dropped", "server healthy after", "[session A A7] dump:",
              "comparison:", "ours (lean): A9 steps", "ours (lean): held-out accept_realized",
              "ours (lean): held-out accept_greedy", "ours (lean): held-out step1", "ours (lean): drafter gate",
              "frozen files byte-identical", "ours (lean): MTP draft manifest sha256", "MTP_DRAFT_MANIFEST_JSON",
              "[session A] complete", "STOP at", "WARNING"],
     "show_lines": 3,
     "print_files": ["mtp-draft/arc3-draft-manifest.json", "session-a.json", "mtp-train/train-report.json"],
     "artifact": false}]}
   ```

**Next steps.**
1. `mtps-ls-1`, fill the sizes, rebuild (`python scripts/_build_mtp_session_kernel.py`), repin, commit; then
   `mtps-push-1` into a free GPU slot when spare quota allows (15.6: never ahead of a scheduled arm's check run).
   **Do not push a v3 of `calamitychasm/arc3-jad-exp081` before the session has run.**
2. `mtps-out-1`. Apply the drafter gate (`ours (lean): drafter gate`): PASS -> step 3; BETWEEN -> no C+D run, the full
   path's probes decide; KILL -> close the line.
3. On PASS: save the printed manifest as a file (the `print_files` text, or `json.dumps(obj, indent=1) + "\n"` from
   `PRINT_FILE_JSON` / `MTP_DRAFT_MANIFEST_JSON`), then
   `python scripts/_build_jad_exp081_draft_kernel.py --draft-manifest <file> --expect-sha <printed sha256>`, pin the
   notebook in `tests/test_jad_exp081_draft.py`, commit notebook + `kaggle_submission_jad_exp081_draft/arc3-draft-manifest.json`,
   push `calamitychasm/arc3-jad-exp081-draft` (ids `jadd-*`) and read it against the C+D gate of 15.6 (markers: arm
   C's six plus `ours: MTP draft`). **Never push a second version of `arc3-mtp-session-a` once C+D pins it**; a
   retrain goes under a new slug.

## 16. Slot plan 10-13..11-02 for P(top 1%) on the private ranking [ANALYSIS + MC, 2026-10-12; nothing built]

Written 2026-10-12 ~07:30 UTC, before arm C draw 1 (57089293) scored. This section proposes a schedule. CLAUDE.md's schedule
is unchanged until the orchestrator adopts the table in 16.7. The Monte Carlo scripts live in the session scratchpad and are not
in the repo; 16.2 gives the model in enough detail to rebuild them.

### 16.1 Inputs

| item | value |
|---|---|
| slots | 21 (10-13..11-02), 1 per UTC day, 1 PENDING at a time, ~9 h to score; submit right after 00:00 UTC |
| incumbent (m2 + LM) | n=5: 33.29, 28.18, 31.03, 29.38, 25.21 (mean 29.42, sd 3.03) |
| arm A (turbo-tail) | n=1: 26.59 |
| arm B (turbo-lossless-tail) | n=0 |
| arm C (exp-081 verbatim, v2) | n=0 of ours, draw 1 pending; JustAdev742's exp-081 draw 34.04 (n=1) |
| same-serving draws without the bundle | 27.97, 26.04, 28.87 (JustAdev742), 26.59 (our A) |
| per-draw sd | ~3 (incumbent 3.03; Franzen copies 1.6-3.9) |
| C-family check runs | C+LM PASS (104 levels / 47.21), C+U PASS (114 / 50.26), both v1; C+hic16 and MTP session A pending |
| leaderboard (06:38 UTC) | top-1% bar 33.60 (rank 42 of 4,193), rising ~0.1-0.2/day; our best 33.29 (rank 57); #1 67.49 |

**Final ranking.** `rules.md` says only "up to 2 Final Submissions selected for judging" and does not say how the two combine.
This plan assumes the usual Kaggle rule: the better of the two selected submissions' private scores counts. The recommended
selection rule (16.6) is also the right one if the two were averaged, because it maximises the expected score as well. Verify on
the live rules page before 11-02.

**Public/private correlation (rho).** Each submission is one run over all 110 games, scored as 55 public + 55 private games, so a
draw's public and private scores share a run. The evidence on rho points both ways and cannot be resolved here:
- Arm C-family full-length public-25 runs spread only ~1.3 (49.21, 48.03, 47.21, 50.26 across C v1/v2, C+LM, C+U). So
  per-game noise over 55 games is small, and most of the hidden sd of ~3 is run-level. That argues for a high rho.
- But in a rerun the D′ gate shares one token budget across all 110 games. A run in which some public games absorb tokens
  starves private ones, which makes that part of the run-level noise zero-sum between the halves. That argues for a low rho.

The plan must therefore hold for rho anywhere in 0-0.6. All results below are reported at rho = 0, 0.3 and 0.6.

### 16.2 Model

**Per draw.**
- public = mu_k + s + e1
- private = mu_k + off_family + off_kernel + s + e2
- var(s) = rho·9 and var(e) = (1-rho)·9, so the per-draw sd is 3.
- off_family ~ N(0, 1): one per family (incumbent; A/B; C). It models arm-specific differences between the public and
  private game sets. off_kernel ~ N(0, 0.5).

**Priors** (on the hidden-set mean per draw):

| kernel | prior | basis |
|---|---|---|
| incumbent | N(29.42, 1.36) | its 5 draws (sd/√5); the 5 public values are fixed and their privates are drawn conditionally |
| B | N(28.5, 2.0) | same serving as A plus lossless; family draws 26-29 |
| C base | **main N(31.8, 2.2)**; **skeptical N(30.0, 2.5)** | main: base family 27.3 ± 1.5, plus a skeptical bundle-effect prior N(+1.5, 3), updated on exp-081's 34.04 (σ 3) |
| C+U | C + dU, dU ~ N(+0.7, 1.0) | §11.4: +1-2 levels per run; mechanism pass |
| C+LM | C + dL, dL ~ N(+1.5, 2.0) | incumbent's +4.8 shrunk for the 03b overlap (§10.2) |
| C+UL (bundle) | C + dU + dL + dI, dI ~ N(-0.3, 1.0) | interaction / token cost |
| + hic16 | + dH ~ N(+0.6, 0.6), with a 3% chance of -10 | §14: +7% tokens = +0.2..+1.5; a host-RAM failure in a 110-game rerun is unmeasured |

There is also an **"effects null" scenario**: dU, dL and dI all have mean 0, with the same spreads.

**In-sim decisions** use the posterior from public draws only. The C family is fitted as one linear-Gaussian model on
(mu_C, dU, dL, dI, dH), so every C-family draw informs mu_C.

**Outcome**: P(max private of the 2 selected > bar), for bars 34, 35 and 36.

**Policies**:

| policy | slots |
|---|---|
| current | CLAUDE.md schedule: B 10-13/15/17, C 10-14/16, then the leader by sample mean; alternate with the incumbent unless the leader is ≥ 1 above it |
| C_only | 21 × C verbatim |
| Cfam | C+U 10-13, C 10-14, C+U 10-15, then the bundle C+UL from 10-16 (check run passes with p 0.85, else C+U). Fall back to the incumbent if the C-base posterior drops below the incumbent's mean − 1 after 10-15 |
| **Cfam_Cmix** | as Cfam, plus C verbatim on 10-19 and 10-25 (C reaches n=4 with 10-12) |
| Cfam_B3 | Cfam with B on 10-13/15/17 |
| separate | C, C+U, C+LM rotated, 7 each |
| B_then_C | B 10-13..15, then the bundle |
| Cfam_Cmix_H | Cfam_Cmix with the bundle + hic16 from 10-19 |

**Selection rules**:
- `each2`: the CLAUDE.md rule, i.e. the top public draw of each of the top-2 arms by posterior, n ≥ 3.
- `best2`: the two highest public draws of the arm with the best posterior, n ≥ 3.

3,000 sims per cell. Arm C's 10-12 draw is simulated, so all results are averaged over its outcome.

### 16.3 Where is the private bar? (field model)

**Fit.** Team means were modelled as N(mu_f, sd_f) for 4,193 teams, each with 1..10 hidden draws (sd 3) since the M2 era
began. The fit is mu_f = 22, sd_f = 2.75, which gives public quantiles 33.5 / 31.1 / 29.9 at ranks 42 / 210 / 420 (observed
33.60 / 31.34 / 29.98). Each team's private score is the best private of its 2 highest public draws.

**Projection to 11-02.** Teams get up to 31 draws, and the field means are shifted so that the public bar reaches 35 or 36:

| public top-1% bar | private top-1% bar, rho 0 | rho 0.3 | rho 0.6 |
|---|---:|---:|---:|
| 33.6 (today) | 32.6 | 33.3 | 33.7 |
| 35 (11-02) | 33.0 | 34.1 | 34.7 |
| 36 (11-02) | 34.0 | 35.1 | 35.9 |

**The public board overstates the private bar by 0-2 points.** The public board ranks each team's best of all its draws; the
private ranking uses best-of-2 selected draws, so the private bar is lower. So the 34-36 bars applied to our private score
below run from realistic to conservative. The field model is crude: Gaussian means, equal noise for every team. Use it only for
the direction and size of the gap.

### 16.4 Monte Carlo results

P(top 1%) at **bar 35**, rule `best2`. In brackets: rule `each2`, then bar 34 / bar 36 for `best2`.

**Main C prior N(31.8, 2.2), main effect priors:**

| policy | rho 0 | rho 0.3 | rho 0.6 |
|---|---|---|---|
| current | 0.35 (each2 0.26; 0.47 / 0.25) | 0.50 (0.42) | 0.65 (0.59) |
| C_only | 0.35 (0.26) | 0.51 (0.41) | 0.66 (0.59) |
| Cfam | 0.53 (0.42) | 0.67 (0.57) | 0.78 (0.73) |
| **Cfam_Cmix** | **0.55** (0.48; 0.64 / 0.45) | **0.67** (0.62; 0.76 / 0.58) | **0.78** (0.76; 0.84 / 0.68) |
| Cfam_B3 | 0.53 (0.42) | 0.66 (0.57) | 0.77 (0.72) |
| separate | 0.55 (0.49) | 0.65 (0.64) | 0.75 (0.75) |
| B_then_C | 0.52 (0.41) | 0.66 (0.57) | 0.77 (0.71) |
| Cfam_Cmix_H | 0.58 (0.56) | 0.70 (0.68) | 0.80 (0.79) |

**Skeptical C prior N(30.0, 2.5):** current 0.23 / 0.36 / 0.50 (rho 0 / 0.3 / 0.6); C_only 0.21 / 0.32 / 0.44; Cfam_Cmix
**0.38 / 0.51 / 0.62**; Cfam_B3 0.36 / 0.50 / 0.62; separate 0.37 / 0.46 / 0.55. The fallback to the incumbent fired in 30% of
these sims, against 6% under the main prior.

**Effects null (dU = dL = dI = 0 on average)**, main C prior: current 0.35 / 0.50 / 0.65; C_only 0.35 / 0.51 / 0.66; Cfam_Cmix
**0.42 / 0.54 / 0.64**; Cfam_B3 0.38 / 0.51 / 0.63; separate 0.44 / 0.54 / 0.66.

**Single arm, more draws (winner's curse).** Arm mean 32; select the top-2 public draws of n; P(max private > 35):

| n | rho 0 | rho 0.3 | rho 0.6 | top public draw → its expected private (rho 0 / 0.3 / 0.6) |
|---:|---:|---:|---:|---|
| 3 | 0.31 | 0.35 | 0.39 | 34.5 → 32.0 / 32.8 / 33.5 |
| 8 | 0.31 | 0.44 | 0.58 | 36.3 → 32.0 / 33.3 / 34.6 |
| 12 | 0.31 | 0.47 | 0.64 | 36.9 → 32.0 / 33.5 / 34.9 |
| 20 | 0.31 | 0.51 | 0.72 | 37.6 → 32.0 / 33.7 / 35.4 |

**What the results say**:

1. **(a) Exploration.** 21 draws resolve about one comparison at the 3-point level.
   - At sd 3, 80% power and one-sided α 0.1 need ~9 draws per arm for +3, ~13 for +2.5 and ~20 for +2.
   - So the slots can settle one question: is the C family above the incumbent? The prior gap is +2.4.
   - Within-family differences (+0.5..+2) cannot be resolved by draws at all. The kernel inside the family is chosen by
     mechanism prior and check-run counts, not by draws.
   - The MC agrees. "separate" (7 draws each of C, C+U, C+LM) is no better than concentrating on one bundle kernel, and is
     worse at rho ≥ 0.3 under `best2`.
2. **(b) Winner's curse / max-of-n.**
   - At rho 0, draws add nothing beyond identifying the arm. Our 33.29 public draw is worth the incumbent's mean on private
     (~29.4).
   - At rho 0.3-0.6, 12-20 draws of the selected kernel raise P(>35) by 12-33 points over 3 draws.
   - So the kernel we will select should also be the one we draw most. That is the bundle. The public top draw of a 12-draw
     kernel overstates its private by 3.4 (rho 0.3) to 4.9 (rho 0) points.
3. **Selection rule.** `best2` beats `each2` by 0.02-0.09 at bar 35 in almost every cell. The exceptions are at rho 0.6 with
   null effects or the "separate" policy, where the two rules are within 0.02 of each other. The second arm (the incumbent, or B) is about 2 points below the C family. A second draw of the best arm beats the top
   draw of a worse arm.
4. **(c) Arm B.** Giving B its 3 explore slots costs 0-0.04 (Cfam_B3 vs Cfam/Cfam_Cmix) and never helps.
   - B's prior (28.5) is below the incumbent's, and it shares A's serving, which drew 26.59.
   - Its one open question was whether acceptance 0.5 is lossy. exp-081's 34.04 at acceptance 0.5 answered that for the
     family we would select.
   - **Drop B from the schedule.**
5. **(d) Bundling.**
   - Cfam_Cmix's lead over C_only (+0.20 at rho 0 under main effects) is mostly the effect priors.
   - Under null effects it shrinks to +0.07 / +0.03 / -0.01 (rho 0 / 0.3 / 0.6). With the bar above our mean, P(top 1%) is
     convex in the arm mean, so the extra variance of an uncertain variant pays at low rho.
   - It never loses more than 0.02 in any scenario.
   - Adding hic16 to the bundle adds 0.02-0.08 if its check run passes.

### 16.5 (d) The bundle kernel C+UL: build, risk, check run

**What it is.** Arm C v2 + C1/C2 (the two source patches in the apply loop, `...successfully: 8`, untried/stale-latch counters)
+ our level-memory install cell and summary dump. It is the union of `_build_jad_exp081_u_kernel.py` and
`_build_jad_exp081_lm_kernel.py`; nothing else changes. Slug proposal: `calamitychasm/arc3-jad-exp081-ulm`.

**Why combine.**
- Neither effect is separable by draws (16.4 item 1).
- Both passed their own full-length check run with errors 0, and serving costs of −2.1% / −2.0% tok/s and +3.2% / +1.6%
  tokens/request.
- They act on disjoint state:
  - C1 is a line inside the patched `_build_user_prompt` body;
  - C2 is the sandbox refusal latch;
  - LM is runtime wrappers on `_build_user_prompt` / `_trim_messages_for_context` / `_ensure_session` that edit the opener's
    tail and `messages[0]`.

**Risks.**
- **Opener composition.** One level-up opener would carry 03b's win record, the I1 line (when due) and our rule request.
  Expect prompt growth of ~+5% tokens/request (≈ 2,085 vs C v2 1,989), on top of the LM block (≤ 8.3 KB system prompt).
- **ours_untried's append-only assumption.** It reads the solver's game history, and LM never touches that history. But LM
  replaces the system message at an eviction, and the I1 hit-counter's matching must survive that.
  - Extend `tests/jad_lm_probe.py` to the 8-patch tree: build the tree with C1/C2, run both install cells, and drive a
    level-up and two evictions.
  - Require LM errors 0, `i1_unmatched` 0 and `stale_refused_after_released` 0.
- **Attribution.** If the bundle draws low, we cannot say which part hurt. Draws could not say that for the parts alone either.
- **Pooling.** Bundle draws inform mu_C only through the model, not as verbatim exp-081 draws. This is why C verbatim keeps
  n=4 (Cmix) as the clean family read and the fallback selection.

**hic16 and the drafter are not in the bundle.** They are serving changes with their own risks: host RAM in a 110-game
rerun (§14.5), and draft quality (§15.5).
- Each may later replace the bundle kernel under the standing "same mechanism, faster" rule, once its own C check run has
  passed and a rebuilt bundle + swap check run passes.
- At most one swap, made no later than 10-24, so that the swapped kernel draws ≥ 8 times.
- Swapped draws pool with the bundle's for selection as "same harness, faster serving", but the swapped kernel needs n ≥ 2
  of its own before one of its draws may be selected.

**Check run** (pre-registered; full-length 25 × 121 min, ~2.3 GPU-h; push into a free GPU slot as soon as C+hic16 and session A
release them):
- **Pass**: all of the following.
  - Markers: arm C's six with `our harness patches applied successfully: 8`, plus `STALE_LATCH_FIX installed`,
    `UNTRIED_NOTICE installed mode=i1` and `LEVEL_MEMORY installed`.
  - Untried counters: `untried_summary.json` errors 0, tracker_errors 0, i2_lines 0, i1_lines ≥ 20; click-within-3 share
    ≥ 15%.
  - Latch counters: `stale_latch_summary.json` errors 0 and `stale_refused_after_released` 0.
  - LM counters: `level_memory_summary.json` errors 0, blocks_applied > 0, rules_captured ≥ 0.5 × levels_recorded.
  - Clean run: 0 tracebacks; 25/25 terminal.
  - Serving: gen tok/s ≥ 750; accept 3.0-3.3; repeated assistant turns ≤ 1; retractions ≤ 4; tokens/request ≤ 2,190.
  - Outcome: ≥ 3 of 4 key levels (sp80 L2, tn36 L2, vc33 L4, cn04 L2); levels ≥ 103; score ≥ 44.
- **Kill**: any one of the following.
  - gen tok/s < 723; levels < 95; score < 40; tokens/request > 2,390.
  - LM errors > 0; blocks_applied 0; i1_lines 0; `stale_refused_after_released` > 0.
  - A missing install marker, or a traceback from `level_memory`, `ours_untried`, the counters or the patched refusal lines.
- **Near-pass** (tok/s 723-750, levels 95-103, score 40-44, click share 10-15%, or 2 of 4 key levels): do not rerun. Keep
  drawing C+U in the bundle's slots. A rerun would cost 2.3 GPU-h and ~2 slots of delay for an effect draws cannot see.
- **Kill or near-pass**: C+U takes every bundle slot in 16.7, and C+LM is not drawn.

### 16.6 (e) Decision rules

**After arm C draw 1 (57089293, x)**: the posterior of the C base under the main prior N(31.8, 2.2), and P(mu_C > mu_incumbent):

| x | posterior mu_C | P(> incumbent) | skeptical prior: posterior, P |
|---:|---:|---:|---|
| 36 | 33.3 ± 1.8 | 0.96 | 32.5, 0.90 |
| 33 | 32.2 | 0.90 | 31.2, 0.78 |
| 30.5 | 31.4 | 0.81 | 30.2, 0.63 |
| 28 | 30.5 | 0.68 | 29.2, 0.46 |
| 25 | 29.4 | 0.50 | 28.0, 0.27 |
| 22 | 28.4 | 0.32 | 26.7, 0.13 |

- **High, x ≥ 33**: run 16.7 unchanged. If the bundle's check run passes in time, it may start on 10-14 (C verbatim moves
  to 10-15).
- **Mid, 28 ≤ x < 33**: run 16.7 unchanged. One draw cannot overturn the prior, and P(C > incumbent) stays ≥ 0.68.
- **Low, x < 28**: do not abandon the family on n=1 (at x = 25 the posterior equals the incumbent's). Swap 10-13 and 10-14:
  C verbatim on 10-13 for a clean second read of the base, then C+U on 10-14. From then on, the standing rule decides.

**Standing rule** (re-evaluated after every C-family score from 10-15). Compute the C-base posterior: the main prior, updated
on every C-family draw after subtracting each variant's prior effect mean, with the variance widened by the effect spreads (the
16.2 joint model). Then:
- **posterior ≥ 28.4** (the incumbent's mean − 1): stay on the schedule.
- **27.4 ≤ posterior < 28.4** with ≥ 3 C-family draws: alternate the bundle and the incumbent (incumbent on odd days). This
  keeps the bundle's option value and adds incumbent draws for selection at rho > 0.
  - Example: C-family draws 28, 26, 27 give 28.8, so stay.
  - Example: 25, 25, 26 give 27.8, so alternate.
- **posterior < 27.4** with ≥ 4 C-family draws: incumbent only.
- Re-check on 10-18 and 10-25 regardless. Pool JustAdev742's exp-081-family hidden draws (exp-085: bundle + I1 + their
  drafter) as extra evidence on mu_C, at half weight (σ 4.2) because of the drafter difference.

**Final selection** (both slots explicitly, by 11-02 12:00 UTC, after the 11-02 draw has scored):
1. Compute posteriors with the 16.2 model for every kernel with n ≥ 3.
2. If the leader's posterior is ≥ 0.5 above the best other family's: select the **two highest public draws among the
   leader's family kernels whose posterior is within 1.0 of the leader's** (`best2`). Normally that is the bundle's top two,
   or the bundle's top plus C verbatim's top.
3. If the top two families are within 0.5 of each other: select the top public draw of each (`each2`).
4. Never select a kernel with n < 3, or a serving-swapped kernel with n < 2 of its own.
5. This replaces "the highest public draw of each of the top-2 arms" in CLAUDE.md. It gains up to 0.09 in P(top 1%) at bar 35,
   and is never more than 0.02 worse (16.4 item 3).

### 16.7 Proposed replacement schedule (for the orchestrator to adopt into CLAUDE.md)

| date | kernel (version) | why |
|---|---|---|
| 10-12 | C (v2), 57089293 | pending; apply 16.6 when it lands |
| 10-13 | **C+U (v1)**; low x: C (v2) | first hidden draw of the passed variant; pools into mu_C |
| 10-14 | **C (v2)**; low x: C+U (v1) | C verbatim n=2 |
| 10-15 | **C+UL (v1)** if its check passed, else C+U | the bundle from here (MC assumed 10-16) |
| 10-16 | C+UL | |
| 10-17 | C+UL | |
| 10-18 | C+UL | checkpoint 1: standing rule, CLAUDE.md "exploit" date |
| 10-19 | **C (v2)** | C verbatim n=3 (selection-eligible fallback) |
| 10-20..10-24 | C+UL ×5 | last date for a serving swap (hic16 or drafter) is 10-24 |
| 10-25 | **C (v2)** | C verbatim n=4; checkpoint 2 |
| 10-26..11-02 | C+UL ×8 | 11-02 submitted at 00:00 UTC, scored by ~09:00; final selection by 12:00 UTC |

**Totals**: C+UL ~16-17, C verbatim 4 (with 10-12), C+U 1-2, incumbent 0, A 0, B 0. If the bundle's check run fails,
replace C+UL with C+U throughout. Each submit request uses the version its check run used: C v2; C+U, C+LM and the bundle v1.

**Expected P(top 1%)** (main C prior, bundle effects as priored, private bar 35): 0.55 / 0.67 / 0.78 at rho 0 / 0.3 / 0.6,
against 0.26 / 0.42 / 0.59 for the current schedule with its current selection rule.
- Under the skeptical C prior: 0.38 / 0.51 / 0.62 vs 0.18 / 0.31 / 0.46.
- Under null effects: 0.42 / 0.54 / 0.64 vs 0.26 / 0.42 / 0.59.

Of the gain at bar 35, about 0.09 comes from the selection rule alone (current schedule, `each2` → `best2`). The rest
(+0.07..+0.20, depending on the effect priors) comes from the schedule: C-family variants instead of B, with the draws
concentrated on the kernel that will be selected.

**Caveats.**
- The priors on mu_C and on the variant effects carry the result. The skeptical and null rows are the floor.
- rho is unknown (16.1). The field model's private bar is crude.
- None of the variant effects will be confirmed by draws. The bundle is chosen on mechanism counts and expected value, not on
  measured score.
