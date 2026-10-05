# Stage 7 — solved-level memory on the milestone-2 fork

**Why.** The milestone-2 harness trims history from the front at ~118K
tokens down to ~59K. With the structured world model switched off, the
conversation is the agent's only memory, so a trim erases what it learned on
earlier levels (only its retained Python helpers survive). Levels build on
earlier mechanics. sirikilohit's Milestone-2 patch M85 pinned each solved
level's rule + winning actions into the system prompt: the only text
addition that helped on his stack (+2.8, 7.14 -> 9.96).

**What.** `kaggle_submission_milestone2_fork/level_memory/level_memory.py`,
installed at runtime on the patched `ToolAgent` (the 8,400-line harness patch
is untouched):
- first prompt after a level clears: record level, action count, last 30
  actions, tail of the clearing turn's reasoning; ask for one line "Rule for
  level N:" (on up to 2 real openers -- never on a resumed turn, whose opener
  is replaced);
- the rule is read back from reply text or reasoning;
- the block is pinned at the end of the system prompt **only when history is
  evicted**, i.e. when the prefix is being rebuilt anyway. Deviation from M85,
  which edits the system prompt immediately: on this harness that would cost a
  full re-prefill (~60-118K tokens) each time. Until the first eviction the
  same facts are still in the conversation.

Tests: `tests/test_level_memory.py` -- 9 unit tests, plus 3 that run the REAL
patched ToolAgent (base bundle + dfranzen patch, `ARC3_M2_SRC`): wrappers
installed, the real prompt carries the request, the real trimmer pins the
block exactly when it evicts and not otherwise. The notebook's install cell
was executed against the real harness locally. Kernel
`calamitychasm/arc3-m2-level-memory`: every upstream cell identical except the
run cell, which only gains a counter dump; one install cell before it.

## Pre-registration

- *Noise.* Copies of the baseline scored mean 25.77, sd 3.93 (n in the
  hundreds). One draw cannot confirm M85's +2.8; the submission is a
  catastrophe check plus a first draw: **falsifier for "safe to keep" = below
  18 (mean - 2 sd)**.
- *Mechanism* (check run, 10 public games x 25 min): `errors == 0`;
  `levels_recorded` > 0; `rules_captured / asks` reported; `blocks_applied`
  reported (may be small -- 25 min may hold few evictions).
- Submission authorised by the user on 2026-10-02 ("Run a submission when
  it's ready"); next slot 2026-10-03 00:00 UTC, fired by a detached
  scheduled task, not by this chat.

## Submission plumbing (2026-10-03)

Kernel v1 pushed ~03:57 UTC (check run in progress). Submission is fired by a
detached one-shot scheduled task (`ARC3Submit_arc3-m2-level-memory`,
`scripts/kaggle_submit_when_ready.py --job ...`), armed for 04:02 UTC: it
waits for the check run, downloads its output, and submits only if the gate
passes -- markers `LEVEL_MEMORY installed`, `priority gate active`, `harness
patch applied successfully`; no traceback outside serving teardown; every game
`won`/`gave_up`/`cancelled`; `level_memory_summary.json` with `errors == 0`.
Steps go to `logs/kaggle_submit.log`; the score lands in
`logs/kaggle_watch_latest.txt` via the watcher.

## Result (2026-10-03): **33.29**, rank 10

Check run: `levels_recorded 47, asks 61, rules_captured 32, rules_missing 12,
blocks_applied 15, block_chars_max 6413, errors 0`; gate passed 04:37 UTC;
the detached task submitted ref **56789553** at 04:37; COMPLETE, **33.29**.
Team rank **10**.

Against the baseline notebook's draw distribution (hundreds of one-shot
copies: mean 25.77, sd 3.93): z = 1.91, **one-sided p = 0.028** that an
unchanged notebook draws >= 33.29. Empirically even rarer: only two teams
outside the prize leaders score above 33.29. But one draw puts the effect at
+7.5 with a 95% interval of [-0.2, +15.2], and the point estimate is inflated
by selection (we report it because it is high). If the true effect were +2.8
(M85 on sirikilohit's stack), P(no effect | this draw) = 0.25 at a 50/50 prior.
Draws for 95% confidence / 80% power: 13 at +2.8, 6 at +4, 4 at +5, 2 at +7.5.
A second draw >= 27.4 makes the two-draw mean significant at p < 0.05
(>= 31.2 for p < 0.01).

## Draw 2 (2026-10-04): **28.18**

Ref 56809165 (kernel v1, byte-identical), submitted 00:02 UTC by the detached
task after the gate passed again. Two draws 33.29, 28.18, mean **30.74**;
vs the copies' 25.77 +/- 3.93: z = 1.79, **one-sided p = 0.037** -- clears the
pre-registered p < 0.05 bar (second draw >= 27.4). Effect estimate +5.0
(95% CI about -0.5 .. +10.4 from two draws).

## Draw 3 (2026-10-05): **31.03**

Ref 56842120 (kernel v1, byte-identical), submitted 04:19 UTC by the detached
task. Three draws: 33.29, 28.18, 31.03 -- mean **30.83**, sd 2.56.
- vs the copies' distribution (25.77, sd 3.93 known from hundreds of draws):
  z = 2.23, **one-sided p = 0.013**; effect **+5.1, 95% CI +0.6 .. +9.5**.
- t-test using only our own spread (df = 2): t = 3.42, p = 0.038.
Both clear p < 0.05; the interval now excludes zero. A 4th draw at the same
mean would reach p < 0.01.
