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
