# Stage 7 — verbatim fork of the Milestone 2 solution

**Why.** `dfranzen/arc-agi-3-milestone-2-solution` (Apache-2.0, published
2026-09-30) scores **31.47** public -- ~7x this project's best (4.58,
sheetu12b + momentum). The leaderboard moved with it: on 2026-10-02 #1 is
52.51 (Tufa Labs), #49 is 28.66, #99 is 27.36, of 3,564 teams. Our
incumbent is no longer near the top 10%.

**What it is (read 2026-10-01).** Duck harness + an 8,400-line patch
(prompts, frame-diff images, undo, game-persistent python functions, no-op
guards, `priority_scheduler.py`), Intel W4A16 AutoRound Qwen3.8-Flash-Next
+ a separate MTP drafter, served by Pennyroyal (an SGLang fork); 128K
context, 12K output; all ~110 games run at once with 10 active model
streams, granted by a harness-side priority gate ranking games by
(current RHAE + continuation value) x chance of finishing the current
level. That gate is a finer version of our momentum allocation; our
momentum cell does not apply to this solver and is not added.

**Kernel.** `calamitychasm/arc3-milestone2-fork` v1: notebook byte-identical
to upstream (`kaggle_submission_milestone2_fork/upstream/`), metadata changed
only in id/title/privacy.

**Pre-registration.** Check run (non-submission mode: 10 public games, 25
min each) must complete with no crashed games. Submission authorised by
the user on 2026-10-02 ("Submit it"). Expectation: well above 4.58; a score
below 10 would point at a fork/environment defect, not variance, and is the
falsifier.

## Check run (kernel v1, 2026-10-02 00:53-01:28 UTC): PASSED

Harness patch applied cleanly (all 16 files), `priority gate active: 10
concurrent streams`, zero tracebacks. 10/10 public games finished (2 won, 8
gave_up); mean 46.22 (upstream's own check run: 36.56) -- public games, not
a ranking. **Submitted: ref 56763055, 2026-10-02 01:29 UTC.**
