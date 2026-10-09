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

## Draw 4 (2026-10-06): **29.38**

Ref 56864478 (kernel v1, byte-identical), submitted 00:02 UTC by the detached
task. Four draws: 33.29, 28.18, 31.03, 29.38 -- mean **30.47**, sd 2.21.
- vs the copies' distribution (25.77, sd 3.93): z = 2.39, **one-sided
  p = 0.008**; effect **+4.7, 95% CI +0.8 .. +8.6**.
- t-test using only our own spread (df = 3): t = 4.25, p ~ 0.012.
Past the p < 0.01 bar on the known-sigma test. Our own draw-to-draw spread
(sd 2.21) is smaller than the copies' (3.93), so the copies' sd is the
conservative choice for the interval.

## Variant: tried facts (`arc3-m2-lm-triedfacts`, flag `LEVEL_MEMORY_TRIED_FACTS=1`)

**Why.** When history is evicted, the agent loses what it already tried on the level it is still
stuck on: how many actions and game overs it has spent, how far each life got, what the fatal runs
looked like, and what it was thinking. The solved-level block only covers *cleared* levels.
Inspired by lordhansolo's `game_overs` objects (his `solver.py` `record_life_action`).

**Mechanism.** At the SAME eviction point as the solved-level block (`apply_on_evict`, only when the
harness says history was dropped, so the prefix is being rebuilt anyway) the system-prompt pin also
gets a facts-only block about the CURRENT level, appended after the solved-level block:
- actions spent on the level (resets included) and the game-over count;
- the action count at each game over, or a plain line when all are equal ("All 3 game overs
  occurred after exactly 42 actions.");
- the last 10 actions of the 3 most recent fatal runs, in the harness's own action notation
  (`HistoryEntry.action`, e.g. `ACTION6(31,14)`);
- the last ~600 chars of the model's own reasoning from the previous turn.

Game overs come from `HistoryEntry.result["game_over"]` (the dfranzen patch stores it per action);
the RESET after a death is not part of any life. Everything is worded as fact, no advice (advice
text hurt scores before). Hard cap 3000 utf-8 bytes: reasoning, then fatal runs, then listed counts
degrade in order before any raw cut. The facts are snapshotted once per turn in the user-prompt
wrapper, so repeated trims inside one turn rebuild the identical block and do not rewrite the
prompt. Nothing before the eviction point is touched; with the flag off none of this runs and
`summary()` has no new keys. `tests/test_level_memory_tried_facts.py` covers content, the cap,
flag-off equivalence, and no mutation of pre-eviction messages. Not run against the real patched
ToolAgent (`ARC3_M2_SRC` not available when this was written): the harness fields used
(`history_entries[i].result["game_over"]`, `.action`, `.frame.level`; `_history_messages`) were read
from the dfranzen patch.

Example (3 deaths at 42 actions on level 2):

```
=== CURRENT LEVEL FACTS (recorded by the harness when older history was trimmed) ===
Level 2: 149 actions spent on it so far (resets included); 3 game overs on it.
All 3 game overs occurred after exactly 42 actions.
Game over #1, last 10 of 42 actions (32 earlier not shown): RIGHT, ACTION6(31,14), DOWN, ...
Game over #2, last 10 of 42 actions (32 earlier not shown): RIGHT, ACTION5, UP, ...
Game over #3, last 10 of 42 actions (32 earlier not shown): DOWN, ACTION5, ACTION5, ...
Your reasoning in the turn before older history was trimmed (tail): ...
=== END CURRENT LEVEL FACTS ===
```

**Build.** `python scripts/_build_m2_level_memory_kernel.py --tried-facts` writes
`kaggle_submission_m2_lm_triedfacts/notebook/` (kernel `calamitychasm/arc3-m2-lm-triedfacts`;
metadata identical to the incumbent except id/title/code_file). The only differences from the
incumbent notebook are the markdown header and the install cell (sets the env flag, asserts it,
prints `TRIED_FACTS installed`). Variant flags compose: each adds a suffix in `VARIANT_ORDER`, so
histcache + triedfacts becomes `arc3-m2-lm-histcache-triedfacts` in
`kaggle_submission_m2_lm_histcache_triedfacts/`. The incumbent notebook in git was not regenerated
(it is what ran as v1); rebuilding it now would embed the new module with the flag off.

**Check-run pass criteria** (10 public games x 25 min, same as the incumbent's check):
- log has `LEVEL_MEMORY installed` and `TRIED_FACTS installed`, plus the incumbent's markers
  (`priority gate active`, `harness patch applied successfully`); every game
  `won`/`gave_up`/`cancelled`;
- `level_memory_summary.json`: `errors == 0`; `blocks_applied` > 0 and `tried_facts_blocks` > 0
  (a short check run may hold few evictions; 0 is inconclusive, not a pass);
- `tried_facts_bytes_max <= 3000`; `tried_facts_truncations` small relative to blocks;
  `tried_facts_avg_bytes` roughly 0.5-3 KB.

Score pre-registration is as for the incumbent: one draw is a catastrophe check (< 18 fails);
effects need several draws against the incumbent's mean 30.47 (sd 2.21, n=4).

**Push / submit** (Kaggle was unreachable when this was written):

```
kaggle kernels push -p kaggle_submission_m2_lm_triedfacts/notebook
python scripts/kaggle_submit_when_ready.py --kernel calamitychasm/arc3-m2-lm-triedfacts --version 1 \
    --message "m2 + level memory + tried facts" --marker "LEVEL_MEMORY installed" \
    --marker "TRIED_FACTS installed" --marker "priority gate active" \
    --marker "harness patch applied successfully" --counters level_memory_summary.json
```

## Variant: history cache (`arc3-m2-lm-histcache`, build flag `--history-cache`)

Variant A': the incumbent plus a cache for the game history. **Zero intended behaviour change**:
the model's prompts, the tool payloads it reads, and everything a snippet can see in the sandbox
are the same as the incumbent's. Only the cost of each action changes. Module:
`kaggle_submission_milestone2_fork/history_cache/history_cache.py`, installed at runtime by its own
notebook cell (right after the level-memory install cell, before the run cell). No vendored
upstream file is edited.

**Why.** For every executed action the harness paid O(history) four times over, all in the one
notebook process every game shares (GIL), while the game holds a GPU stream:
1. the solver rewrote the whole history to `tool_runtime_state.json` with `json.dumps(indent=2)`
   after every action of a batch (`runtime_state.write_runtime_state`, called from
   `_HarnessGameSession._execute_action`);
2. every python tool call re-read and re-parsed that file (`tool_agent.load_runtime_state`);
3. after every `action()` the tool re-read it again, rebuilt ascii + grid payloads for every past
   frame (`_ascii_history_view_payload`) and sent the whole history to the sandbox as one JSON line;
4. the sandbox re-parsed that line and rebuilt every frame view, inside its 30 s budget (31 s CPU
   rlimit).

**Mechanism** (ported from sirikilohit's M86, cell 38 of his Milestone-2 solution, Apache-2.0, and
extended; the incremental sandbox protocol and the counters are new here):
- *write*: each history entry is serialized once; the state file is written compactly (same JSON
  content, valid at all times between writes); an unchanged state is not rewritten; later writes
  replace only the file's tail in place; inside one `step_env` batch the per-action writes collapse
  into one write when the batch ends (nothing reads the file mid-batch).
- *read*: `load_runtime_state` returns in-memory objects made by exactly the round trip a file read
  performs, keyed by the state path and checked against the file's stat; anything unknown or stale
  falls back to parsing the file.
- *ascii*: each distinct frame is formatted once (ascii, grid JSON) and reused.
- *sandbox*: after an `action()` the reply carries only the entries the sandbox does not have yet
  (`history_delta`, with `base_len` + `base_last_step` checked by the sandbox), whenever the host
  can prove by object identity that the new history extends what it already sent on that pipe;
  otherwise -- and always for a new sandbox process (every python tool call starts one), and after
  a timeout kill -- the full payload. The sandbox keeps the raw entries and rebuilds FRESH view
  objects on every refresh, exactly as before, so a snippet that mutates a history object sees it
  restored after its next `action()`, as it did when the history was re-parsed.
- *counters* (`HISTORY_CACHE.summary()`, written to `history_cache_summary.json` after `bm.run`,
  and a `HISTORY_CACHE stats` line every 250 python calls): writes / appended / skipped / deferred /
  fallbacks, loads cached / file / stale, view hits / misses, payloads full / delta / plain,
  `sandbox_timeout_kills`, `sandbox_died`, `errors`, and per history-length bucket
  (0-99, 100-299, 300-999, 1000+) the host time per python call and per state sent. Startup marker:
  `HISTORY_CACHE installed`.

**Equivalence tests** (`tests/test_history_cache.py`, on the REAL patched ToolAgent and the REAL
sandbox subprocess; `tests/m2_harness.py` rebuilds dfranzen's patched `src/` by `git apply`-ing the
patch embedded in the upstream notebook onto the base bundle checkout, as the notebook's setup cell
does on Kaggle). Each scenario is played twice in one process, without and with the cache, and
must give identical sandbox-visible state (history, transitions, frames with raw grids,
segmentation, results, `last_*` globals, printed by the snippet), identical model-visible tool
payloads, identical next user prompt, step summary, state-file JSON and `load_runtime_state`
result. Covered: N = 1, 50 and 1000 history entries; deltas after `action()`; a new sandbox per
call; a sandbox killed by the tool timeout mid-snippet; RESETs (incl. the auto-reset after a game
over) and level changes; a snippet that mutates history objects before acting. Plus: the file is
valid JSON equal to the original writer's at every write; a file rewritten behind the cache's back
is detected; a replaced (non-extending) history rebuilds; batch writes collapse to one and still
happen when the batch raises; install/uninstall restores every patched attribute; the built
level-memory + history-cache cells, executed verbatim, compose on the real harness (all markers,
both modules' wrappers in place); every committed variant notebook equals a fresh build and differs
from the incumbent only by its additions. No behaviour difference was found.

Two test-harness defects in the unfinished first version were fixed while verifying this: its
synthetic boards were uniformly random 64x64 grids, on which the sandbox's segmentation takes ~9 s
per frame, so every scenario snippet hit the 30 s tool timeout and the "identical stdout" checks
were comparing two empty strings (now ARC-like boards: background + rectangles, and the tests
assert exactly the one deliberate timeout and an observation line in every other call); and a
chunked/continuous action-cycle mismatch in the file-equality test.

The same harness now also runs `tests/test_level_memory.py`'s three real-harness tests (previously
skipped without `ARC3_M2_SRC`). The committed `arc3-m2-lm-triedfacts` notebook was one line stale
against `level_memory.py` (a `.lstrip()` added after the build); it is rebuilt here.

**Overhead** (`python scripts/bench_m2_history_cache.py`, this 4-CPU container, real harness +
sandbox, one python call issuing 12 `action()`s that each change the board):

| N history | arm | wall per action | host CPU per action (GIL) | payload per action |
|---:|---|---:|---:|---:|
| 100 | incumbent | 0.201 s | 0.179 s | 2.29 MB |
| 100 | cache | 0.009 s | 0.003 s | 0.04 MB |
| 300 | incumbent | 0.538 s | 0.464 s | 6.46 MB |
| 300 | cache | 0.017 s | 0.003 s | 0.04 MB |
| 1000 | incumbent | 2.031 s | 1.724 s | 20.65 MB |
| 1000 | cache | 0.065 s | 0.005 s | 0.04 MB |

Host CPU per action is flat in N; what remains O(N) is the sandbox rebuilding fresh view objects
(~0.05 ms per entry, in the sandbox process, not under the notebook's GIL) and the one full payload
each python call still sends at sandbox start (the sandbox is a new process per call). In the
full run ~10 games act concurrently on one GIL, so the incumbent's host cost at a few hundred
history entries is ~0.5 s of serialized CPU per action per game. What this buys in score is not
predicted here: the run is decode-bound, and this frees host CPU and wall time between decode
steps, plus snippets that previously died on the 30 s sandbox budget at long histories.

**Check-run pass criteria** (10 public games x 25 min, same as the incumbent's check):
- log has `LEVEL_MEMORY installed`, `HISTORY_CACHE installed`, `priority gate active`,
  `harness patch applied successfully` (plus `TRIED_FACTS installed` for the combined kernel);
  every game `won`/`gave_up`/`cancelled`; no traceback outside serving teardown;
- `history_cache_summary.json`: `errors == 0`, `write_fallbacks == 0`, `payload_plain == 0`,
  `view_misses == 0`, `loads_stale == 0`; `payload_delta` >> `payload_full` (one full payload per
  python call; every `action()` reply after it a delta -- expect several deltas per full);
  `by_history` host ms per state roughly flat across buckets (single-digit ms);
- `sandbox_timeout_kills` <= the incumbent's: count `Tool timed out after` in the incumbent check
  run's transcripts and in this one's (same games, same 25 min);
- `level_memory_summary.json`: `errors == 0` (as for the incumbent).

Score pre-registration: as for the incumbent, one draw is a catastrophe check only (< 18 fails);
an effect needs several draws against the incumbent's mean 30.47 (sd 2.21, n=4). Because this
variant is designed not to change what the model sees, a score change would come only from more
model time per game (less host stall), so expect at most a small effect; prefer it over the
incumbent only on mechanism (counters above) plus draws that are not worse.

**Build** (all three are written by the same script; the incumbent notebook is not regenerated):

```
python scripts/_build_m2_level_memory_kernel.py --history-cache                # arc3-m2-lm-histcache
python scripts/_build_m2_level_memory_kernel.py --tried-facts                  # arc3-m2-lm-triedfacts
python scripts/_build_m2_level_memory_kernel.py --history-cache --tried-facts  # arc3-m2-lm-histcache-triedfacts
```

**Push / check / submit** (on the dev box; new slugs must be first-pushed into a free GPU slot --
see the CLAUDE.md gotcha -- so queue them via `scripts/kaggle_push_queue.py` or push only when
`kaggle kernels status` shows fewer than 2 GPU sessions running):

```
set PYTHONUTF8=1
kaggle kernels push -p kaggle_submission_m2_lm_histcache/notebook
kaggle kernels status calamitychasm/arc3-m2-lm-histcache
python scripts/kaggle_submit_when_ready.py --kernel calamitychasm/arc3-m2-lm-histcache --version 1 \
    --message "m2 + level memory + history cache" --marker "LEVEL_MEMORY installed" \
    --marker "HISTORY_CACHE installed" --marker "priority gate active" \
    --marker "harness patch applied successfully" \
    --counters level_memory_summary.json --counters history_cache_summary.json

kaggle kernels push -p kaggle_submission_m2_lm_histcache_triedfacts/notebook
kaggle kernels status calamitychasm/arc3-m2-lm-histcache-triedfacts
python scripts/kaggle_submit_when_ready.py --kernel calamitychasm/arc3-m2-lm-histcache-triedfacts --version 1 \
    --message "m2 + level memory + history cache + tried facts" --marker "LEVEL_MEMORY installed" \
    --marker "HISTORY_CACHE installed" --marker "TRIED_FACTS installed" --marker "priority gate active" \
    --marker "harness patch applied successfully" \
    --counters level_memory_summary.json --counters history_cache_summary.json
```

(add `--arm "YYYY-MM-DD HH:MM"` in UTC to schedule). `--counters` may now be repeated: the gate
requires every listed counters file to exist and report `errors == 0` (a single `--counters` and
old job files with a string value behave as before). The tried-facts commands are in the section
above. The gate does not check the payload/timeout criteria; read `history_cache_summary.json` from
the check-run output before arming a submission.
