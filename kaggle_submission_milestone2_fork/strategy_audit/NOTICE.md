# Strategy audit: provenance (lordhansolo, Milestone 2)

`strategy_audit.py` is **ours**. The prompt text in it (`PROMPT`) is lordhansolo's `STRATEGY_AUDIT_PROMPT` with three
harness-specific references adapted, and the trigger re-implements his `_should_audit_strategy` on a token clock.

Source: Kaggle kernel `lordhansolo/arc-agi-3-milestone-2` (one of the three ARC Prize 2026 Milestone-2 winners, public
LB 23.84) and its dataset `lordhansolo/taaf-kaggle-source`, files
`src/ARC3-Inference/inference/agent/prompts.py` (lines 11-46, sha256 of the file `51972ca4f823e8de...`),
`.../agent/tool_agent.py` (lines 711-712: appended to the turn's state lines) and `.../framework/solver.py`
(lines 231-250, 310, 437-477: the timer). Read from the public mirror github.com/tonghuikang/daniel-franzen-arc-agi-3
@ `f472820`, `kaggle/lordhansolo/dataset-taaf-kaggle-source/` (that repository is Apache-2.0).

License: the bundle itself carries no LICENSE file. It was published as a public competition notebook and dataset for
the Milestone-2 prize, which requires a public notebook under an open-source license, and the competition rules treat
publicly shared competition code as licensed under an OSI-approved license that permits commercial use (`rules.md`).
It is built on Tufa Labs' Duck harness / TAAF (Apache-2.0). We credit it here and in the notebook header. **Before a
final selection, confirm the license shown on lordhansolo's Kaggle notebook page**; if it is not Apache/MIT-style,
replace `PROMPT` with our own wording (the trigger is ours).

## Text changes (everything else verbatim)

| lordhansolo | ours | why |
|---|---|---|
| "Saved notes and passing tests of your own code do not establish that the assumed rules are complete." | "Your own notes, retained helper functions and passing checks of your own code do not establish ..." | our harness keeps retained helper functions, not saved modules with tests |
| "put the hypothesis and predicted outcomes in the python call's plan and execute it with action(...)" | "write the hypothesis and predicted outcomes as comments at the top of the python snippet and execute it with action(...)" | our `python` tool takes only `code`; there is no `plan` argument |
| "Include the finding and supporting observation in that action's plan so they remain available in history after compaction." | "Write the finding and supporting observation as a comment in that turn's snippet so they remain available in your history." | same |
| "If revising a saved rule, add the observed counterexample to its tests." | "If a retained helper function encodes a revised rule, fix that function." | same |

## Trigger changes

| lordhansolo | ours |
|---|---|
| wall clock since the level started (or since the last audit turn) >= 25% of the game's initial runtime limit (3,918 s cap -> ~980 s) | generated tokens on the level since it started (or since the last confirmed audit) >= `ARC3_STRATEGY_AUDIT_TOKENS` (default 56,000 = 25% of ~228K expected tokens per game on the turbo-lossless stack) |
| timer restarts at the audit turn's start once the turn completes without a retryable failure | clock restarts at the audit's delivery once the model generated tokens after it; a turn that generated nothing does not count and the audit is re-sent |
| appended to the turn's state lines | appended to the end of the turn opener (user prompt); a resumed turn re-sends the full opener in this harness, the audit is not re-sent there unless the clock is due again |

Our clock is in tokens because the milestone-2 priority gate time-shares ~110 games over 10-14 streams: wall time on a
level is mostly time parked behind other games.
