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
