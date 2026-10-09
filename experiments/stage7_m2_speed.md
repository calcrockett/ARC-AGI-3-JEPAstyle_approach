# Stage 7 — serving speed on the incumbent (milestone-2 fork + level memory)

**Why.** The run is decode-bound and short on tokens (~200K/game vs 27-67K
per level). Our check run: decode plateaus ~700-760 tok/s at 8-10 running
requests, KV pool peaks at 91%, 4.25 GB GPU memory left free, speculative
accept length ~2.7 of 4. The other two Milestone-2 winners peak at 1,135 and
1,159 tok/s.

**Variants** (`scripts/_build_m2_speed_kernels.py`; serving knobs only; every
variant plays all 25 public games for 25 min so more games compete than there
are streams, as in the hidden run; never submitted):

| variant | change |
|---|---|
| base | incumbent serving: 3 draft steps, mem 0.96, 10 streams |
| spec4 | 4 draft steps (lossless) |
| s12 | mem 0.98, 12 streams |
| s14 | mem 0.98, 14 streams, Mamba cache 72 |
| s14hic | s14 + 32 GB hierarchical KV cache in system RAM (~64 GB free of 177) |

Kaggle allows 2 GPU sessions: base and spec4 ran together (fair pair); the
rest are pushed by a detached queue (`scripts/kaggle_push_queue.py`, task
`ARC3PushQueue`). Report: `scripts/m2_speed_report.py` -> `logs/m2_speed_report.md`.

**Pre-registration.** Primary metric: generated tokens/s over the run (from
the harness summary), with decode tok/s by concurrent requests from the server
log as the mechanism. A variant is adopted only if it beats base on generated
tokens/s by > 5% with no tracebacks, no retraction storm (retracts within 2x
base) and prefix reuse >= 90%. spec4 is lossless, so it may ship on speed alone;
stream/memory variants change cache pressure and are judged on all four.
s14hic must show `hicache_attached=True`, else it is a no-op, not a result.
Public-game scores are reported but cannot rank (one 25-game run each).

## Round 1 results (2026-10-04)

- **spec4: impossible on this model.** Server start failed in CUDA-graph
  capture: "Qwen QSA requires speculative_num_draft_tokens <= the QSA compress
  ratio (4) ... got 5". The incumbent's 3 steps (4 draft tokens) is already the
  architectural maximum. Option closed.
- **base** (25 games, 10 streams): 577.8 generated tok/s; decode 779 tok/s at
  10 running (median running 10); accept length 2.62; prefix reuse 92.4%;
  **KV pool peak 0.99** (0.91 in the 10-game check) -- with more games than
  streams the cache, not compute, is the binding constraint.
- **s12 / s14 / s14hic: not run.** All three died at setup: the bundle dataset
  was not mounted (`cp: cannot stat .../taaf-kaggle-source-bundle-copy`),
  despite identical metadata and an unchanged dataset (last updated 09-28);
  base/spec4, pushed ~1 h earlier, mounted it. Treated as a transient Kaggle
  mount failure; re-queued 15:00 UTC (s12 pushed as v2).

## Round 2 results (2026-10-05)

- **Throughput reproduces.** base re-run (v2, 11:43 UTC): 576.5 generated
  tok/s vs 577.8 in v1 (0.2%); decode 770 tok/s at 10 running; KV peak 0.98;
  prefix reuse 93.2%.
- **The dataset-mount failures were kernel-specific, not Kaggle-wide.** A CPU
  diagnostic kernel with the same inputs mounted the bundle; base v2 mounted
  it. The old s12/s14/s14hic slugs failed 3x each. Rebuilt under fresh slugs
  (`arc3-m2-spd2-*`): **s14 mounted fine; spd2-s12 failed again -- and its very
  first push was the one rejected at the 2-GPU-session limit** (18:19 UTC),
  exactly like the original three. Working rule: a kernel whose first push is
  rejected at the session limit never mounts its datasets; push only into a
  free slot (the queue now checks).
- **s14 (mem 0.98, 14 streams): server crashed with CUDA OOM** on the first
  requests ("Tried to allocate 160 MiB ... 49 MiB free"): Triton kernels load
  lazily after start-up and need the headroom that mem 0.96 leaves (~4.25 GB
  free). KV pool would have been 1,104,576 (+9%). **mem 0.98 is not viable**;
  more streams need memory from elsewhere (smaller KV dtype, shorter retained
  context, or a host-RAM tier), not from the static fraction.
- **s14hic: cancelled by Kaggle** (CANCEL_ACKNOWLEDGED, no output); not
  measured. It shares s14's mem 0.98 and would hit the same OOM -- needs a
  rebuild at mem 0.96 before it can answer the system-RAM question.

**Status of the speed questions:** #1 (more draft tokens) closed -- impossible.
#2 (more streams) and system-RAM tier both still open; next attempt must keep
mem 0.96 and find KV room another way.
