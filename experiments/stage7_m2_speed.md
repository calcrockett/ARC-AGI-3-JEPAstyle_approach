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

## Round 3 (2026-10-09): mem 0.97, 12 streams, 32 GB host KV tier -- built, not yet run

Three kernels from the same builder (`python scripts/_build_m2_speed_kernels.py m97s12 m96s12hic m97s12hic`),
each = the incumbent (milestone-2 fork + level memory **only**; no history cache, no tried facts) with
serving knobs changed, the 25-game x 25-min speed shape, metadata identical to the incumbent except
`id`/`title`/`code_file`. Directories `kaggle_submission_m2_spd_<name>/notebook/`.

| kernel (`calamitychasm/...`) | mem | streams (`--max-running-requests`, `ARC3_MAX_ACTIVE_STREAMS`, graph max bs) | Mamba cache | host KV tier |
|---|---|---|---|---|
| `arc3-m2-spd-m97s12` | 0.97 | 12 | 72 | none |
| `arc3-m2-spd-m96s12hic` | 0.96 | 12 | 72 | `--hicache-size 32` |
| `arc3-m2-spd-m97s12hic` | 0.97 | 12 | 72 | `--hicache-size 32` |

The two single-factor questions: does 12 streams on a ~1.03M-token pool beat 10 (`m97s12` vs base), and does
the host tier make the resulting KV oversubscription cheap (`m96s12hic` vs `m97s12`; `m97s12hic` = both).

### Flags, checked against Pennyroyal v2.5.3 (d00d88e, the pin dfranzen's bundle builds)

`--enable-hierarchical-cache --hicache-size 32 --hicache-write-policy write_through --hicache-io-backend kernel`
(`server_args.py` ~L2796-2850; the last two and `--hicache-mem-layout page_first` are the defaults, spelled out so the
log shows them; `--hicache-ratio` defaults to 2.0 and is overridden by `--hicache-size`, which is GB of host RAM for
**all** host pools, split by device-pool bytes). Identical to sirikilohit's working boot (`--hicache-size 48`; his
`sglang-main.log`: `Attached hybrid pool stack to UnifiedRadixCache: pools=KV + MAMBA + QSA compressed`,
`hicache_attached=True`, `speculative_algorithm=EAGLE`, `mamba_radix_cache_strategy=extra_buffer`, `fp8_e4m3` KV,
`mem_fraction_static=0.97`, `max_running_requests=16`). Source path: `registry.py:_create_unified_radix_cache`
calls `UnifiedRadixCache.init_hicache` -> `hybrid_pool_assembler.build_hybrid_mamba_stack` (KV host pool incl. the
MTP draft layer, `MambaPoolHost`, `QSACompressedPoolHost`). Checks that bite: hicache is rejected with
`--disable-radix-cache` and `--enable-int8-mamba-checkpoint` (neither is used), and `tree_components` stays
`{FULL, MAMBA}` with hicache on, so dfranzen's Mamba-retention patch (it edits `unified_radix_cache.py`,
`schedule_batch.py`, `schedule_policy.py`, `mamba_component.py` -- not `registry.py`; its
`mamba_prefill_final_only` gate is `set(tree_components) == {FULL, MAMBA}`) stays active. Untested interaction:
the patch's early return in `cache_unfinished_req` skips the tree insert for non-branch chunks, so those chunks are
not write-through-backed to the host tier (less coverage, not a correctness problem); the check run's traceback
scan is the evidence. We run `NEXTN` (an alias of EAGLE) with a quantised draft; sirikilohit ran EAGLE with the
same QSA/MTP pools.

### Host RAM budget (why 32 GB, and why the builder refuses > 40)

Kaggle RTX Pro 6000 host: `MemTotal` 176.9 GiB (sirikilohit's `sglang_boot_result.json`; our census line prints the
same). dfranzen keeps the **BF16 PLE tables (102.4 GB = 95.4 GiB, pinned by `--ple-offload-embedding`)**; sirikilohit
swapped to FP8 PLE (about half). His boot ended with `MemAvailable` 55.7 of 174.3 GiB free at start -> 118.6 GiB
used = FP8 PLE ~47.7 (inferred) + hicache 48 GB (44.7 GiB) + **~26 GiB everything else**. For this stack:

| item | GiB |
|---|---:|
| BF16 PLE, pinned | 95.4 |
| processes / venv / misc (from his residual) | ~26 |
| **free before the host tier** | **~53** (the older "~64 free" note is the optimistic end) |
| `--hicache-size 32` (32 GB) | 29.8 |
| **free after** (harness, 25-110 game threads, sandboxes) | **~23** |
| `--hicache-size 48` | 44.7 -> ~8 free: unsafe |

SGLang checks each pool against `MemAvailable - 10 GiB` at allocation, so a too-large request fails loudly at
boot (`Not enough host memory`), but a host OOM *after* boot would kill a process mid-run; 32 GB keeps ~23 GiB
for the harness. `_build_m2_speed_kernels.py` refuses `HICACHE_GB > 40`. Pass criterion below: the 10-minute
census (`RAM used/total`) must keep >= 10 GiB available.

Host pool sizes at 32 GB (scaling his 48 GB split: KV 33.54 / Mamba 12.39 / QSA 2.10 GB): Mamba ~7.6 GB (~137
slots vs 72 on the device), KV+QSA ~24.4 GB ~ **1.7M tokens (~1.65x the device pool)**.

### What the stream count changes (harness + server audit)

| thing | depends on streams? | at 12 |
|---|---|---|
| `ARC3_MAX_ACTIVE_STREAMS` -> `_PriorityGate(slots)` | yes, the only harness knob | 12; log line `priority gate active: 12 concurrent streams` |
| `--max-running-requests`, `--cuda-graph-max-bs-decode` | must be >= gate slots (launcher asserts graph bs) | 12; graph list `{1,2,4,7,8,9,10,12}` (11 pads to 12) |
| **Mamba cache** `--max-mamba-cache-size` | **SGLang caps running requests at `size // 5`** (3 base + 2 overlap `extra_buffer` ping-pong, `kv_cache_configurator._calculate_mamba_ratio`) | incumbent 60 = 6/stream; 60 at 12 streams = exactly the floor, zero slots for retained prefix checkpoints -> **72** (+12 slots ~ +0.67 GB GPU, ~ -50K KV tokens). Builder refuses `MAMBA_CACHE // 5 < streams` |
| context trim: window 128K, `ARC3_CONTEXT_DRAIN_TOKENS` 58K, history 150/30 turns | **no** -- fixed token counts; nothing in the 8,400-line patch reads the pool size (no `max_total_num_tokens`, `server_info`, metrics) | per-stream context still saw-tooths 59K-128K |
| `bm.solver.concurrency` | no -- game threads (25 in this shape, 110 hidden) | unchanged |

So KV demand scales linearly with streams while the pool barely grows.

### KV oversubscription (quantified)

Incumbent pool 1,011,264 FP8 tokens (0.96); peak usage 0.91 in the 10-game check (~920K tokens), 0.98-0.99 in the
25-game speed shape. Per-stream footprint at that peak ~92-100K tokens (running contexts plus retained prefixes of
parked games). Pool estimate (his run: ~12.9 KB/token all pools, 1% of 95 GiB ~ 0.95 GiB ~ +73K tokens):
`m97s12` and `m97s12hic` ~ 1.011M + 73K - 52K (Mamba 72) ~ **1.03M (+2%)**; `m96s12hic` ~ 0.96M-0.98M... i.e. the
0.96 variant has ~ 1.01M - 52K ~ **0.96M (-5%)**; the log prints the true `#tokens`. Demand at 12 streams ~
1.10-1.20M => **107-116% of the pool** (vs 91-99% at 10); the worst case 12 x 128K = 1.54M. Without the host tier
the surplus is paid by radix evictions (re-prefill of parked games, prefix reuse falls from the ~93% baseline) and,
if no evictable pages remain, retractions. With the tier, evicted pages are already write-through-backed and are
loaded back over PCIe instead of re-prefilled. Decode itself was flat 9 -> 10 streams (757 -> 709 median tok/s),
and each step reads streams x context x ~12.9 KB of KV, so more streams may also not add aggregate tok/s at
128K contexts (sirikilohit's 899 tok/s @ 16 streams was at 69K contexts). Expect `m97s12` to be null-or-worse and
the answer to be in the hic pair.

### Free check run: pass criteria (25 games x 25 min, same shape as base; one run each)

Primary: generated tok/s (summary.txt) **>= 606.7 (= +5% over 577.8; base reproduced at 576.5)**, with
- retraction lines in `serve.log` **<= 2x base** (`python scripts/m2_speed_report.py base`),
- no OOM: no `CUDA out of memory` / `Not enough host memory` / `Killed`, serving stays up (m97 variants: lazily
  loaded Triton kernels need the headroom 0.97 leaves ~3.3 GB; mem 0.98 failed at ~2.3 GB),
- KV pool `#tokens` and usage peak recorded (peak < 1.0; a pinned 1.0 with retractions = oversubscribed),
- prefix reuse >= 90% (base 92.4-93.2%), median `#running-req` >= 11 (admission not Mamba-capped),
- markers in the notebook log: `LEVEL_MEMORY installed`, `priority gate active: 12 concurrent streams`,
  `harness patch applied successfully`; `level_memory_summary.json` `errors == 0`; no traceback outside serving
  teardown; every game won/gave_up/cancelled,
- hic variants only: `hicache_attached=True`, `Allocating kv hierarchical KV host pool`, `Attached hybrid pool
  stack to UnifiedRadixCache: pools=KV + MAMBA + QSA compressed` in `serve.log`, and `MemAvailable` >= 10 GiB in
  every census line.
Public-25 levels/score are reported, not used to rank. Adopt only if the primary and all guards pass; a hic
variant that passes but is not faster still tells us the tier is free (needed before spending the freed KV on
longer history). A kernel whose mount fails (`cp: cannot stat .../taaf-kaggle-source-bundle-copy`) is a dead slug:
rebuild under a new name rather than re-pushing.

### Dev-box commands (Windows; needs a free GPU slot for the first push -- see Gotchas 2026-10-05)

```
cd C:\path\to\ARC-AGI-3-JEPAstyle_approach
git pull --no-rebase origin claude/modest-ride-gut4vo
set PYTHONUTF8=1
python scripts\_build_m2_speed_kernels.py m97s12 m96s12hic m97s12hic     # regenerates (idempotent)
:: queue (pushes only into a free slot; order = first two run as a pair)
echo kaggle_submission_m2_spd_m97s12/notebook>> logs\kaggle_push_queue.txt
echo kaggle_submission_m2_spd_m96s12hic/notebook>> logs\kaggle_push_queue.txt
echo kaggle_submission_m2_spd_m97s12hic/notebook>> logs\kaggle_push_queue.txt
echo calamitychasm/arc3-m2-spd-m97s12>> logs\kaggle_watch_kernels.txt
echo calamitychasm/arc3-m2-spd-m96s12hic>> logs\kaggle_watch_kernels.txt
echo calamitychasm/arc3-m2-spd-m97s12hic>> logs\kaggle_watch_kernels.txt
python scripts\kaggle_push_queue.py            :: one tick; or let task ARC3PushQueue tick every 10 min
:: results (once COMPLETE)
kaggle kernels output calamitychasm/arc3-m2-spd-m96s12hic -p logs\m2_speed\m96s12hic
python scripts\m2_speed_report.py base m97s12 m96s12hic m97s12hic
```
Do not run a submission queue slot and these at the same time; the 2-session limit is shared.
