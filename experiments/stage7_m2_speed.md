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
`m97s12` and `m97s12hic` ~ 1.011M + 73K - 52K (Mamba 72) ~ **1.03M (+2%)**; `m96s12hic` (no extra 1%) ~ 1.011M - 52K ~ **0.96M (-5%)**; the server log prints the true `#tokens`. Demand at 12 streams ~ 1.10-1.20M => **107-116% of the 1.03M pool (115-125% of 0.96M)**, against 91-99% at 10 streams; the worst case 12 x 128K = 1.54M. Without the host tier
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

## vLLM stack: the incumbent harness on lordhansolo's serving (2026-10-09, built, not yet run)

**What.** `scripts/_build_m2_vllm_kernel.py` -> `kaggle_submission_m2_vllm_s12/notebook` (`calamitychasm/arc3-m2-vllm-s12`)
and `..._s14/notebook` (`calamitychasm/arc3-m2-vllm-s14`). The incumbent notebook (milestone-2 harness + level memory,
from `scripts/_build_m2_level_memory_kernel.py`) with only the serving swapped for lordhansolo's
(`lordhansolo/arc-agi-3-milestone-2`): vLLM nightly e975732 (0.29.1rc1) unpacked from Docker layers plus his
hash-pinned 32-file overlay (rc2, `e3a6fe0d...`), his NVFP4/FP8 mixed checkpoint with its built-in MTP head (3 draft
tokens, 32k draft vocabulary), FP8 KV, Mamba prefix caching in align mode, gpu-mem-util 0.98, max-num-batched-tokens
2048, async scheduling, PLE + input embeddings in host RAM, his watchdog (5 restarts) and GPU shard prefetcher.
Every harness cell, the level-memory install and the run cell are byte-identical to the incumbent (pinned by
`tests/test_m2_vllm_kernel.py`). Never submitted until a check run passes the criteria below.

**Mechanism.** The run is decode-bound and KV-starved (base: 577.8 generated tok/s, KV peak 0.99 at 10 streams).
His stack decodes faster per GPU (peak 1,135 tok/s at 14 streams vs our 946) and holds a bigger pool
(~1.42M tokens vs 1.01M: no separate draft model/draft KV, the input embeddings off the GPU, Mamba state shared with
the KV pool), so 12-14 streams fit where SGLang OOMed at mem 0.98 (round 2).

**How it is wired.** His setup step (`setup_commands.json[0]` of `lordhansolo/taaf-kaggle-source`) is vendored verbatim
(`kaggle_submission_m2_vllm/vendor/`, sha-pinned, provenance in `PROVENANCE.json`) and patched at build time by exact
anchors only:

| knob | his value | ours | why |
|---|---|---|---|
| `--max-model-len` | 147,072 | **139,264** | the incumbent's SGLang context: harness window (116+12)K + 8K headroom; read from the incumbent cell, not retyped |
| `--max-num-seqs` (= `ARC3_MAX_ACTIVE_STREAMS`) | 14 | **12 / 14** | two kernels |
| `max_pixels` | 65,536 (256 px boards, ~66 tokens) | **409,600** | our boards are 640 px (~402 tokens); below this vLLM silently downscales them |
| `--limit-mm-per-prompt` | `{"video":0}` | `{"image":64,"video":0}` | our harness attaches grid + diff + death + animation boards in one request; his sent one |
| `--default-chat-template-kwargs` | `preserve_thinking`, `reasoning_effort: xhigh` | **`preserve_thinking` only** | the incumbent's default; the harness sends `enable_thinking`/`preserve_thinking` per request and never `reasoning_effort` |
| `--enable-prompt-tokens-details` | test runs only | **always** | `cached_tokens` for the prefix-reuse metric |
| served model name | `primitive-ai/...` | `flashnext` (+ his name as alias) | the name the incumbent's cell 5 / pickled benchmark request |
| cache tracing plugin (test runs) | on | **opt-in** (`ARC3_VLLM_CACHE_DIAGNOSTICS=1`) | per-lookup tracing would bias the speed measurement |
| overlay check | `RuntimeError` | **`VLLM_OVERLAY_HASH_MISMATCH <found>` and exit 86** before unpacking | the tar is hashed, then his manifest/overlay validation and the applier run under the same marker |
| his harness env export (temperature 0.6, `xhigh`, 4x boards, 1K tool output, 81K target) | written | **dropped** | the incumbent's cell 5 configures the harness |

Sampling stays the incumbent's: the harness sends temperature 0.7 / top_p 0.95 / top_k 20 per request, and
`--generation-config vllm` only removes the model's defaults for fields a request omits (the harness sends no
`min_p`/`repetition_penalty`/`seed`; SGLang filled omitted fields from `generation_config.json`, which the boot probe
prints so the dev box can see whether anything there differed). Cell 5: port 1234, `ARC3_REASONING_HISTORY_KEY=reasoning`
(vLLM's field; vLLM also accepts `reasoning_content` -- the probe checks both). The launcher cell runs the setup step
detached and keeps the incumbent's release at 12 min (`ARC3_HTTP_RETRY_INITIAL_SECONDS=900` grace); a setup failure
raises (`VLLM_SETUP_FAILED rc=N`). It refuses a Python mismatch between the notebook image and the unpacked runtime
(`VLLM_PYTHON_ABI_MISMATCH`; fix: drop `docker_image` from kernel-metadata.json). The check run plays all 25 public
games x 25 min (the speed shape, comparable to base); a submission is unaffected.

**Boot probes (check runs only, logged before the benchmark when the server is up by the deadline):**
`TEMPLATE_PROBE` (tokenizer / chat template sha, `matches_m2_pin` against the incumbent's pinned tokenizer, the
model's `generation_config.json`; files copied to `vllm_model_meta/`), `REASONING_ECHO_PROBE` (does the served
template render a prior turn's reasoning under each key; rendered text in `vllm_rendered_probe.txt`),
`IMAGE_TOKENS_PROBE` (prompt-token delta of one 640 px board), `MULTI_IMAGE_PROBE` (six boards in one request),
`PREFIX_CACHE_PROBE identical=` (one ~8K-token multi-turn prompt with an image and prior reasoning at temperature 0:
cold, warm, warm; `cached_tokens` per call; `under_load=True` if games were already running), then
`VLLM_SERVING active ... ready_after=Ns`. A monitor thread snapshots `/metrics` every 2 min to `vllm_metrics.jsonl`
(preemptions, prefix hits/queries, spec-decode acceptance) and echoes the watchdog log as `[vllm-watchdog]` lines.
`python scripts/m2_speed_report.py base vllm-s12 vllm-s14` reads all of it (vLLM stat lines, metrics, watchdog log,
probes) into the same table as the SGLang runs.

**Tokenizer / chat template.** Neither model's tokenizer or `chat_template.jinja` is in the local clones, and
huggingface.co is blocked from the build container, so they could not be diffed here. Known from configuration:
both servers load `chat_template.jinja` from their own model directory; his default kwargs add `reasoning_effort: xhigh`
(removed); his MTP draft uses a 32k draft vocabulary where ours used a 64k FR-Spec hot-token map (acceptance only,
never the text); our tokenizer is pinned at `06b95093...` and `TEMPLATE_PROBE matches_m2_pin` reports whether his
tokenizer.json is byte-identical. The diff happens on the first check run (see dev-box steps).

**Risks.**
- **Model swap: quality unknown.** NVFP4/FP8 mixed weights and a different MTP head replace W4A16 AutoRound + the
  INT4 drafter. Speed is measurable in one check run; solvability is not (public-25 cannot rank; the hidden-set noise
  floor is sd ~3-4 per draw). This is a two-variable change (server + weights) and cannot be separated without a
  vLLM-on-W4A16 or SGLang-on-NVFP4 arm.
- **Per-stream decode may be slower at our contexts.** His speed is at <= 81K-token prompts; ours saw-tooth
  59K -> 118K (each decode step reads streams x context of KV). 12 x ~90K average ~ 1.08M tokens (s12), 14 x ~90K ~
  1.26M (s14) against ~1.42M, but peaks reach 1.54M / 1.79M: s14 can preempt, and vLLM V1 preemption recomputes the
  whole prompt. The 640 px image profile may also shrink the pool slightly (read `GPU KV cache size`).
- **Prefix-cache correctness on Mamba/GDN.** The old vLLM NVFP4 Duck stack had prefix caching produce identical
  actions and -47% score (closed lines, section 9). His overlay exists to make align-mode caching correct; the
  determinism probe is the only check we have before scores.
- `--max-num-batched-tokens 2048` (vs SGLang chunk 8192): a 60K re-prefill after a history drain takes ~30 steps.
- Server restarts: his harness retries a failed request every second until its deadline; ours retries 3 x 5 s, so a
  watchdog restart (weights reload, minutes) costs every in-flight game more here than in his run.
- The datasets are referenced unversioned: a new version of his runtime with another overlay fails the gate loudly
  (by design); a new bundle version could move the draft vocabulary.

**Day-0 dev-box checks (before queueing; no GPU needed):**
```
kaggle datasets files lordhansolo/vllm-main-e975732-arc3          :: runtime-manifest.json, overlay .tar.blob, layer blobs, applier
kaggle datasets files lordhansolo/taaf-kaggle-source              :: src/ARC3-Inference/configs/draft_vocab_32k.json present
kaggle models instances versions files lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1
kaggle datasets download lordhansolo/vllm-main-e975732-arc3 -f arc3_vllm_main_e975732_arc3_overlay.tar.blob -p tmp\vllm
kaggle datasets download lordhansolo/vllm-main-e975732-arc3 -f runtime-manifest.json -p tmp\vllm
certutil -hashfile tmp\vllm\arc3_vllm_main_e975732_arc3_overlay.tar.blob SHA256   :: must be e3a6fe0d9f010bc1...e065e0cb
type tmp\vllm\runtime-manifest.json | findstr dist_packages                       :: python3.X must match the notebook image
```
(A downloaded single file may arrive zipped; unzip before hashing.)

**Push (pushes only into a free GPU slot; Gotchas 2026-10-05):**
```
git pull --no-rebase origin claude/modest-ride-gut4vo
set PYTHONUTF8=1
python scripts\_build_m2_vllm_kernel.py                       :: idempotent; tests pin the committed copies
echo kaggle_submission_m2_vllm_s12/notebook>> logs\kaggle_push_queue.txt
echo kaggle_submission_m2_vllm_s14/notebook>> logs\kaggle_push_queue.txt
echo calamitychasm/arc3-m2-vllm-s12>> logs\kaggle_watch_kernels.txt
echo calamitychasm/arc3-m2-vllm-s14>> logs\kaggle_watch_kernels.txt
python scripts\kaggle_push_queue.py
:: once COMPLETE
kaggle kernels output calamitychasm/arc3-m2-vllm-s12 -p logs\m2_speed\vllm-s12
python scripts\m2_speed_report.py base vllm-s12 vllm-s14
```
Template diff after the first run: `logs\m2_speed\vllm-s12\vllm_model_meta\chat_template.jinja` against the incumbent
model's (`https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/resolve/main/chat_template.jinja`), and
`vllm_rendered_probe.txt` for what the template does with prior reasoning.

**Check-run pass / kill criteria (one run each; any kill item ends the arm):**
- kill: `VLLM_OVERLAY_HASH_MISMATCH`, `VLLM_PYTHON_ABI_MISMATCH`, `VLLM_INPUT_MISSING` or `VLLM_SETUP_FAILED`;
  `VLLM_SERVING active ... ready_after` > 900 s (ready < 15 min from notebook start);
  any watchdog exit/restart or CUDA fault (`restarts`, `cuda_faults` in the report);
  `PREFIX_CACHE_PROBE identical=False` with `under_load=False`;
  `IMAGE_TOKENS_PROBE` far from ~402 (e.g. ~66 = downscaled) or `MULTI_IMAGE_PROBE ok=False`;
  `REASONING_ECHO_PROBE ... rendered=False` for the harness key.
- pass (all): generated tok/s (summary.txt) **>= 665** (+15% over base 577.8; the model swap needs a larger margin
  than a pure serving knob); server-side prefix hit / request `cache_pct` **>= 85%**; preemptions (`vllm_metrics.jsonl`)
  **<= ~50**; tool-failure and truncation rate **<= 2x base**; `level_memory_summary.json` `errors == 0`; the usual
  markers `LEVEL_MEMORY installed`, `priority gate active: N concurrent streams`, `harness patch applied successfully`,
  plus `VLLM_SERVING active`; no traceback outside serving teardown; every game won/gave_up/cancelled.
- If both pass: the gated submitter takes `--marker "VLLM_SERVING active"` in addition to the usual three. Even then,
  ranking needs hidden-set draws (n >= 3, interleaved with the incumbent), because the weights changed.

## Turbo kernel: JustAdev742's measured serving config on our stack (2026-10-09, built, not yet run)

Kernel **`calamitychasm/arc3-m2-turbo`** (`kaggle_submission_m2_turbo/notebook/`), built by
`python scripts/_build_m2_level_memory_kernel.py --turbo` = the incumbent (milestone-2 fork + level memory)
**+ history cache + sandbox-timeout fix + REAP-448 + MTP acceptance 0.5 + ARC FR-Spec map + 14 streams**, with the
check run in the speed kernels' shape (25 public games x 25 min; the competition rerun is unaffected). Every piece
is a separate builder flag, so subsets build too (`--timeout-fix`, `--reap`, `--spec-accept X`, `--arc-hotmap`,
`--streams N`, `--check-all25`; e.g. `--reap --streams 14` -> `arc3-m2-lm-reap-s14`). The existing kernels
(histcache / triedfacts / both) rebuild byte-identical. Provenance and the vendored files:
`kaggle_submission_milestone2_fork/turbo/` (`NOTICE.md`; JustAdev742, Apache-2.0).

### Source and why it transfers

JustAdev742 (github.com/JustAdev742/Arc-Agi-3-Kaggle-comp; Kaggle team "Jovian Game Studios", kernels
`scottmahony/...`) runs **exactly our serving stack**: Franzen's notebook (same sha256 `7b76c194...` as our
`upstream/` copy), the same Pennyroyal v253 wheel (`sglang-0.5.19+gd00d88efc8d6`), Intel's W4A16 checkpoint, the
albucino MTP draft, and the same base FR-Spec map (`becfa41d...` = our `TOKEN_MAP_SHA`). Their harness differs:
they run D' (Franzen's notebook with a different slot priority), we run Franzen's priority gate + level memory
(+ history cache here).

### Mechanism

| change | what it does | where in the notebook |
|---|---|---|
| **REAP-448** | the target loads 448 of its 512 routed experts per layer (the set the public `REAP-k448` build keeps; recovered bit-exactly from router rows, checked at load against per-layer router sha256s). Frees 7.31 GiB of weights -> KV pool ~1.01M -> **1,477,888 tokens** (their serve.log). The MTP draft keeps 512 experts (`--speculative-draft-model-override-args '{}'`). | cell before the launcher writes `sglang_reap_patch.py`, the kept list (shipped as the 64 pruned ids per layer, rebuilt byte-identical, sha256 checked) and its meta; launcher: `apply` on the installed `qwen4_exp.py` right after `env.update` (sha256-locked file, two anchors; **raises, stopping the cell before the server starts**, on any mismatch), `ARC3_REAP_KEPT_EXPERTS` in the server env, `--json-model-override-args '{"text_config": {"num_experts": 448}}'` |
| **14 streams** | MAXREQ 14, CUDA-graph max bs 14 (graph list `{1,2,4,7,8,9,10,14}`; 11-13 pad to 14), Mamba cache 84 (= 6/stream; 84 // 5 = 16 >= 14), `ARC3_MAX_ACTIVE_STREAMS` 14; **mem fraction stays 0.96** | CFG + setup cell |
| **acceptance 0.5** | `--speculative-accept-threshold-single/acc 0.5` (incumbent 1.0 = lossless): a draft token is also accepted when the target is confident enough -- roughly a lower effective temperature. **Lossy by design.** | CFG |
| **ARC FR-Spec map** | the MTP draft may only propose tokens in its 64k hot map; Pennyroyal's generic map misses 1.2-1.5% of this harness's output tokens ("Hmm", " BFS", ".ascii", grid runs). JustAdev742's map (same size) covers 99.9%. Lossless. | cell before the launcher writes it (runs of ids, base64, no zlib; torch.save zip layout with fixed records -> sha256 `ec15348b1186...`, byte-identical to what their builder writes); `TOKEN_MAP_SHA` replaced, so the launcher's own assert checks it |
| **timeout fix** | a python call that hits the 30 s sandbox timeout returns no `keepable_functions`, and `_record_retained_functions` then dropped **every** retained helper (Franzen's demo: r11l lost 13 helpers to one timeout, then 10 min with 1 action). Now the previous functions stay and the payload says so; `time` joins SAFE_MODULES. Installed at runtime (`timeout_fix.py` wraps the method and edits the bootstrap string), so the harness patch cell stays verbatim; raises on a missing anchor. | own cell after the history cache cell |

**Stream count chosen: 14**, their measured best on this stack. 16 is not built (the builder refuses > 14): in their
full-length run at 14 the pool peaked at 0.99 with 12.3 of 14 running and a queue of at most 1, so 16 would mostly add
retractions at long contexts; nobody measured it. The builder also refuses > 12 streams without `--reap` (the 1.01M
pool is oversubscribed past 12; see Round 3) and Mamba cache // 5 < streams. Mem fraction is left at 0.96 (0.98 OOMs,
Round 2); their 14-stream runs at 0.96 had >= 0.95 GiB minimum free device memory (Franzen's v3: 0.87 GiB).

### Their evidence (same conditions = D', 25 public games x 25 min; their `docs/research_log.md`, 2026-10-07/08)

| gate | output tok/s | decode p50 at 10 running | mean running | MTP accept length |
|---|---:|---:|---:|---:|
| base, 10 streams (exp-072a) | 641.9 | 783 | 9.28 / 10 | 2.64 |
| REAP-448 + 14 streams (exp-072f) | 733.2 (+14%) | 811 | 12.58 / 14 | 2.64 |
| + acceptance 0.5 (exp-072g) | **819.3 (+28%)** | 920-1004 at 12-14 running | 12.57 / 14 | 3.06 |

Output tokens per request unchanged (1,718 vs 1,718): no sign of longer/looping outputs under lossy acceptance.
Full length (121 min/game, public 25): REAP + 14 alone 49.45 (735 tok/s); + acceptance 0.5 **56.00** (791 tok/s, accept
3.10) and a repeat **42.89** -- against Franzen v3's four passes 45.6-47.5. The FR-Spec map: accept 3.15 vs 3.09-3.10,
decode +2% (exp-077). Their elasticity estimate: score ~ 0.6-0.8 x tokens.

**Fidelity probe** (154 logged requests, greedy, logprobs, prefix cache flushed): REAP-448 mean |delta logprob| on the
agreed prefix 0.048 vs a cross-run noise floor 0.038 (ratio ~1.3), prefix share before divergence 0.075 vs 0.12-0.13;
the shift is **confined to image turns** (fresh-frame turns 0.054 vs 0.037; text/tool turns 0.038 vs 0.038), largest
on vc33/ft09/tn36. They kept REAP (no score loss visible; vc33/tn36 won in the 56.00 run).

**Their LB draw of this serving config** (exp-074t = D' + REAP-448 + 14 streams + acceptance 0.5 + timeout fix):
submission 56980485, 2026-10-09 00:14 UTC -- **still PENDING** in their log at the time of this port (their repo
`fe2ad06`, 03:53 UTC). Read it before the first turbo submission; their D' copy at 10 streams drew 28.87 (2026-10-08).

### Risks

- **Quality (REAP)**: calibrated on agentic text traffic, not ARC; the measurable shift lives on image turns, which
  every turn of ours has. Public-25 scores (n=2, 56.00 / 42.89) cannot resolve a few points.
- **Quality (acceptance 0.5)**: changes the sampled distribution (lossy). Measured output length unchanged; quality
  only through scores.
- **Our harness differs from theirs** (Franzen priority + level memory + history cache vs D'): KV demand per stream
  may differ; read the KV peak and retractions.
- **Startup failure**: an apply failure raises in the launcher (no server, no run) -- intentional; a failure inside
  the server (router sha mismatch) leaves the notebook running against a dead server, as upstream does for any
  server failure. The launcher prints `REAP448 applied kept=448 | ...` only when serve.log shows the server's own
  `ARC3 REAP: kept 448 of 512 ... router sha256 verified` line, else `REAP448 NOT CONFIRMED` (gate fails).
- **Input mount layout**: JustAdev742 saw 5 of 9 GPU sessions get the older `/kaggle/input/<slug>` layout (their
  `--input-fallback`). Our notebooks hardcode the newer `/kaggle/input/datasets/...` layout, which every one of our
  runs so far got; a mount failure shows as `cp: cannot stat` in the first minute (not specific to turbo).
- **Check-run confound**: base 577.8 was measured without the history cache; turbo includes it (host-side only).

### Check run: pass / kill criteria (25 games x 25 min, the base shape; one run)

- **Markers** (notebook log; `python scripts/_build_m2_level_memory_kernel.py --turbo` prints the list):
  `LEVEL_MEMORY installed`, `priority gate active: 14 concurrent streams`, `harness patch applied successfully`,
  `HISTORY_CACHE installed`, `TIMEOUT_FIX installed`, `REAP448 applied kept=448`, `SPEC_ACCEPT 0.5`,
  `ARC_HOTMAP sha=ec15348b11863ec6fb94b655e4f9ddc4c0ce457fb11f77807b0c5c2d391da70f`; no `REAP448 NOT CONFIRMED`.
- **Primary**: generated tok/s (summary.txt) **>= 664.5 (+15% over base 577.8)** on the same shape; their
  same-conditions gain was +28%, so 664 is a deliberately loose floor. Kill below 606.7 (+5%).
- **Guards**: retraction lines in serve.log **<= 2x base**; no OOM (`CUDA out of memory`, `Killed`), serving stays up;
  KV pool `#tokens` ~1.45-1.5M and its peak recorded; prefix reuse >= 90%; median `#running-req` >= 12; MTP accept
  length reported (expect ~3.0-3.15 vs base 2.62); serve.log `server_args` show
  `'json_model_override_args': '{"text_config": {"num_experts": 448}}'` and
  `'speculative_draft_model_override_args': '{}'`; the draft's `Load weight end ... Qwen4ExpForCausalLMMTP` memory
  unchanged (~3.79 GB, proof the draft kept 512 experts).
- **Counters**: `history_cache_summary.json` as in the histcache criteria (`errors`, `write_fallbacks`,
  `payload_plain`, `view_misses`, `loads_stale` all 0; `payload_delta` >> `payload_full`);
  `level_memory_summary.json` `errors == 0`; `timeout_fix_summary.json` `errors == 0` (`timeouts_kept` is reported:
  how many timeouts would have wiped helpers).
- **Lossy-acceptance guards (new 2026-10-09; turbo only, because only turbo changes the sampled distribution)**, both
  from the check run's request logs (`<game>_p0_requests.jsonl`, event `response`; the check run saves them):
  (a) **output tokens per request within +-10% of base** (mean `usage.completion_tokens`; JustAdev742 saw 1,718 vs
  1,718, so a drift means acceptance 0.5 changed how long the model talks); (b) **0 exact repeated assistant turns**
  (an assistant turn whose content + tool calls exactly equal the previous assistant turn of the same game; a
  loop detector for lossy acceptance). Trip either and the acceptance step is suspect: do not submit turbo, run
  turbo-lossless. Automated, see "Computing the guards" below. The base value for (a) must come from a base
  check run with request logs (`python scripts\m2_speed_report.py base turbo`).
- No traceback outside serving teardown; every game won/gave_up/cancelled.

**Score pre-registration.** If the serving gain transfers (+28% tokens) and their elasticity holds, the expected
effect is roughly +17-22% score over the incumbent's 30.47 (sd 2.21, n=4), i.e. ~35-37 -- but REAP and lossy
acceptance can cost quality that no throughput number shows. One draw is a catastrophe check only (< 22 fails:
more than 2 sd of the fork's draw noise, 3.93, below the incumbent's mean). Adoption needs n >= 3 draws interleaved in time with the incumbent
(or histcache) arm; the final-selection rules in CLAUDE.md apply. If the bundle loses, ablate on the check-run shape
first (`--reap --streams 14` vs `--spec-accept 0.5` alone), not on submissions.

### Commands

Build (idempotent; tests pin the committed notebook): `python scripts/_build_m2_level_memory_kernel.py --turbo`.

Dev box:
```
set PYTHONUTF8=1
echo kaggle_submission_m2_turbo/notebook>> logs\kaggle_push_queue.txt
echo calamitychasm/arc3-m2-turbo>> logs\kaggle_watch_kernels.txt
python scripts\kaggle_push_queue.py
kaggle kernels output calamitychasm/arc3-m2-turbo -p logs\m2_speed\turbo
python scripts\m2_speed_report.py base turbo
python scripts\kaggle_submit_when_ready.py --kernel calamitychasm/arc3-m2-turbo --version 1 ^
    --message "m2 turbo draw 1" --marker "LEVEL_MEMORY installed" --marker "priority gate active: 14 concurrent streams" ^
    --marker "harness patch applied successfully" --marker "HISTORY_CACHE installed" --marker "TIMEOUT_FIX installed" ^
    --marker "REAP448 applied kept=448" --marker "SPEC_ACCEPT 0.5" --marker "ARC_HOTMAP sha=ec15348b11863ec6fb94b655e4f9ddc4c0ce457fb11f77807b0c5c2d391da70f" ^
    --counters level_memory_summary.json --counters history_cache_summary.json --counters timeout_fix_summary.json
```
(read `history_cache_summary.json` by hand; this gate does not check the zero-fields -- the kaggle-ops `submit` op does,
via `require_zero`). GitHub Actions (kaggle-ops) request examples are in CLAUDE.md ("Operating Kaggle from the cloud").

### Computing the guards (output length, loops)

`scripts/m2_speed_report.py: request_log_metrics(dir)` reads every `*_requests.jsonl` in a downloaded kernel output and
returns `completion_tokens_per_request` (mean of `usage.completion_tokens` over `response` events),
`repeated_assistant_turns`, `repeated_assistant_turns_long` (those of >= 200 serialised characters: the same one-line
tool call twice is not a loop) and up to 3 `repeat_examples`. It is in the `m2_speed_report.py` table (columns
`completion_tokens_per_request`, `repeated_assistant_turns`, `repeated_assistant_turns_long`; `m2_speed_report.py base
turbo turbo-lossless`) and in the kaggle-ops `kernel_output` digest (`request_log`). Method: every request carries the
(possibly trimmed) history, so per game the assistant turns are reassembled by overlap (largest k with the sequence's
last k turns == the request's first k turns), then adjacent equal turns are counted; a retried request does not count.
By hand: `jq -c 'select(.event=="response") | .usage.completion_tokens' <game>_p0_requests.jsonl` for the mean, and
diff the assistant messages of consecutive responses for repeats. If `repeated_assistant_turns` is non-zero only in
short tool-call turns (`_long` = 0) read the examples before calling it a loop; base's own count (same method on the
base check run) is the reference for that judgement. Unit tests: `tests/test_m2_turbo_lossless.py`.

### Tail check-run results (2026-10-09, kaggle-ops request `tail-eval-1009b`, workflow run 37994047836)

Both tail variants PASS (all markers, 0 tracebacks, 25/25 gave_up, 2 retractions, 0 repeated assistant turns, history_cache
zero-counters 0, timeout_fix / level_memory errors 0). One public-25 run cannot rank candidates; per-game figures not checked.
- **turbo-tail v1**: 772.4 gen tok/s (+34% vs 577.8; target 664.5), 1779 completion tokens/request (+3.6% vs 1718), public-25 31 levels / mean_score 6.62, payload_delta 1005 vs payload_full 738 (only ~1.4x: "delta >> full" weakly met), KV peak 0.93, accept median 3.02, prefix cache 92.6%.

## Turbo-lossless kernel: turbo without the lossy acceptance (2026-10-09, built, not yet run)

Kernel **`calamitychasm/arc3-m2-turbo-lossless`** (`kaggle_submission_m2_turbo_lossless/notebook/`), built by
`python scripts/_build_m2_level_memory_kernel.py --turbo-lossless` = turbo's components (`--history-cache --timeout-fix
--reap --arc-hotmap --streams 14 --check-all25`, plus the input resolver) with **`--spec-accept` omitted**: upstream
`SPEC_ACCEPT_SINGLE` / `SPEC_ACCEPT_ACC` stay 1.0 and no `SPEC_ACCEPT` marker is printed. Against turbo only two cells
differ (the header blurb and the launcher's acceptance values plus its `SPEC_ACCEPT` print/asserts); tests pin both.

**Why.** Acceptance 0.5 is the one lossy, behaviour-changing piece of turbo, and it is also the largest throughput step
in JustAdev742's gates (733.2 -> 819.3 tok/s). Nobody has verified a hidden-set gain from any serving speedup, and
their hidden draw of the full bundle (submission 56980485) was pending at 2026-10-09 03:53 UTC. REAP-448 + 14 streams
alone was +14% (733.2 vs 641.9 tok/s) with public-25 full-length 49.45 vs Franzen 45.6-47.5. Turbo-lossless keeps the
lossless-by-construction part (plus the ARC FR-Spec map and REAP, whose own quality shift is confined to image turns).
If their draw lands <= ~28 (their D' copy at 10 streams drew 28.87), treat acceptance 0.5 as suspect and prefer this
kernel.

### Check run: pass / kill criteria (25 games x 25 min; one run)

- **Markers**: as turbo minus `SPEC_ACCEPT 0.5` (`python scripts/_build_m2_level_memory_kernel.py --turbo-lossless` prints
  the list): `LEVEL_MEMORY installed`, `priority gate active: 14 concurrent streams`, `harness patch applied
  successfully`, `INPUT_RESOLVED`, `HISTORY_CACHE installed`, `TIMEOUT_FIX installed`, `REAP448 applied kept=448`,
  `ARC_HOTMAP sha=ec15348b11863ec6fb94b655e4f9ddc4c0ce457fb11f77807b0c5c2d391da70f`; no `REAP448 NOT CONFIRMED`.
  The launcher prints no acceptance line in this build; the serve.log `server_args` (if it echoes the thresholds)
  should show 1.0.
- **Primary**: generated tok/s **>= 640** (+10.8% over base 577.8; JustAdev742 measured +14% for REAP-448 + 14 streams on
  this stack, 641.9 -> 733.2 at their base, so ~659 is the central expectation here). **Kill below 606.7** (+5%).
  Between 606.7 and 640: do not submit on the check alone; read the guards and decide.
- **Guards, as turbo**: retraction lines <= 2x base; no OOM; KV pool `#tokens` ~1.45-1.5M and its peak recorded;
  prefix reuse >= 90%; median `#running-req` >= 12; MTP accept length ~2.6 (**no** rise to 3.0+: that would mean
  acceptance is not 1.0); `json_model_override_args` 448 experts and `speculative_draft_model_override_args` `{}`;
  draft load ~3.79 GB.
- **Output length and loops** (sanity for a lossless build; should be trivially met): mean completion tokens per
  request within +-10% of base; 0 exact repeated assistant turns (`repeated_assistant_turns_long`).
- **Counters**: `history_cache_summary.json` `errors`, `write_fallbacks`, `payload_plain`, `view_misses`, `loads_stale`
  all 0, `payload_delta` >> `payload_full`; `level_memory_summary.json` `errors == 0`; `timeout_fix_summary.json`
  `errors == 0`.
- No traceback outside serving teardown; every game won/gave_up/cancelled.

**Score pre-registration.** Elasticity 0.6-0.8 x tokens on +14% -> roughly +8-11% over the incumbent's 30.47 (~33-34),
quality risk only from REAP on image turns. One draw is a catastrophe check only (< 22 fails); adoption needs n >= 3
draws interleaved in time with the incumbent (and turbo, if both pass); final-selection rules in CLAUDE.md apply.

### Commands

Build (idempotent; tests pin the committed notebook): `python scripts/_build_m2_level_memory_kernel.py --turbo-lossless`.

Dev box (push right after turbo's check run, when a GPU slot is free):
```
set PYTHONUTF8=1
echo kaggle_submission_m2_turbo_lossless/notebook>> logs\kaggle_push_queue.txt
echo calamitychasm/arc3-m2-turbo-lossless>> logs\kaggle_watch_kernels.txt
python scripts\kaggle_push_queue.py
kaggle kernels output calamitychasm/arc3-m2-turbo-lossless -p logs\m2_speed\turbo-lossless
python scripts\m2_speed_report.py base turbo turbo-lossless
python scripts\kaggle_submit_when_ready.py --kernel calamitychasm/arc3-m2-turbo-lossless --version 1 ^
    --message "m2 turbo-lossless draw 1" --marker "LEVEL_MEMORY installed" --marker "priority gate active: 14 concurrent streams" ^
    --marker "harness patch applied successfully" --marker "INPUT_RESOLVED" --marker "HISTORY_CACHE installed" ^
    --marker "TIMEOUT_FIX installed" --marker "REAP448 applied kept=448" ^
    --marker "ARC_HOTMAP sha=ec15348b11863ec6fb94b655e4f9ddc4c0ce457fb11f77807b0c5c2d391da70f" ^
    --counters level_memory_summary.json --counters history_cache_summary.json --counters timeout_fix_summary.json
```
kaggle-ops request examples are in CLAUDE.md ("Operating Kaggle from the cloud").

### Tail check-run result (2026-10-09, request `tail-eval-1009b`, workflow run 37994047836)

**turbo-lossless-tail v1: PASS.** All markers, 0 tracebacks, 25/25 gave_up, 657.4 gen tok/s (+13.8%; target 640, thin margin),
2 retractions, 1579 tokens/request (-8.1%, within +-10%), 0 repeated turns, history_cache zero-counters 0 (delta 996 vs full 714),
errors 0, public-25 36 levels / mean_score 8.17, KV peak 0.92, accept median 2.66. One public-25 run cannot rank candidates.

## Priority-gate tail variant combinable with turbo (2026-10-09, built, not yet run)

`--prio-tail` adds `ARC3_PRIORITY_HUMAN_ACTIONS=60` and last-level B = 5 (runtime, install cell) to any build:
`arc3-m2-turbo-tail`, `arc3-m2-turbo-lossless-tail`, `arc3-m2-lm-tail`. Scheduling only, no throughput effect; the
check run passes on the base kernel's criteria plus the `PRIORITY_TAIL installed` marker. Why and the replay
(+0.4..+3.9% RHAE simulated, median ~+2.2%): `experiments/stage7_milestone2_improvements.md` section 4.

## Pre-flight review of turbo-tail / turbo-lossless-tail (2026-10-09)

Cell-by-cell diff of `arc3-m2-turbo-tail` against the incumbent and against JustAdev742's own builder output
(`scripts/build_franzen_nb.py --input-fallback --reap-kept ... --cfg MAXREQ=14 --cfg CUDAGRAPH_MAXBS=14
--cfg MAMBA_CACHE=84 --cfg SPEC_ACCEPT_*=0.5 --env ARC3_MAX_ACTIVE_STREAMS=14 --hot-tokens ... --full25 25
--patch ours-sandbox-timeout-keeps-work.patch`, run against da-fr's repo): the launcher cell matches theirs line for
line apart from comments and our marker prints (same REAP apply point, same override args with the draft kept at
512, same hot-map sha `ec15348b...`, same streams/graph/Mamba values, MEMFRAC 0.96 so the freed weights go to the
KV pool: their measured 1.01M -> 1.48M tokens). Every code cell compiles under Python 3.12 (the pinned image), and
the history-cache / level-memory / timeout-fix / turbo test files pass under 3.12 as well as 3.13; the built
install + history-cache + timeout-fix cells of the tail kernel compose on the real harness (gate reports 14 streams,
`_priority_human_actions()` 60, B 8/7/5/5).

Two changes made:
- **REAP marker re-checked after the run.** The launcher's serve.log check runs right after the health loop, and
  the loop releases the benchmark at 12 min after notebook start (a normal boot is ~9 min). A server still loading
  then would have left a healthy check run without `REAP448 applied kept=448` and failed the gate. The run cell now
  re-reads serve.log after `bm.run` (`REAP448 applied kept=448 (post-run) | ...`; never raises).
- **`docker_image_pinning_type: "original"`** in every variant kernel's metadata (incumbent v1 untouched), as in
  JustAdev742's lesson 0029: Kaggle's latest image moved to Python 3.13 while the wheelhouse is cp312 only.

## FP4 KV cache on this stack: CLOSED, not supported (2026-10-09, source audit, nothing built)

Question: can turbo-tail (FP8 KV, 1,477,888-token pool, 14 streams, KV peak 0.93) swap to a 4-bit KV cache and run
16-18 streams? Audited Pennyroyal at the pin dfranzen's bundle builds (`d00d88efc8d6`,
`jpezzulli/sglang-rtxpro6000`; paths below are under `python/sglang/srt/`). **Answer: no.** The flag parses and the
pool allocates, but the attention kernels this model uses cannot read an FP4 pool.

- **Flag values.** `server_args.py` L722-746 lists `nvfp4`, `fp4_mx_block16`, `fp4_e2m1`. `fp4_e2m1` raises
  "deprecated, use fp4_mx_block16" (`layers/quantization/fp4_kv_cache_quant_method.py` L814-818). `nvfp4` is
  SM100/SM120 only (`server_args.py` L6635-6641; the RTX Pro 6000 is SM120, so the check passes).
- **The full-attention layers always run `QwenSparseAttnBackend`.** For a hybrid-GDN model with a QSA config,
  `layers/attention/attention_registry.py` L459-467 (`is_qwen_qsa`, L461) replaces the full-attention backend with QSA regardless of
  `--attention-backend`. The FP4 access rules
  (`fp4_kv_cache_quant_method.py` L714-719, L777-790) cover only `flashinfer` prefill + `trtllm_mha` decode (nvfp4) and
  `triton/torch_native/flex_attention/trtllm_mha/fa4` (fp4_mx_block16). Only `flashinfer_backend.py` L324 and
  `trtllm_mha_backend.py` L143 consult them; `qwen_sparse_attn_backend.py` never does (zero `fp4` references in it
  or in `layers/attention/qsa/`).
- **What QSA would actually read.** Prefill (`qwen_sparse_attn_backend.py` L1633, L1664) and decode (L1877 ->
  `_forward_trtllm_sparse`, `bmm2_scale=1.0` at L1845) both call `pool.get_key_buffer(layer)` /
  `get_value_buffer(layer)` on the **whole layer** and hand it to Triton kernels that upcast elementwise
  (`qsa/sparse_attn.py`) or to trtllm-gen with no KV scale.
  - `nvfp4`: the buffer comes back as packed `uint8` `[tokens, 2, 128]` with separate FP8 block scales the QSA path
    never reads (no PLAIN access rule, so `memory_pool.py` L2440-2453 returns the raw buffer). Head-dim mismatch or
    silent garbage.
  - `fp4_mx_block16`: it declares PLAIN-with-storage, so `get_key_buffer` **dequantizes the entire layer pool on
    every call** (`memory_pool.py` L2443-2450 -> `kvfp4_tensor.py` `batched_dequantize`, which materialises `uint8`,
    `int64` (L137 `magnitude_idx.long()`) and `float32` copies of every element). At ~2.6M FP4 tokens that is 1.35G
    elements per K per layer, i.e. >10 GB of int64 alone, against ~4 GB free at mem 0.96: **OOM on the first
    forward**. Even if memory allowed it, 12 layers x 2 x a full-pool dequant per forward would cost more than the
    current ~50 ms decode step.
- **Why FP8 works when the old note said "QSA requires a BF16 main KV cache".** That note (CLAUDE.md "Closed lines",
  2026-09) was the **vLLM** Duck/NVFP4 stack. Pennyroyal's QSA path has no dtype gate. FP8 e4m3 works because it is
  a plain elementwise dtype with scale 1.0: the Triton kernels cast on load, and trtllm-gen decode takes FP8 KV
  natively. The indexer's own compressed-key cache is always BF16 (`mem_cache/qsa_kv_pool.py` L36), independent of
  `--kv-cache-dtype`. No other smaller dtype exists: `fp8_e5m2` is the same 1 B/elem, and `mxfp8` is larger.
- **sirikilohit** (`ext/LohitSiriki_arc-agi-3-milestone2-solution`, README L17, WRITEUP L90-100,
  `run/sglang_boot_result.json`): **FP8 e4m3** KV, 1,004,288 tokens, **16 streams**, mem 0.97, `--hicache-size 48`,
  and **context 69,632** (trimmed at 57,344 back to 45,056). His 16 streams come from roughly half our per-stream
  context, not from a smaller KV dtype.

**KV arithmetic (for the record).** Geometry from his boot log: FP8 K 5.75 GiB / 1,004,288 tokens = 6,144 B
= 12 full-attention layers x 2 KV heads x 256 dim; K+V = 12,288 B/token (+1,024 B for the MTP draft layer). FP4
block-16 = 0.5625 B/elem -> 6,912 B/token, 1.78x the tokens in the same bytes: turbo's 1.48M would become ~2.63M
(+1.15M). At turbo-tail's measured peak (0.93 x 1.48M / 14 streams = ~98K tokens/stream) that would hold ~25
streams. The arithmetic is moot without FP4 reads in the QSA kernels; a port of them is not a check-run-sized change.

**Next best route to more streams: the host KV tier on top of turbo-tail.** turbo-tail + `--enable-hierarchical-cache
--hicache-size 32` (host budget unchanged by REAP: ~23 GiB left after the tier, see Round 3) + 16 streams (Mamba
cache 96 = 6/stream as now). Demand ~16 x 98K = 1.57M vs the 1.48M device pool (6% over) is the regime the ~1.75M-token
host tier exists to absorb, and it is exactly sirikilohit's working combination (16 streams + host tier + FP8). Risk:
the host tier has never run on our stack (the incumbent-based `m96s12hic` is built, never run), and the Mamba-retention
patch skips write-through on non-branch chunks. Built as `arc3-m2-turbo-tail-hic16` (next section).

## Turbo-tail-hic16: host KV tier + 16 streams on arm A (built 2026-10-09; check run read 2026-10-10: near-pass)

`python scripts/_build_m2_level_memory_kernel.py --turbo --prio-tail --hicache-gb` -> `calamitychasm/arc3-m2-turbo-tail-hic16`
(`kaggle_submission_m2_turbo_tail_hic16/notebook/`; metadata = the incumbent's except identity, plus
`docker_image_pinning_type: "original"`). `--hicache-gb [N]` (default 32, 8..40; the cap is the speed builder's
`HICACHE_MAX_GB`) adds the variant token `hic<N>`; with a turbo preset it replaces the preset's 14 streams (default 16;
15-16 streams are allowed only with the tier and `--reap`) and the slug gets `-hic<streams>` (`-hic<N>gb-s<streams>` for a
size other than 32). The kernel differs from `arc3-m2-turbo-tail` in exactly four cells (tested):

| cell | change |
|---|---|
| header | blurbs for `s16` and `hic32` |
| setup | `ARC3_MAX_ACTIVE_STREAMS` 14 -> 16 |
| launcher | `MAXREQ` / `CUDAGRAPH_MAXBS` 16, `MAMBA_CACHE` 96 (6/stream as now; 96 // 5 >= 16); `--enable-hierarchical-cache --hicache-size 32 --hicache-write-policy write_through --hicache-io-backend kernel` (the spd *hic kernels' `hicache_line`, reused); request print `HICACHE requested --hicache-size 32 GB`; after the health loop, reads serve.log for SGLang's `Tree cache initialized: ... hicache_attached=<bool>` (`mem_cache/registry.py` L237-255) and prints `HICACHE_TIER attached hicache_attached=True | ...` only if SGLang said True (`HICACHE_TIER NOT ATTACHED` / `NOT CONFIRMED` otherwise; never raises) |
| run | the same serve.log check re-run after `bm.run` (`... (post-run)`), as for REAP (the launcher's check can run before the boot finishes) |

Turbo's flags SGLang rejects with hicache (`--disable-radix-cache`, `--enable-int8-mamba-checkpoint`) are absent
(tested). The REAP patch, acceptance 0.5, hot map, priority-gate fix, history cache and timeout fix are unchanged.

**Markers (submit gate):** turbo-tail's markers with `priority gate active: 16 concurrent streams` in place of 14, plus
`HICACHE_TIER attached hicache_attached=True` and `STREAMS max_running_requests=16 cuda_graph_bs=` (the server, not only
the harness gate, got 16). Counters as turbo-tail.

**Pass (all of):**
- generated tok/s **>= 834.2** (+8% over turbo-tail's 772.4; the speed digest);
- `HICACHE_TIER attached hicache_attached=True` (post-run line accepted);
- `MemAvailable` >= 10 GiB in **every** `[sys]` census line (RAM used <= total - 10);
- retractions <= 2x turbo-tail's (2 -> <= 4);
- output tokens/request within +-10% of turbo-tail's 1779 (1601..1957) and **0** exact repeated assistant turns
  (`request_log` digest);
- turbo-tail's other guards: no traceback outside serving teardown, no OOM (GPU or `Not enough host memory`), every game
  won/gave_up/cancelled, history_cache zero-counters 0, level_memory / timeout_fix errors 0, all markers present; KV pool
  `#tokens` and peak recorded (a lower device pool than 1.48M from the larger Mamba cache is expected; a peak pinned at 1.0
  with retractions means the tier is not absorbing the overflow).

**Kill:** gen tok/s < 772.4 (no better than turbo-tail), or any pass guard tripped. Between 772.4 and 834.2: not adopted,
no slots; read decode tok/s at 15-16 running requests in serve.log before deciding whether 15 streams is worth a run.

**Result (check run COMPLETE, read via kaggle-ops `daily-1010`, 2026-10-10): near-pass -- between kill and pass.**

| criterion | bar | measured | |
|---|---|---|---|
| gen tok/s | >= 834.2 (kill < 772.4) | **831.6** (+7.7% vs 772.4) | 2.6 short, not a pass |
| `hicache_attached=True` | yes | yes (post-run line; `UnifiedRadixCache`, `hybrid_ssm=True`) | ok |
| `MemAvailable` >= 10 GiB in every `[sys] RAM` line | yes | 15 lines, min 29.9 GiB / median 32.1 / last 31.4 (of 176.9) | ok |
| retractions | <= 4 | 2 | ok |
| tokens/request | 1601..1957 | 1736 (-2.4%) | ok |
| repeated assistant turns | 0 | 0 (760 turns) | ok |
| tracebacks / `Not enough host memory` | 0 | 0 / absent | ok |
| games | all won/gave_up/cancelled | 25/25 gave_up | ok |
| histcache zero-counters, timeout_fix / level_memory errors | 0 | all 0 | ok |
| public-25 | recorded | 43 levels, mean_score 12.83, 1429 actions, 26m25s (turbo-tail: 31 / 6.62) | |

Server: KV pool 1,427,968 tokens, peak 0.99, Mamba peak 0.67, accept median 3.06, prefix cache 92.9%, decode median 986 /
p90 1252, running median 16. Decode tok/s by running requests: 12 -> 1106, 13 -> 983, 14 -> 1051, 15 -> 1012, 16 -> 988, so
there is no visible per-request gain from 15-16 running requests over 12-14 in this single run. A peak of 0.99 with only 2
retractions suggests the tier absorbed the overflow. By the rule above: not adopted, no slots. Not a KILL either (the tier
attached and nothing broke); the missing `MemAvailable` check and a +7.7% vs +8% bar miss are within one run's noise.
**OOM lines (kaggle-ops `hic16-oom-2`, 2026-10-10).** The digest's `mem_problems.oom = 13` counts `\bOOM\b` over all top-level
`*.log` files. All 13 are one benign SGLang advisory in serve.log, 22:47:45-22:48:58: "Triton kernel '_qsa_graph_layout_kernel' /
'alloc_extend_kernel' / 'assign_req_to_token_pool' device-loaded after serving started (free device mem: 0.96, then 0.84 GiB).
Pre-load it during engine init to avoid CUDA OOM." No allocation failure, no OOM-driven retraction, no process kill
(`killed` 0, `Not enough host memory` 0, 0 tracebacks); the single `retract` hit is the `server_args` echo. The only
signal is thin device headroom (0.84 GiB free at lazy kernel load, cf. the mem 0.98 OOM in section 14). Tooling: the
digest's `grep` reports present/absent only; the op's new `show_lines: N` prints the matching lines from all `*.log`.

**Verdict:** near-pass, not adopted under the pre-registered rule; candidate to replace arm A only if arm A's hidden
draws hold up and a second check run confirms >= +8%.

## MTP drafter fine-tune (feasibility, 2026-10-09; desk study, nothing built or run)

**Question.** Can a fine-tuned MTP draft head raise the lossless accept length of arm B (median 2.66, 657.4 gen tok/s)
toward arm A's lossy 3.02 (772.4), and should we build one before 2026-11-02?

### 1. JustAdev742's plan (repo `JustAdev742/Arc-Agi-3-Kaggle-comp`, Apache-2.0, branch `claude/admiring-ride-b4dq3i` @ fe2ad06;
`docs/research/beat-tufa/mtp-drafter-finetune.md` sections 0-11, `docs/research_log.md` 2026-10-08 19:05 .. 10-09 00:18)

- **What is trained.** The MTP head's dense weights only: 88.9M of the 90.6M BF16 `mtp.*` tensors (attention, gated
  residuals, input fusion `fc_embedding`/`fc_hidden`, router, shared expert; the QSA indexer frozen). The 512 routed
  experts stay albucino's INT4 RTN g32 file byte-identical (dequantized in the trainer, so the dense weights learn around
  the experts' ~10% RTN error). `embed_tokens`/`lm_head` are the target's (shared at load). **No new FR-Spec map is
  trained**: the draft is trained *for* the map it will be served with (their ARC map, sha `ec15348b...` = our
  `ARC_HOTMAP`). Starting point = albucino as shipped (its dense tensors are bit-identical to Intel's BF16 original MTP).
- **Data.** Their own Save & Run request logs (`<game>_p0_requests.jsonl`: exact messages, base64 images, tools,
  `chat_template_kwargs`) from loop-free **lossless** runs (exp-073: 25 games x 121 min, 1.48 GB of logs), mounted as
  `kernel_sources`. The logs supply contexts only: an env-gated Pennyroyal patch on `qwen4_exp.py` dumps the target's
  final hyper-connection state `H` (10,240 values/token, FP8 + per-row scale) while a no-spec, no-radix, no-CUDA-graph
  server replays maximal snapshots with `max_tokens 1`. Labels are the target's own distribution (forward KL over the
  65,536-row hot vocab, recomputed from `H` through the target's mixer + `lm_head`), chained 3-step training-time test,
  step weights 0.51/0.31/0.18, AdamW 5e-5, 2 epochs over ~2-3M loss rows; 11 public games held out. Loop-turn filter.
- **Compute.** One Kaggle RTX session ("session A", D' + 11 step cells, A0-A11): two server boots, dump ~15-25 min,
  replica check (GO/NO-GO against SGLang's own accept length on 16 probe requests), train 30-75 min, export a 4.1 GB
  albucino-format draft to `/kaggle/working`. Estimate 1.6-2.8 GPU-h (7 h hard cap); plan total ~3.5 GPU-h incl. a
  probe gate (0.45), a 25x25 production gate (0.7) and one retry (1.0). ~2 engineering days, already spent.
- **Expected gain.** Their lossless accept 2.78 -> 2.92-3.05 (+5-10%), "close to today's 0.5/0.5 speed (3.10) without
  its distortion"; perfect-draft lossless ceiling 3.51 (from probe logprobs). Literature: AngelSpec 2.54 -> 2.90 at
  T 0.9 (one shared MTP block, D = 3), FastMTP 1.21x -> 1.81x from a weaker base. Their own warning: under 0.5/0.5 a
  sharper draft pushes decoding toward greedy (loop risk); ship lossless first.
- **Status.** All code built and CPU-tested (`scripts/{sglang_hc_dump_patch,hc_dump_driver,mtp_replica,mtp_train,
  mtp_write_draft,mtp_probe_dump,mtp_session_a,build_mtp_session}.py`, ~5.9k lines, 150 tests). **Nothing has run on a
  GPU; no accept-length result exists.** Session A is first in line after their Sat 2026-10-10 00:00 UTC quota reset
  (their week's 30 h quota is spent). Open GPU-only risks they list: kernel-output mount paths, dump speed, the hook
  under the overlap scheduler, `/tmp`/`/dev/shm` capacity, training speed, the replica matching SGLang's fused kernels.
- **Licence.** Their code: Apache-2.0 (portable with NOTICE, as for turbo). The trained weights are a derivative of
  Qwen's MTP (Qwen Community License 1.0), the same status as albucino and Intel's checkpoint.

### 2. Could we do the same?

- **Data: yes, already on Kaggle.** Our check runs set `bm.solver.save_request_logs = True` (off only when
  `TRUE_SUBMISSION`), and the harness patch writes the same record format as theirs (`messages`, `tools`,
  `chat_template_kwargs`, `usage`, `finish_reason`; `request_log_metrics` in `scripts/m2_speed_report.py` and the
  `request_log` digest in `scripts/kaggle_ops.py` read the `"event": "response"` records). Best source:
  `arc3-m2-turbo-lossless-tail` v1 (lossless, REAP-448 target, ARC map, 25 games x 25 min, 0 repeated turns); then the
  lossless base speed runs and the incumbent's check runs. Because the target's `H` and logits are recomputed at dump
  time, logs from an unpruned or generic-map run are still valid contexts. Avoid turbo/turbo-tail logs (lossy 0.5).
  Supply is roughly 1-2M output tokens per 25x25 run (not counted here), against their 2-3M-row target: two or three of
  our outputs, or one full-length run, cover it.
- **Training notebook.** A competition-attached notebook (needed for the RTX Pro 6000; RTX sessions have no internet)
  with the Pennyroyal dataset, the Intel and albucino models and our check-run kernel as `kernel_sources`. Internet is
  not needed (the wheelhouse is a dataset). Output goes to `/kaggle/working` (19.5 GB; the draft is 4.1 GB).
- **Format and loading: yes.** Export keeps albucino's 12 files (only `mtp-dense.safetensors` rewritten, 181 MB BF16;
  config, index, 1.4 GB INT4 expert file byte-identical), so our launcher's `prepare_draft_view` checks pass unchanged.
  Our notebooks already resolve `DRAFT_MODEL_DIR` through `resolve_input('draft_model', ...)`; a `--draft SRC` builder
  option swaps the model source for the training kernel's output (or, for the scored run, a private Kaggle model made
  from it; scored reruns have no internet, inputs must be attached). Never mount two drafts under one directory.
- **Throughput conversion.** With one chain (topk 1, 4 draft tokens), tok/s = B x L / (t_verify(B) + 3 t_draft(B)); a
  retrained head with the same shapes, experts and map leaves the denominator unchanged, so tok/s scales with L.
  Our numbers: base 577.8 / 2.62 = 220.5 verify-cycles/s (10 streams, ~45 ms/cycle); turbo-lossless-tail
  657.4 / 2.66 = 247.1; turbo-tail 772.4 / 3.02 = 255.8 (14 streams, ~55 ms/cycle). The A/B pair differs only in
  acceptance and moved tok/s 1.27x as much as L in log terms (cycles/s +3.5%, within run-to-run spread at 14 streams);
  use 1.0 as central, 1.27 as an optimistic bound.
- **Estimate (lossless, arm B).** Scaling their +5-10% to our base: L 2.66 -> 2.79-2.93, tok/s 657 -> **690-723**
  (+5-10%; 699-741 at the 1.27 slope). Their ceiling scaled to our base (3.51 x 2.66/2.78 = 3.36) bounds it at ~830.
  Matching arm A's 3.02 (+13.5%) would need the top of their range and then some; plan on landing between B and A.
  Score: at the caller-cited full-budget elasticity ~0.25, +1.3-2.5% (~+0.4-0.8 LB points on ~32); at JustAdev742's
  25-min public-run elasticity 0.6-0.8, +3-8% (~+1-2.5). **Undetectable on the LB** (sd 2.2/draw); it must be adopted
  on mechanism: lossless by construction, measured as tok/s and accept length in one check run.
- **Arm A with a fine-tuned draft** (0.5/0.5): L 3.02 -> ~3.15-3.3, but lossy, and a sharper draft raises snap/loop
  risk (their exp-076: accept 3.70, 69% repeated turns with a mismatched draft). Not recommended without a loop gate
  and >= 2 full runs.
- **Risks.** (1) Replica NO-GO (their trainer mis-matches SGLang's fused kernels) -> no draft, ~1 extra day on their side.
  (2) The gain is small or capacity-bound (one layer). (3) Hidden games differ from the public 25: harmless for quality
  (verification stays lossless), it only shrinks the speed gain; the game holdout measures that. (4) Kaggle GPU quota:
  1.6-2.8 h training + ~1 h check run on top of our A/B/hic16/audit check runs; push only into a free slot (2-session
  limit, Gotchas 2026-10-05). (5) Pairing: a draft is valid for one target + one map (REAP-448 + ARC map = arms A/B;
  not the incumbent v1, which serves the unpruned target with the generic map). (6) Our builder is not D'; porting
  their session builder means re-deriving cells from our notebook, not reusing `build_franzen_nb.py`.

### 3. Recommendation: NO-GO on building our own now; conditional GO on adopting theirs

Their stack is ours: same Pennyroyal wheel, Intel target, albucino draft, REAP-448 kept list and ARC map (our turbo is
their exp-074t), so their session-A output is a drop-in draft for arms A/B. Duplicating an untested 5.9k-line pipeline
before its first GPU run spends our quota on their debugging.

- **Trigger:** their session A reports a replica GO and held-out `accept_expected` up >= ~4% (expected 2026-10-10/11;
  watch their `docs/research_log.md`).
- **Path 1 (they publish the draft as a public Kaggle model/dataset):** add a `--draft SRC` builder option + tests
  (~2-3 agent-h), one 25x25 check run of turbo-lossless-tail with the new draft (~1 GPU-h). Pass: accept median
  >= 2.79 and gen tok/s >= 690, plus all turbo-lossless guards (0 repeated turns, tokens/request within +-10%).
  Earliest check run **2026-10-11/12**.
- **Path 2 (not published):** port their scripts (Apache-2.0, NOTICE) onto our notebook base with our
  turbo-lossless-tail logs as `kernel_sources` (~8-12 agent-h incl. tests), one session A (~2-3 GPU-h, cap 7), then the
  check run (~1 GPU-h); ~4 GPU-h with a retry margin. Earliest check run **~2026-10-13/14**, inside the explore phase.
- **Slots:** none of its own. Per the schedule's rule ("same mechanism with better measured throughput"), a passing
  lossless draft takes over arm B's remaining slots from the next B day; it does not need fresh LB draws to establish
  quality. Worth doing before 11-02 only via one of these two paths; if their replica check is NO-GO or their gain
  is < +4%, drop it (expected value ~+0.5-1 point does not justify a from-scratch build).

## REAP image-turn shift (feasibility, 2026-10-09; desk study, nothing built or run)

Question: arms A/B serve REAP-448 (JustAdev742's kept list, `kaggle_submission_milestone2_fork/turbo/`), whose
fidelity probe found a small logprob shift on image turns. Can we shrink it without losing the KV/throughput gain,
and is it worth doing before 11-02? Sources: JustAdev742 repo (`claude/admiring-ride-b4dq3i` @ fe2ad06):
`docs/research/beat-tufa/reap-at-load.md`, `fidelity-probe.md`, `serving.md` arm 3, `intel-oct8.md` item 3,
`docs/research_log.md` 2026-10-07 23:24 .. 10-09 03:53; `scripts/reap_kept_experts.py`.

### 1. How the REAP-448 kept list was chosen

- **Nobody on their side ran REAP.** The list is *recovered*, not computed: `scripts/reap_kept_experts.py` reads only
  the router tensors (HTTP range reads) of the public build `lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448@8d565c90`
  and of Intel's W4A16 checkpoint, and maps every pruned router row to the unique bit-identical Intel row
  (48/48 layers, 448 distinct ids, MTP router identical). We vendored that JSON unchanged.
- **Saliency:** standard REAP (router-weighted expert activation norm, per expert), computed and pruned **per layer**
  (64 per layer in every layer; 92.65% of REAP mass retained, mean 7.35% removed, worst layer 15 at 9.11%; 28 dead
  experts pruned first, 5 "super experts" force-kept). Per-expert scores are **not published** -- only the list.
- **Calibration (model card, not reproducible by them):** 16.49 M tokens of the model's own agentic production traffic
  **including the multimodal path (626 images)**. So it is not strictly text-only, as their research-log entries say;
  it is image-poor and contains no ARC boards. On NVIDIA's NVFP4 quant (same experts/router as Intel W4A16).
- Calibration choice moves the list a lot: sh0wie's public REAP-384 (686 K tokens of agentic coding) keeps a median
  15.5 per layer (3-36) of lee-chang's 64 pruned experts.

### 2. Is the shift from pruning vision-relevant experts, and can it be fixed?

- **Evidence it is image-specific (3-way probe, 154 replayed ARC requests, greedy, 10 streams, lossless):** headline
  |dlp| 0.0480 vs cross-run floor 0.0376 (ratio 1.28); fresh-frame (user+image) turns 0.0536 vs 0.0372, text/tool
  turns 0.0383 vs 0.0383 (no shift). Largest by game vc33 0.094/0.041, ft09 0.075/0.042, tn36 0.045/0.017; lp85, tu93
  none. Accept length unchanged (2.77 vs 2.78). Image tokens go through the same 48 MoE layers, so an image-poor
  calibration under-scoring experts that ARC board tokens route to is the plausible mechanism; it is **not shown**
  (nobody has router statistics on ARC traffic).
- **They discuss the fix but have not tried it.** 2026-10-08 16:03: "A REAP calibrated on our own logged ARC traffic
  (image turns included) is the obvious way to shrink it; that is a follow-up, not a blocker." REAP-384 entry
  (19:23): "Revisit with an ARC-calibrated REAP." `intel-oct8.md` item 3 plans it (re-score on logged requests incl.
  images, then 384 + 18-20 streams; ~2-3 GPU-h), behind MTP session A. `serving.md` arm 3(b) sketches the method:
  accumulate router-weighted expert-output norms while replaying Franzen's request logs; cheapest first step is
  route-only statistics (how much top-10 routed weight the 64 dropped experts carry per layer).
- **Best shape for us: a swap list at K = 448**, not a re-prune. Measure per-layer routed weight of all 512 experts on
  our ARC traffic (unpruned server); in each layer swap the least-used kept experts for the most-used pruned ones
  where the margin is clear. Same count => same 7.31 GiB, same 14 streams, same throughput, and the loader patch is
  unchanged (it checks Intel's router sha256 per layer and `num_experts == len(kept)`, not which ids).
- **REAP-480 is worse value.** REAP-384 (128 pruned) gave 2.2x REAP-448's excess |dlp| and spread it to text turns,
  so ~half the shift at 32 pruned is plausible -- but (a) the 32 to restore still need a ranking nobody published
  (an arbitrary half just halves the shift), i.e. the same calibration session as the swap; (b) it frees only
  3.65 GiB: measured REAP-448+14 pool 1.478 M -> ~1.18 M, vs 14-stream peak need ~1.35 M (114%), so it supports ~12
  streams (~1.24 M vs ~1.16 M need). Their decode plateau (flat from ~10 running; REAP-14's gain was keeping ~12.5
  running) suggests ~+8-10% output tokens instead of +14%. It trades throughput for fidelity; the swap does not.

### 3. What our own list would take

| Item | Need | Effort |
|---|---|---|
| Data | ARC prompts with images: our check runs' `*_requests.jsonl` (Franzen harness logs every request with inline base64 PNGs; the tail runs of 2026-10-09 have them, artifact retention 7 days). Sample with a port of their `fidelity_sample.py` (stdlib, Apache-2.0). Prompts made by a REAP server are fine: we read the *unpruned* model's routing on them. | 1-2 agent-h |
| Router-stat patch | Anchored edit in the installed Pennyroyal MoE block (as `sglang_reap_patch.py` / `sglang_hc_dump_patch.py` do): per layer x 512 experts, sum of renormalised top-10 gate weight and counts, split image vs text tokens (image-token mask from input ids, as the hc-dump patch does); dump at exit. Router-weighted frequency is a proxy: the fused Marlin MoE does not expose per-expert output norms that true REAP needs. Radix cache off, max_tokens 1 (prefill only). | 4-6 agent-h incl. tests (~250 lines) |
| Calibration notebook + list builder | Replay driver cell, swap-list builder, `.meta.json` (Intel router sha256 unchanged), builder `--reap-kept FILE` option, re-pinned tests, marker `REAP448 applied kept=448` unchanged | 3-4 agent-h |
| Verification | Port of their fidelity probe/compare (stdlib); their probe dataset is private, so we need our own base floor: base + old REAP + new REAP arms (~25 min each) | 2-3 agent-h, ~1.3 GPU-h |
| GPU | Stats session ~0.5-1 GPU-h (9 min start, ~5-10 M prompt tokens at ~10 k tok/s); probe ~1.3; one 25x25 check run of the arm with the new list ~0.7 | **~3-3.5 GPU-h**, ~4.5 with a retry |
| Total | | **~11-16 agent-h**; earliest check run ~2026-10-13/14 (2-session limit, arms A/B and the speed queue ahead) |

Risk of breaking things: low if K stays 448 (loader, markers, gate unchanged); the stats patch lives only in a
calibration notebook. Real risks: a swap that introduces a **text-turn** shift where REAP-448 has none (the probe must
show text turns at the floor), and calibrating on the 25 public games (hidden games share the 64x64 16-colour board
format, so image-token routing should transfer, but this is untested).

### 4. Is the image-turn shift score-relevant?

No evidence either way, and no evidence of harm:
- vc33 -58.9 / tn36 -49.9 / tr87 -36.0 came from one run (exp-073, REAP lossless, 49.45). The next REAP run
  (exp-073b, + acceptance 0.5) **won all three** (vc33 21 -> 100, tn36 4 -> 100), and exp-075 (same config) then
  dropped tn36 100 -> 3.6. Per-game SD between runs is 22-23 points; one 25-game run's mean SD ~4.5.
- REAP public-25 runs: 49.45 (lossless), 56.00 / 42.89 (acceptance 0.5), against Franzen v3 passes 45.6-47.5 (not
  independent draws). No equal-stream test (REAP-10 vs base-10) was ever run, so quality and throughput are confounded.
- Our tail check runs (A 31 levels / 6.62, B 36 / 8.17, 25 min) cannot resolve anything.
- Size: excess |dlp| ~0.016 nats on image turns, below the sampling noise of T=0.7; it does flip confident tokens
  (0/145 REAP divergences were near-ties), so it is a real model change, of unknown consequence.
- Hidden set: JustAdev742's first REAP draw (56980485) was pending at 03:53 UTC; our A/B have none. With per-draw sd
  2.2-3.9, a sub-point effect (the most a fix to a niche shift plausibly buys) needs > 50 draws per arm: **not
  measurable before 11-02**; any decision rests on mechanism.

### 5. Recommendation: NO-GO now (conditional)

- The benefit is unmeasured and likely small; the cost (~11-16 agent-h, ~3.5-4.5 GPU-h, earliest result ~10-13/14)
  competes with the queued check runs for the 2 GPU slots and weekly quota, and its effect can never be confirmed on
  the LB in time. REAP-480 is dominated by the K=448 swap.
- JustAdev742 has the same fix on their plan (`intel-oct8.md` item 3); their list would be a drop-in JSON for our
  patch (same Intel router fingerprints).
- **Revisit (GO) if:** (a) JustAdev742 publishes an ARC-calibrated list -> adopt it: swap the JSON + meta, re-pin tests
  (~2 agent-h), probe base/old/new (~1.3 GPU-h) and one check run; or (b) after >= 3 hidden draws, arm B (lossless
  REAP) sits > 1 sd (2.2) below the incumbent while its throughput guards pass -- REAP quality then becomes the leading
  suspect and the swap list is the targeted fix; or (c) GPU quota is idle late in a week with nothing else queued.

## Per-game check-run levels (2026-10-10)

Read with the `per_game` digest in `scripts/kaggle_ops.py` `kernel_output` (request `pergame-1`, workflow run 38024038318; source: `benchmark.json` `game_runs`, `levels_completed` / `final_score`). Check runs of 2026-10-09, 25 public games x 25 min, v1 of each kernel. Hard-15 / easy-10 split from JustAdev742 `docs/research/beat-tufa/intel-oct10.md` section 2.3 (hard 15: bp35 cd82 cn04 dc22 g50t ka59 lf52 ls20 m0r0 s5i5 sk48 sp80 su15 tn36 wa30; easy 10: ar25 ft09 lp85 r11l re86 sb26 sc25 tr87 tu93 vc33). Totals match the earlier digests (B 36, A 31). One run per kernel; the SE of one public-25 run is about 2.5 points, so single-game differences are noise-level.

| kernel | total levels | hard-15 levels | easy-10 levels | hard-15 score sum | easy-10 score sum | vc33 | tn36 | tr87 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B (turbo-lossless-tail) | 36 | 15 | 21 | 72.5 | 131.8 | 3 | 1 | 0 |
| A (turbo-tail) | 31 | 10 | 21 | 34.9 | 130.5 | 2 | 0 | 0 |
| hic16 (turbo-tail-hic16) | 43 | 16 | 27 | 104.6 | 216.1 | 3 | 0 | 0 |

Levels per game:

| game | set | B | A | hic16 |
|---|---|---:|---:|---:|
| ar25 | easy | 2 | 2 | 2 |
| bp35 | hard | 1 | 0 | 0 |
| cd82 | hard | 0 | 0 | 0 |
| cn04 | hard | 3 | 1 | 2 |
| dc22 | hard | 1 | 0 | 3 |
| ft09 | easy | 2 | 1 | 2 |
| g50t | hard | 0 | 0 | 0 |
| ka59 | hard | 1 | 1 | 2 |
| lf52 | hard | 1 | 1 | 1 |
| lp85 | easy | 4 | 5 | 6 |
| ls20 | hard | 0 | 1 | 0 |
| m0r0 | hard | 2 | 1 | 2 |
| r11l | easy | 2 | 2 | 2 |
| re86 | easy | 1 | 2 | 2 |
| s5i5 | hard | 1 | 1 | 1 |
| sb26 | easy | 1 | 1 | 7 |
| sc25 | easy | 3 | 3 | 0 |
| sk48 | hard | 1 | 0 | 0 |
| sp80 | hard | 1 | 1 | 3 |
| su15 | hard | 1 | 2 | 1 |
| tn36 | hard | 1 | 0 | 0 |
| tr87 | easy | 0 | 0 | 0 |
| tu93 | easy | 3 | 3 | 3 |
| vc33 | easy | 3 | 2 | 3 |
| wa30 | hard | 1 | 1 | 1 |

B's hard-15 (15) / easy-10 (21) are the control for the reasoning-effort check run (`stage7_milestone2_improvements.md` section 6.3): advance needs total >= 42 and hard-15 >= 21; kill if hard-15 is not above 15.
