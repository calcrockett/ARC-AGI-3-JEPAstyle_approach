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
