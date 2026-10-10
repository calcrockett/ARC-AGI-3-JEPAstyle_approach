# Arm C: `calamitychasm/arc3-jad-exp081` -- a verbatim fork of JustAdev742's exp-081

exp-081 is JustAdev742's notebook `scottmahony/arc3-dprime-r14a05-harness4-percept-full` (team "Jovian Game Studios"),
hidden-set draw **34.04** (submission 57026621, 2026-10-10) and a full-length public-25 run of 50.00 / 113 levels.
It is Franzen's Milestone-2 notebook on the public D' base with REAP-448 at load, 14 streams, MTP acceptance 0.5/0.5,
their ARC FR-Spec map, their sandbox-timeout source patch (ours-01) and the harness bundle 02 / 04 / 03b / 05 / 08b
(budget meter, search helper, win ledger, level scratch memory, perception helpers). Analysis:
`experiments/stage7_milestone2_improvements.md` section 9.

## Source

- Repository: https://github.com/JustAdev742/Arc-Agi-3-Kaggle-comp, commit **`c7a462be8fe3a22910804d4d61614f43a510fdfb`**
  (`c7a462b`, 2026-10-10), licensed under the **Apache License 2.0** (`vendor/LICENSE`, their file).
- `vendor/` holds their files **unmodified**, at their repo-relative paths, so their builder runs as is
  (`vendor/SHA256SUMS` lists every file; all match `c7a462b`):
  - `scripts/build_franzen_nb.py` (their builder), `scripts/franzen_tree.py` (the tree rebuild + apply check),
    `scripts/sglang_reap_patch.py` (REAP at load);
  - `kaggle/franzen/`: Franzen's Milestone-2 notebook (Apache-2.0) + `NOTICE.md`, `bundle/` (Tufa's source bundle
    delta + manifest; see `bundle/NOTICE.md`), the six patches exp-081 applies, `reap448_kept_experts.json` (+ meta),
    `hot_tokens_64k_arc.pt` (+ meta);
  - `kaggle/dprime/`: the public D' notebook (shiiin9 / AFF AI CLUB, Apache-2.0) + `NOTICE.md`.
- Third-party code and weights keep their own licences (Franzen Apache-2.0, Tufa Labs MIT, Pennyroyal, Intel/Albucino
  checkpoints).

## Build

```
python scripts/_build_jad_exp081_kernel.py          # -> notebook/arc3-jad-exp081.ipynb + kernel-metadata.json
```

`scripts/_build_jad_exp081_kernel.py` (ours) runs `vendor/scripts/build_franzen_nb.py` with exp-081's flags, i.e.
their `scripts/build_candidates.sh` exp-084 command without `--draft/--draft-manifest` (their research log
2026-10-10 04:10: exp-084 is "exp-081's exact build ... plus the fine-tuned draft and the 2400 s grace"):

```
--base dprime --full25 121 --input-fallback --wait-inputs 120 --env ARC3_MAX_ACTIVE_STREAMS=14
--cfg MAXREQ=14 --cfg CUDAGRAPH_MAXBS=14 --cfg MAMBA_CACHE=84 --cfg SPEC_ACCEPT_SINGLE=0.5 --cfg SPEC_ACCEPT_ACC=0.5
--reap-kept kaggle/franzen/reap448_kept_experts.json --hot-tokens kaggle/franzen/hot_tokens_64k_arc.pt --fail-fast
--env-add OURS_BUDGET_METER=1 --env-add OURS_WIN_LEDGER=1 --env-add OURS_SEARCH_HELPER=1 --env-add OURS_LEVEL_MEM=1
--env-add OURS_PERCEPTION=1
--patch ours-sandbox-timeout-keeps-work.patch --patch ours-02-budget-meter.patch --patch ours-04-search-helper.patch
--patch ours-03b-win-ledger-on-02-04.patch --patch ours-05-level-mem.patch
--patch ours-08b-perception-on-01-02-04-03b-05.patch --compact
```

The apply check (their patches applied on top of Franzen's patch, on the exact tree the notebook builds) needs
Franzen's repo `da-fr/arc-agi-3-solution` @ `10882e3` (`$FRANZEN_REPO`, default `/home/user/ext/da-fr_arc-agi-3-solution`);
`--no-apply-check` writes the same bytes without it. `tests/test_jad_exp081.py` pins the output.

## Fidelity check (2026-10-10)

- Their `c7a462b` builder reproduces their pinned exp-084 notebook byte for byte in this environment
  (`tests/test_build_candidates.py` sha256 `846a5677...`), so the toolchain is exact.
- exp-081 itself was built on 2026-10-09 ~01:35 UTC with their builder at `48a1e18` (last builder change `8226ad4`).
  Rebuilding exp-081 with that builder and diffing it against our notebook leaves exactly three differences:
  1. cell 0 (markdown): the change list / our `--note`;
  2. cell 11: `os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '2400'` instead of `'900'` (**our deviation**);
  3. cell 32: the `--fail-fast` watchdog limit 55 min (`health_deadline_s=3300`) instead of 35 min (2100), from their
     2026-10-10 builder change `0a595b7`. This watchdog runs only in a Save & Run; a competition rerun never starts it.

## Deviations from exp-081 (all of them)

| what | exp-081 | arm C | why |
|---|---|---|---|
| kernel id / title / slug | `scottmahony/arc3-dprime-r14a05-harness4-percept-full` | `calamitychasm/arc3-jad-exp081` (private) | our account |
| `ARC3_HTTP_RETRY_INITIAL_SECONDS` | 900 (D' default) | **2400** | our standing rule (CLAUDE.md 2026-10-10); their exp-083/084 do the same; inert unless the server boots late |
| fail-fast health limit (Save & Run only) | 35 min | 55 min | inherited from their `c7a462b` builder (slow-storage boots ~27-29 min) |
| cell 0 markdown | their change list | + a note naming the fork | documentation |

Everything else is their bytes: the same 36 cells, the same six patches (sha256-checked in the notebook), the same
REAP list, FR-Spec map, D' module, docker image pin and inputs. Our level memory, history cache and priority tail are
**not** in arm C.

## Inputs (all public, all mounted by our other arms)

`dfranzen/pennyroyal-v253`, `dfranzen/taaf-kaggle-source-bundle-copy` (datasets), the competition
`arc-prize-2026-arc-agi-3`, models `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1` and
`dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1`. No kernel sources and nothing private of theirs
(exp-084's fine-tuned draft, which mounts their private kernel output `scottmahony/arc3-mtp-session-a`, is not used).

## Mount layouts

Their `--input-fallback` resolves all 6 `/kaggle/input` paths in either Kaggle layout
(`/kaggle/input/{datasets/<owner>,competitions,models/...}/<slug>` or `/kaggle/input/<slug>`; prints
`ours: <path> -> <alt>` when it falls back) and `--wait-inputs 120` waits up to 120 s for the mounts
(`ours: inputs mounted after N s`). Our own resolver is not added and the notebook prints no `INPUT_RESOLVED` line.

## Check-run shape

The Save & Run (non-submission mode) plays **all 25 public games at 121 min per game** (`--full25 121`, D''s 14 slots;
~2.2-2.4 GPU-h), exactly the shape of their nine full-length public-25 runs (exp-081: 50.00 / 113 levels, 803 output
tok/s, accept 3.14). It is NOT comparable to our 25 games x 25 min check runs. The competition rerun is D''s own path,
unchanged.
