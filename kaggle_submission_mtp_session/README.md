# `calamitychasm/arc3-mtp-session-a`: lean MTP-draft session A (section 15.6, step L0)

Fine-tunes the dense weights of the MTP draft head on arm C v2's own request logs, following JustAdev742's
session A (their `docs/research/beat-tufa/mtp-drafter-finetune.md`), on the **lean path** of
`experiments/stage7_milestone2_improvements.md` section 15.6: one server start (the training dump), the training
dump (A6-A7), training (A8-A9) and the export (A11). The probe dump and the replica checks (A2-A5, A10) are dropped.

## Layout

- `vendor/`: their files at `ccb7214`, unmodified (`SHA256SUMS`, `LICENSE`; see `NOTICE`).
- `arc3-jad-exp081-v2-request-logs.json`: the training input, arm C v2's 25 request logs (kernel source
  `calamitychasm/arc3-jad-exp081`, version 2, the latest). Names pinned; byte sizes `null` until a kaggle-ops
  listing (`mtps-ls-1`) fills them; response count pinned at 2,952 (checked by our data cell).
- `notebook/`: the built kernel (`arc3-mtp-session-a.ipynb`, `kernel-metadata.json`), pinned by
  `tests/test_mtp_session_lean.py`.

## Build

```
python scripts/_build_mtp_session_kernel.py            # -> notebook/
```

The wrapper runs their `build_mtp_session.build()` in a fresh interpreter and replaces, for that run only:
`LOGS_MANIFEST` (our logs), `STEP_CELLS` (A2, A3, A4, A5, A10 dropped), `_setup_cell` (inputs = the logs only, no
private probe-prompts dataset or replica reference; build record names the wrapper), `worst_case_minutes` (lean
steps: one boot) and `_steps_markdown`. With their `--set`: `session_hours=4.5`, `train.reserve_minutes=20`,
`storage.probe_gb=0.0`. Everything else is their v1 recipe: REAP-448, hold-out of 11 games (ar25 ft09 lp85 r11l re86
sb26 sc25 tn36 tr87 tu93 vc33), repeat / "stuck in a loop" hygiene, 34 GB dump cap, 75 min training cap, the ARC
FR-Spec map, the exporter. Machine `NvidiaRtxPro6000`, their pinned image, private, no internet.

Our four cells (marked `# >>> calamitychasm (lean path)`):
1. after their setup cell: data check (25 names present, waits up to `wait_inputs_s`; response count must be
   2,952, else the session stops: another version of arm C mounted); prints `ours (lean): data loaded N responses`.
2. before A6: the A5 stand-in (records the skipped replica check, lets `require_go` pass).
3. after A9: the drafter gate from A9's held-out evaluation (PASS / BETWEEN / KILL, section 15.6).
4. after A11: manifest sha256, dense sha256, frozen files byte-identical to the stock draft, and the manifest on one
   line (`MTP_DRAFT_MANIFEST_JSON`).

## Next

`scripts/_build_jad_exp081_draft_kernel.py --draft-manifest <pulled arc3-draft-manifest.json>` builds the C+D
check-run kernel `calamitychasm/arc3-jad-exp081-draft` once this session has run. Never push a second version of
this kernel after the session the C+D build pins has run: a kernel source mounts the latest version.
