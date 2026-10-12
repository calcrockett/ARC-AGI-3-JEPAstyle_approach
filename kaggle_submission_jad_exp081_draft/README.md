# `calamitychasm/arc3-jad-exp081-draft` (arm C+D)

Arm C (`calamitychasm/arc3-jad-exp081` v2) serving our own fine-tuned MTP draft, the output of the lean session
`calamitychasm/arc3-mtp-session-a` v1 (`kaggle_submission_mtp_session/`; `experiments/stage7_milestone2_improvements.md`
section 15). Session A passed its drafter gate on 2026-10-12 (held-out realized accept 2.894 -> 3.123, step-1 KL -55%,
top-1 up at all three steps).

Files:

- `arc3-draft-manifest.json`: session A v1's `mtp-draft/arc3-draft-manifest.json`, byte-exact (2998 bytes, sha256
  `6f408d58fd813bce415edc55802f174ebac73fef292455ac388104c404dc1d45`; `mtp-dense.safetensors` sha256
  `b677e0c4234c6344fe29f147a972b192de241e557e83a12452409da13967a985`; 26 dense tensors replaced, 12 files). Rebuilt as
  `json.dumps(obj, indent=1) + "\n"` from the `PRINT_FILE_JSON mtp-draft/arc3-draft-manifest.json` line of kaggle-ops run
  38243454868 (`read-1012-h16`) and checked against the file text printed in the same log.
- `notebook/`: built by

  ```
  python scripts/_build_jad_exp081_draft_kernel.py --draft-manifest kaggle_submission_jad_exp081_draft/arc3-draft-manifest.json \
      --expect-sha 6f408d58fd813bce415edc55802f174ebac73fef292455ac388104c404dc1d45
  ```

  (the builder refuses to build without a real manifest; no placeholder exists). Pinned in `tests/test_jad_exp081_draft.py`,
  which also rebuilds it byte-identically from the committed manifest.

The notebook differs from arm C v2 only in cell 4's `DRAFT_MODEL_DIR` (a wait of up to 600 s for
`arc3-draft-manifest.json` under `/kaggle/input/notebooks/calamitychasm/arc3-mtp-session-a/mtp-draft` or
`/kaggle/input/arc3-mtp-session-a/mtp-draft`, then a sha256 check of the manifest and of `mtp-dense.safetensors`,
printing `ours: MTP draft <path> after N s (manifest 6f408d58fd81, dense b677e0c4234c)`) and markdown cell 0; the
metadata swaps the stock-draft model source for the session kernel source (`kernel_sources` cannot pin a version, so
the kernel mounts the session's latest version: **never push a v2 of `arc3-mtp-session-a`**; the sha check would stop
the notebook if its output changed).
