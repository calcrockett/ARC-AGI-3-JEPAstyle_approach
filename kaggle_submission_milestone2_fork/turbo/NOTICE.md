# Turbo kernel: provenance (JustAdev742, Apache-2.0)

The files below come from **JustAdev742's** ARC-AGI-3 Milestone-2 repository,
https://github.com/JustAdev742/Arc-Agi-3-Kaggle-comp (commit `fe2ad06`, 2026-10-09; Kaggle team
"Jovian Game Studios", notebooks under `scottmahony/...`), licensed under the **Apache License 2.0**
(copy: `LICENSE-JustAdev742-Apache-2.0.txt`). Their work runs on exactly our serving stack: Daniel
Franzen's Milestone-2 notebook (Apache-2.0) with John Pezzulli's Pennyroyal SGLang fork
(`dfranzen/pennyroyal-v253`, wheel `sglang-0.5.19+gd00d88efc8d6`), Intel's Qwen3.8-Flash-Next W4A16
AutoRound checkpoint and Albucino's MTP draft. Third-party code and weights keep their own licenses.

| file here | their path | changed? |
|---|---|---|
| `sglang_reap_patch.py` | `scripts/sglang_reap_patch.py` | verbatim |
| `reap448_kept_experts.json` (+ `.meta.json`) | `kaggle/franzen/` | verbatim (the notebook ships the kept list as the 64 pruned ids per layer and rebuilds this exact file, sha256 `e7e6a28b27b1...`) |
| `hot_tokens_64k_arc.pt` (+ `.meta.json`) | `kaggle/franzen/` | verbatim (sha256 `9c77419ad575...`); the notebook writes the same 65,536 ids in the record layout of their builder's `HOT_WRITER`, sha256 `ec15348b1186...` -- byte-identical to what their builder writes |
| `ours-sandbox-timeout-keeps-work.patch` | `kaggle/franzen/patches/` | verbatim, reference only |
| `timeout_fix.py` | -- | **ours**: the same fix applied at runtime by wrapping `ToolAgent._record_retained_functions` and editing the sandbox bootstrap string, so the upstream harness patch cell stays verbatim |

`scripts/_build_m2_level_memory_kernel.py` (flags `--reap`, `--spec-accept`, `--arc-hotmap`,
`--streams`, `--timeout-fix`, `--turbo`) is ours; its launcher-cell edits follow their
`scripts/build_franzen_nb.py` (`_reap`, `_hot_tokens`): the same anchors, server flags and file paths.

What they measured (their `docs/research_log.md`, same conditions: D' base, 25 public games x 25 min):
base 641.9 output tok/s; REAP-448 + 14 streams 733.2 (+14%); + MTP acceptance 0.5 819.3 (+28%, output
length per request unchanged); the ARC FR-Spec map ~+2% at full length (exp-077). Full-length (121 min
per game) public-25 scores: 56.00 and 42.89 for REAP + 14 + acceptance 0.5 vs Franzen v3's 45.6-47.5.
Fidelity probe: REAP-448 shifts chosen-token logprobs on image turns (mean |dlogprob| 0.048 vs a
0.038 noise floor), not on text turns. Their LB draw of this serving config (submission 56980485,
2026-10-09 00:14 UTC) was still PENDING in their log at the time of the port.

The pruned expert list is the one the public build `lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448`
keeps (recovered by JustAdev742 from exact router-row matches against Intel's checkpoint).
