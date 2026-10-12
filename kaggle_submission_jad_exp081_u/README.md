# Arm C+U: `calamitychasm/arc3-jad-exp081-u`

Arm C v2 (`calamitychasm/arc3-jad-exp081`, our verbatim fork of JustAdev742's exp-081) plus two harness changes.
Analysis and pre-registered check-run criteria: `experiments/stage7_milestone2_improvements.md` section 12.

- **C1, the I1 "not yet tried on this level" notice**: JustAdev742's `ours-10-untried.patch`, run with
  `OURS_UNTRIED=i1` (the I1 line only; I2 off).
- **C2, the stale-state latch fix** (ours): `patches/calamitychasm-stale-latch-fix.patch`.

## Third-party file (vendored verbatim)

| file | source | sha256 |
|---|---|---|
| `vendor/kaggle/franzen/patches/ours-10-untried.patch` | github.com/JustAdev742/Arc-Agi-3-Kaggle-comp, path `kaggle/franzen/patches/ours-10-untried.patch`, commit `c7a462be8fe3a22910804d4d61614f43a510fdfb` (identical at `ccb7214862bbe7fab3a24a7e7e2ba6653d931cac`; added in their commit `85bc492`) | `a3a7d89fbb2cb253afcfc8eb9b4d43f763c45ea8b8624139db54949e7bc6a52e` |
| `vendor/LICENSE` | same repository, `LICENSE` (Apache License 2.0), same commits | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `vendor/kaggle/franzen/NOTICE.md` | same repository, `kaggle/franzen/NOTICE.md` (the patch modifies Daniel Franzen's Apache-2.0 harness), same commits | `e2db507c745f5cee9324b092dba1b80c79391996448afc478c1ae7a629dd0ba8` |

`vendor/SHA256SUMS` lists all three. The patch needs nothing beyond what arm C's vendored builder (`c7a462b`) already
has: it adds `inference/utils/ours_untried.py` (stdlib only) and two 3-line call sites in `tool_agent.py`, and applies
on top of exp-081's six patches with line offsets only. No newer builder code is used.

NOTICE: this kernel contains `ours-10-untried.patch` by JustAdev742 (Apache License 2.0), unmodified, applied to
Daniel Franzen's Milestone-2 harness (Apache License 2.0), which builds on Tufa Labs' Duck harness (MIT). Our own
changes are the latch-fix patch, `cu_counters.py` and the notebook edits listed below.

## Build

```
python scripts/_build_jad_exp081_u_kernel.py          # -> notebook/arc3-jad-exp081-u.ipynb + kernel-metadata.json
```

It reads arm C's committed notebook (sha256-pinned, never rewritten) and makes these edits only:

1. after arm C's last patch cell: the ours-10 cell (made by their builder's own `_file_cell`, compact, as their
   `--patch` ships it) and a plain `%%writefile` cell with the latch fix;
2. setup cell: `'OURS_UNTRIED': 'i1',  # ours (--env-add)`, and the apply loop over all eight patches (their
   `_apply_code`), whose success line becomes `our harness patches applied successfully: 8`;
3. a cell before the run cell that embeds `cu_counters.py`, checks both changes in the imported harness, prints
   `STALE_LATCH_FIX installed ...` and `UNTRIED_NOTICE installed ...`, and installs count-only wrappers;
4. the run cell writes `untried_summary.json`, `stale_latch_summary.json` and `untried_firings.json` after `bm.run`.

`kernel-metadata.json` differs from arm C's only in `id`, `title` and `code_file`. Tests: `tests/test_jad_exp081_u.py`
(+ the probe `tests/jad_u_probe.py`, which builds the real harness tree from the notebook's own patch cells).
