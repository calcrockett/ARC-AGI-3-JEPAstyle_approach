# `calamitychasm/arc3-jad-exp081-draft` (arm C+D): NOT BUILT YET

Arm C (`calamitychasm/arc3-jad-exp081` v2) serving our own fine-tuned MTP draft, the output of the lean session
`calamitychasm/arc3-mtp-session-a` (`kaggle_submission_mtp_session/`; `experiments/stage7_milestone2_improvements.md`
section 15). The notebook does not exist until that session has run: it pins the sha256 of the session's
`mtp-draft/arc3-draft-manifest.json`, and `scripts/_build_jad_exp081_draft_kernel.py` refuses to build without the
real file (no placeholder exists).

After session A passes its drafter gate:

```
python scripts/_build_jad_exp081_draft_kernel.py --draft-manifest <pulled arc3-draft-manifest.json> --expect-sha <sha the session printed>
```

writes `notebook/` and copies the manifest to `arc3-draft-manifest.json` here; commit all three together and pin the
notebook's sha256 in `tests/test_jad_exp081_draft.py`. The notebook differs from arm C v2 only in cell 4's
`DRAFT_MODEL_DIR` (a wait of up to 600 s for the mounted draft, sha256-checked, printing `ours: MTP draft ...`) and
markdown cell 0; the metadata swaps the stock-draft model source for the session kernel source.
