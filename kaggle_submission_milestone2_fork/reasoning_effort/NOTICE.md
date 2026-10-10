# Static reasoning effort: provenance

`reasoning_effort.py` is **ours** (no copied code). It sends the served chat template's `reasoning_effort` kwarg
with every request by wrapping `ToolAgent._harness_template_kwargs` at runtime; the harness patch is untouched.

- **The knob.** The Qwen3.8-Flash-Next `chat_template.jinja` (`reasoning_effort|default('xhigh')`; accepted
  values xhigh, medium, low) prepends "Reasoning effort is set to xhigh. Please think carefully through the task,
  validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity
  in the final answer." to the system prompt at the default; "medium" adds nothing; "low" adds "Keep your thinking
  brief and focused ...". Read in JustAdev742's verbatim fixture of the template (sha256 `c3cf9e34...`, from
  RadixArk/Qwen3.8-Flash-Next-NVFP4; their mtp-drafter-finetune.md records the same hash for the template served on
  Franzen's Intel W4A16 stack) and their lesson 0021. Our kernel re-checks the template it actually serves at
  install time (`REASONING_EFFORT_TEMPLATE`) and the live server's handling (`REASONING_EFFORT_SERVER`).
- **Where Franzen's harness sets it.** `inference/agent/tool_agent.py` `ToolAgent._harness_template_kwargs`
  (from the harness patch in `kaggle_submission_milestone2_fork/upstream/`): only `preserve_thinking`, plus
  `reasoning_effort` from the truncation ladder `ARC3_REASONING_EFFORT_LADDER`, which the incumbent leaves unset.
- **The idea and the evidence.** juliancamilovilla's Kaggle notebooks `arc-agi3-e1-med` / `arc-agi3-e1-med-g15`
  (Oct 9, env `ARC3_STATIC_REASONING_EFFORT`), as read and tabulated by JustAdev742 in
  `docs/research/beat-tufa/intel-oct10.md` section 2.3; JustAdev742's own patch
  `kaggle/franzen/patches/ours-09-reasoning-effort.patch` (env `OURS_REASONING_EFFORT`) and run exp-082
  (github.com/JustAdev742/Arc-Agi-3-Kaggle-comp, Apache-2.0). We copy neither text: the behaviour (static value,
  the ladder still steps below it) is a single dict key.
