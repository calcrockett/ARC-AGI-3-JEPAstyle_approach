#!/usr/bin/env python
"""Build our lean MTP-draft session A (experiments/stage7_milestone2_improvements.md section 15.6, step L0) as
calamitychasm/arc3-mtp-session-a.

    python scripts/_build_mtp_session_kernel.py                 # writes kaggle_submission_mtp_session/notebook/
    python scripts/_build_mtp_session_kernel.py --out DIR       # elsewhere (the tests rebuild into a tmp dir)

It runs JustAdev742's session builder ``build_mtp_session.py``, vendored unmodified with everything it ships and
imports in kaggle_submission_mtp_session/vendor/ (github.com/JustAdev742/Arc-Agi-3-Kaggle-comp @ ccb7214,
Apache-2.0; see the README and NOTICE there). Their files are not edited. This wrapper runs their ``build()`` in a
fresh interpreter and replaces, at module level and only for that run:

- ``LOGS_MANIFEST``: the training logs are arm C v2's own request logs (kernel source ``calamitychasm/arc3-jad-exp081``,
  mounted at its latest version, 2), from kaggle_submission_mtp_session/arc3-jad-exp081-v2-request-logs.json, instead
  of their private exp-073 output;
- ``STEP_CELLS``: the lean path drops A2-A5 (probe dump server, probe dump, its stop, the replica check) and A10 (the
  trained replica check), keeping A1, A6-A9, A11 and the summary;
- ``_setup_cell``: the session's inputs are the logs only (their private probe-prompts dataset and replica reference
  are not inputs), and the build record names this wrapper;
- ``worst_case_minutes``: the lean steps' worst case (one server boot, no A3/A5/A10), so a 4.5 h deadline is accepted;
- ``_steps_markdown``: the step list's header describes the lean path.

and with their own ``--set``: ``session_hours=4.5``, ``train.reserve_minutes=20`` (only A11 runs after A9) and
``storage.probe_gb=0.0`` (no probe dump). Hold-out (11 games), hygiene (repeat / "stuck in a loop" turns dropped),
the dump caps (34 GB, 2 epochs, 75 min training cap) and the export are theirs, unchanged.

After their build it inserts four cells of ours (marked ``# >>> calamitychasm (lean path)``): the data check after
their setup cell (25 log names, response count pinned to arm C v2's 2,952), the A5 stand-in before A6 (no replica GO
is needed), the drafter gate after A9 (section 15.6, read from A9's held-out evaluation) and the export check after
A11 (manifest sha256, frozen files byte-identical). It rewrites the kernel metadata (our id, arm C v2 as the only
kernel source, no private inputs) and appends a fork note to markdown cell 0.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KDIR = ROOT / "kaggle_submission_mtp_session"
VENDOR = KDIR / "vendor"
OUT = KDIR / "notebook"
LOGS_MANIFEST = KDIR / "arc3-jad-exp081-v2-request-logs.json"
OWNER = "calamitychasm"
SLUG = "arc3-mtp-session-a"
LOGS_KERNEL = "calamitychasm/arc3-jad-exp081"
UPSTREAM_COMMIT = "ccb7214"
DROPPED = ("A2", "A3", "A4", "A5", "A10")
SETTINGS = ["session_hours=4.5", "train.reserve_minutes=20", "storage.probe_gb=0.0"]
LEAN_DATA_MIN = 10       # our data check (reads arm C v2's ~1.5-2.5 GB of logs once)
LEAN_EXPORT_MIN = 5      # our export check (sha256 of the stock draft's frozen files)
LEAN_BEGIN = "# >>> calamitychasm (lean path)"
LEAN_END = "# <<< calamitychasm (lean path)"
NOTE = ("Built for calamitychasm by scripts/_build_mtp_session_kernel.py (lean path L0, "
        "experiments/stage7_milestone2_improvements.md 15.6) through JustAdev742's unmodified "
        f"scripts/build_mtp_session.py @ {UPSTREAM_COMMIT} (Apache-2.0)")
# section 15.6's drafter gate, read from A9's held-out evaluation (accept-length proxies of mtp_train.py)
GATE = {"pass_realized": 0.20, "pass_greedy": 0.20, "pass_kl_drop": 0.40, "kill_realized": 0.10}


def lean_worst_case_minutes(config: dict) -> float:
    """build_mtp_session.worst_case_minutes for the lean steps: inputs, A0, our data check, A1, ONE boot (A6), A7,
    A8, A9 with its grace, A11 and our export check."""
    b, d, t = config["budgets_min"], config["dump"], config["train"]
    return (config["wait_inputs_s"] / 60 + 5 + LEAN_DATA_MIN
            + b["A1"] + max(b["boot"], b["boot_max"])
            + d["holdout_minutes"] + 5 + d["minutes"] + 20
            + b["A8"] + t["minutes"] + t["grace_minutes"] + b["A11"] + LEAN_EXPORT_MIN)


def lean_steps_markdown(config: dict) -> str:
    return (
        "## Session A (lean path): steps A1, A6-A9, A11\n\n"
        "Each cell below is one step of JustAdev742's session A (`scripts/mtp_session_a.py`, unmodified), run on "
        "arm C v2's own request logs. The lean path (calamitychasm, experiments/stage7_milestone2_improvements.md "
        "15.6) drops A2-A5 (probe dump and replica check) and A10 (trained replica check): one server start, the "
        "training dump (A6-A7), training (A8-A9) and the export (A11). Our cells, marked `calamitychasm (lean "
        "path)`, check the data, stand in for A5, print the drafter gate from A9's held-out evaluation and check the "
        "export. Every command's output is also in `/kaggle/working/logs/`; `/kaggle/working/session-a.json` is "
        f"rewritten after every step. The module's own deadline is {config['session_hours']:g} h after the notebook "
        "started.")


def logs_spec(manifest: dict) -> dict:
    """The session's logs input: sizes pinned when the manifest lists them all, else names checked by our cell."""
    files = manifest["files"]
    if all(isinstance(v, int) and not isinstance(v, bool) for v in files.values()):
        return {"kind": "kernel", "id": manifest["kernel"], "files": dict(files)}
    if any(v is not None for v in files.values()):
        raise SystemExit(f"{LOGS_MANIFEST.name}: list every size or none")
    return {"kind": "kernel", "id": manifest["kernel"], "files": {}}


# ------------------------------------------------------------------------------------------------- our cells

def _cell(code: str) -> str:
    return code if code.endswith("\n") else code + "\n"


def data_cell(manifest: dict) -> str:
    names = sorted(manifest["files"])
    return _cell(f'''{LEAN_BEGIN}: the training data, arm C v2's request logs ({manifest["kernel"]} version {manifest["version"]},
# a kernel source: it mounts the latest version). Check the {len(names)} log names and the response count (pinned to v2's
# {manifest["responses"]:,}; another version of the kernel output stops the session here), and print what A7 will read.
import json as _lean_json
import time as _lean_time
from pathlib import Path as _LeanPath
_LEAN_LOG_NAMES = {names!r}
_LEAN_EXPECT = {{"responses": {manifest["responses"]!r}, "completion_tokens": {manifest["completion_tokens"]!r}}}


def _lean_data(session):
    root = _LeanPath(session.inputs["logs"]["dir"])
    holdout = set(session.cfg["dump"]["holdout_games"])
    t0 = _lean_time.time()
    while True:  # a kernel source may still be mounting: wait as long as the inputs step does
        files = sorted(p for p in root.rglob("*_requests.jsonl") if p.is_file())
        missing = sorted(set(_LEAN_LOG_NAMES) - {{p.name for p in files}})
        if not missing or _lean_time.time() - t0 > float(session.cfg.get("wait_inputs_s", 300)):
            break
        _lean_time.sleep(10)
    if missing:
        raise RuntimeError(f"ours (lean): request logs missing under {{root}}: {{missing}} (not arm C v2's output?)")
    split = {{"train": {{"games": set(), "responses": 0, "completion_tokens": 0}},
             "holdout": {{"games": set(), "responses": 0, "completion_tokens": 0}}}}
    for path in files:
        game = path.name.split("-", 1)[0]
        part = split["holdout" if game in holdout else "train"]
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if '"event": "response"' not in line:
                    continue
                try:
                    rec = _lean_json.loads(line)
                except ValueError:
                    continue
                tokens = (rec.get("usage") or {{}}).get("completion_tokens")
                if tokens is None:
                    continue
                part["games"].add(game)
                part["responses"] += 1
                part["completion_tokens"] += int(tokens or 0)
    responses = sum(p["responses"] for p in split.values())
    tokens = sum(p["completion_tokens"] for p in split.values())
    size = sum(p.stat().st_size for p in files)
    record = {{"dir": str(root), "files": [p.name for p in files], "bytes": size, "responses": responses,
              "completion_tokens": tokens, "seconds": round(_lean_time.time() - t0, 1),
              "split": {{k: {{**v, "games": sorted(v["games"])}} for k, v in split.items()}}}}
    session.state["lean_data"] = record
    session.save()
    print(f"ours (lean): data loaded {{responses}} responses ({{tokens}} completion tokens) from {{len(files)}} request "
          f"logs, {{size / 1e9:.2f}} GB, in {{root}} ({{record['seconds']:.0f}} s)")
    for key in ("train", "holdout"):
        part = record["split"][key]
        print(f"ours (lean): {{key}} split {{len(part['games'])}} games, {{part['responses']}} responses, "
              f"{{part['completion_tokens']}} completion tokens: {{' '.join(part['games'])}}")
    if responses != _LEAN_EXPECT["responses"]:
        raise RuntimeError(f"ours (lean): {{responses}} responses, arm C v2 has {{_LEAN_EXPECT['responses']}} "
                           "(another version of calamitychasm/arc3-jad-exp081 mounted?)")
    if tokens != _LEAN_EXPECT["completion_tokens"]:
        session.warn(f"lean data: {{tokens}} completion tokens, the 25-log count of v2 is "
                     f"{{_LEAN_EXPECT['completion_tokens']}}")


_lean_data(SESSION)
{LEAN_END}''')


def skip_cell() -> str:
    return _cell(f'''{LEAN_BEGIN}: A2-A5 (probe dump server, probe dump, stop, replica check) and A10 are not in this
# notebook. Their module lets A6-A11 run only after A5's GO (Session.require_go reads the replica verdict); on the
# lean path there is no replica check, so this session's require_go accepts every step. go() still stops the
# notebook after a failed step. The drafter gate is read from A9's held-out evaluation (next cells).
SESSION.state["replica_check"] = {{"skipped": "lean path: no probe dump and no replica check (A2-A5, A10 dropped)"}}
SESSION.state["lean_path"] = {{"dropped_steps": {list(DROPPED)!r}, "gate": "A9 held-out evaluation, section 15.6"}}
SESSION.save()
SESSION.require_go = lambda step: None
SESSION_A_GO = True
print("ours (lean): A2-A5 and A10 dropped (no probe dump, no replica check); A6-A9 and A11 run without a replica GO")
{LEAN_END}''')


def gate_cell() -> str:
    return _cell(f'''{LEAN_BEGIN}: the drafter gate of experiments/stage7_milestone2_improvements.md 15.6, from A9's
# held-out evaluation (mtp_train.py's train-report.json: the 11 held-out games, original weights against trained).
# PASS: realized accept +{GATE["pass_realized"]:.2f} or more, greedy +{GATE["pass_greedy"]:.2f} or more, step-1 KL down {GATE["pass_kl_drop"]:.0%} or more, top-1
# agreement up at all three steps. KILL: realized below +{GATE["kill_realized"]:.2f}. Anything else: BETWEEN (no C+D run; the
# full path's probes decide). The export (A11) runs either way; only a PASS feeds the C+D build.
_LEAN_GATE = {GATE!r}


def _lean_gate(session):
    path = session.working / "mtp-train" / "train-report.json"
    report = _lean_json.loads(path.read_text()) if path.is_file() else {{}}
    cmp = report.get("comparison") or {{}}

    def get(key, field):
        value = cmp.get(key)
        return value.get(field) if isinstance(value, dict) else None

    def fmt(x, signed=False):
        return "-" if x is None else f"{{x:+.4f}}" if signed else f"{{x:.4f}}"

    print(f"ours (lean): A9 steps {{report.get('steps_done')}} of {{report.get('total_steps')}}, "
          f"{{report.get('train_seconds')}} s; windows {{_lean_json.dumps(report.get('windows'))[:300]}}")
    for key in ("accept_realized", "accept_greedy", "accept_expected"):
        print(f"ours (lean): held-out {{key}} before {{fmt(get(key, 'original'))}} after {{fmt(get(key, 'trained'))}} "
              f"delta {{fmt(get(key, 'delta'), True)}}")
    for k in (1, 2, 3):
        print(f"ours (lean): held-out step{{k}} " + "; ".join(
            f"{{name}} {{fmt(get(f'step{{k}}_{{name}}', 'original'))}} -> {{fmt(get(f'step{{k}}_{{name}}', 'trained'))}}"
            for name in ("kl", "top1_target_full", "top1_realized", "rate_expected")))
    realized, greedy = get("accept_realized", "delta"), get("accept_greedy", "delta")
    kl0, kl1 = get("step1_kl", "original"), get("step1_kl", "trained")
    kl_drop = (1 - kl1 / kl0) if kl0 and kl1 is not None else None
    top1 = [get(f"step{{k}}_top1_target_full", "delta") for k in (1, 2, 3)]
    top1_up = all(x is not None and x > 0 for x in top1)
    if realized is None:
        verdict = "UNREADABLE"
    elif realized < _LEAN_GATE["kill_realized"]:
        verdict = "KILL"
    elif (realized >= _LEAN_GATE["pass_realized"] and greedy is not None and greedy >= _LEAN_GATE["pass_greedy"]
          and kl_drop is not None and kl_drop >= _LEAN_GATE["pass_kl_drop"] and top1_up):
        verdict = "PASS"
    else:
        verdict = "BETWEEN"
    gate = {{"verdict": verdict, "realized_delta": realized, "greedy_delta": greedy, "step1_kl_drop": kl_drop,
            "top1_target_full_delta": top1, "thresholds": _LEAN_GATE}}
    session.state["lean_gate"] = gate
    session.save()
    print(f"ours (lean): drafter gate {{verdict}}: realized {{fmt(realized, True)}}, greedy {{fmt(greedy, True)}}, step-1 KL drop "
          f"{{'-' if kl_drop is None else f'{{kl_drop:.0%}}'}}, top-1 (target, full) delta per step "
          f"{{[None if x is None else round(x, 4) for x in top1]}}")


_lean_gate(SESSION)
{LEAN_END}''')


def export_cell() -> str:
    return _cell(f'''{LEAN_BEGIN}: the export. Print the manifest's sha256 (the C+D build pins it: scripts/
# _build_jad_exp081_draft_kernel.py --draft-manifest), the dense shard's sha256, and check that every other file of
# the draft is byte-identical to the stock draft's and to its manifest entry (the frozen tensors; the dense shard's
# frozen tensors are checked by their exporter's own check in A11). The manifest is also printed on one line.
import hashlib as _lean_hashlib


def _lean_export(session):
    draft = session.working / "mtp-draft"
    raw = (draft / "arc3-draft-manifest.json").read_bytes()
    manifest = _lean_json.loads(raw)
    files = manifest.get("files") or {{}}
    stock = _LeanPath(session.launcher["albucino"])
    frozen = {{n: s for n, s in files.items() if n != "mtp-dense.safetensors"}}
    same = sorted(n for n, s in frozen.items() if (stock / n).is_file() and (draft / n).is_file()
                  and arc3_sa.sha256_file(draft / n) == s == arc3_sa.sha256_file(stock / n))
    differ = sorted(set(frozen) - set(same))
    sha = _lean_hashlib.sha256(raw).hexdigest()
    session.state["lean_export"] = {{"manifest_sha256": sha, "manifest_bytes": len(raw),
                                    "dense_sha256": files.get("mtp-dense.safetensors"),
                                    "replaced": len(manifest.get("replaced") or []), "frozen_identical": same,
                                    "frozen_differ": differ}}
    session.save()
    print(f"ours (lean): MTP draft manifest sha256 {{sha}} ({{len(raw)}} bytes); mtp-dense.safetensors sha256 "
          f"{{files.get('mtp-dense.safetensors')}}; {{len(manifest.get('replaced') or [])}} dense tensors replaced; "
          f"frozen files byte-identical to the stock draft {{len(same)}}/{{len(frozen)}}")
    print("MTP_DRAFT_MANIFEST_JSON " + _lean_json.dumps(manifest, separators=(",", ":")))
    if differ or not frozen:
        raise RuntimeError(f"ours (lean): draft files differ from the stock draft: {{differ or 'none listed'}}")


_lean_export(SESSION)
{LEAN_END}''')


# -------------------------------------------------------------------------------------------------- the build

def _inner(out: Path, manifest_path: Path) -> None:
    """Runs in a fresh interpreter: import their builder, replace the five module attributes, build, add our cells."""
    sys.path.insert(0, str(VENDOR / "scripts"))
    import build_mtp_session as bm  # noqa: E402  (theirs, unmodified)

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kernel") != LOGS_KERNEL or manifest.get("version") != 2:
        raise SystemExit(f"{manifest_path}: expected {LOGS_KERNEL} version 2")
    spec = logs_spec(manifest)
    orig_setup = bm._setup_cell

    def setup_cell(config: dict) -> str:
        config["inputs"] = {"logs": copy.deepcopy(spec)}
        config["build"].update(
            builder="scripts/build_mtp_session.py (JustAdev742 @ ccb7214, unmodified)",
            wrapper="scripts/_build_mtp_session_kernel.py (calamitychasm, lean path L0)",
            dropped_steps=list(DROPPED), reference_source=None,
            logs_pinned="sizes" if spec["files"] else "names and response count (our data cell)")
        return orig_setup(config)

    bm.LOGS_MANIFEST = manifest_path
    bm.STEP_CELLS = [s for s in bm.STEP_CELLS if s[0] not in DROPPED]
    bm._setup_cell = setup_cell
    bm.worst_case_minutes = lean_worst_case_minutes
    bm._steps_markdown = lean_steps_markdown
    with tempfile.TemporaryDirectory() as tmp:
        result = bm.build(Path(tmp), SLUG, settings=SETTINGS, note=NOTE)
        nb = json.loads(result["notebook"].read_text())
    cells = nb["cells"]

    def at(needle: str) -> int:
        hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and needle in "".join(c["source"])]
        if len(hits) != 1:
            raise SystemExit(f"{needle!r} found in cells {hits} (expected one)")
        return hits[0]

    def code(template: int, text: str) -> dict:
        cell = copy.deepcopy(cells[template])
        cell["outputs"], cell["execution_count"] = [], None
        cell["source"] = text.splitlines(keepends=True)
        return cell

    for needle, text, after in (("SESSION.write_draft()", export_cell(), True), ("SESSION.train()", gate_cell(), True),
                                ("SESSION.start_server('A6', 'train')", skip_cell(), False),
                                ("SESSION = arc3_sa.Session(", data_cell(manifest), True)):
        i = at(needle)
        cells.insert(i + 1 if after else i, code(i, text))
    steps = [s[0] for s in bm.STEP_CELLS]
    if steps != ["A1", "A6", "A7", "A8", "A9", "A11", "end"]:
        raise SystemExit(f"unexpected lean steps {steps}")

    # cell 0: their builder's change list is kept; the inputs entry and the owner are ours
    head = "".join(cells[0]["source"])
    mark = "\n\n**Our session-A notebook (scottmahony, built by scripts/build_mtp_session.py"
    if head.count(mark) != 1:
        raise SystemExit("cell 0: their change list not found once")
    head = head.split(mark)[0]
    changes = [c for c in result["changes"] if not c.startswith("inputs:")]
    changes.append(f"input: {LOGS_KERNEL} (kernel source, version 2: arm C v2's {len(manifest['files'])} request "
                   "logs; " + ("byte sizes pinned" if spec["files"] else "names and the response count checked by "
                                                                          "our data cell") + ")")
    changes.append("lean path: steps " + ", ".join(DROPPED) + " dropped (no probe dump, no replica check, so no "
                   "probe-prompts dataset and no replica reference); four cells of ours added (data check, A5 "
                   "stand-in, drafter gate from A9's held-out evaluation, export check)")
    cells[0]["source"] = [head + "\n\n**calamitychasm/arc3-mtp-session-a (built by JustAdev742's "
                          "scripts/build_mtp_session.py from the unmodified dprime notebook, through our wrapper):** "
                          + "; ".join(changes) + f". {NOTE}\n"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    bf = bm.bf
    sources = copy.deepcopy(bf.SOURCES)
    sources["kernel_sources"] = [LOGS_KERNEL]
    meta = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "keywords": [], **sources, "docker_image": bf.IMAGE,
            "docker_image_pinning_type": "original", "machine_shape": bf.MACHINE_SHAPE}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"bytes": (out / f"{SLUG}.ipynb").stat().st_size,
                      "worst_case_minutes": result["worst_case_minutes"],
                      "session_hours": result["config"]["session_hours"]}))


def build(out: Path = OUT, *, logs_manifest: Path = LOGS_MANIFEST) -> dict:
    """Build into OUT through a fresh interpreter (their modules are imported and patched there, never here)."""
    with tempfile.TemporaryDirectory() as tmp:
        run = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--inner", "--out", tmp,
                              "--logs-manifest", str(Path(logs_manifest).resolve())],
                             capture_output=True, text=True, check=False)
        if run.returncode != 0:
            raise SystemExit(f"the session build failed ({run.returncode}):\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}")
        info = json.loads(run.stdout.strip().splitlines()[-1])
        out.mkdir(parents=True, exist_ok=True)
        for f in (f"{SLUG}.ipynb", "kernel-metadata.json"):
            shutil.copyfile(Path(tmp) / f, out / f)
    print(f"built {out / (SLUG + '.ipynb')} ({info['bytes']:,} bytes; lean worst case "
          f"{info['worst_case_minutes']} min of {info['session_hours']:g} h)")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--logs-manifest", type=Path, default=LOGS_MANIFEST)
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.inner:
        _inner(a.out, a.logs_manifest)
    else:
        build(a.out, logs_manifest=a.logs_manifest)


if __name__ == "__main__":
    main()
