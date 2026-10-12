"""Lean MTP-draft session A (calamitychasm/arc3-mtp-session-a; experiments/stage7_milestone2_improvements.md 15.6 L0).

Pins the committed notebook and its rebuild, the vendored upstream files (JustAdev742 @ ccb7214, unmodified), the
wrapper's overrides (inputs, dropped steps, owner, deadline), and runs the notebook's own lean cells in order against
the real session module with THEIR stub scripts and stub server (their tests/test_mtp_session_a.py Bed). CPU only."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_mtp_session_kernel as B  # noqa: E402
import kaggle_ops  # noqa: E402

KDIR = ROOT / "kaggle_submission_mtp_session"
VENDOR = KDIR / "vendor"
NB = KDIR / "notebook" / "arc3-mtp-session-a.ipynb"
META = KDIR / "notebook" / "kernel-metadata.json"
NB_SHA = "f11d1ea9223e1545a6c75b935d7f1b0ea44377cf6f04b18c23556532f170d336"
META_SHA = "a37adfc33bdf3ccd9120cd058b20c2ea0d487d08cdf0e2066bdc49f34f4dd284"
UPSTREAM = Path("/home/user/ext/JustAdev742_Arc-Agi-3-Kaggle-comp")
LAUNCH_ANCHOR = "CFG = dict(\n"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def theirs():
    """Their session module, tree helper and stub-session test bed, loaded from the vendored files."""
    tests = _load("_mtps_their_session_tests", VENDOR / "tests" / "test_mtp_session_a.py")
    tree = _load("_mtps_franzen_tree", VENDOR / "scripts" / "franzen_tree.py")
    return SimpleNamespace(sa=tests.sa, Bed=tests.Bed, launcher_args=tests.launcher_args, tests=tests, tree=tree)


@pytest.fixture(scope="module")
def nb():
    data = json.loads(NB.read_text(encoding="utf-8"))
    return data, ["".join(c["source"]) for c in data["cells"]]


def _index(cells: list[str], needle: str) -> int:
    hits = [i for i, s in enumerate(cells) if needle in s]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def _config(cells: list[str]) -> dict:
    setup = cells[_index(cells, "SESSION = arc3_sa.Session(")]
    literal = setup.split("SESSION_A_CONFIG = ", 1)[1].split("\nSESSION = ", 1)[0]
    return ast.literal_eval(literal)


# ------------------------------------------------------------------------------------------------ pins

def test_committed_notebook_is_pinned():
    assert _sha(NB) == NB_SHA
    assert _sha(META) == META_SHA
    assert NB.stat().st_size < 900_000  # Kaggle refuses notebooks near 1 MB (their lesson 0033)


def test_rebuild_is_byte_identical(tmp_path):
    info = B.build(tmp_path)
    assert _sha(tmp_path / "arc3-mtp-session-a.ipynb") == NB_SHA
    assert _sha(tmp_path / "kernel-metadata.json") == META_SHA
    assert info["worst_case_minutes"] <= info["session_hours"] * 60


def test_vendored_files_are_unmodified():
    listed = {}
    for line in filter(None, (VENDOR / "SHA256SUMS").read_text().split("\n")):
        sha, name = line.split(maxsplit=1)
        listed[name.removeprefix("./")] = sha
    on_disk = {str(p.relative_to(VENDOR)) for p in VENDOR.rglob("*")
               if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts}
    assert on_disk == set(listed)
    for name, sha in listed.items():
        assert _sha(VENDOR / name) == sha, name
    for name in ("scripts/build_mtp_session.py", "scripts/mtp_session_a.py", "scripts/sglang_hc_dump_patch.py",
                 "scripts/hc_dump_driver.py", "scripts/mtp_train.py", "scripts/mtp_write_draft.py",
                 "scripts/mtp_replica.py", "LICENSE"):
        assert name in listed, name
    if (UPSTREAM / ".git").is_dir():  # the clone at ccb7214: every vendored file is theirs, byte for byte
        head = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "--short=7", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip()
        if head == B.UPSTREAM_COMMIT:
            for name, sha in listed.items():
                assert _sha(UPSTREAM / name) == sha, name


def test_vendored_arm_c_builder_is_the_same_file():
    """Arm C was built with their c7a462b builder; ccb7214's is byte-identical (section 15.1)."""
    assert _sha(VENDOR / "scripts" / "build_franzen_nb.py") == \
        _sha(ROOT / "kaggle_submission_jad_exp081" / "vendor" / "scripts" / "build_franzen_nb.py")


def test_their_cpu_session_tests_pass_on_the_vendored_tree():
    """Their own stub-session, builder, dump-driver and exporter tests, run unmodified in vendor/."""
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c", os.devnull,
                          "--rootdir", str(VENDOR), "tests"], cwd=VENDOR, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=900, check=False)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]
    assert " passed" in run.stdout and "failed" not in run.stdout


# ------------------------------------------------------------------------------------------- overrides

def test_metadata_is_ours_with_arm_c_v2_as_the_only_kernel_source():
    meta = json.loads(META.read_text())
    assert meta["id"] == "calamitychasm/arc3-mtp-session-a" and meta["title"] == "arc3-mtp-session-a"
    assert meta["code_file"] == "arc3-mtp-session-a.ipynb"
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is False
    assert meta["machine_shape"] == "NvidiaRtxPro6000"  # their session's machine (RTX PRO 6000)
    assert meta["kernel_sources"] == ["calamitychasm/arc3-jad-exp081"]
    assert meta["dataset_sources"] == ["dfranzen/pennyroyal-v253", "dfranzen/taaf-kaggle-source-bundle-copy"]
    assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
    assert len(meta["model_sources"]) == 2 and all(m.startswith("dfranzen/") for m in meta["model_sources"])
    assert "scottmahony" not in META.read_text()
    arm_c = json.loads((ROOT / "kaggle_submission_jad_exp081" / "notebook" / "kernel-metadata.json").read_text())
    assert meta["docker_image"] == arm_c["docker_image"] and meta["machine_shape"] == arm_c["machine_shape"]
    assert kaggle_ops.is_gpu_family(meta["id"])


def test_session_config_inputs_deadline_and_their_recipe(nb):
    data, cells = nb
    cfg = _config(cells)
    sizes = json.loads(B.LOGS_MANIFEST.read_text())["files"]  # listed by kaggle-ops mtps-ls-1 (run 38214680909)
    assert len(sizes) == 25 and sum(sizes.values()) == 1_798_106_223
    assert all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in sizes.values())
    assert cfg["inputs"] == {"logs": {"kind": "kernel", "id": "calamitychasm/arc3-jad-exp081", "files": sizes}}
    assert cfg["session_hours"] == 4.5 and cfg["train"]["reserve_minutes"] == 20 and cfg["storage"]["probe_gb"] == 0.0
    assert B.lean_worst_case_minutes(cfg) <= cfg["session_hours"] * 60
    # their v1 recipe and hold-out, unchanged
    assert cfg["dump"]["holdout_games"] == ["ar25", "ft09", "lp85", "r11l", "re86", "sb26", "sc25", "tn36", "tr87",
                                            "tu93", "vc33"]
    assert cfg["storage"]["train_gb"] == 34.0 and cfg["train"]["minutes"] == 75 and cfg["server"]["reap"] is True
    assert cfg["train"]["map_sha256"] == "ec15348b11863ec6fb94b655e4f9ddc4c0ce457fb11f77807b0c5c2d391da70f"
    assert cfg["build"]["dropped_steps"] == ["A2", "A3", "A4", "A5", "A10"]
    text = "\n".join(cells)
    for private in ("arc3-fidelity-prompts", "arc3-fidelity-base", "arc3-dprime-reap448-r14-full"):
        assert private not in text.split("# >>> ours (--compact)")[0] and private not in json.dumps(cfg), private
    assert "scottmahony" not in cells[0]
    assert "calamitychasm/arc3-mtp-session-a" in cells[0] and "ccb7214" in cells[0]
    assert data["metadata"]["kaggle"]["accelerator"] == "nvidiaRtxPro6000"


def test_lean_steps_and_our_cells_in_order(nb):
    data, cells = nb
    launch = _index(cells, LAUNCH_ANCHOR)
    steps = [s for c, s in zip(data["cells"][launch + 1:], cells[launch + 1:]) if c["cell_type"] == "code"]
    labels = [("ours" if s.startswith(B.LEAN_BEGIN) else s.split(":")[0].split()[-1]) for s in steps]
    assert labels == ["A1", "ours", "A6", "A7", "A8", "A9", "ours", "A11", "ours", "end"]
    for gone in ("SESSION.start_server('A2'", "SESSION.probe_dump()", "SESSION.stop('A4')", "replica_gate()",
                 "replica_check('A10'"):
        assert gone not in "\n".join(cells), gone
    data_cell = _index(cells, "_lean_data(SESSION)")
    assert data_cell == _index(cells, "SESSION = arc3_sa.Session(") + 1
    names = json.loads(B.LOGS_MANIFEST.read_text())["files"]
    assert len(names) == 25 and all(n in cells[data_cell] for n in names)
    assert '"responses": 2952' in cells[data_cell]


def test_every_code_cell_compiles_and_shipped_files_round_trip(nb, theirs, tmp_path):
    data, cells = nb
    shipped = {}
    for cell, src in zip(data["cells"], cells):
        if cell["cell_type"] == "code" and not src.startswith("%%writefile"):
            compile("\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))), "cell",
                    "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        written = theirs.tree.written_file(src)
        if written is not None:
            shipped[written[0]] = written[1]
    for name in ("mtp_session_a.py", "sglang_hc_dump_patch.py", "hc_dump_driver.py", "fidelity_probe.py",
                 "fidelity_sample.py", "mtp_probe_dump.py", "mtp_replica.py", "mtp_train.py", "mtp_write_draft.py"):
        assert shipped[f"/kaggle/arc3-mtp/{name}"] == (VENDOR / "scripts" / name).read_text(), name
    assert shipped["/kaggle/arc3-reap-kept.json"] == \
        (VENDOR / "kaggle" / "franzen" / "reap448_kept_experts.json").read_text()


def test_kaggle_ops_counts_the_session_as_a_gpu_kernel():
    assert "arc3-mtp-" in kaggle_ops.GPU_KERNEL_PREFIXES
    assert kaggle_ops.is_gpu_family("calamitychasm/arc3-mtp-session-a")
    assert kaggle_ops.is_gpu_family("calamitychasm/arc3-jad-exp081-draft")


# ------------------------------------------------------------------------- a lean session on their stubs

TRAIN_STUB = '''
cmd = sys.argv[1]
if cmd == "plan":
    print(json.dumps({"requests": 3, "train": {"windows": 12}, "holdout": {"windows": 2}, "total_steps": 8}, indent=1))
    sys.exit(0)
out = Path(opt("--out"))
out.mkdir(parents=True, exist_ok=True)
assert Path(opt("--token-map")).is_file() and opt("--target-dir") and opt("--draft")
(out / "eval-original.json").write_text(json.dumps({"seconds": 1.0}))
with open(out / "train-log.jsonl", "w") as f:
    for i in range(1, 9):
        f.write(json.dumps({"step": i, "seconds": 0.01, "loss": 1.0 / i}) + "\\n")
(out / "trained-dense.safetensors").write_bytes(b"dense")
cmp = json.loads((HERE / "comparison.json").read_text())
(out / "train-report.json").write_text(json.dumps({"windows": {"train": 12, "holdout": 2}, "total_steps": 8,
                                                   "steps_done": 8, "train_seconds": 1.0, "comparison": cmp}))
'''

WRITE_STUB = '''
import hashlib, shutil
out, src = Path(opt("--out")), Path(opt("--draft"))
assert sys.argv[1] == "write" and Path(opt("--trained")).is_file()
out.mkdir(parents=True)
for p in sorted(src.iterdir()):
    shutil.copyfile(p, out / p.name)
(out / "mtp-dense.safetensors").write_bytes(b"trained dense")
if (HERE / "corrupt.txt").exists():  # a frozen file that is not the stock draft's (and recorded as such)
    (out / "config.json").write_text("{}")
files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}
manifest = {"format": "arc3-mtp-draft", "source": str(src), "trained_file": opt("--trained"),
            "trained_sha256": "0" * 64, "replaced": ["mtp.fc.weight", "mtp.norm.weight"], "copied_extra": [],
            "files": files}
(out / "arc3-draft-manifest.json").write_text(json.dumps(manifest, indent=1) + "\\n")
print("check: ok")
'''


def comparison(realized=0.25, greedy=0.24, kl=(0.40, 0.20), top1=(0.03, 0.02, 0.01)) -> dict:
    out = {"accept_realized": {"original": 2.75, "trained": 2.75 + realized, "delta": realized},
           "accept_greedy": {"original": 2.70, "trained": 2.70 + greedy, "delta": greedy},
           "accept_expected": {"original": 2.80, "trained": 2.95, "delta": 0.15},
           "step1_kl": {"original": kl[0], "trained": kl[1], "delta": kl[1] - kl[0]}}
    for k, d in zip((1, 2, 3), top1):
        out[f"step{k}_top1_target_full"] = {"original": 0.80, "trained": 0.80 + d, "delta": d}
    return out


def _write_logs(folder: Path, responses: int, tokens: int) -> None:
    names = sorted(json.loads(B.LOGS_MANIFEST.read_text())["files"])
    for p in folder.glob("*"):
        p.unlink()
    per, rest = divmod(responses, len(names))
    left = tokens
    for i, name in enumerate(names):
        n = per + (1 if i < rest else 0)
        lines = []
        for j in range(n):
            t = left if (i == len(names) - 1 and j == n - 1) else 1989
            left -= t
            lines.append(json.dumps({"event": "request", "messages": []}))
            lines.append(json.dumps({"event": "response", "usage": {"completion_tokens": t}}))
        (folder / name).write_text("\n".join(lines) + "\n")
    (folder / "stray_requests.jsonl").write_text(json.dumps({"event": "response", "usage": {}}) + "\n")


def _lean_bed(theirs, tmp_path, *, responses=2952, tokens=5871402, corrupt=False):
    bed = theirs.Bed(tmp_path)
    head = theirs.tests.STUB_COMMON
    (bed.scripts / "mtp_train.py").write_text(head + TRAIN_STUB)
    (bed.scripts / "mtp_write_draft.py").write_text(head + WRITE_STUB)
    (bed.scripts / "comparison.json").write_text(json.dumps(comparison()))
    if corrupt:
        (bed.scripts / "corrupt.txt").write_text("1")
    (bed.draft / "mtp-dense.safetensors").write_bytes(b"stock dense")
    (bed.draft / "model.safetensors.index.json").write_text("{}")
    _write_logs(bed.logs, responses, tokens)
    # the lean config: the logs are the only input (names checked by our cell), no probe dump
    bed.config["inputs"] = {"logs": {"kind": "kernel", "id": "me/arc3-full-run", "files": {}}}
    bed.config["storage"]["probe_gb"] = 0.0
    bed.config["dump"]["holdout_games"] = ["ar25", "ft09", "lp85", "r11l", "re86", "sb26", "sc25", "tn36", "tr87",
                                           "tu93", "vc33"]
    return bed


def _run_lean_cells(theirs, nb, bed, capsys):
    data, cells = nb
    s = bed.session()
    s.find_inputs(root=str(bed.input), poll_s=0.1)
    s.storage()
    args = theirs.launcher_args(sglang=sys.executable, port=bed.port)
    args[1:2] = [str(bed.scripts / "sglang.py"), "serve"]
    ns = {"SESSION": s, "arc3_sa": theirs.sa, "args": args,
          "env": dict(os.environ, ARC3_REAP_KEPT_EXPERTS="/k/kept.json"), "LOG": bed.working / "serve.log",
          "SERVED_MODEL_PORT": bed.port, "SERVED_MODEL_NAME": "flashnext", "PYTHON": sys.executable,
          "VENV": str(bed.venv), "MODEL_DIR": str(bed.tmp / "target"), "DRAFT_MODEL_DIR": str(bed.tmp / "albucino"),
          "tok": bed.generic, "precache_model_thread": None}
    launch = _index(cells, LAUNCH_ANCHOR)
    run = [cells[_index(cells, "_lean_data(SESSION)")]]
    run += [src for c, src in zip(data["cells"][launch + 1:], cells[launch + 1:]) if c["cell_type"] == "code"]
    capsys.readouterr()
    error = None
    try:
        for src in run:
            exec(src, ns)
    except Exception as exc:  # noqa: BLE001
        error = exc
    out = capsys.readouterr().out
    state = json.loads((bed.working / "session-a.json").read_text())
    return s, state, out, error


def test_the_lean_cells_run_a_session_on_their_stubs(theirs, nb, tmp_path, capsys):
    bed = _lean_bed(theirs, tmp_path)
    s, state, out, error = _run_lean_cells(theirs, nb, bed, capsys)
    assert error is None, error
    assert state["verdict"] == "complete" and s.server is None
    assert [x["step"] for x in state["steps"]] == ["inputs", "A0", "A1", "A6", "A7", "A8", "A9", "A11"]
    assert all(x["status"] == "ok" for x in state["steps"])
    # one server, the training dump (REAP kept); no probe dump, no replica check
    servers = bed.servers()
    assert len(servers) == 1 and servers[0]["argv"] == theirs.sa.dump_server_args(s.launcher["args"], reap=True)[2:]
    scripts = [c["script"] for c in bed.calls()]
    assert "mtp_probe_dump.py" not in scripts and "mtp_replica.py" not in scripts
    assert scripts[0] == "sglang_hc_dump_patch.py" and scripts[-1] == "mtp_write_draft.py"
    assert scripts.count("mtp_train.py") == 2
    for call in bed.calls():  # every command line uses only options the real script has
        if call["script"] in ("mtp_train.py", "mtp_write_draft.py", "hc_dump_driver.py", "sglang_hc_dump_patch.py"):
            real = theirs.tests._real_options(call["script"])
            used = {a for a in call["argv"] if a.startswith("--")}
            assert used <= real, (call["script"], used - real)
    # our cells' records and markers
    assert state["lean_data"]["responses"] == 2952 and state["lean_data"]["completion_tokens"] == 5871402
    assert state["lean_data"]["split"]["holdout"]["games"] == sorted(bed.config["dump"]["holdout_games"])
    assert len(state["lean_data"]["split"]["train"]["games"]) == 14
    assert state["replica_check"] == {"skipped": "lean path: no probe dump and no replica check (A2-A5, A10 dropped)"}
    assert state["lean_gate"]["verdict"] == "PASS"
    manifest = bed.working / "mtp-draft" / "arc3-draft-manifest.json"
    assert state["lean_export"]["manifest_sha256"] == _sha(manifest)
    assert state["lean_export"]["frozen_differ"] == [] and len(state["lean_export"]["frozen_identical"]) == 2
    for marker in ("ours (lean): data loaded 2952 responses (5871402 completion tokens) from 26 request logs",
                   "ours (lean): train split 14 games", "ours (lean): holdout split 11 games",
                   "ours (lean): A2-A5 and A10 dropped", "ours (lean): held-out accept_realized before 2.7500 after "
                   "3.0000 delta +0.2500", "ours (lean): held-out step1 kl 0.4000 -> 0.2000",
                   "ours (lean): drafter gate PASS", f"ours (lean): MTP draft manifest sha256 {_sha(manifest)}",
                   "frozen files byte-identical to the stock draft 2/2", "MTP_DRAFT_MANIFEST_JSON {"):
        assert marker in out, marker
    line = next(ln for ln in out.splitlines() if ln.startswith("MTP_DRAFT_MANIFEST_JSON "))
    rebuilt = json.dumps(json.loads(line.split(" ", 1)[1]), indent=1) + "\n"  # the file, from the log line
    assert hashlib.sha256(rebuilt.encode()).hexdigest() == _sha(manifest)


def test_the_data_cell_stops_on_another_version_of_the_logs(theirs, nb, tmp_path, capsys):
    bed = _lean_bed(theirs, tmp_path, responses=2950)
    bed.config["wait_inputs_s"] = 0
    _, state, out, error = _run_lean_cells(theirs, nb, bed, capsys)
    assert isinstance(error, RuntimeError) and "another version" in str(error)
    assert state["steps"][-1]["step"] == "A0" and "lean_gate" not in state and not bed.servers()


def test_the_data_cell_stops_on_a_missing_log(theirs, nb, tmp_path, capsys):
    bed = _lean_bed(theirs, tmp_path)
    bed.config["wait_inputs_s"] = 0
    next(bed.logs.glob("sp80-*")).unlink()
    _, _, _, error = _run_lean_cells(theirs, nb, bed, capsys)
    assert isinstance(error, RuntimeError) and "sp80-589a99af_p0_requests.jsonl" in str(error)


def test_the_export_cell_stops_when_a_frozen_file_differs(theirs, nb, tmp_path, capsys):
    bed = _lean_bed(theirs, tmp_path, corrupt=True)
    _, state, _, error = _run_lean_cells(theirs, nb, bed, capsys)
    assert isinstance(error, RuntimeError) and "config.json" in str(error)
    assert state["lean_export"]["frozen_differ"] == ["config.json"]


@pytest.mark.parametrize("kw, verdict", [
    ({}, "PASS"),
    ({"realized": 0.05}, "KILL"),
    ({"realized": 0.15}, "BETWEEN"),
    ({"greedy": 0.10}, "BETWEEN"),
    ({"kl": (0.40, 0.30)}, "BETWEEN"),
    ({"top1": (0.03, -0.01, 0.01)}, "BETWEEN"),
])
def test_the_gate_cell_applies_section_15_6(nb, tmp_path, capsys, kw, verdict):
    _, cells = nb
    train = tmp_path / "mtp-train"
    train.mkdir()
    (train / "train-report.json").write_text(json.dumps({"comparison": comparison(**kw)}))
    session = SimpleNamespace(working=tmp_path, state={}, save=lambda: None)
    exec(cells[_index(cells, "_lean_gate(SESSION)")], {"SESSION": session, "_lean_json": json})
    assert session.state["lean_gate"]["verdict"] == verdict
    assert f"ours (lean): drafter gate {verdict}" in capsys.readouterr().out
