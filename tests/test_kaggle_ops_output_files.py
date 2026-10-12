"""scripts/kaggle_ops.py kernel_output's read-only file options (added for the lean MTP session, section 15.6 L0):
list_files (names, sizes and sha256 of output files matching a glob, e.g. arm C v2's request logs) and print_files
(small text files in full, e.g. the session's arc3-draft-manifest.json). No network: the Kaggle API is a fake."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import kaggle_ops as ko  # noqa: E402
from test_kaggle_ops import FakeApi, runner  # noqa: E402

K = "calamitychasm/arc3-mtp-session-a"
MANIFEST = {"format": "arc3-mtp-draft", "replaced": ["mtp.fc.weight"], "files": {"mtp-dense.safetensors": "a" * 64}}


def write(p: Path) -> None:
    (p / "arc3-mtp-session-a.log").write_text(json.dumps([{"stream_name": "stdout", "data": "ours (lean): data\n"}]))
    (p / "ar25-0c556536_p0_requests.jsonl").write_text("x" * 10)
    (p / "bp35-0a0ad940_p0_requests.jsonl").write_text("y" * 7)
    (p / "mtp-draft").mkdir()
    (p / "mtp-draft" / "arc3-draft-manifest.json").write_text(json.dumps(MANIFEST, indent=1) + "\n")
    (p / "notes.txt").write_text("no final newline")


def test_parse_accepts_the_options_and_refuses_unsafe_paths():
    ok = {"op": "kernel_output", "kernel": K, "list_files": "*_requests.jsonl",
          "print_files": ["mtp-draft/arc3-draft-manifest.json", "session-a.json"], "artifact": False}
    ko.parse_request(json.dumps({"id": "x", "ops": [ok]}))
    for key, bad in (("list_files", "../x"), ("list_files", "/abs/*"), ("list_files", 3), ("print_files", "a.json"),
                     ("print_files", []), ("print_files", ["../etc/passwd"]), ("print_files", ["/abs"]),
                     ("print_files", [f"f{i}" for i in range(11)])):
        with pytest.raises(ko.RequestError):
            ko.parse_request(json.dumps({"id": "x", "ops": [dict(ok, **{key: bad})]}))


def test_kernel_output_lists_and_prints_files(tmp_path, capsys):
    api = FakeApi(output={K: write}, kernels=[NS(ref=K, current_version_number=1)])
    r = runner(api, tmp_path).op_kernel_output(
        {"op": "kernel_output", "kernel": K, "list_files": "*_requests.jsonl",
         "print_files": ["mtp-draft/arc3-draft-manifest.json", "notes.txt", "missing.json"], "artifact": False})
    out = capsys.readouterr().out
    assert r["result"] == "OK" and r["artifact_dir"] is None
    assert r["list_files"] == {"pattern": "*_requests.jsonl", "count": 2, "bytes": 17}
    line = next(ln for ln in out.splitlines() if ln.startswith("LIST_FILES "))
    listed = json.loads(line.split(" ", 1)[1])
    assert listed["ar25-0c556536_p0_requests.jsonl"] == {"bytes": 10,
                                                         "sha256": hashlib.sha256(b"x" * 10).hexdigest()}
    raw = (json.dumps(MANIFEST, indent=1) + "\n").encode()
    pm = r["print_files"]["mtp-draft/arc3-draft-manifest.json"]
    assert pm == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "printed": True}
    assert f"sha256 {hashlib.sha256(raw).hexdigest()}" in out and '"format": "arc3-mtp-draft"' in out
    js = next(ln for ln in out.splitlines() if ln.startswith("PRINT_FILE_JSON mtp-draft/arc3-draft-manifest.json "))
    rebuilt = json.dumps(json.loads(js.split(" ", 2)[2]), indent=1) + "\n"
    assert hashlib.sha256(rebuilt.encode()).hexdigest() == pm["sha256"]  # the file, from the log line
    assert "(no final newline)" in out and r["print_files"]["missing.json"] is None
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").rglob("*.jsonl"))  # artifact deleted
