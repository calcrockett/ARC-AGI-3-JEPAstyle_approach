#!/usr/bin/env python
"""Build arm C+D: arm C (JustAdev742's exp-081 fork) serving OUR fine-tuned MTP draft, as
calamitychasm/arc3-jad-exp081-draft (experiments/stage7_milestone2_improvements.md section 15.4 / 15.6, step L2).

    python scripts/_build_jad_exp081_draft_kernel.py --draft-manifest FILE [--expect-sha SHA256]
        # writes kaggle_submission_jad_exp081_draft/notebook/ and copies FILE to
        # kaggle_submission_jad_exp081_draft/arc3-draft-manifest.json
    python scripts/_build_jad_exp081_draft_kernel.py --draft-manifest FILE --out DIR     # elsewhere (tests)

FILE is session A's ``mtp-draft/arc3-draft-manifest.json`` exactly as written by calamitychasm/arc3-mtp-session-a
(pull it with kaggle-ops ``kernel_output`` ``print_files``, or rebuild it from the session log's
``MTP_DRAFT_MANIFEST_JSON`` line as ``json.dumps(obj, indent=1) + "\\n"``; pass the sha256 the session printed as
``--expect-sha`` to check the bytes). **There is no default and no placeholder: without a real manifest the builder
refuses to write anything.** It also refuses a manifest that is not the exporter's format (``format`` arc3-mtp-draft,
a non-empty ``replaced`` list, the sha256 of mtp-dense.safetensors and of at least one frozen file) or that carries a
``not_for_push`` / ``placeholder`` key.

The notebook is arm C's build, unchanged: THEIR builder vendored for arm C (kaggle_submission_jad_exp081/vendor/,
c7a462b, byte-identical to ccb7214's) with exp-081's flags (scripts/_build_jad_exp081_kernel.builder_args, including
our 2400 s grace), plus their own ``--draft calamitychasm/arc3-mtp-session-a --draft-manifest FILE`` (their exp-084
recipe), a new slug and a fork note. Their ``--draft`` then: drops the stock-draft model source and adds the session
kernel as a kernel source; replaces cell 4's DRAFT_MODEL_DIR with a wait of up to 600 s for the mounted
``mtp-draft/`` (either mount layout) that checks the manifest's and the dense shard's sha256 and prints
``ours: MTP draft <path> after N s (manifest <sha12>, dense <sha12>)``. Everything else is byte-identical to arm C v2
(the six patches, D', REAP-448, acceptance 0.5/0.5, ARC map, 14 streams, 25 x 121 min Save & Run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_jad_exp081_kernel as C  # noqa: E402  (arm C's builder args and vendored builder)

KDIR = ROOT / "kaggle_submission_jad_exp081_draft"
OUT = KDIR / "notebook"
OWNER = "calamitychasm"
SLUG = "arc3-jad-exp081-draft"
SESSION_KERNEL = "calamitychasm/arc3-mtp-session-a"
DENSE = "mtp-dense.safetensors"
NOTE = ("Arm C+D by calamitychasm: arm C (calamitychasm/arc3-jad-exp081, a fork of JustAdev742's exp-081; "
        "github.com/JustAdev742/Arc-Agi-3-Kaggle-comp, Apache-2.0) built with their builder and exp-081's flags plus "
        f"their --draft: the MTP draft is our own fine-tune, the output of {SESSION_KERNEL} (their session A, lean "
        "path, trained on arm C v2's request logs). Deviations from exp-081: the draft, first-request grace 2400 s "
        "(ARC3_HTTP_RETRY_INITIAL_SECONDS; theirs 900), the c7a462b builder's 55-min Save & Run fail-fast limit")


def check_manifest(path: Path, expect_sha: str | None = None) -> dict:
    """The manifest's sha256 and dense sha256, or SystemExit when it is missing, a placeholder or not the exporter's."""
    if path is None:
        raise SystemExit("--draft-manifest is required: the C+D notebook is built only from a real session-A manifest")
    path = Path(path)
    try:
        raw = path.read_bytes()
        man = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--draft-manifest {path}: unreadable ({exc!r})") from None
    sha = hashlib.sha256(raw).hexdigest()
    problems = []
    if not isinstance(man, dict):
        raise SystemExit(f"--draft-manifest {path}: not a JSON object")
    if {"not_for_push", "placeholder"} & set(man):
        problems.append("marked as a placeholder (not_for_push / placeholder)")
    if man.get("format") != "arc3-mtp-draft":
        problems.append(f"format {man.get('format')!r}, not 'arc3-mtp-draft'")
    if not (isinstance(man.get("replaced"), list) and man["replaced"]):
        problems.append("no replaced dense tensors")
    files = man.get("files") if isinstance(man.get("files"), dict) else {}
    if not re.fullmatch(r"[0-9a-f]{64}", str(files.get(DENSE, ""))):
        problems.append(f"no sha256 for {DENSE}")
    if not any(n != DENSE and re.fullmatch(r"[0-9a-f]{64}", str(s)) for n, s in files.items()):
        problems.append("no frozen file with a sha256")
    if expect_sha is not None and sha != expect_sha.lower():
        problems.append(f"sha256 {sha}, the session printed {expect_sha}")
    if problems:
        raise SystemExit(f"--draft-manifest {path}: " + "; ".join(problems))
    return {"sha256": sha, "dense_sha256": files[DENSE], "replaced": len(man["replaced"])}


def builder_args(out: Path, manifest: Path) -> list[str]:
    args = C.builder_args(out)
    for flag, value in (("--slug", SLUG), ("--note", NOTE)):
        args[args.index(flag) + 1] = value
    i = args.index("--compact")
    return args[:i] + ["--draft", SESSION_KERNEL, "--draft-manifest", str(Path(manifest).resolve())] + args[i:]


def build(manifest: Path | None, out: Path = OUT, *, expect_sha: str | None = None, his_repo: Path | None = None,
          no_apply_check: bool = False) -> dict:
    info = check_manifest(manifest, expect_sha)
    with tempfile.TemporaryDirectory() as tmp:
        args = [sys.executable, "-I", "-B", str(C.VENDOR / "scripts" / "build_franzen_nb.py"),
                *builder_args(Path(tmp), manifest)]
        if no_apply_check:
            args.append("--no-apply-check")
        else:
            args += ["--his-repo", str(his_repo or C.DEFAULT_HIS_REPO)]
        run = subprocess.run(args, cwd=C.VENDOR, capture_output=True, text=True, check=False)
        if run.returncode != 0:
            raise SystemExit(f"their builder failed ({run.returncode}):\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}")
        meta_path = Path(tmp) / "kernel-metadata.json"
        meta = json.loads(meta_path.read_text())
        if meta["id"] != f"scottmahony/{SLUG}":
            raise SystemExit(f"unexpected kernel id from their builder: {meta['id']!r}")
        meta["id"] = f"{OWNER}/{SLUG}"
        meta["title"] = SLUG
        if meta["kernel_sources"] != [SESSION_KERNEL]:
            raise SystemExit(f"unexpected kernel sources {meta['kernel_sources']}")
        meta_path.write_text(json.dumps(meta, indent=1) + "\n")
        out.mkdir(parents=True, exist_ok=True)
        for f in (f"{SLUG}.ipynb", "kernel-metadata.json"):
            shutil.copyfile(Path(tmp) / f, out / f)
    if Path(out).resolve() == OUT.resolve():
        shutil.copyfile(manifest, KDIR / "arc3-draft-manifest.json")
    print(f"built {out / (SLUG + '.ipynb')} (manifest sha256 {info['sha256'][:12]}, dense {info['dense_sha256'][:12]}, "
          f"{info['replaced']} dense tensors replaced)")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--draft-manifest", type=Path, default=None, metavar="FILE",
                    help="session A's mtp-draft/arc3-draft-manifest.json, as pulled (required)")
    ap.add_argument("--expect-sha", default=None, metavar="SHA256", help="the manifest sha256 session A printed")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--his-repo", type=Path, default=None)
    ap.add_argument("--no-apply-check", action="store_true")
    a = ap.parse_args()
    build(a.draft_manifest, a.out, expect_sha=a.expect_sha, his_repo=a.his_repo, no_apply_check=a.no_apply_check)


if __name__ == "__main__":
    main()
