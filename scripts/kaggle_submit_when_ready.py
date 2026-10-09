"""Detached, gated Kaggle submission -- no AI, no chat session.

Fired once by Windows Task Scheduler (see --arm). It:
  1. waits (up to --wait-hours) for the kernel's check run to reach COMPLETE;
  2. downloads its output and applies the gate: the log must show the expected
     install marker(s), no traceback outside the known vLLM/SGLang teardown, no
     game ending in an error state, and (if given) a JSON counters file whose
     "errors" field is 0;
  3. submits the kernel version, retrying on the daily-quota error;
  4. appends every step to logs/kaggle_submit.log. The watcher
     (scripts/kaggle_watch.py) then records the score when it lands.

    python scripts/kaggle_submit_when_ready.py --kernel OWNER/SLUG --version 1 \
        --message "..." --marker "LEVEL_MEMORY installed" --counters level_memory_summary.json
    (--marker and --counters may each be repeated; every counters file must report errors == 0)
    python scripts/kaggle_submit_when_ready.py --arm "2026-10-03 00:02" [same args]   # schedule (UTC)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "kaggle_submit.log"
COMPETITION = "arc-prize-2026-arc-agi-3"
TEARDOWN_OK = re.compile(r"teardown|bounded terminal gate|serving_teardown", re.I)


def log(msg: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    line = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S UTC} {msg}"
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def log_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        return "".join(e.get("data", "") for e in json.loads(raw))
    except Exception:  # noqa: BLE001
        return raw


def gate(out: Path, markers: list[str], counters) -> list[str]:
    problems = []
    logs = list(out.glob("*.log"))
    if not logs:
        return ["no notebook log in the output"]
    text = log_text(logs[0])
    for m in markers:
        if m not in text:
            problems.append(f"marker missing: {m!r}")
    for tb in re.finditer(r"Traceback \(most recent call last\):", text):
        window = text[tb.start(): tb.start() + 1500]
        if not TEARDOWN_OK.search(window):
            problems.append("traceback outside teardown: " + window[:200].replace("\n", " | "))
    bench = out / "benchmark.json"
    if bench.exists():
        states = [r.get("state") for r in json.loads(bench.read_text()).get("game_runs", [])]
        bad = [s for s in states if s not in ("won", "gave_up", "cancelled")]
        if not states:
            problems.append("benchmark.json has no game runs")
        if bad:
            problems.append(f"games in bad states: {bad}")
    else:
        problems.append("no benchmark.json")
    for name in ([counters] if isinstance(counters, str) else list(counters or [])):
        f = out / name
        if not f.exists():
            problems.append(f"counters file missing: {name}")
        else:
            c = json.loads(f.read_text())
            if c.get("errors", 0) != 0:
                problems.append(f"counters report errors ({name}): {c}")
            else:
                log(f"counters ({name}): {c}")
    return problems


def run(args) -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    deadline = time.time() + args.wait_hours * 3600
    while True:
        status = str(getattr(api.kernels_status(args.kernel), "status", "")).split(".")[-1]
        if status == "COMPLETE":
            break
        if status in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"):
            log(f"ABORT: kernel {args.kernel} check run ended {status}; not submitting")
            return 1
        if time.time() > deadline:
            log(f"ABORT: kernel still {status} after {args.wait_hours} h; not submitting")
            return 1
        time.sleep(300)
    out = Path(tempfile.mkdtemp(prefix="kaggle_gate_"))
    api.kernels_output(args.kernel, path=str(out))
    problems = gate(out, args.marker, args.counters)
    if problems:
        log("ABORT: gate failed -> " + " ; ".join(problems))
        return 1
    log(f"gate passed for {args.kernel} (output in {out})")
    for attempt in range(args.retries):
        try:
            api.competition_submit_cli(file_name="submission.parquet", message=args.message,
                                       competition=COMPETITION, kernel=args.kernel, version=str(args.version))
            ref = api.competition_submissions(COMPETITION)[0]
            log(f"SUBMITTED {args.kernel} v{args.version}: ref {ref.ref} status {ref.status}")
            return 0
        except Exception as exc:  # noqa: BLE001
            body = getattr(getattr(exc, "response", None), "text", repr(exc))
            log(f"submit attempt {attempt + 1} failed: {str(body)[:300]}")
            time.sleep(600)
    log("ABORT: all submit attempts failed")
    return 1


def arm(args, when_utc: str) -> None:
    """Register a one-shot scheduled task at a UTC time (converted to local for schtasks). The
    arguments go to a job file: schtasks caps the task command at 261 characters."""
    t_utc = dt.datetime.strptime(when_utc, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
    t_local = t_utc.astimezone()
    name = "ARC3Submit_" + args.kernel.split("/")[-1]
    job = ROOT / "logs" / "submit_jobs" / f"{name}.json"
    job.parent.mkdir(parents=True, exist_ok=True)
    job.write_text(json.dumps({k: v for k, v in vars(args).items() if k not in ("arm", "job")}, indent=1))
    py = Path(sys.executable).with_name("pythonw.exe")
    py = py if py.exists() else Path(sys.executable)
    tr = f'"{py}" "{Path(__file__).resolve()}" --job {name}'
    subprocess.run(["schtasks", "/Create", "/F", "/TN", name, "/SC", "ONCE",
                    "/SD", t_local.strftime("%m/%d/%Y"), "/ST", t_local.strftime("%H:%M"), "/TR", tr], check=True)
    log(f"armed task {name} for {when_utc} UTC ({t_local:%Y-%m-%d %H:%M} local), job {job}")


def main() -> int:
    if "--job" in sys.argv:
        ref = sys.argv[sys.argv.index("--job") + 1]
        path = Path(ref) if ref.endswith(".json") else ROOT / "logs" / "submit_jobs" / f"{ref}.json"
        job = json.loads(path.read_text())
        return run(argparse.Namespace(**job))
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--message", required=True)
    ap.add_argument("--marker", action="append", default=[])
    ap.add_argument("--counters", action="append", default=[])
    ap.add_argument("--wait-hours", type=float, default=6.0)
    ap.add_argument("--retries", type=int, default=12)
    ap.add_argument("--arm")
    args = ap.parse_args()
    if args.arm:
        arm(args, args.arm)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
