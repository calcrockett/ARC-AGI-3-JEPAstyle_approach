"""Detached Kaggle watcher -- no AI, no chat session.

Run once per invocation (Windows Task Scheduler fires it every 10 minutes):
records the competition's recent submissions and the status of every kernel
listed in logs/kaggle_watch_kernels.txt, and appends any CHANGE to
logs/kaggle_watch.jsonl. logs/kaggle_watch_latest.txt always holds the
current snapshot, so anyone (a person, or a later session) can just read it.

    python scripts/kaggle_watch.py            # one poll
    python scripts/kaggle_watch.py --install  # register the scheduled task
    python scripts/kaggle_watch.py --uninstall
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
EVENTS = LOGS / "kaggle_watch.jsonl"
LATEST = LOGS / "kaggle_watch_latest.txt"
STATE = LOGS / "kaggle_watch_state.json"
KERNELS = LOGS / "kaggle_watch_kernels.txt"
COMPETITION = "arc-prize-2026-arc-agi-3"
TASK = "ARC3KaggleWatch"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def poll() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    LOGS.mkdir(exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    snap: dict[str, str] = {}
    for s in api.competition_submissions(COMPETITION)[:10]:
        snap[f"submission {s.ref}"] = (
            f"{str(s.status).split('.')[-1]} score={s.public_score or '-'} "
            f"date={s.date} :: {(s.description or '')[:70]}"
        )
    kernels = [k.strip() for k in KERNELS.read_text().splitlines()] if KERNELS.exists() else []
    for k in [k for k in kernels if k and not k.startswith("#")]:
        try:
            st = api.kernels_status(k)
            snap[f"kernel {k}"] = str(getattr(st, "status", st)).split(".")[-1]
        except Exception as exc:  # noqa: BLE001
            snap[f"kernel {k}"] = f"status-error {type(exc).__name__}"

    old = json.loads(STATE.read_text()) if STATE.exists() else {}
    stamp = now()
    with EVENTS.open("a", encoding="utf-8") as fh:
        for key, val in snap.items():
            if old.get(key) != val:
                fh.write(json.dumps({"t": stamp, "key": key, "old": old.get(key), "new": val}) + "\n")
    STATE.write_text(json.dumps(snap, indent=1))
    LATEST.write_text(f"polled {stamp}\n" + "\n".join(f"{k}: {v}" for k, v in snap.items()) + "\n", encoding="utf-8")


def install() -> None:
    py = Path(sys.executable).with_name("pythonw.exe")
    py = py if py.exists() else Path(sys.executable)
    cmd = f'"{py}" "{Path(__file__).resolve()}"'
    subprocess.run(["schtasks", "/Create", "/F", "/TN", TASK, "/SC", "MINUTE", "/MO", "10", "/TR", cmd], check=True)
    print(f"installed task {TASK}: every 10 min -> {cmd}")


def main() -> None:
    if "--install" in sys.argv:
        install()
    elif "--uninstall" in sys.argv:
        subprocess.run(["schtasks", "/Delete", "/F", "/TN", TASK], check=True)
    else:
        try:
            poll()
        except Exception as exc:  # noqa: BLE001
            LOGS.mkdir(exist_ok=True)
            with EVENTS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": now(), "key": "watcher", "new": f"error {type(exc).__name__}: {exc}"}) + "\n")


if __name__ == "__main__":
    main()
