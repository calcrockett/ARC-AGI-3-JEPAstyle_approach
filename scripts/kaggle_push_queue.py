"""Detached kernel push queue -- no AI, no chat session.

Kaggle allows 2 concurrent batch GPU sessions. Windows Task Scheduler runs this every 10 minutes:
it tries to push the first kernel directory listed in logs/kaggle_push_queue.txt; on success the
line is removed, on "session count" it stays for the next tick. Every attempt goes to
logs/kaggle_push_queue.log; the watcher (scripts/kaggle_watch.py) tracks the pushed kernels.

    python scripts/kaggle_push_queue.py            # one tick
    python scripts/kaggle_push_queue.py --install  # register the scheduled task
    python scripts/kaggle_push_queue.py --uninstall
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "logs" / "kaggle_push_queue.txt"
LOG = ROOT / "logs" / "kaggle_push_queue.log"
KAGGLE = ROOT / "venv" / "Scripts" / "kaggle.exe"
TASK = "ARC3PushQueue"


def log(msg: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S UTC} {msg}\n")


def tick() -> None:
    if not QUEUE.exists():
        return
    lines = [ln.strip() for ln in QUEUE.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        return
    target = lines[0]
    res = subprocess.run([str(KAGGLE), "kernels", "push", "-p", str(ROOT / target)],
                         capture_output=True, text=True)
    out = (res.stdout + res.stderr).strip().splitlines()
    last = out[-1] if out else ""
    if "successfully pushed" in last:
        log(f"PUSHED {target}: {last}")
        QUEUE.write_text("\n".join(lines[1:]) + ("\n" if lines[1:] else ""))
    elif "session count" in last:
        log(f"waiting (GPU sessions full): {target}")
    else:
        log(f"push failed for {target}: {last}")


def main() -> None:
    if "--install" in sys.argv:
        py = Path(sys.executable).with_name("pythonw.exe")
        py = py if py.exists() else Path(sys.executable)
        subprocess.run(["schtasks", "/Create", "/F", "/TN", TASK, "/SC", "MINUTE", "/MO", "10",
                        "/TR", f'"{py}" "{Path(__file__).resolve()}"'], check=True)
    elif "--uninstall" in sys.argv:
        subprocess.run(["schtasks", "/Delete", "/F", "/TN", TASK], check=True)
    else:
        try:
            tick()
        except Exception as exc:  # noqa: BLE001
            log(f"error {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
