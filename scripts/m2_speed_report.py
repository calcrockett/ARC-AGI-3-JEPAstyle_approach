"""Compare the serving-speed test kernels (scripts/_build_m2_speed_kernels.py).

Downloads each COMPLETE kernel's output (cached under logs/m2_speed/<variant>/) and reports:
generated tokens/s over the run, decode throughput by concurrent requests and its p90,
speculative accept length, prefix-cache reuse, KV / Mamba pool peaks, retractions,
hierarchical-cache evidence, tracebacks, and the public-game levels/score (noisy).

    python scripts/m2_speed_report.py [variant ...]
"""

from __future__ import annotations

import glob
import json
import re
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "venv" / "Scripts" / "kaggle.exe"
CACHE = ROOT / "logs" / "m2_speed"
VARIANTS = ["base", "spec4", "s12", "s14", "s14hic", "m97s12", "m96s12hic", "m97s12hic"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _build_m2_speed_kernels import layout  # noqa: E402  (variant -> kernel slug)


def slug(v: str) -> str:
    return layout(v)[2]


def fetch(v: str) -> Path | None:
    d = CACHE / v
    if (d / "summary.txt").exists():
        return d
    status = subprocess.run([str(KAGGLE), "kernels", "status", f"calamitychasm/{slug(v)}"],
                            capture_output=True, text=True).stdout
    if "COMPLETE" not in status:
        print(f"{v}: not complete ({status.strip()[-60:]})")
        return None
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(KAGGLE), "kernels", "output", f"calamitychasm/{slug(v)}", "-p", str(d)],
                   capture_output=True, text=True)
    return d if (d / "summary.txt").exists() else None


def notebook_log(d: Path) -> str:
    logs = list(d.glob("*.log"))
    if not logs:
        return ""
    raw = logs[0].read_text(encoding="utf-8", errors="replace")
    try:
        return "".join(e.get("data", "") for e in json.loads(raw))
    except Exception:  # noqa: BLE001
        return raw


def analyse(v: str, d: Path) -> dict:
    r: dict = {"variant": v}
    summ = (d / "summary.txt").read_text(encoding="utf-8", errors="replace")
    for key, pat in [("gen_tok_s", r"generated tokens/sec: ([\d.]+)"), ("mean_score", r"mean score:\s+([\d.]+)"),
                     ("actions", r"total actions: (\d+)"), ("tokens", r"total tokens:\s+(\d+)"),
                     ("duration", r"duration:\s+(.+)")]:
        m = re.search(pat, summ)
        r[key] = m.group(1).strip() if m else None
    levels = [float(x) for x in re.findall(r"levels=([\d.]+)/", summ)]
    r["levels"] = sum(levels)
    serve = (d / "serve.log").read_text(encoding="utf-8", errors="replace") if (d / "serve.log").exists() else ""
    by = defaultdict(list)
    tp, acc, fu, mu = [], [], [], []
    for ln in serve.splitlines():
        if "Decode batch" not in ln:
            continue
        try:
            n = int(re.search(r"#running-req: (\d+)", ln).group(1))
            t = float(re.search(r"gen throughput \(token/s\): ([\d.]+)", ln).group(1))
        except AttributeError:
            continue
        by[n].append(t)
        tp.append(t)
        m = re.search(r"accept len: ([\d.]+)", ln)
        if m:
            acc.append(float(m.group(1)))
        m = re.search(r"full token usage: ([\d.]+)", ln)
        if m:
            fu.append(float(m.group(1)))
        m = re.search(r"mamba usage: ([\d.]+)", ln)
        if m:
            mu.append(float(m.group(1)))
    r["decode_med"] = round(st.median(tp)) if tp else None
    r["decode_p90"] = round(sorted(tp)[int(0.9 * len(tp))]) if tp else None
    r["by_running"] = {n: round(st.median(x)) for n, x in sorted(by.items()) if len(x) >= 5}
    r["running_med"] = st.median([n for n, x in by.items() for _ in x]) if by else None
    r["accept_med"] = round(st.median(acc), 2) if acc else None
    r["kv_peak"] = max(fu) if fu else None
    r["mamba_peak"] = max(mu) if mu else None
    r["retracts"] = len(re.findall(r"retract", serve, re.I))
    r["kv_pool"] = (re.findall(r"max_total_num_tokens=(\d+)", serve) or [None])[0]
    r["hicache_lines"] = re.findall(r"(?i)hicache[^\n]{0,120}", serve)[:3]
    P = C = 0
    for f in glob.glob(str(d / "*_p0_requests.jsonl")):
        for line in open(f, encoding="utf-8"):
            if '"event": "response"' not in line:
                continue
            u = json.loads(line).get("usage") or {}
            P += u.get("prompt_tokens", 0)
            C += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    r["cache_pct"] = round(100 * C / P, 1) if P else None
    log = notebook_log(d)
    r["tracebacks"] = len(re.findall(r"Traceback \(most recent call last\)", log))
    return r


def main() -> None:
    wanted = sys.argv[1:] or VARIANTS
    rows = []
    for v in wanted:
        d = fetch(v)
        if d:
            rows.append(analyse(v, d))
    cols = ["variant", "gen_tok_s", "decode_med", "decode_p90", "running_med", "accept_med", "cache_pct",
            "kv_peak", "mamba_peak", "retracts", "kv_pool", "levels", "mean_score", "tracebacks"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
    out = "\n".join(lines)
    for r in rows:
        out += f"\n\n{r['variant']}: decode by #running {r['by_running']}"
        if r["hicache_lines"]:
            out += f"\n  hicache: {r['hicache_lines']}"
    print(out)
    (ROOT / "logs" / "m2_speed_report.md").write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
