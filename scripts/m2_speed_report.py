"""Compare the serving-speed test kernels (scripts/_build_m2_speed_kernels.py).

Downloads each COMPLETE kernel's output (cached under logs/m2_speed/<variant>/) and reports:
generated tokens/s over the run, decode throughput by concurrent requests and its p90,
speculative accept length, prefix-cache reuse, KV / Mamba pool peaks, retractions,
hierarchical-cache evidence, tracebacks, and the public-game levels/score (noisy).
vLLM kernels (scripts/_build_m2_vllm_kernel.py, variants vllm-s12 / vllm-s14) are read from
vllm-openai-server.log, vllm_metrics.jsonl and the watchdog log instead (parse_vllm_* below):
same columns, plus server-side prefix hit rate, preemptions, watchdog restarts and boot probes.

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
VARIANTS = ["base", "spec4", "s12", "s14", "s14hic", "m97s12", "m96s12hic", "m97s12hic", "vllm-s12", "vllm-s14"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _build_m2_speed_kernels import layout  # noqa: E402  (variant -> kernel slug)


def slug(v: str) -> str:
    if v.startswith("vllm-"):
        from _build_m2_vllm_kernel import kernel_slug
        return kernel_slug(v[len("vllm-"):])
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


def _pct(m) -> float | None:
    return float(m.group(1)) if m else None


def parse_vllm_server_log(text: str) -> dict:
    """vLLM V1 periodic stat lines ("Avg generation throughput: X tokens/s, Running: N reqs, ...
    KV cache usage: U%, Prefix cache hit rate: H%"), SpecDecoding lines, the KV pool size, and
    watchdog restarts / CUDA faults. Intervals with nothing running are idle, not throughput."""
    by = defaultdict(list)
    tp, kv, hit, acc, waiting = [], [], [], [], []
    for ln in text.splitlines():
        if "Avg generation throughput" in ln:
            t = re.search(r"Avg generation throughput: ([\d.]+) tokens/s", ln)
            n = re.search(r"Running: (\d+) req", ln)
            if not (t and n):
                continue
            u = _pct(re.search(r"KV cache usage: ([\d.]+)%", ln))
            if u is not None:
                kv.append(u / 100)
            h = _pct(re.search(r"Prefix cache hit rate: ([\d.]+)%", ln))
            if h is not None:
                hit.append(h)
            w = re.search(r"Waiting: (\d+) req", ln)
            if w:
                waiting.append(int(w.group(1)))
            if int(n.group(1)) > 0 and float(t.group(1)) > 0:
                by[int(n.group(1))].append(float(t.group(1)))
                tp.append(float(t.group(1)))
        elif "Mean acceptance length" in ln:
            m = re.search(r"Mean acceptance length: ([\d.]+)", ln)
            if m:
                acc.append(float(m.group(1)))
    pool = re.findall(r"KV cache size: ([\d,]+) tokens", text)
    return {
        "decode_med": round(st.median(tp)) if tp else None,
        "decode_p90": round(sorted(tp)[int(0.9 * len(tp))]) if tp else None,
        "by_running": {n: round(st.median(x)) for n, x in sorted(by.items()) if len(x) >= 5},
        "running_med": st.median([n for n, x in by.items() for _ in x]) if by else None,
        "accept_med": round(st.median(acc), 2) if acc else None,
        "kv_peak": max(kv) if kv else None,
        "prefix_hit_last": hit[-1] if hit else None,
        "waiting_max": max(waiting) if waiting else None,
        "kv_pool": pool[0].replace(",", "") if pool else None,
        "preempt_lines": len(re.findall(r"preempt", text, re.I)),
        "restarts": len(re.findall(r"==== watchdog restart", text)),
        "cuda_faults": len(re.findall(r"illegal memory access|CUDA error", text)),
    }


def parse_vllm_metrics(lines) -> dict:
    """Last /metrics snapshot of the launcher's monitor (vllm_metrics.jsonl): counters are
    cumulative, so the last line is the run total."""
    last = {}
    for line in lines:
        line = line.strip()
        if line:
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    q, h = last.get("vllm:prefix_cache_queries_total"), last.get("vllm:prefix_cache_hits_total")
    drafts, acc = last.get("vllm:spec_decode_num_drafts_total"), last.get("vllm:spec_decode_num_accepted_tokens_total")
    return {
        "preempts": int(last["vllm:num_preemptions_total"]) if "vllm:num_preemptions_total" in last else None,
        "prefix_hit": round(100 * h / q, 1) if q else None,
        "accept_len": round(1 + acc / drafts, 2) if drafts else None,
        "generated": int(last["vllm:generation_tokens_total"]) if "vllm:generation_tokens_total" in last else None,
    }


PROBES = (r"PREFIX_CACHE_PROBE identical=(\w+)", r"IMAGE_TOKENS_PROBE (-?\d+)",
          r"REASONING_ECHO_PROBE harness_key=\S+ rendered=(\w+)", r"TEMPLATE_PROBE tokenizer_sha=\S+ matches_m2_pin=(\w+)",
          r"VLLM_OVERLAY_HASH_(MISMATCH|OK)", r"VLLM_SERVING (active)")


def parse_vllm_probes(log: str) -> dict:
    out = {}
    for pat in PROBES:
        m = re.search(pat, log)
        out[pat.split(" ")[0].split("(")[0].rstrip("_")] = m.group(1) if m else None
    return out


def analyse_vllm(d: Path) -> dict:
    def read(name):
        p = d / name
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    r = parse_vllm_server_log(read("vllm-openai-server.log"))
    r["restarts"] += len(re.findall(r"restarting vLLM server", read("vllm-watchdog.log")))
    met = parse_vllm_metrics(read("vllm_metrics.jsonl").splitlines())
    r["preempts"] = met["preempts"]
    r["prefix_hit"] = met["prefix_hit"] if met["prefix_hit"] is not None else r.pop("prefix_hit_last")
    r.pop("prefix_hit_last", None)
    if r["accept_med"] is None:
        r["accept_med"] = met["accept_len"]
    r["probes"] = parse_vllm_probes(notebook_log(d))
    return r


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
    if (d / "vllm-openai-server.log").exists():
        r.update(analyse_vllm(d))
    return r


def main() -> None:
    wanted = sys.argv[1:] or VARIANTS
    rows = []
    for v in wanted:
        d = fetch(v)
        if d:
            rows.append(analyse(v, d))
    cols = ["variant", "gen_tok_s", "decode_med", "decode_p90", "running_med", "accept_med", "cache_pct",
            "kv_peak", "mamba_peak", "retracts", "kv_pool", "levels", "mean_score", "tracebacks",
            "prefix_hit", "preempts", "restarts"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
    out = "\n".join(lines)
    for r in rows:
        out += f"\n\n{r['variant']}: decode by #running {r['by_running']}"
        if r["hicache_lines"]:
            out += f"\n  hicache: {r['hicache_lines']}"
        if r.get("probes"):
            out += f"\n  vLLM probes: {r['probes']}; cuda faults {r.get('cuda_faults')}; max waiting {r.get('waiting_max')}"
    print(out)
    (ROOT / "logs" / "m2_speed_report.md").write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
