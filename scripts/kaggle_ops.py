"""Operate Kaggle from GitHub Actions: execute the ops in .github/kaggle-ops/request.json.

The cloud dev container cannot reach kaggle.com; Actions runners can. Committing a new
request.json on the ops branch runs .github/workflows/kaggle-ops.yml, which calls this script.
Results go to the job log (each op prints a block; the last line is
`KAGGLE_OPS_RESULT {json}`) and kernel outputs to a workflow artifact.

Request: {"id": "<unique>", "ops": [op, ...]}, executed in order. Ops:
  {"op": "status", "kernels": [...optional extra owner/slug]}
  {"op": "push_kernel", "dir": "kaggle_submission_.../notebook"}
  {"op": "kernel_output", "kernel": "owner/slug", "grep": ["marker", ...], "artifact": true}
  {"op": "list_kernels", "sort_by": "scoreDescending|dateCreated|voteCount|...", "page_size": 50,
   "pages": 2, "search": "optional"}     (read-only: public kernels of the competition)
  {"op": "pull_kernel", "kernel": "owner/slug", "max_lines": 200}   (read-only: downloads the
   notebook source, prints cells that mention score/prompt/context/stream/audit/memory)
  {"op": "submit", "kernel": "owner/slug", "version": N, "message": "...",
   "markers": [...], "counters": ["level_memory_summary.json", ...],
   "require_zero": {"history_cache_summary.json": ["write_fallbacks", ...]},
   "force_gate": false}

Safety: a submit is refused if any submission is already dated today (UTC) -- there is no
override for that -- and refused when the gate fails unless force_gate is true. A push is
skipped when 2 of our GPU kernels are already running/queued (a first push rejected at the
session limit never mounts its datasets). Mutating ops (push_kernel, submit) are skipped when
the triggering push did not change request.json (REQUEST_CHANGED=false), so editing the
workflow never replays a request.

    python scripts/kaggle_ops.py [--request .github/kaggle-ops/request.json] [--out kaggle_ops_out]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_submit_when_ready as ksr  # noqa: E402

COMPETITION = ksr.COMPETITION
OWNER = "calamitychasm"
TEAM_PATTERNS = ("how bad can it go", "calamitychasm")
REQUEST = ROOT / ".github" / "kaggle-ops" / "request.json"
OPS = ("status", "push_kernel", "kernel_output", "submit", "list_kernels", "pull_kernel")
MUTATING = ("push_kernel", "submit")
BUSY = ("RUNNING", "QUEUED", "CANCEL_REQUESTED")
GPU_LIMIT = 2
# Only the milestone-2 family can hold a GPU session; the old graph-explorer / hypothesis / serving
# benchmark kernels are skipped (their status calls only fed Kaggle's rate limiter, run 37994686371).
GPU_KERNEL_PREFIXES = ("arc3-m2-", "arc3-milestone2")
STATUS_SPACING_S = 0.5
LIST_SORTS = ("hotness", "commentCount", "dateCreated", "dateRun", "relevance", "scoreAscending",
              "scoreDescending", "viewCount", "voteCount")
PULL_KEYWORDS = ("score", "prompt", "context", "stream", "audit", "memory")
PULL_MAX_LINES = 200
PULL_CELL_LINES = 40
RATE_LIMIT_BACKOFF_S = (5, 10, 20)
_RATE_LIMITED = ("429", "too many requests", "rate limit", "rate-limit", "quota exceeded")


class RequestError(ValueError):
    pass


# ---------------------------------------------------------------- request parsing

def parse_request(raw: str) -> dict:
    """Validate the request file. Raises RequestError with a readable reason."""
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RequestError(f"request is not valid JSON: {exc}") from exc
    if not isinstance(req, dict):
        raise RequestError("request must be a JSON object")
    rid = req.get("id")
    if not isinstance(rid, str) or not rid.strip():
        raise RequestError("request needs a non-empty string 'id'")
    ops = req.get("ops")
    if not isinstance(ops, list) or not ops:
        raise RequestError("request needs a non-empty 'ops' list")
    for i, op in enumerate(ops):
        if not isinstance(op, dict) or op.get("op") not in OPS:
            raise RequestError(f"ops[{i}]: 'op' must be one of {OPS}")
        kind = op["op"]
        if kind == "push_kernel":
            _check_dir(op.get("dir"), i)
        if kind in ("kernel_output", "submit"):
            k = op.get("kernel")
            if not isinstance(k, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", k):
                raise RequestError(f"ops[{i}]: 'kernel' must look like owner/slug")
        if kind == "submit":
            v = op.get("version")
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise RequestError(f"ops[{i}]: 'version' must be a positive integer")
            if not isinstance(op.get("message"), str) or not op["message"].strip():
                raise RequestError(f"ops[{i}]: 'message' is required")
            for key in ("markers", "counters"):
                if not isinstance(op.get(key, []), list):
                    raise RequestError(f"ops[{i}]: '{key}' must be a list")
            if not isinstance(op.get("require_zero", {}), dict):
                raise RequestError(f"ops[{i}]: 'require_zero' must be an object")
            if not isinstance(op.get("force_gate", False), bool):
                raise RequestError(f"ops[{i}]: 'force_gate' must be true/false")
        if kind == "list_kernels":
            _check_list_kernels(op, i)
        if kind == "pull_kernel":
            k = op.get("kernel")
            if not isinstance(k, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", k):
                raise RequestError(f"ops[{i}]: 'kernel' must look like owner/slug")
            ml = op.get("max_lines", PULL_MAX_LINES)
            if not isinstance(ml, int) or isinstance(ml, bool) or not 1 <= ml <= 1000:
                raise RequestError(f"ops[{i}]: 'max_lines' must be an integer in 1..1000")
        if kind == "kernel_output" and not isinstance(op.get("grep", []), list):
            raise RequestError(f"ops[{i}]: 'grep' must be a list")
    return req


def _check_list_kernels(op: dict, i: int) -> None:
    if op.get("sort_by", "scoreDescending") not in LIST_SORTS:
        raise RequestError(f"ops[{i}]: 'sort_by' must be one of {LIST_SORTS}")
    for key, hi in (("page_size", 100), ("pages", 10)):
        v = op.get(key, 1)
        if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= hi:
            raise RequestError(f"ops[{i}]: '{key}' must be an integer in 1..{hi}")
    if op.get("search") is not None and not isinstance(op["search"], str):
        raise RequestError(f"ops[{i}]: 'search' must be a string")


def _check_dir(d: Any, i: int) -> None:
    if not isinstance(d, str) or not d:
        raise RequestError(f"ops[{i}]: 'dir' is required")
    p = Path(d)
    if p.is_absolute() or ".." in p.parts or not p.parts[0].startswith("kaggle_submission"):
        raise RequestError(f"ops[{i}]: 'dir' must be a repo-relative kaggle_submission*/ path")


# ---------------------------------------------------------------- credentials

def setup_credentials(env=os.environ, home: Path | None = None) -> str | None:
    """Return the credential source in use, or None. KAGGLE_JSON is written to
    ~/.kaggle/kaggle.json (mode 600). Never prints a secret."""
    home = home or Path.home()
    kj = (env.get("KAGGLE_JSON") or "").strip()
    if kj:
        try:
            d = json.loads(kj)
        except json.JSONDecodeError:
            print("KAGGLE_JSON is set but is not valid JSON; ignoring it")
        else:
            if d.get("key"):
                print("::add-mask::" + str(d["key"]))
            p = home / ".kaggle" / "kaggle.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(d), encoding="utf-8")
            os.chmod(p, 0o600)
            return "KAGGLE_JSON"
    if (env.get("KAGGLE_USERNAME") or "").strip() and (env.get("KAGGLE_KEY") or "").strip():
        return "KAGGLE_USERNAME+KAGGLE_KEY"
    if (env.get("KAGGLE_API_TOKEN") or "").strip():
        return "KAGGLE_API_TOKEN"
    if (home / ".kaggle" / "kaggle.json").exists():
        return "~/.kaggle/kaggle.json"
    return None


def secret_presence(env=os.environ) -> dict[str, bool]:
    return {k: bool((env.get(k) or "").strip())
            for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_JSON", "KAGGLE_API_TOKEN")}


# ---------------------------------------------------------------- pure helpers

def enum_name(x: Any) -> str:
    return str(x).split(".")[-1] if x is not None else ""


def as_utc(d: Any) -> dt.datetime | None:
    if d is None or d == "":
        return None
    if isinstance(d, str):
        try:
            d = dt.datetime.fromisoformat(d.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(d, dt.datetime):
        return None
    return d.replace(tzinfo=dt.timezone.utc) if d.tzinfo is None else d.astimezone(dt.timezone.utc)


def submitted_today(subs: list, now: dt.datetime) -> list:
    """Submissions dated on the current UTC day. Errored submissions count too: Kaggle
    charges them against the daily quota."""
    today = now.astimezone(dt.timezone.utc).date()
    return [s for s in subs if (t := as_utc(getattr(s, "date", None))) is not None and t.date() == today]


def percentile_rank(n: int, frac: float) -> int:
    """The worst rank still inside the top `frac` of n teams (at least 1)."""
    return max(1, min(n, math.ceil(n * frac - 1e-9)))


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def leaderboard_summary(rows: list[dict], team_patterns=TEAM_PATTERNS, top: int = 40) -> dict:
    def rank(r):
        try:
            return int(r.get("Rank") or 0)
        except ValueError:
            return 0
    if rows and all(rank(r) for r in rows):
        rows = sorted(rows, key=rank)
    else:  # no usable Rank column: order by score, best first
        rows = sorted(rows, key=lambda r: -(_num(r.get("Score")) or float("-inf")))
    n = len(rows)

    def brief(r, i):
        return {"rank": rank(r) or i + 1, "score": _num(r.get("Score")), "team": r.get("TeamName", ""),
                "subs": r.get("SubmissionCount"), "last": r.get("LastSubmissionDate")}

    out: dict = {"teams": n, "first": brief(rows[0], 0) if rows else None, "bars": {}, "ours": [], "top": []}
    for label, frac in (("top1%", 0.01), ("top5%", 0.05), ("top10%", 0.10)):
        if n:
            i = percentile_rank(n, frac) - 1
            out["bars"][label] = brief(rows[i], i)
    for i, r in enumerate(rows):
        hay = (r.get("TeamName", "") + " " + r.get("TeamMemberUserNames", "")).lower()
        if any(p in hay for p in team_patterns):
            out["ours"].append(brief(r, i))
    out["top"] = [brief(r, i) for i, r in enumerate(rows[:top])]
    return out


def read_leaderboard_zip(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        return list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8", newline="")))


_NOT_PUSHED = ("permission", "denied", "not found", "404", "403", "does not exist", "cannot access")


def _never_pushed(exc: Exception) -> bool:
    """Status error that means 'this kernel does not exist / is not ours yet', not 'unknown state'."""
    return any(t in f"{type(exc).__name__} {exc}".lower() for t in _NOT_PUSHED)


def _rate_limited(exc: Exception) -> bool:
    return any(t in f"{type(exc).__name__} {exc}".lower() for t in _RATE_LIMITED)


def is_gpu_family(ref: str) -> bool:
    return ref.split("/", 1)[-1].lower().startswith(GPU_KERNEL_PREFIXES)


def claude_md_kernels(text: str) -> list[str]:
    """Incumbent plus every kernel in the CURRENT STATUS kernel tables of CLAUDE.md."""
    head = text.split("## HISTORY", 1)[0]
    found = re.findall(r"^\|\s*kernel\s*\|\s*`(calamitychasm/[\w.-]+)`", head, re.M)
    found += [f"{OWNER}/{s}" for s in re.findall(r"^\|\s*`(arc3-[\w.-]+)`\s*\|", head, re.M)]
    return list(dict.fromkeys(found))


def tracebacks(text: str) -> tuple[int, int]:
    """(all tracebacks, tracebacks outside the known serving teardown)."""
    total = outside = 0
    for tb in re.finditer(r"Traceback \(most recent call last\):", text):
        total += 1
        if not ksr.TEARDOWN_OK.search(text[tb.start(): tb.start() + 1500]):
            outside += 1
    return total, outside


def zero_problems(out: Path, require_zero: dict) -> list[str]:
    problems = []
    for name, fields in (require_zero or {}).items():
        f = out / name
        if not f.exists():
            problems.append(f"require_zero file missing: {name}")
            continue
        c = json.loads(f.read_text(encoding="utf-8"))
        for k in fields:
            if c.get(k, None) != 0:
                problems.append(f"{name}: {k} = {c.get(k)!r}, required 0")
    return problems


# ---------------------------------------------------------------- the runner

def kernel_row(k: Any) -> dict:
    """One listed kernel as a plain dict. kagglesdk getters return 0/None/"" for unset fields, so a
    zero vote count is real-or-unset; a score is reported only if the object carries one (the
    ApiKernelMetadata of kagglesdk 0.1.x has no score field, so this is usually None)."""
    when = as_utc(getattr(k, "last_run_time", None))
    score = None
    for attr in ("best_public_score", "bestPublicScore", "public_score", "score"):
        v = getattr(k, attr, None)
        if v not in (None, "", 0, 0.0):
            score = v
            break
    return {"ref": getattr(k, "ref", "") or "", "title": getattr(k, "title", "") or "",
            "author": getattr(k, "author", "") or "",
            "last_run": f"{when:%Y-%m-%d %H:%M}" if when else None,
            "votes": getattr(k, "total_votes", None), "score": score,
            "version": getattr(k, "current_version_number", None) or None}


def notebook_cells(path: Path) -> list[tuple[str, str]]:
    """(kind, source) for each cell of an .ipynb, or one 'code' cell for a plain script."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix != ".ipynb":
        return [("md" if path.suffix in (".md", ".Rmd") else "code", text)]
    try:
        nb = json.loads(text)
    except json.JSONDecodeError:
        return []
    cells = []
    for c in nb.get("cells", []):
        src = c.get("source", "")
        cells.append(("md" if c.get("cell_type") == "markdown" else "code",
                      "".join(src) if isinstance(src, list) else str(src)))
    return cells


def matching_cells(cells: list[tuple[str, str]], keywords=PULL_KEYWORDS) -> list[tuple[int, str, str, list[str]]]:
    out = []
    for idx, (kind, src) in enumerate(cells):
        low = src.lower()
        hit = [w for w in keywords if w in low]
        if hit:
            out.append((idx, kind, src, hit))
    return out



def default_cli(args: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    env = dict(os.environ, PYTHONUTF8="1")
    res = subprocess.run(["kaggle", *args], capture_output=True, text=True, cwd=cwd, env=env)
    return res.returncode, (res.stdout + res.stderr)


class Runner:
    def __init__(self, api, out_dir: Path, cli: Callable[[list[str]], tuple[int, str]] = default_cli,
                 now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
                 request_changed: bool = True, claude_md: Path = ROOT / "CLAUDE.md",
                 sleep: Callable[[float], None] = time.sleep):
        self.api, self.out_dir, self.cli, self.now = api, out_dir, cli, now
        self.sleep = sleep
        self.request_changed, self.claude_md = request_changed, claude_md
        self._mine: list | None = None
        self._versions: dict[str, tuple[int | None, str]] = {}

    # -- shared lookups
    def submissions(self) -> list:
        return list(self.api.competition_submissions(COMPETITION, page_size=50) or [])

    def my_kernels(self) -> list:
        if self._mine is None:
            got: list = []
            for page in range(1, 6):
                batch = list(self.api.kernels_list(mine=True, page=page, page_size=100) or [])
                got += batch
                if len(batch) < 100:
                    break
            self._mine = got
        return self._mine

    def _listed(self) -> list | None:
        try:
            return self.my_kernels()
        except Exception:  # noqa: BLE001
            return None

    def current_version(self, kernel: str) -> int | None:
        """Latest version number, or None when Kaggle does not say (see ksr.kernel_version:
        kagglesdk reports an unpopulated current_version_number as 0, never a real version)."""
        key = kernel.lower()
        if key not in self._versions:
            self._versions[key] = ksr.kernel_version(self.api, kernel, self._listed())
        return self._versions[key][0]

    def status_of(self, kernel: str) -> tuple[str, str]:
        st = self.api.kernels_status(kernel)
        return enum_name(getattr(st, "status", "")), (getattr(st, "failure_message", "") or "")

    def candidate_kernels(self, extra: list[str] | None = None) -> list[str]:
        """Every kernel that could hold a GPU session: the incumbent plus the CLAUDE.md tables (the
        same list the status op reports), any ``extra`` refs (the kernel being pushed), and every
        ``arc3-m2-*`` / ``arc3-milestone2*`` kernel the listing knows about. Old non-m2 kernels are
        not checked."""
        try:
            found = claude_md_kernels(self.claude_md.read_text(encoding="utf-8"))
        except OSError:
            found = []
        listed = self._listed() or []
        found += [r for r in (getattr(k, "ref", "") or "" for k in listed)
                  if is_gpu_family(r)]
        found += list(extra or [])
        seen: dict[str, str] = {}
        for ref in found:
            seen.setdefault(ref.lower(), ref)
        return list(seen.values())

    def status_with_retry(self, kernel: str) -> tuple[str, str]:
        """``status_of`` retried with backoff (RATE_LIMIT_BACKOFF_S) when Kaggle answers 429."""
        for delay in (*RATE_LIMIT_BACKOFF_S, None):
            try:
                return self.status_of(kernel)
            except Exception as exc:  # noqa: BLE001
                if delay is None or not _rate_limited(exc):
                    raise
                print(f"  busy-check {kernel}: rate limited (429), retrying in {delay}s")
                self.sleep(delay)
        raise AssertionError("unreachable")

    def busy_gpu_kernels(self, extra: list[str] | None = None) -> list[str]:
        """Our GPU kernels currently running/queued, decided by ``kernels_status`` per kernel (the call
        the status op uses). NOT by ``kernels_list`` fields: kagglesdk's ``enable_gpu`` getter returns
        ``_enable_gpu or False`` and the list endpoint leaves it unset, so a filter on it dropped every
        kernel (Actions run 37973474712 pushed with ``busy_before: []`` while turbo-tail was RUNNING).
        A status error that is not permission-denied/not-found on a kernel that has been pushed (it is
        in the listing, or the listing is unavailable) is counted busy, to be conservative.
        Prints the evidence for every candidate."""
        listing = self._listed()
        pushed = None if listing is None else {(getattr(k, "ref", "") or "").lower() for k in listing}
        busy = []
        unknown = []
        for n, ref in enumerate(self.candidate_kernels(extra)):
            if n:
                self.sleep(STATUS_SPACING_S)
            try:
                status, _ = self.status_with_retry(ref)
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                if _never_pushed(exc):
                    print(f"  busy-check {ref}: not accessible / never pushed -> ignored ({msg[:90]})")
                elif pushed is not None and ref.lower() not in pushed:
                    print(f"  busy-check {ref}: status error, not in our listing -> ignored ({msg[:90]})")
                else:
                    print(f"  busy-check {ref}: status UNKNOWN ({msg[:90]}) -> counted busy (conservative)")
                    busy.append(f"{ref} UNKNOWN")
                    unknown.append(ref)
                continue
            print(f"  busy-check {ref}: {status}" + (" -> BUSY" if status in BUSY else ""))
            if status in BUSY:
                busy.append(f"{ref} {status}")
        if unknown:
            print(f"  busy-check: status still UNKNOWN after retries for {len(unknown)} kernel(s): {unknown}")
        return busy

    # -- ops
    def op_status(self, op: dict) -> dict:
        res: dict = {"op": "status", "result": "OK"}
        print("== submissions (latest 25) ==")
        try:
            subs = self.submissions()
            for s in subs[:25]:
                t = as_utc(s.date)
                when = f"{t:%Y-%m-%d %H:%M}Z" if t else str(s.date)
                err = getattr(s, "error_description", None)
                print(f"  {when}  ref {s.ref}  {enum_name(s.status):10s} public={s.public_score or '-'}  "
                      f"{(s.description or '')[:70]}" + (f"  ERR: {err[:120]}" if err else ""))
            today = submitted_today(subs, self.now())
            print(f"  submitted today (UTC {self.now():%Y-%m-%d}): {len(today)}")
            res["submitted_today"] = len(today)
            res["latest_submission"] = {"ref": subs[0].ref, "status": enum_name(subs[0].status),
                                        "score": subs[0].public_score} if subs else None
        except Exception as exc:  # noqa: BLE001
            print(f"  submissions error: {type(exc).__name__}: {exc}")
            res["result"] = "PARTIAL"
        print("== leaderboard ==")
        try:
            tmp = Path(tempfile.mkdtemp(prefix="lb_"))
            self.api.competition_leaderboard_download(COMPETITION, str(tmp))
            zips = list(tmp.glob("*.zip"))
            rows = read_leaderboard_zip(zips[0]) if zips else list(csv.DictReader(next(tmp.glob("*.csv")).open(encoding="utf-8")))
            lb = leaderboard_summary(rows)
            print(f"  teams: {lb['teams']}  #1: {lb['first']}")
            for label, r in lb["bars"].items():
                print(f"  {label} bar: rank {r['rank']} score {r['score']} ({r['team']})")
            for r in lb["ours"] or [{"team": "NOT FOUND"}]:
                print(f"  OUR TEAM: {r}")
            print("  top 40:")
            for r in lb["top"]:
                print(f"    {r['rank']:>5} {r['score']!s:>8}  {r['team'][:40]}  subs={r['subs']}")
            res["leaderboard"] = {k: lb[k] for k in ("teams", "first", "bars", "ours")}
        except Exception as exc:  # noqa: BLE001
            print(f"  leaderboard error: {type(exc).__name__}: {exc}")
            res["result"] = "PARTIAL"
        print("== kernels ==")
        try:
            kernels = claude_md_kernels(self.claude_md.read_text(encoding="utf-8"))
        except OSError:
            kernels = []
        kernels = list(dict.fromkeys(kernels + list(op.get("kernels") or [])))
        res["kernels"] = {}
        for k in kernels:
            try:
                status, fail = self.status_of(k)
            except Exception as exc:  # noqa: BLE001
                status, fail = f"status-error {type(exc).__name__}", str(exc)[:120]
            v = self.current_version(k)
            vs = f"v{v}" if v is not None else "v?"
            print(f"  {k:55s} {vs:4s}  {status}" + (f"  ({fail[:120]})" if fail else ""))
            res["kernels"][k] = f"{vs} {status}"
        return res

    def op_push_kernel(self, op: dict) -> dict:
        d = ROOT / op["dir"]
        res: dict = {"op": "push_kernel", "dir": op["dir"]}
        if not (d / "kernel-metadata.json").exists():
            return {**res, "result": "ERROR", "reason": "no kernel-metadata.json in dir"}
        try:
            meta_id = json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8")).get("id")
        except (OSError, ValueError):
            meta_id = None
        busy = self.busy_gpu_kernels([meta_id] if meta_id else None)
        print(f"  running/queued GPU kernels: {busy or 'none'}")
        if len(busy) >= GPU_LIMIT:
            return {**res, "result": "SKIPPED_GPU_BUSY", "busy": busy}
        rc, out = self.cli(["kernels", "push", "-p", str(d)])
        last = [ln for ln in out.strip().splitlines() if ln.strip()][-1:] or [""]
        print("  " + "\n  ".join(out.strip().splitlines()[-8:]))
        m = re.search(r"version (\d+)", last[0])
        if "successfully pushed" in last[0].lower():
            return {**res, "result": "PUSHED", "version": int(m.group(1)) if m else None, "busy_before": busy}
        if "session count" in last[0].lower():
            return {**res, "result": "SKIPPED_GPU_BUSY", "reason": last[0][:200]}
        return {**res, "result": "ERROR", "reason": last[0][:300], "rc": rc}

    def op_kernel_output(self, op: dict) -> dict:
        k = op["kernel"]
        res: dict = {"op": "kernel_output", "kernel": k}
        status, fail = self.status_of(k)
        res.update(status=status, version=self.current_version(k))
        print(f"  status {status} version {res['version']}" + (f" failure: {fail[:200]}" if fail else ""))
        dest = self.out_dir / re.sub(r"[^\w.-]+", "_", k)
        dest.mkdir(parents=True, exist_ok=True)
        self.api.kernels_output(k, path=str(dest), force=True)
        res.update(digest(dest, op.get("grep") or []))
        res["artifact_dir"] = str(dest)
        if op.get("artifact", True) is False:
            res["artifact_dir"] = None
            for f in sorted(dest.rglob("*"), reverse=True):
                f.unlink() if f.is_file() else f.rmdir()
        res["result"] = "OK"
        return res

    def op_list_kernels(self, op: dict) -> dict:
        sort_by = op.get("sort_by", "scoreDescending")
        size, pages = op.get("page_size", 50), op.get("pages", 1)
        res: dict = {"op": "list_kernels", "sort_by": sort_by, "competition": COMPETITION}
        rows: list[dict] = []
        for page in range(1, pages + 1):
            kw: dict = dict(competition=COMPETITION, sort_by=sort_by, page=page, page_size=size)
            if op.get("search"):
                kw["search"] = op["search"]
            batch = list(self.api.kernels_list(**kw) or [])
            rows += [kernel_row(k) for k in batch if k is not None]
            if len(batch) < size:
                break
        print(f"  {len(rows)} kernels, sort_by={sort_by}" + (f", search={op['search']!r}" if op.get("search") else ""))
        print(f"  {'#':>3} {'last run (UTC)':16} {'votes':>5} {'score':>7}  ref | title | author")
        for n, r in enumerate(rows, 1):
            print(f"  {n:>3} {r['last_run'] or '?':16} {r['votes'] if r['votes'] is not None else '?':>5} "
                  f"{str(r['score'] if r['score'] is not None else '-'):>7}  {r['ref']} | {r['title'][:70]} | {r['author']}")
        res.update(count=len(rows), kernels=rows, result="OK")
        return res

    def op_pull_kernel(self, op: dict) -> dict:
        k = op["kernel"]
        budget = op.get("max_lines", PULL_MAX_LINES)
        res: dict = {"op": "pull_kernel", "kernel": k}
        dest = self.out_dir / ("pull_" + re.sub(r"[^\w.-]+", "_", k))
        dest.mkdir(parents=True, exist_ok=True)
        self.api.kernels_pull(k, path=str(dest), metadata=True, quiet=True)
        files = sorted(p for p in dest.rglob("*") if p.is_file())
        res["files"] = [str(p.relative_to(dest)) for p in files]
        code = [p for p in files if p.suffix in (".ipynb", ".py", ".R", ".Rmd", ".md")]
        print(f"  pulled {len(files)} files: {res['files'][:20]}")
        shown, matched = 0, 0
        for p in code:
            cells = notebook_cells(p)
            hits = matching_cells(cells)
            matched += len(hits)
            print(f"  == {p.name}: {len(cells)} cells, {len(hits)} mention {PULL_KEYWORDS}")
            for idx, kind, src, words in hits:
                if shown >= budget:
                    break
                lines = src.splitlines()[:PULL_CELL_LINES]
                lines = lines[:budget - shown]
                shown += len(lines) + 1
                print(f"  --- cell {idx} [{kind}] matches {','.join(words)}")
                for ln in lines:
                    print("    " + ln[:200])
        res.update(matched_cells=matched, printed_lines=shown, result="OK")
        return res

    def op_submit(self, op: dict, submitted_this_run: bool) -> dict:
        k, v = op["kernel"], op["version"]
        res: dict = {"op": "submit", "kernel": k, "version": v}
        if submitted_this_run:
            return {**res, "result": "SKIPPED_ALREADY_SUBMITTED_TODAY", "reason": "an earlier op in this run submitted"}
        try:
            subs = self.submissions()
        except Exception as exc:  # noqa: BLE001
            return {**res, "result": "SKIPPED_CANNOT_VERIFY_DAILY_LIMIT", "reason": f"{type(exc).__name__}: {exc}"[:200]}
        today = submitted_today(subs, self.now())
        if today:
            s = today[0]
            print(f"  already submitted today: ref {s.ref} {enum_name(s.status)} {(s.description or '')[:60]}")
            return {**res, "result": "SKIPPED_ALREADY_SUBMITTED_TODAY", "today_refs": [str(t.ref) for t in today]}
        verdict, problems, detail = ksr.version_check(self.api, k, v, self._listed())
        res["version_check"] = verdict
        if verdict == "VERSION_UNKNOWN":
            warn = (f"VERSION_UNKNOWN: {detail}; requested v{v}. Gating on the latest check run's markers, "
                    f"tracebacks, counters and game states only -- confirm by hand that it is v{v}'s run.")
            print("  " + "!" * 70 + f"\n  {warn}\n  " + "!" * 70)
            res["warnings"] = [warn]
        else:
            print(f"  {verdict}: {detail}")
        dest = self.out_dir / ("gate_" + re.sub(r"[^\w.-]+", "_", k))
        dest.mkdir(parents=True, exist_ok=True)
        gate_problems, out = ksr.check_ready(self.api, k, list(op.get("markers") or []),
                                             list(op.get("counters") or []), out=dest)
        problems += gate_problems
        if out is not None:
            problems += zero_problems(out, op.get("require_zero") or {})
            for name in op.get("counters") or []:
                f = out / name
                if f.exists():
                    print(f"  counters {name}: {f.read_text(encoding='utf-8')[:600]}")
        for p in problems:
            print(f"  GATE: {p}")
        res["gate_problems"] = problems
        if problems and not op.get("force_gate", False):
            return {**res, "result": "GATE_FAILED"}
        if problems:
            print("  force_gate=true: submitting despite the gate problems above")
        try:
            msg = self.api.competition_submit_cli(file_name="submission.parquet", message=op["message"],
                                                  competition=COMPETITION, kernel=k, version=str(v))
        except Exception as exc:  # noqa: BLE001
            resp = getattr(exc, "response", None)
            body = getattr(resp, "text", None) or repr(exc)
            code = getattr(resp, "status_code", None)
            print(f"  SUBMIT FAILED: HTTP {code}: {body[:800]}")
            return {**res, "result": "ERROR", "http_status": code, "error_body": str(body)[:800]}
        print(f"  submit response: {msg}")
        try:
            latest = self.submissions()[0]
            res["ref"], res["status"] = str(latest.ref), enum_name(latest.status)
            print(f"  latest submission: ref {latest.ref} {enum_name(latest.status)} {latest.description}")
        except Exception:  # noqa: BLE001
            pass
        return {**res, "result": "SUBMITTED", "response": str(msg)[:300]}

    def run(self, req: dict) -> list[dict]:
        results = []
        submitted = False
        for i, op in enumerate(req["ops"]):
            kind = op["op"]
            print(f"\n######## op {i}: {json.dumps(op)[:300]}")
            if kind in MUTATING and not self.request_changed:
                r = {"op": kind, "result": "SKIPPED_REQUEST_UNCHANGED",
                     "reason": "this run was not triggered by a change to request.json"}
            else:
                try:
                    if kind == "status":
                        r = self.op_status(op)
                    elif kind == "push_kernel":
                        r = self.op_push_kernel(op)
                    elif kind == "kernel_output":
                        r = self.op_kernel_output(op)
                    elif kind == "list_kernels":
                        r = self.op_list_kernels(op)
                    elif kind == "pull_kernel":
                        r = self.op_pull_kernel(op)
                    else:
                        r = self.op_submit(op, submitted)
                        submitted = submitted or r.get("result") == "SUBMITTED"
                except Exception as exc:  # noqa: BLE001
                    r = {"op": kind, "result": "ERROR", "reason": f"{type(exc).__name__}: {exc}"[:400]}
            print(f"  -> {r.get('result')}")
            results.append(r)
        return results


def digest(out: Path, markers: list[str]) -> dict:
    """Concise summary of a downloaded kernel output directory."""
    d: dict = {"files": len([p for p in out.rglob("*") if p.is_file()])}
    nb = ksr.notebook_log(out)
    text = ksr.log_text(nb) if nb else ""
    d["notebook_log"] = nb.name if nb else None
    d["markers"] = {m: (m in text) for m in markers}
    d["tracebacks"], d["tracebacks_outside_teardown"] = tracebacks(text)
    bench = out / "benchmark.json"
    if bench.exists():
        try:
            runs = json.loads(bench.read_text(encoding="utf-8")).get("game_runs", [])
            d["game_states"] = dict(Counter(str(r.get("state")) for r in runs))
        except (json.JSONDecodeError, AttributeError):
            d["game_states"] = "unreadable"
    d["counters"] = {}
    for f in sorted(out.glob("*_summary.json")):
        try:
            d["counters"][f.name] = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            d["counters"][f.name] = "unreadable"
    if (out / "summary.txt").exists():
        try:
            import m2_speed_report as rep
            m = rep.analyse(out.name, out)
            d["speed"] = {k: m.get(k) for k in ("gen_tok_s", "decode_med", "decode_p90", "running_med",
                                                 "accept_med", "cache_pct", "kv_peak", "mamba_peak", "retracts",
                                                 "kv_pool", "levels", "mean_score", "actions", "duration",
                                                 "prefix_hit", "preempts", "restarts")}
            d["speed"]["by_running"] = m.get("by_running")
        except Exception as exc:  # noqa: BLE001
            d["speed"] = f"error {type(exc).__name__}: {exc}"[:200]
    if any(out.glob("*_requests.jsonl")):
        try:
            import m2_speed_report as rep
            d["request_log"] = rep.request_log_metrics(out)
        except Exception as exc:  # noqa: BLE001
            d["request_log"] = f"error {type(exc).__name__}: {exc}"[:200]
    tail = [ln for ln in text.strip().splitlines() if ln.strip()][-15:]
    print(f"  files {d['files']}, notebook log {d['notebook_log']}")
    print(f"  markers: {d['markers']}")
    print(f"  tracebacks: {d['tracebacks']} ({d['tracebacks_outside_teardown']} outside teardown)")
    print(f"  game states: {d.get('game_states')}")
    for name, c in d["counters"].items():
        print(f"  {name}: {json.dumps(c)[:800]}")
    if "speed" in d:
        print(f"  speed: {json.dumps(d['speed'])[:900]}")
    if "request_log" in d:
        print(f"  request log: {json.dumps(d['request_log'])[:600]}")
    print("  notebook log tail:\n    " + "\n    ".join(t[:200] for t in tail))
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", default=str(REQUEST))
    ap.add_argument("--out", default=str(ROOT / "kaggle_ops_out"))
    args = ap.parse_args(argv)
    try:
        req = parse_request(Path(args.request).read_text(encoding="utf-8"))
    except (OSError, RequestError) as exc:
        print(f"BAD REQUEST: {exc}")
        print("KAGGLE_OPS_RESULT " + json.dumps({"id": None, "result": "BAD_REQUEST", "reason": str(exc)}))
        return 1
    print(f"request id: {req['id']} ({len(req['ops'])} ops)")
    changed = os.environ.get("REQUEST_CHANGED", "true").lower() != "false"
    print(f"request changed in the triggering push: {changed}")
    source = setup_credentials()
    if source is None:
        missing = [k for k, v in secret_presence().items() if not v]
        print("NO_CREDENTIALS: no Kaggle credentials found. Add repository secrets "
              "KAGGLE_USERNAME + KAGGLE_KEY, or KAGGLE_JSON (the kaggle.json contents), or KAGGLE_API_TOKEN. "
              f"Unset: {', '.join(missing)}")
        print("KAGGLE_OPS_RESULT " + json.dumps({"id": req["id"], "result": "NO_CREDENTIALS",
                                                  "secrets": secret_presence()}))
        return 0
    print(f"credentials from {source}")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit:
        print("KAGGLE_OPS_RESULT " + json.dumps({"id": req["id"], "result": "AUTH_FAILED"}))
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = Runner(api, out, request_changed=changed).run(req)
    summary = {"id": req["id"], "at": f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}", "ops": results}
    (out / "result.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print("\nKAGGLE_OPS_RESULT " + json.dumps(summary, default=str))
    return 1 if any(r.get("result") == "ERROR" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
