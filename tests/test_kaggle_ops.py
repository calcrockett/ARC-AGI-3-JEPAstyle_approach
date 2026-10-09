"""scripts/kaggle_ops.py: the GitHub Actions Kaggle operator. No network: the Kaggle API and CLI
are fakes."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_ops as ko  # noqa: E402
import kaggle_submit_when_ready as ksr  # noqa: E402

NOW = dt.datetime(2026, 10, 9, 12, 0, tzinfo=dt.timezone.utc)
K = "calamitychasm/arc3-m2-level-memory"
MARKERS = ["LEVEL_MEMORY installed", "priority gate active"]


def sub(ref, when, status="COMPLETE", score="30.0", desc="d"):
    return NS(ref=ref, date=when, status=f"SubmissionStatus.{status}", public_score=score,
              description=desc, error_description=None)


def write_output(path: Path, *, markers=MARKERS, states=("won", "gave_up"), errors=0, traceback=None,
                 extra_logs=True):
    text = "\n".join(markers) + "\n" + (traceback or "")
    (path / "arc3-m2-level-memory.log").write_text(json.dumps([{"stream_name": "stdout", "data": text}]))
    if extra_logs:
        (path / "serve.log").write_text("Decode batch ... no markers here\n")
    (path / "benchmark.json").write_text(json.dumps({"game_runs": [{"state": s} for s in states]}))
    (path / "level_memory_summary.json").write_text(json.dumps({"errors": errors, "blocks": 15}))


class FakeApi:
    def __init__(self, subs=(), kernels=(), statuses=None, output=None, submit_exc=None):
        self.subs = list(subs)
        self.kernels = list(kernels)
        self.statuses = statuses or {}
        self.output = output or {}
        self.submit_exc = submit_exc
        self.submitted = []

    def competition_submissions(self, comp, page_size=20):
        return list(self.subs)

    def kernels_list(self, mine=False, page=1, page_size=20, search=None):
        if search is not None:
            return [k for k in self.kernels if search in (getattr(k, "ref", "") or "")]
        return self.kernels if page == 1 else []

    def kernels_status(self, kernel):
        return NS(status=f"KernelWorkerStatus.{self.statuses.get(kernel, 'COMPLETE')}", failure_message="")

    def kernels_output(self, kernel, path, force=False):
        self.output.get(kernel, lambda p: None)(Path(path))

    def competition_submit_cli(self, **kw):
        if self.submit_exc:
            raise self.submit_exc
        self.submitted.append(kw)
        self.subs.insert(0, sub("999", NOW, status="PENDING", score=None, desc=kw["message"]))
        return "Successfully submitted"


def runner(api, tmp_path, cli=None, changed=True):
    return ko.Runner(api, tmp_path / "out", cli=cli or (lambda a: (0, "")), now=lambda: NOW,
                     request_changed=changed, claude_md=ROOT / "CLAUDE.md")


SUBMIT = {"op": "submit", "kernel": K, "version": 1, "message": "m2 level memory draw 5",
          "markers": MARKERS, "counters": ["level_memory_summary.json"]}


# ---------------------------------------------------------------- request parsing

def test_parse_valid_request_and_committed_request_file():
    req = ko.parse_request(json.dumps({"id": "x1", "ops": [{"op": "status"}, SUBMIT,
                                                          {"op": "push_kernel", "dir": "kaggle_submission_m2_lm_histcache/notebook"},
                                                          {"op": "kernel_output", "kernel": K, "grep": ["a"]}]}))
    assert [o["op"] for o in req["ops"]] == ["status", "submit", "push_kernel", "kernel_output"]
    ko.parse_request((ROOT / ".github" / "kaggle-ops" / "request.json").read_text())


@pytest.mark.parametrize("raw, why", [
    ("not json", "valid JSON"),
    ('{"ops": [{"op": "status"}]}', "'id'"),
    ('{"id": "a", "ops": []}', "'ops'"),
    ('{"id": "a", "ops": [{"op": "delete_everything"}]}', "'op'"),
    ('{"id": "a", "ops": [{"op": "push_kernel", "dir": "../etc"}]}', "'dir'"),
    ('{"id": "a", "ops": [{"op": "push_kernel", "dir": "/abs/kaggle_submission_x"}]}', "'dir'"),
    ('{"id": "a", "ops": [{"op": "submit", "kernel": "o/s", "version": "1", "message": "m"}]}', "'version'"),
    ('{"id": "a", "ops": [{"op": "submit", "kernel": "o/s", "version": 1}]}', "'message'"),
    ('{"id": "a", "ops": [{"op": "submit", "kernel": "noslash", "version": 1, "message": "m"}]}', "'kernel'"),
    ('{"id": "a", "ops": [{"op": "submit", "kernel": "o/s", "version": 1, "message": "m", "force_gate": "yes"}]}',
     "force_gate"),
])
def test_parse_rejects_bad_requests(raw, why):
    with pytest.raises(ko.RequestError, match=why):
        ko.parse_request(raw)


# ---------------------------------------------------------------- credentials

def test_credentials_from_kaggle_json_writes_private_file(tmp_path, capsys):
    env = {"KAGGLE_JSON": json.dumps({"username": "u", "key": "sekrit"})}
    assert ko.setup_credentials(env, home=tmp_path) == "KAGGLE_JSON"
    p = tmp_path / ".kaggle" / "kaggle.json"
    assert json.loads(p.read_text())["key"] == "sekrit"
    assert p.stat().st_mode & 0o777 == 0o600
    out = capsys.readouterr().out
    assert out.count("sekrit") == 1 and "::add-mask::sekrit" in out  # only the mask command


def test_credentials_other_sources_and_none(tmp_path):
    assert ko.setup_credentials({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}, home=tmp_path) == "KAGGLE_USERNAME+KAGGLE_KEY"
    assert ko.setup_credentials({"KAGGLE_API_TOKEN": "t"}, home=tmp_path) == "KAGGLE_API_TOKEN"
    assert ko.setup_credentials({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": ""}, home=tmp_path) is None


def test_main_without_credentials_exits_cleanly(tmp_path, monkeypatch, capsys):
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_JSON", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(ko.Path, "home", classmethod(lambda cls: tmp_path))
    req = tmp_path / "r.json"
    req.write_text('{"id": "init-status", "ops": [{"op": "status"}]}')
    assert ko.main(["--request", str(req), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "NO_CREDENTIALS" in out
    last = [ln for ln in out.splitlines() if ln.startswith("KAGGLE_OPS_RESULT ")][-1]
    assert json.loads(last.split(" ", 1)[1])["result"] == "NO_CREDENTIALS"


# ---------------------------------------------------------------- daily limit

def test_submitted_today_uses_utc_day_and_counts_errors():
    subs = [sub("1", dt.datetime(2026, 10, 9, 0, 1), status="ERROR"),        # naive = UTC
            sub("2", dt.datetime(2026, 10, 8, 23, 59)),
            sub("3", "2026-10-08T23:30:00-02:00")]                           # = 10-09 01:30 UTC
    assert [s.ref for s in ko.submitted_today(subs, NOW)] == ["1", "3"]


def test_submit_refused_when_already_submitted_today(tmp_path):
    api = FakeApi(subs=[sub("56900000", dt.datetime(2026, 10, 9, 0, 3))], output={K: write_output})
    r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
    assert r["result"] == "SKIPPED_ALREADY_SUBMITTED_TODAY" and api.submitted == []


def test_submit_refused_even_with_force_gate_when_already_submitted(tmp_path):
    api = FakeApi(subs=[sub("1", NOW - dt.timedelta(hours=2))])
    r = runner(api, tmp_path).op_submit({**SUBMIT, "force_gate": True}, False)
    assert r["result"] == "SKIPPED_ALREADY_SUBMITTED_TODAY" and api.submitted == []


def test_second_submit_in_one_request_is_refused(tmp_path):
    api = FakeApi(subs=[sub("1", NOW - dt.timedelta(days=1))], output={K: write_output},
                  kernels=[NS(ref=K, current_version_number=1, enable_gpu=True, last_run_time=None)])
    res = runner(api, tmp_path).run({"id": "x", "ops": [dict(SUBMIT), dict(SUBMIT)]})
    assert [r["result"] for r in res] == ["SUBMITTED", "SKIPPED_ALREADY_SUBMITTED_TODAY"]
    assert len(api.submitted) == 1


def test_submit_refused_when_submissions_cannot_be_read(tmp_path):
    api = FakeApi()
    api.competition_submissions = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
    assert r["result"] == "SKIPPED_CANNOT_VERIFY_DAILY_LIMIT" and api.submitted == []


# ---------------------------------------------------------------- gate

def test_gate_pass_submits_with_version(tmp_path):
    api = FakeApi(subs=[sub("1", NOW - dt.timedelta(days=1))], output={K: write_output},
                  kernels=[NS(ref=K, current_version_number=1, enable_gpu=True, last_run_time=None)])
    r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
    assert r["result"] == "SUBMITTED", r
    assert api.submitted == [{"file_name": "submission.parquet", "message": SUBMIT["message"],
                              "competition": ko.COMPETITION, "kernel": K, "version": "1"}]
    assert r["ref"] == "999"


@pytest.mark.parametrize("writer, expect", [
    (lambda p: write_output(p, markers=MARKERS[:1]), "marker missing"),
    (lambda p: write_output(p, errors=3), "counters report errors"),
    (lambda p: write_output(p, states=("won", "error")), "bad states"),
    (lambda p: write_output(p, traceback="Traceback (most recent call last):\n  File x\nKeyError: 1\n"),
     "traceback outside teardown"),
])
def test_gate_fail_refuses(tmp_path, writer, expect):
    api = FakeApi(subs=[], output={K: writer})
    r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
    assert r["result"] == "GATE_FAILED" and api.submitted == []
    assert any(expect in p for p in r["gate_problems"]), r["gate_problems"]


def test_gate_fails_when_check_run_not_complete_or_version_differs(tmp_path):
    api = FakeApi(statuses={K: "RUNNING"})
    r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
    assert r["result"] == "GATE_FAILED" and "not COMPLETE" in r["gate_problems"][0]
    api = FakeApi(output={K: write_output}, kernels=[NS(ref=K, current_version_number=2)])
    r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
    assert r["result"] == "GATE_FAILED" and "latest version" in r["gate_problems"][0]


def test_teardown_traceback_is_allowed_and_force_gate_overrides(tmp_path):
    tb = "Traceback (most recent call last):\n  serving_teardown\n"
    api = FakeApi(output={K: lambda p: write_output(p, traceback=tb)})
    assert runner(api, tmp_path).op_submit(dict(SUBMIT), False)["result"] == "SUBMITTED"
    api = FakeApi(output={K: lambda p: write_output(p, errors=1)})
    r = runner(api, tmp_path).op_submit({**SUBMIT, "force_gate": True}, False)
    assert r["result"] == "SUBMITTED" and r["gate_problems"]


def test_require_zero_fields(tmp_path):
    def w(p):
        write_output(p)
        (p / "history_cache_summary.json").write_text(json.dumps({"errors": 0, "view_misses": 2, "loads_stale": 0}))
    api = FakeApi(output={K: w})
    op = {**SUBMIT, "require_zero": {"history_cache_summary.json": ["view_misses", "loads_stale", "write_fallbacks"]}}
    r = runner(api, tmp_path).op_submit(op, False)
    assert r["result"] == "GATE_FAILED"
    assert any("view_misses" in p for p in r["gate_problems"])
    assert any("write_fallbacks" in p for p in r["gate_problems"])  # absent field != 0


def test_notebook_log_preferred_over_serve_log(tmp_path):
    (tmp_path / "a_serve.log").write_text("plain text server log\n")
    (tmp_path / "z_kernel.log").write_text(json.dumps([{"data": "x"}]))
    assert ksr.notebook_log(tmp_path).name == "z_kernel.log"


def test_submit_error_body_is_reported(tmp_path):
    exc = RuntimeError("400 Client Error")
    exc.response = NS(status_code=400, text='{"message": "daily Submission allowance (1)"}')
    api = FakeApi(output={K: write_output}, submit_exc=exc)
    r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
    assert r["result"] == "ERROR" and r["http_status"] == 400 and "allowance" in r["error_body"]


# ---------------------------------------------------------------- push / GPU slots

def kmeta(ref, hours_ago=1.0, gpu=True):
    return NS(ref=ref, enable_gpu=gpu, last_run_time=NOW - dt.timedelta(hours=hours_ago), current_version_number=1)


def test_push_skipped_when_two_gpu_kernels_busy(tmp_path):
    calls = []
    api = FakeApi(kernels=[kmeta("o/a"), kmeta("o/b"), kmeta("o/c")],
                  statuses={"o/a": "RUNNING", "o/b": "QUEUED", "o/c": "COMPLETE"})
    r = runner(api, tmp_path, cli=lambda a: calls.append(a) or (0, "")).op_push_kernel(
        {"op": "push_kernel", "dir": "kaggle_submission_m2_lm_histcache/notebook"})
    assert r["result"] == "SKIPPED_GPU_BUSY" and calls == []


def test_push_ignores_old_and_cpu_kernels_then_pushes(tmp_path):
    calls = []

    def cli(a):
        calls.append(a)
        return 0, "Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/x"
    api = FakeApi(kernels=[kmeta("o/a"), kmeta("o/old", hours_ago=100), kmeta("o/cpu", gpu=False)],
                  statuses={"o/a": "RUNNING", "o/old": "RUNNING", "o/cpu": "RUNNING"})
    r = runner(api, tmp_path, cli=cli).op_push_kernel({"op": "push_kernel", "dir": "kaggle_submission_m2_lm_histcache/notebook"})
    assert r["result"] == "PUSHED" and r["version"] == 1
    assert calls == [["kernels", "push", "-p", str(ROOT / "kaggle_submission_m2_lm_histcache/notebook")]]


def test_push_session_limit_message_is_a_skip(tmp_path):
    api = FakeApi()
    cli = lambda a: (1, "400 - Bad Request: Maximum batch GPU session count of 2 reached.")  # noqa: E731
    r = runner(api, tmp_path, cli=cli).op_push_kernel({"op": "push_kernel", "dir": "kaggle_submission_m2_lm_histcache/notebook"})
    assert r["result"] == "SKIPPED_GPU_BUSY"


def test_mutating_ops_skipped_when_request_unchanged(tmp_path):
    api = FakeApi(output={K: write_output})
    res = runner(api, tmp_path, changed=False).run({"id": "x", "ops": [
        dict(SUBMIT), {"op": "push_kernel", "dir": "kaggle_submission_m2_lm_histcache/notebook"}]})
    assert [r["result"] for r in res] == ["SKIPPED_REQUEST_UNCHANGED"] * 2 and api.submitted == []


# ---------------------------------------------------------------- leaderboard / status / output

def test_percentile_rank():
    assert ko.percentile_rank(3564, 0.10) == 357
    assert ko.percentile_rank(2870, 0.10) == 287
    assert ko.percentile_rank(3564, 0.01) == 36
    assert ko.percentile_rank(3564, 0.05) == 179
    assert ko.percentile_rank(5, 0.01) == 1
    assert ko.percentile_rank(100, 0.05) == 5


def test_leaderboard_summary():
    rows = [{"Rank": str(i), "TeamName": f"t{i}", "Score": str(100 - i * 0.01), "TeamMemberUserNames": ""}
            for i in range(1, 1001)]
    rows[9]["TeamName"] = "How bad can it go?"
    rows = rows[::-1]  # unsorted input
    lb = ko.leaderboard_summary(rows)
    assert lb["teams"] == 1000 and lb["first"]["rank"] == 1
    assert lb["bars"]["top1%"]["rank"] == 10 and lb["bars"]["top10%"]["rank"] == 100
    assert lb["ours"][0]["rank"] == 10 and len(lb["top"]) == 40


def test_claude_md_kernel_list_has_incumbent_and_challengers():
    ks = ko.claude_md_kernels((ROOT / "CLAUDE.md").read_text(encoding="utf-8"))
    assert ks[0] == K
    assert "calamitychasm/arc3-m2-lm-histcache" in ks


def test_kernel_output_digest(tmp_path, capsys):
    def w(p):
        write_output(p, states=("won", "won", "gave_up"))
        (p / "history_cache_summary.json").write_text(json.dumps({"errors": 0, "payload_delta": 900}))
    api = FakeApi(output={K: w}, kernels=[NS(ref=K, current_version_number=3)])
    r = runner(api, tmp_path).op_kernel_output({"op": "kernel_output", "kernel": K, "grep": MARKERS + ["absent"]})
    assert r["result"] == "OK" and r["version"] == 3 and r["status"] == "COMPLETE"
    assert r["markers"] == {MARKERS[0]: True, MARKERS[1]: True, "absent": False}
    assert r["game_states"] == {"won": 2, "gave_up": 1} and r["tracebacks"] == 0
    assert r["counters"]["history_cache_summary.json"]["payload_delta"] == 900
    assert r["notebook_log"] == "arc3-m2-level-memory.log"


def test_status_op_end_to_end(tmp_path, monkeypatch):
    import io
    import zipfile

    def lb_download(comp, path):
        buf = io.StringIO()
        buf.write("Rank,TeamId,TeamName,LastSubmissionDate,Score,SubmissionCount,TeamMemberUserNames\n")
        for i in range(1, 51):
            buf.write(f"{i},{i},{'How bad can it go?' if i == 10 else 't%d' % i},2026-10-08,{60 - i},3,"
                      f"{'calamitychasm' if i == 10 else 'x'}\n")
        with zipfile.ZipFile(Path(path) / f"{comp}.zip", "w") as z:
            z.writestr(f"{comp}-publicleaderboard.csv", buf.getvalue())
    api = FakeApi(subs=[sub("1", NOW - dt.timedelta(hours=3))])
    api.competition_leaderboard_download = lb_download
    r = runner(api, tmp_path).op_status({"op": "status"})
    assert r["result"] == "OK" and r["submitted_today"] == 1
    assert r["leaderboard"]["teams"] == 50 and r["leaderboard"]["ours"][0]["rank"] == 10
    assert K in r["kernels"]


# ---------------------------------------------------------------- version lookup (kaggle 2.2.4 / kagglesdk)

class GetKernelApi(FakeApi):
    """FakeApi plus the single-kernel GET that `kaggle kernels pull --metadata` uses."""

    def __init__(self, get_versions=None, search_kernels=None, **kw):
        super().__init__(**kw)
        self.get_versions = get_versions or {}
        self.search_kernels = search_kernels
        api = self

        class Client:
            def __enter__(self):
                return NS(kernels=NS(kernels_api_client=NS(get_kernel=api._get_kernel)))

            def __exit__(self, *a):
                return False
        self._client = Client

    def build_kaggle_client(self):
        return self._client()

    def _get_kernel(self, req):
        ref = f"{req.user_name}/{req.kernel_slug}"
        if ref not in self.get_versions:
            raise RuntimeError("404")
        return NS(metadata=NS(ref=ref, current_version_number=self.get_versions[ref]), blob=None)

    def kernels_list(self, mine=False, page=1, page_size=20, search=None):
        if search is not None and self.search_kernels is not None:
            return list(self.search_kernels)
        return super().kernels_list(mine=mine, page=page, page_size=page_size, search=search)


def test_version_zero_from_listing_is_unknown_not_a_refusal(tmp_path, capsys):
    # The 2026-10-09 Actions run: kernels_list(mine=True) reported current_version_number 0 for a
    # kernel whose latest version is 1, and the gate refused "latest version ... is v0".
    api = FakeApi(output={K: write_output}, kernels=[NS(ref=K, current_version_number=0)])
    r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
    assert r["result"] == "SUBMITTED", r
    assert r["version_check"] == "VERSION_UNKNOWN" and "VERSION_UNKNOWN" in r["warnings"][0]
    assert not any("latest version" in p for p in r["gate_problems"])
    assert "VERSION_UNKNOWN" in capsys.readouterr().out


def test_version_none_and_absent_are_unknown(tmp_path):
    for kernels in ([NS(ref=K, current_version_number=None)], []):
        api = FakeApi(output={K: write_output}, kernels=kernels)
        r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
        assert r["result"] == "SUBMITTED" and r["version_check"] == "VERSION_UNKNOWN", r


def test_version_unknown_still_enforces_the_rest_of_the_gate(tmp_path):
    api = FakeApi(output={K: lambda p: write_output(p, errors=2)}, kernels=[NS(ref=K, current_version_number=0)])
    r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
    assert r["result"] == "GATE_FAILED" and r["version_check"] == "VERSION_UNKNOWN" and api.submitted == []
    assert any("counters report errors" in p for p in r["gate_problems"])
    api = FakeApi(statuses={K: "RUNNING"}, kernels=[NS(ref=K, current_version_number=0)])
    r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
    assert r["result"] == "GATE_FAILED" and any("not COMPLETE" in p for p in r["gate_problems"])


def test_real_kagglesdk_metadata_unset_version_reads_as_unknown():
    svc = pytest.importorskip("kagglesdk.kernels.types.kernels_api_service")
    meta = svc.ApiKernelMetadata()
    meta.ref = K
    assert meta.current_version_number == 0  # the getter's `or 0` -- the bug's source
    assert ksr._version_of(meta) is None
    meta.current_version_number = 1
    assert ksr._version_of(meta) == 1


def test_get_kernel_version_overrides_a_zero_listing(tmp_path, capsys):
    api = GetKernelApi(get_versions={K: 1}, output={K: write_output},
                       kernels=[NS(ref=K, current_version_number=0)])
    r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
    assert r["result"] == "SUBMITTED" and r["version_check"] == "VERSION_OK", r
    assert "warnings" not in r
    assert "VERSION_OK: v1 via get_kernel" in capsys.readouterr().out


def test_search_listing_used_when_get_kernel_has_no_version(tmp_path):
    api = GetKernelApi(get_versions={K: 0}, search_kernels=[NS(ref=K, current_version_number=1)],
                       output={K: write_output}, kernels=[])
    assert ksr.kernel_version(api, K) == (1, "kernels_list(search)")
    r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
    assert r["result"] == "SUBMITTED" and r["version_check"] == "VERSION_OK"


def test_genuine_version_mismatch_refuses(tmp_path):
    for api in (GetKernelApi(get_versions={K: 2}, output={K: write_output}, kernels=[NS(ref=K, current_version_number=0)]),
                FakeApi(output={K: write_output}, kernels=[NS(ref=K, current_version_number=2)])):
        r = runner(api, tmp_path).op_submit(dict(SUBMIT), False)
        assert r["result"] == "GATE_FAILED" and r["version_check"] == "VERSION_MISMATCH" and api.submitted == []
        assert any("latest version of" in p and "is v2" in p and "not v1" in p for p in r["gate_problems"])


def test_version_check_verdicts():
    assert ksr.version_check(GetKernelApi(get_versions={K: 1}), K, 1)[0] == "VERSION_OK"
    assert ksr.version_check(GetKernelApi(get_versions={K: 3}), K, 1)[0] == "VERSION_MISMATCH"
    verdict, problems, detail = ksr.version_check(FakeApi(kernels=[NS(ref=K, current_version_number=0)]), K, 1)
    assert verdict == "VERSION_UNKNOWN" and problems == [] and "version field empty/0" in detail


def test_status_shows_version_for_freshly_pushed_kernel(tmp_path, capsys):
    tt = "calamitychasm/arc3-m2-turbo-tail"
    api = GetKernelApi(get_versions={tt: 1}, kernels=[NS(ref=K, current_version_number=0)])
    api.competition_leaderboard_download = lambda comp, path: None
    r = runner(api, tmp_path).op_status({"op": "status", "kernels": [tt]})
    assert r["kernels"][tt].startswith("v1 ")
    assert r["kernels"][K].startswith("v? ")  # unknown is shown as v?, never v0 / vNone
    assert not any(v.startswith(("v0", "vNone")) for v in r["kernels"].values())
