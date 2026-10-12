"""scripts/kaggle_ops.py: the opt-in `transcripts` option of kernel_output (per-game timeline, tool-result counts,
excerpts of the level the game ended on). Synthetic *_requests.jsonl fixtures only; no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_ops as ko  # noqa: E402

from test_kaggle_ops import FakeApi, runner  # noqa: E402

K = "calamitychasm/arc3-jad-exp081"
SECRET = "sk-ABCDEF0123456789abcdef0123"
LONG = "Zq9" + "aB3xY7" * 8          # 51 chars, letters and digits: looks like a token


def opener(step, level, extra=""):
    return {"role": "user", "content": [
        {"type": "text", "text": f"Turn start.\nCurrent state: step {step}, level {level}, 12.5 min elapsed.\n{extra}"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 400}}]}


def asst(text, code=None, reasoning=None):
    m = {"role": "assistant", "content": text}
    if code:
        m["tool_calls"] = [{"id": "c", "type": "function",
                            "function": {"name": "python", "arguments": json.dumps({"code": code})}}]
    if reasoning:
        m["reasoning_content"] = reasoning
    return m


def tool(text):
    return {"role": "tool", "tool_call_id": "c", "content": text}


def record(messages, comp=100, **kw):
    return json.dumps({"event": "response", "messages": [{"role": "system", "content": "sys"}] + messages,
                       "tools": [], "usage": {"prompt_tokens": 1000, "completion_tokens": comp}, **kw})


def conversation(code2="action(1)\naction(2)"):
    """Level 1: two turns, then cleared; level 2: three turns, game ends there."""
    h = []
    recs = []

    def step(msgs, **kw):
        h.extend(msgs)
        recs.append(record(list(h), **kw))
    step([opener(1, 1), asst("click the blue bar", "action(6, 10, 20)", "think A " * 40)], action=1)
    step([tool("changed: yes"), opener(2, 1), asst("again", code2)], action=2)
    step([tool("StaleStateActionError: refused\nStaleStateActionError: refused again\nNameError: x"),
          opener(3, 2), asst("L2 first turn " + "w" * 600, "action(6, 3, 4)", "reasoning L2 first")], action=3)
    step([tool("[view] scrolled by 5\nGAME_OVER\nRESET done"), opener(4, 2), asst("L2 second", "action(5)")], action=4)
    step([tool("Traceback (most recent call last):\nNameError: y"), opener(5, 2),
          asst("api_key = " + SECRET + " token " + LONG, "action(7)", "R3 " * 200)], action=5)
    step([tool("[view] scrolled by 2"), opener(6, 2), asst("L2 fourth", "action(1)")], action=6)
    step([tool("GAME_OVER"), opener(7, 2), asst("L2 LAST turn", "action(2)", "final thoughts")], action=7)
    return recs


def write_game(d: Path, name="sp80-0605ab9e_p0_requests.jsonl", recs=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("\n".join(recs or conversation()) + "\n", encoding="utf-8")


def report(tmp_path, cfg, capsys):
    res = ko.transcripts_report(tmp_path, cfg)
    return res, capsys.readouterr().out


def test_timeline_counts_and_ended_level(tmp_path, capsys):
    write_game(tmp_path)
    res, out = report(tmp_path, {"games": ["sp80"], "max_lines_per_game": 80}, capsys)
    g = res["games"]["sp80"]
    assert g["records"] == 7 and g["ended_level"] == 2 and g["markers"] == 7
    assert g["levels"]["1"]["reqs"] == 2 and g["levels"]["2"]["reqs"] == 5
    # tool results are counted once each even though every record repeats the history
    assert g["counts"] == {"StaleStateActionError": 2, "NameError": 2, "Traceback": 1, "[view] scrolled": 2,
                           "GAME_OVER": 2, "RESET": 1}
    # a tool result is counted against the level the call was made in (before the next opener's marker)
    # (the refusals came back for level 1's last call; the GAME_OVER / RESET for level 2's first)
    assert g["levels"]["1"]["counts"] == {"StaleStateActionError": 2, "NameError": 1}
    assert g["levels"]["2"]["counts"] == {"[view] scrolled": 2, "GAME_OVER": 2, "RESET": 1, "Traceback": 1, "NameError": 1}
    assert g["levels"]["1"]["action_calls"] == 3 and g["levels"]["2"]["action_calls"] == 5
    assert "L1: 2 req, 200 tok" in out and "cleared" in out and "ENDED HERE, not cleared" in out
    assert "counter 1-2" in out and "steps 1-2" in out
    assert "StaleStateActionError=2 (in 1 results, max 2/result)" in out
    assert "12.5 min elapsed" in out          # the state line carries the time
    assert "action() args (first 3): 6, 10, 20; 1; 2" in out


def test_excerpt_first_two_and_last_three_turns_truncated(tmp_path, capsys):
    write_game(tmp_path)
    _, out = report(tmp_path, {"games": ["sp80"], "turn_chars": 400}, capsys)
    ex = [ln for ln in out.splitlines() if ln.strip().startswith("[rec ")]
    assert len(ex) == 5
    assert [int(e.split("[rec ")[1].split(",")[0]) for e in ex] == [3, 4, 5, 6, 7]   # first 2 of 5 + last 3 = all 5
    assert "L2 first turn" in ex[0] and "L2 LAST turn" in ex[-1]
    assert all(len(e) <= 400 + 60 for e in ex)
    assert "w" * 400 not in out               # truncated, not dumped
    assert "final thoughts" in ex[-1] and "action(2)" in ex[-1]


def test_excerpt_with_few_turns_shows_all_once(tmp_path, capsys):
    recs = conversation()[:3]          # ends on level 2 after a single turn there
    write_game(tmp_path, recs=recs)
    _, out = report(tmp_path, {"games": ["sp80"]}, capsys)
    assert len([ln for ln in out.splitlines() if ln.strip().startswith("[rec ")]) == 1


def test_credentials_images_and_long_tokens_are_scrubbed(tmp_path, capsys):
    write_game(tmp_path)
    _, out = report(tmp_path, {"games": ["sp80"], "turn_chars": 2000, "max_lines_per_game": 100}, capsys)
    assert SECRET not in out and LONG not in out and "A" * 100 not in out
    assert "[REDACTED]" in out
    assert ko.tx_scrub("Authorization: Bearer abc.def.ghi") == "Authorization: Bearer [REDACTED]"
    assert ko.tx_scrub("KAGGLE_KEY=0123456789abcdef0123456789abcdef") == "KAGGLE_KEY=[REDACTED]"
    assert ko.tx_scrub("password: 'hunter2'") == "password: [REDACTED]"
    assert ko.tx_scrub("ARC key 6f2a3b4c-1111-2222-3333-444455556666") == "ARC key [uuid]"
    assert ko.tx_scrub("data:image/png;base64,AAAA+/==") == "[b64]"
    assert ko.tx_scrub("/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy") == \
        "/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy"      # ordinary paths survive


def test_history_trimming_does_not_double_count(tmp_path, capsys):
    """A request whose history lost its oldest turns (trim) and gained a pinned block still counts new results once."""
    recs = conversation()
    last = json.loads(recs[-1])
    pinned = {"role": "user", "content": "Rule block (pinned) step 0, level 0"}
    trimmed = [m for m in last["messages"] if m["role"] != "system"][6:]
    extra = [tool("GAME_OVER again"), opener(8, 2), asst("L2 after trim", "action(3)")]
    rec = record([pinned] + trimmed + extra, action=8)
    write_game(tmp_path, recs=recs + [rec])
    res, _ = report(tmp_path, {"games": ["sp80"]}, capsys)
    c = res["games"]["sp80"]["counts"]
    assert c["GAME_OVER"] == 3 and c["NameError"] == 2 and c["StaleStateActionError"] == 2
    assert res["games"]["sp80"]["levels"]["2"]["reqs"] == 6


def test_unrelated_messages_without_overlap_fall_back_to_unseen_only(tmp_path, capsys):
    a = record([opener(1, 1), asst("one", "action(1)")])
    b = record([tool("GAME_OVER"), opener(5, 2), asst("two", "action(2)")])      # shares nothing with `a`
    write_game(tmp_path, recs=[a, b, b])
    res, out = report(tmp_path, {"games": ["sp80"]}, capsys)
    assert res["games"]["sp80"]["counts"]["GAME_OVER"] == 1
    assert "shared no message" in out


def test_layouts_nested_flat_and_all(tmp_path, capsys):
    write_game(tmp_path, "sp80_p0_requests.jsonl")
    write_game(tmp_path / "lf52-deadbeef" / "logs", "x_requests.jsonl")
    write_game(tmp_path, "sp800-1_p0_requests.jsonl")                       # must not match "sp80"
    assert [p.name for p in ko.tx_find_files(tmp_path, "sp80")] == ["sp80_p0_requests.jsonl"]
    assert [p.name for p in ko.tx_find_files(tmp_path, "lf52")] == ["x_requests.jsonl"]
    res, out = report(tmp_path, {"games": ["sp80", "lf52", "g50t"]}, capsys)
    assert set(res["games"]) == {"sp80", "lf52", "g50t"} and res["games"]["g50t"] is None
    assert "g50t: no matching requests file. Files:" in out and "sp800-1_p0_requests.jsonl" in out
    res, out = report(tmp_path, {"games": "all"}, capsys)
    assert res["files"] == 3 and len(res["games"]) == 3


def test_no_state_markers_pools_levels_and_shows_the_first_prompt(tmp_path, capsys):
    rec = record([{"role": "user", "content": "Plain opener without markers"}, asst("hi", "action(1)")])
    write_game(tmp_path, recs=[rec])
    res, out = report(tmp_path, {"games": ["sp80"]}, capsys)
    assert res["games"]["sp80"]["markers"] == 0 and "L?: 1 req" in out
    assert "NO 'step N, level L' state markers" in out and "Plain opener without markers" in out


def test_line_caps(tmp_path, capsys):
    h, recs = [], []
    for lvl in range(1, 40):
        h += [opener(lvl, lvl), asst(f"turn {lvl}", "action(1)"), tool("ok")]
        recs.append(record(list(h)))
    write_game(tmp_path, recs=recs)
    _, out = report(tmp_path, {"games": ["sp80"], "max_lines_per_game": 20}, capsys)
    game_lines = [ln for ln in out.splitlines() if not ln.startswith("  == transcripts")]
    assert len(game_lines) <= 20 and "earlier timeline lines omitted" in out and "L39:" in out and "L1:" not in out
    # the global cap
    for n in range(8):
        write_game(tmp_path, f"g{n:03d}-x_p0_requests.jsonl", recs=recs)
    old = ko.TX_TOTAL_LINES
    ko.TX_TOTAL_LINES = 30
    try:
        res = ko.transcripts_report(tmp_path, {"games": "all", "max_lines_per_game": 20})
    finally:
        ko.TX_TOTAL_LINES = old
    assert res["lines_printed"] <= 30


def test_extra_count_strings_and_bad_lines(tmp_path, capsys):
    recs = conversation()
    write_game(tmp_path, recs=recs[:2] + ['{"event": "response", "messages": [tr'] + recs[2:])
    res, out = report(tmp_path, {"games": ["sp80"], "count": ["refused"]}, capsys)
    assert res["games"]["sp80"]["counts"]["refused"] == 2 and "1 unreadable lines" in out


def test_samples_grep_text_and_excerpt_level(tmp_path, capsys):
    write_game(tmp_path)
    cfg = {"games": ["sp80"], "samples": {"StaleStateActionError": 2}, "grep_text": {"blue bar": 3, "nomatch_xyz": 1},
           "excerpt_levels": {"sp80": 1}, "max_lines_per_game": 100}
    _, out = report(tmp_path, cfg, capsys)
    assert "excerpt of level 1: 2 assistant turns there" in out and "[rec 1, step 1]" in out and "L2 LAST turn" not in out
    assert "first 1 tool results containing 'StaleStateActionError'" in out
    assert "2x in this result, prior code had except=False" in out and "prior code tail: action(1) action(2)" in out
    assert "matching /blue bar/ (first 1):" in out and "[L1 rec 1 step 1]" in out and "click the blue bar" in out
    assert "matching /nomatch_xyz/ (first 0):" in out and "(no match)" in out
    # the default "*" key, and a wrapped call is flagged
    write_game(tmp_path, recs=conversation("try:\n action(1)\nexcept Exception:\n pass"))
    _, out = report(tmp_path, {"games": ["sp80"], "samples": {"NameError": 1}, "excerpt_levels": {"*": 2}}, capsys)
    assert "prior code had except=True" in out and "excerpt of level 2" in out


def test_parse_request_validates_new_transcript_keys():
    ok = {"op": "kernel_output", "kernel": K, "transcripts": {
        "games": ["sp80"], "excerpt_levels": {"sp80": 2, "*": 1}, "samples": {"StaleStateActionError": 3},
        "grep_text": {"SPACE": 4}}}
    ko.parse_request(json.dumps({"id": "x", "ops": [ok]}))
    for bad in ({"excerpt_levels": {"sp80": "2"}}, {"excerpt_levels": []}, {"samples": {"a": 0}}, {"samples": {"a": 9}},
                {"grep_text": {"": 1}}, {"grep_text": {"a": True}}, {"samples": ["a"]}):
        with pytest.raises(ko.RequestError):
            ko.parse_request(json.dumps({"id": "x", "ops": [dict(ok, transcripts=dict(ok["transcripts"], **bad))]}))


def test_sample_chars_widens_the_sample(tmp_path, capsys):
    rec = record([opener(1, 1), asst("x", "action(1)"), tool("StaleStateActionError " + "p" * 600 + " TAIL_MARK")])
    write_game(tmp_path, recs=[rec])
    _, narrow = report(tmp_path, {"games": ["sp80"], "samples": {"StaleStateActionError": 1}}, capsys)
    _, wide = report(tmp_path, {"games": ["sp80"], "samples": {"StaleStateActionError": 1}, "sample_chars": 1000}, capsys)
    assert "TAIL_MARK" not in narrow and "TAIL_MARK" in wide
    with pytest.raises(ko.RequestError):
        ko.parse_request(json.dumps({"id": "x", "ops": [{"op": "kernel_output", "kernel": K,
                                                         "transcripts": {"games": "all", "sample_chars": 5}}]}))


def test_files_without_request_logs(tmp_path, capsys):
    res, out = report(tmp_path, {"games": "all"}, capsys)
    assert res == {"files": 0} and "no *requests.jsonl" in out


def test_parse_request_validates_transcripts():
    ok = {"op": "kernel_output", "kernel": K, "grep": ["x"],
          "transcripts": {"games": ["sp80", "lf52"], "max_lines_per_game": 80, "count": ["a"]}}
    assert ko.parse_request(json.dumps({"id": "x", "ops": [ok]}))["id"] == "x"
    ok2 = dict(ok, transcripts={"games": "all"})
    ko.parse_request(json.dumps({"id": "x", "ops": [ok2]}))
    for bad in ({"games": []}, {"games": "sp80"}, {"games": ["../x"]}, {"games": ["a"], "max_lines_per_game": 5},
                {"games": ["a"], "max_lines_per_game": 5000}, {"games": ["a"], "turn_chars": True}, "all",
                {"games": ["a"], "count": "x"}):
        with pytest.raises(ko.RequestError):
            ko.parse_request(json.dumps({"id": "x", "ops": [dict(ok, transcripts=bad)]}))


def test_kernel_output_op_runs_transcripts_and_keeps_the_digest(tmp_path, capsys):
    def w(p):
        (p / "arc3-jad-exp081.log").write_text(json.dumps([{"stream_name": "stdout", "data": "[view] scrolled\n"}]))
        write_game(p)
    api = FakeApi(output={K: w}, kernels=[NS(ref=K, current_version_number=2)])
    r = runner(api, tmp_path).op_kernel_output(
        {"op": "kernel_output", "kernel": K, "grep": ["[view] scrolled"], "transcripts": {"games": ["sp80", "zz99"]}})
    out = capsys.readouterr().out
    assert r["result"] == "OK" and r["version"] == 2 and r["markers"] == {"[view] scrolled": True}
    assert r["transcripts"]["games"]["sp80"]["ended_level"] == 2 and r["transcripts"]["games"]["zz99"] is None
    assert "==== sp80:" in out and "ENDED HERE" in out


def test_kernel_output_without_the_option_prints_no_transcripts(tmp_path, capsys):
    api = FakeApi(output={K: lambda p: write_game(p)}, kernels=[NS(ref=K, current_version_number=2)])
    r = runner(api, tmp_path).op_kernel_output({"op": "kernel_output", "kernel": K})
    assert "transcripts" not in r and "== transcripts" not in capsys.readouterr().out


def test_transcript_failure_does_not_lose_the_digest(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ko, "transcripts_report", lambda *a, **k: 1 / 0)
    api = FakeApi(output={K: lambda p: write_game(p)}, kernels=[NS(ref=K, current_version_number=2)])
    r = runner(api, tmp_path).op_kernel_output(
        {"op": "kernel_output", "kernel": K, "transcripts": {"games": ["sp80"]}})
    assert r["result"] == "OK" and r["transcripts"].startswith("error ZeroDivisionError")
