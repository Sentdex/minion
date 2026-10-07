#!/usr/bin/env python3
"""Run budgets (MINION_MAX_RUN_SECONDS / _TOKENS / MINION_MAX_TOOL_CALLS) end
the run as budget_exhausted/<name> after exactly one forced-final turn, and
usage is counted on every model_turn exit path (5.1)."""
import io
import os
import sys
import tempfile

import pytest

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None, reasoning=None):
        self.content = content
        self.tool_calls = tool_calls
        self.model_extra = {"reasoning_content": reasoning} if reasoning else {}
        self.reasoning_content = reasoning


class _FakeChoice:
    def __init__(self, delta, finish_reason):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, delta=None, finish_reason="stop", usage=None):
        self.choices = [_FakeChoice(delta, finish_reason)] if delta is not None else []
        self.usage = usage
        self.model_extra = {}


class _Usage:
    def __init__(self, prompt, completion):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = None
        self.output_tokens_details = None


class _NoSpinner:
    _t = None

    def __init__(self, **kw):
        pass

    def start(self):
        pass

    def stop(self):
        pass


def _quiet(monkeypatch):
    monkeypatch.setattr(m, "_interrupt_watcher", lambda: None)
    monkeypatch.setattr(m, "LifeSpinner", _NoSpinner)
    monkeypatch.setattr(m, "VERIFY_CMD", "")
    for k in ("MAX_RUN_SECONDS", "MAX_RUN_TOKENS", "MAX_TOOL_CALLS"):
        monkeypatch.setattr(m, k, 0)


def _fake_turns(monkeypatch, on_normal):
    """model_turn stub: normal turns run `on_normal` and return TURN_TOOL;
    forced-final turns just finish. Returns the list of forced flags."""
    calls = []

    def fake_model_turn(messages, rl, ms, empty_turn_count=0,
                        forced_final=False, recovery_sampling=False):
        calls.append(forced_final)
        if forced_final:
            return m._turn_done("completed_unverified")
        on_normal(len(calls))
        return m.TURN_TOOL

    monkeypatch.setattr(m, "model_turn", fake_model_turn)
    return calls


def test_wall_time_budget(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "MAX_RUN_SECONDS", 60)

    def tick(n):  # "patched time": the 2nd turn pushes the run past 60 s
        if n == 2:
            m._RUN_STATS["started"] -= 1000

    calls = _fake_turns(monkeypatch, tick)
    res = m._run_model_turn_loop([{"role": "user", "content": "go"}])
    assert res["terminal"] == "budget_exhausted"
    assert res["budget"] == "wall_time"
    assert calls == [False, False, True]  # exactly one forced call


def test_tool_call_budget(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "MAX_TOOL_CALLS", 2)
    monkeypatch.setattr(m, "list_dir", lambda path=".": "a\nb")
    monkeypatch.setitem(m.DISPATCH, "list_dir", lambda path=".": "a\nb")
    calls = _fake_turns(monkeypatch, lambda n: m.run_tool("list_dir", {"path": "."}))
    res = m._run_model_turn_loop([{"role": "user", "content": "go"}])
    assert res["terminal"] == "budget_exhausted"
    assert res["budget"] == "tool_calls"
    assert res["tool_calls"] == 2
    assert calls == [False, False, True]


def test_tool_calls_past_the_cap_are_refused(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "MAX_TOOL_CALLS", 1)
    ran = []
    monkeypatch.setitem(m.DISPATCH, "list_dir", lambda path=".": ran.append(path) or "ok")
    m._RUN_STATS = m._new_run_stats()
    assert m.run_tool("list_dir", {"path": "a"}) == "ok"
    out = m.run_tool("list_dir", {"path": "b"})
    assert out.startswith("ERROR [budget_exhausted]"), out
    assert ran == ["a"]


def test_token_budget_from_stream_usage(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "MAX_RUN_TOKENS", 100)
    monkeypatch.setattr(m, "_METRICS", None)
    tc = type("TC", (), {"index": 0, "id": "c1",
                         "function": type("F", (), {"name": "list_dir",
                                                    "arguments": '{"path": "."}'})()})()
    monkeypatch.setitem(m.DISPATCH, "list_dir", lambda path=".": "x")
    forced = []

    def stream(messages, tools=None, **k):
        if tools is not None:  # forced final
            forced.append(1)
            return iter([_FakeChunk(_FakeDelta(content="stopping"))])
        return iter([_FakeChunk(_FakeDelta(tool_calls=[tc]), "tool_calls"),
                     _FakeChunk(None, usage=_Usage(80, 40))])

    monkeypatch.setattr(m, "open_stream", stream)
    res = m._run_model_turn_loop([{"role": "user", "content": "go"}])
    assert res["terminal"] == "budget_exhausted"
    assert res["budget"] == "tokens"
    assert res["tokens"] >= 120
    assert len(forced) == 1


def test_stalled_turn_usage_is_counted(monkeypatch):
    # 5.1: a reasoning-only stall returns early; its usage must still count.
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "REASONING_ONLY_RETRY_LIMIT", 3)

    class _Rec:
        def __init__(self):
            self.deltas = []

        def add(self, d):
            self.deltas.append(d)

    rec = _Rec()
    monkeypatch.setattr(m, "_METRICS", rec)
    m._RUN_STATS = m._new_run_stats()
    stream = iter([_FakeChunk(_FakeDelta(reasoning="hmm " * 10), None),
                   _FakeChunk(None, usage=_Usage(50, 7))])
    monkeypatch.setattr(m, "open_stream", lambda *a, **k: stream)
    status = m.model_turn([{"role": "user", "content": "go"}])
    assert status == m.TURN_STALL
    assert m._RUN_STATS["input_tokens"] == 50
    assert m._RUN_STATS["output_tokens"] == 7
    assert m._RUN_STATS["api_calls"] == 1
    assert rec.deltas and rec.deltas[0]["input_tokens"] == 50


def test_one_shot_builds_metrics_and_exits_4_on_budget(monkeypatch, tmp_path):
    _quiet(monkeypatch)
    prompt = tmp_path / "task.txt"
    prompt.write_text("do it\n")
    monkeypatch.setattr(m, "_PROMPT_FILE", str(prompt))
    monkeypatch.setattr(m, "ACTIVE", object())
    monkeypatch.setattr(m, "_ensure_ctx_probe", lambda src: None)
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(m, "HEADLESS_APPROVAL", False)
    monkeypatch.setattr(m, "_METRICS", None)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(m, "MAX_RUN_SECONDS", 1)
    _fake_turns(monkeypatch, lambda n: m._RUN_STATS.__setitem__(
        "started", m._RUN_STATS["started"] - 10))
    with pytest.raises(SystemExit) as exc:
        m._run_one_shot()
    assert exc.value.code == 4
    assert isinstance(m._METRICS, m._Metrics)
