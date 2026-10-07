#!/usr/bin/env python3
"""_run_model_turn_loop returns a typed terminal state instead of None.

model_turn keeps returning plain TURN_* values; each TURN_DONE site records why
(_LAST_TURN_REASON) and the loop maps it to completed_unverified | blocked |
model_error | budget_exhausted | cancelled | recovery_failed | truncated.
"""
import os
import sys
import tempfile

import httpx

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.model_extra = {}
        self.reasoning_content = None


class _FakeChoice:
    def __init__(self, delta, finish_reason):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, delta=None, finish_reason="stop"):
        self.choices = [_FakeChoice(delta, finish_reason)] if delta is not None else []
        self.usage = None
        self.model_extra = {}


class _FakeTC:
    def __init__(self, index, id_, name, args):
        self.index = index
        self.id = id_
        self.function = type("F", (), {"name": name, "arguments": args})()


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


def test_open_stream_failure_is_model_error(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "open_stream", lambda *a, **k: None)
    res = m._run_model_turn_loop([{"role": "user", "content": "hi"}])
    assert res["terminal"] == "model_error"
    assert res["budget"] is None


def test_stream_timeout_is_model_error(monkeypatch):
    _quiet(monkeypatch)

    def stream(*a, **k):
        yield _FakeChunk(_FakeDelta(content="par"), finish_reason=None)
        raise httpx.ReadTimeout("stalled")

    monkeypatch.setattr(m, "open_stream", stream)
    res = m._run_model_turn_loop([{"role": "user", "content": "hi"}])
    assert res["terminal"] == "model_error"


def test_plain_answer_is_completed_unverified(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "open_stream",
                        lambda *a, **k: iter([_FakeChunk(_FakeDelta(content="all done"))]))
    res = m._run_model_turn_loop([{"role": "user", "content": "hi"}])
    # 5.1/5.2 added run stats (tool_calls, tokens, wall_s) to the result.
    assert {k: res[k] for k in ("terminal", "budget", "steps")} == \
        {"terminal": "completed_unverified", "budget": None, "steps": 0}
    assert res["tool_calls"] == 0


def test_tool_forever_hits_turn_budget(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "MAX_MODEL_TURNS", 3)
    monkeypatch.setattr(m, "model_turn", lambda *a, **k: m.TURN_TOOL)
    res = m._run_model_turn_loop([{"role": "user", "content": "hi"}])
    assert res["terminal"] == "budget_exhausted"
    assert res["budget"] == "turns"
    assert res["steps"] == 3


def test_final_answer_blocked_status_is_blocked(monkeypatch):
    _quiet(monkeypatch)
    args = '{"answer": "Need the DB password to continue.", "status": "blocked"}'
    monkeypatch.setattr(m, "open_stream", lambda *a, **k: iter([
        _FakeChunk(_FakeDelta(tool_calls=[_FakeTC(0, "f1", "final_answer", args)]),
                   finish_reason="tool_calls")]))
    messages = [{"role": "user", "content": "deploy"}]
    res = m._run_model_turn_loop(messages, force_final=True)
    assert res["terminal"] == "blocked"
    assert messages[-1]["content"].startswith("Need the DB password")


def test_esc_is_cancelled(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "model_turn", lambda *a, **k: m.TURN_ESC)
    res = m._run_model_turn_loop([{"role": "user", "content": "hi"}])
    assert res["terminal"] == "cancelled"


def test_malformed_stream_exhausted_is_recovery_failed(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "MALFORMED_STREAM_RETRY_LIMIT", 1)

    def stream(*a, **k):
        yield _FakeChunk(_FakeDelta(content="x"), finish_reason=None)
        raise httpx.RemoteProtocolError("bad SSE frame")

    monkeypatch.setattr(m, "open_stream", stream)
    res = m._run_model_turn_loop([{"role": "user", "content": "hi"}])
    assert res["terminal"] == "recovery_failed"
    assert res["steps"] == 1


def test_run_end_is_logged(monkeypatch):
    _quiet(monkeypatch)
    seen = []
    monkeypatch.setattr(m, "_log_event", lambda d, p: seen.append((d, p)))
    monkeypatch.setattr(m, "model_turn", lambda *a, **k: m.TURN_DONE)
    res = m._run_model_turn_loop([{"role": "user", "content": "hi"}])
    assert ("run_end", res) in seen
