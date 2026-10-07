#!/usr/bin/env python3
"""In-loop autocompress and context-overflow routing (plan item 6.3).

  - After a TURN_TOOL whose prompt filled the window past the threshold,
    compress(auto=True) runs before the next model turn.
  - MINION_AUTOCOMPRESS_IN_LOOP=0 turns that off.
  - An overflow error from the server is not retried without tools=: exactly
    one create() call, open_stream returns CONTEXT_FULL, model_turn returns
    TURN_CONTEXT_FULL, and the loop compresses once and retries; a second
    overflow ends the run as model_error/context_exhausted.
"""
import os
import sys
import tempfile

import httpx
import openai

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


class _Src:
    name = "fake"

    def __init__(self, window):
        self._context_window = window

    def extra_request_kwargs(self):
        return {}


def _loop(monkeypatch, statuses, compress_result=(2, 4, 10)):
    """Run the loop over scripted model_turn statuses; return the event log."""
    events = []

    def fake_model_turn(messages, *a, forced_final=False, **k):
        n = sum(1 for e in events if e[0] == "turn")
        events.append(("turn", forced_final))
        st = statuses[n] if n < len(statuses) else m.TURN_DONE
        if st == m.TURN_TOOL:
            m._LAST_PROMPT_TOKENS = 9000
        return st

    def fake_compress(messages, keep=m.COMPRESS_KEEP, auto=False, session_id=None):
        events.append(("compress", auto, session_id))
        return compress_result

    monkeypatch.setattr(m, "model_turn", fake_model_turn)
    monkeypatch.setattr(m, "compress", fake_compress)
    monkeypatch.setattr(m, "ACTIVE", _Src(10000))
    monkeypatch.setattr(m, "AUTOCOMPRESS_PERCENT", 85)
    monkeypatch.setattr(m, "_LAST_PROMPT_TOKENS", 0)
    res = m._run_model_turn_loop([{"role": "user", "content": "go"}], session_id="sid1")
    return events, res


def test_autocompress_runs_between_tool_turns(monkeypatch):
    events, res = _loop(monkeypatch, [m.TURN_TOOL, m.TURN_DONE])
    assert events == [("turn", False), ("compress", True, "sid1"), ("turn", False)]
    assert res["terminal"] == "completed_unverified"
    assert m._LAST_PROMPT_TOKENS == 0  # reset so a stale count can't re-fold


def test_autocompress_in_loop_can_be_disabled(monkeypatch):
    monkeypatch.setattr(m, "AUTOCOMPRESS_IN_LOOP", 0)
    events, _ = _loop(monkeypatch, [m.TURN_TOOL, m.TURN_DONE])
    assert [e[0] for e in events] == ["turn", "turn"]


def test_context_full_compresses_once_and_retries(monkeypatch):
    events, res = _loop(monkeypatch, [m.TURN_CONTEXT_FULL, m.TURN_DONE])
    assert events == [("turn", False), ("compress", True, "sid1"), ("turn", False)]
    assert res["terminal"] == "completed_unverified"


def test_context_full_twice_is_context_exhausted(monkeypatch):
    events, res = _loop(monkeypatch, [m.TURN_CONTEXT_FULL, m.TURN_CONTEXT_FULL])
    assert [e[0] for e in events] == ["turn", "compress", "turn"]
    assert (res["terminal"], res["budget"]) == ("model_error", "context_exhausted")


def test_context_full_when_nothing_to_fold(monkeypatch):
    events, res = _loop(monkeypatch, [m.TURN_CONTEXT_FULL], compress_result=None)
    assert [e[0] for e in events] == ["turn", "compress"]
    assert (res["terminal"], res["budget"]) == ("model_error", "context_exhausted")
    assert m.ONE_SHOT_EXIT_CODES[res["terminal"]] == 5


def _overflow_error(msg):
    req = httpx.Request("POST", "http://fake/v1/chat/completions")
    return openai.BadRequestError(msg, response=httpx.Response(400, request=req),
                                  body={"error": {"message": msg}})


class _RaisingClient:
    base_url = "http://fake"

    def __init__(self, exc):
        self.exc = exc
        self.calls = 0
        self.chat = type("Ch", (), {})()
        self.chat.completions = self

    def create(self, **kw):
        self.calls += 1
        raise self.exc


def test_open_stream_overflow_is_one_call(monkeypatch):
    for msg in ("This model's maximum context length is 8192 tokens. However, "
                "you requested 9000 tokens",
                "the request exceeds the available context size, try increasing it"):
        fake = _RaisingClient(_overflow_error(msg))
        monkeypatch.setattr(m, "client", fake)
        monkeypatch.setattr(m, "ACTIVE", None)
        out = m.open_stream([{"role": "user", "content": "hi"}])
        assert out is m.CONTEXT_FULL
        assert fake.calls == 1, "overflow must not be retried without tools="


def test_open_stream_other_rejection_still_falls_back(monkeypatch):
    fake = _RaisingClient(_overflow_error("tools param not supported"))
    monkeypatch.setattr(m, "client", fake)
    monkeypatch.setattr(m, "ACTIVE", None)
    assert m.open_stream([{"role": "user", "content": "hi"}]) is None
    assert fake.calls == 3  # tools, no-tools, no-tools-no-extra


def test_model_turn_maps_context_full(monkeypatch):
    monkeypatch.setattr(m, "open_stream", lambda *a, **k: m.CONTEXT_FULL)
    assert m.model_turn([{"role": "user", "content": "hi"}]) == m.TURN_CONTEXT_FULL


def test_overflow_end_to_end_one_create_then_compress(monkeypatch):
    """Real model_turn + open_stream: the overflow costs one create(), then the
    loop compresses; a repeat overflow ends as context_exhausted."""
    fake = _RaisingClient(_overflow_error("maximum context length is 100 tokens"))
    order = []
    real_create = fake.create

    def create(**kw):
        order.append("create")
        return real_create(**kw)

    fake.create = create
    monkeypatch.setattr(m, "client", fake)
    monkeypatch.setattr(m, "ACTIVE", None)

    def fake_compress(messages, keep=m.COMPRESS_KEEP, auto=False, session_id=None):
        order.append("compress")
        return (1, 1, 1)

    monkeypatch.setattr(m, "compress", fake_compress)
    res = m._run_model_turn_loop([{"role": "user", "content": "go"}])
    assert order == ["create", "compress", "create"]
    assert res["budget"] == "context_exhausted"
