#!/usr/bin/env python3
"""Recovery branches of model_turn / _run_model_turn_loop: the malformed-stream
cap at its limit (stream error and malformed tool args), repetition collapse,
the intermediate reasoning-only TURN_STALL tiers, and reasoning_loop_cuts
surviving a TURN_TOOL (stall → tool → stall must still escalate). Also checks
the per-run recovery counters that feed the MINION_RESULT_FILE trial row."""
import os
import sys
import tempfile

import httpx
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
        self.model_extra = {}
        self.reasoning_content = reasoning


class _FakeChoice:
    def __init__(self, delta, finish_reason):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, delta, finish_reason=None):
        self.choices = [_FakeChoice(delta, finish_reason)]
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


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(m, "_interrupt_watcher", lambda: None)
    monkeypatch.setattr(m, "LifeSpinner", _NoSpinner)
    monkeypatch.setattr(m, "run_tool", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("recovery branches must not execute tools")))
    monkeypatch.setattr(m, "_RUN_STATS", m._new_run_stats())


def _msgs():
    return [{"role": "system", "content": "sys"},
            {"role": "user", "content": "do it"}]


def _stream(chunks, error=None):
    def gen():
        yield from chunks
        if error is not None:
            raise error
    return lambda *a, **k: gen()


# --- malformed cap ----------------------------------------------------------

@pytest.mark.parametrize("cuts,expect", [(0, "cut"), (1, "cut"), (2, "done")])
def test_stream_error_malformed_cap(monkeypatch, cuts, expect):
    monkeypatch.setattr(m, "MALFORMED_STREAM_RETRY_LIMIT", 2)
    monkeypatch.setattr(m, "open_stream", _stream(
        [_FakeChunk(_FakeDelta(content="partial"))],
        error=httpx.RemoteProtocolError("peer closed connection")))
    messages = _msgs()
    status = m.model_turn(messages, malformed_stream_cut_count=cuts)
    if expect == "cut":
        assert status == m.TURN_STREAM_CUT
        assert "malformed" in messages[-1]["content"]
    else:
        assert status == m.TURN_DONE
        assert m._LAST_TURN_REASON == "recovery_failed"
    assert not any(x["role"] == "assistant" for x in messages)


@pytest.mark.parametrize("cuts,expect", [(1, m.TURN_STREAM_CUT), (2, m.TURN_DONE)])
def test_malformed_tool_args_cap(monkeypatch, cuts, expect):
    monkeypatch.setattr(m, "MALFORMED_STREAM_RETRY_LIMIT", 2)
    monkeypatch.setattr(m, "open_stream", _stream([_FakeChunk(_FakeDelta(tool_calls=[
        _FakeTC(0, "c1", "write_file", '{"path": "x", "content": "unterminated')]),
        "tool_calls")]))
    messages = _msgs()
    status = m.model_turn(messages, malformed_stream_cut_count=cuts)
    assert status == expect
    if expect == m.TURN_DONE:
        assert m._LAST_TURN_REASON == "recovery_failed"
    assert not any(x["role"] in ("assistant", "tool") for x in messages)


# --- collapse ---------------------------------------------------------------

def test_repetition_collapse_discards_and_nudges(monkeypatch):
    monkeypatch.setattr(m, "REPEAT_CHUNK_LIMIT", 4)
    monkeypatch.setattr(m, "open_stream", _stream(
        [_FakeChunk(_FakeDelta(content="import")) for _ in range(10)]))
    messages = _msgs()
    status = m.model_turn(messages)
    assert status == m.TURN_STREAM_CUT
    assert not any(x["role"] == "assistant" for x in messages)  # spam never kept
    assert "repetition collapse" in messages[-1]["content"]
    assert m._RUN_STATS["collapses"] == 1


def test_collapse_under_forced_final_is_not_retried(monkeypatch):
    monkeypatch.setattr(m, "REPEAT_CHUNK_LIMIT", 4)
    monkeypatch.setattr(m, "open_stream", _stream(
        [_FakeChunk(_FakeDelta(content="import")) for _ in range(10)]))
    status = m.model_turn(_msgs(), forced_final=True)
    assert status == m.TURN_DONE
    assert m._RUN_STATS["collapses"] == 0


# --- reasoning-only stall tiers ----------------------------------------------

@pytest.mark.parametrize("cuts,expect", [
    (0, m.TURN_STALL), (1, m.TURN_STALL), (2, m.TURN_FORCE_FINAL), (3, m.TURN_DONE)])
def test_reasoning_stall_tiers(monkeypatch, cuts, expect):
    monkeypatch.setattr(m, "REASONING_ONLY_RETRY_LIMIT", 3)
    monkeypatch.setattr(m, "REASONING_ONLY_CHAR_LIMIT", 3)
    monkeypatch.setattr(m, "open_stream", _stream(
        [_FakeChunk(_FakeDelta(reasoning="thinking hard"))]))
    messages = _msgs()
    status = m.model_turn(messages, reasoning_loop_cut_count=cuts)
    assert status == expect
    if expect == m.TURN_STALL:
        # Intermediate stall: reasoning kept as context, nudge to ACT appended.
        assert messages[-2]["role"] == "assistant"
        assert messages[-1]["content"] == m.REASONING_STALL_CONTINUE_NOTE
    if expect == m.TURN_DONE:
        assert m._LAST_TURN_REASON == "recovery_failed"


# --- loop counters ------------------------------------------------------------

def _scripted_loop(monkeypatch, statuses):
    seen = []
    it = iter(statuses)

    def fake(messages, reasoning_cuts=0, malformed_cuts=0, empty_turn_count=0,
             forced_final=False, recovery_sampling=False):
        seen.append((reasoning_cuts, malformed_cuts, forced_final))
        st = next(it)
        if st == m.TURN_DONE:
            return m._turn_done("completed_unverified")
        return st
    monkeypatch.setattr(m, "model_turn", fake)
    monkeypatch.setattr(m, "AUTOCOMPRESS_IN_LOOP", 0)
    monkeypatch.setattr(m, "VERIFY_CMD", "")
    return seen


def test_reasoning_cuts_survive_tool_turns_but_malformed_resets(monkeypatch):
    seen = _scripted_loop(monkeypatch, [
        m.TURN_STALL, m.TURN_TOOL, m.TURN_STREAM_CUT, m.TURN_TOOL,
        m.TURN_STALL, m.TURN_TOOL, m.TURN_DONE])
    res = m._run_model_turn_loop([{"role": "user", "content": "go"}])
    assert [s[0] for s in seen] == [0, 1, 1, 1, 1, 2, 2]  # never reset by TURN_TOOL
    assert [s[1] for s in seen] == [0, 0, 0, 1, 0, 0, 0]  # reset by TURN_TOOL
    assert res["terminal"] == "completed_unverified"
    assert res["counters"]["stalls"] == 2
    assert res["counters"]["malformed"] == 1


def test_counters_and_run_id_in_result(monkeypatch):
    seen = _scripted_loop(monkeypatch, [
        m.TURN_EMPTY, m.TURN_FORCE_FINAL, m.TURN_DONE])
    res = m._run_model_turn_loop([{"role": "user", "content": "go"}], run_id="run-x")
    assert seen[-1][2] is True  # the escalation ran as a forced final
    assert res["run_id"] == "run-x"
    assert res["counters"]["empty_turns"] == 1
    assert res["counters"]["stalls"] == 1
    assert res["counters"]["forced_finals"] == 1
    assert m._RUN_ID is None  # cleared after the run


def test_log_event_carries_run_id(monkeypatch):
    lines = []
    monkeypatch.setattr(m, "_llog", type("L", (), {"write": lambda self, s: lines.append(s)})())

    def fake(messages, *a, **k):
        m._log_event("req", {"x": 1})
        return m._turn_done("completed_unverified")
    monkeypatch.setattr(m, "model_turn", fake)
    monkeypatch.setattr(m, "VERIFY_CMD", "")
    m._run_model_turn_loop([{"role": "user", "content": "go"}], run_id="run-abc")
    assert '"run": "run-abc"' in lines[0]
    assert any('"dir": "run_end"' in ln and "run-abc" in ln for ln in lines)
