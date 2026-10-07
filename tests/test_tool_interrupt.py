#!/usr/bin/env python3
"""Ctrl-C (KeyboardInterrupt) while a tool runs: the current call is recorded
as INTERRUPTED (outcome unknown), the rest as SKIPPED, the turn returns
TURN_ESC, and the REPL still saves the session."""
import builtins
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402
from test_esc_approval import _FakeChunk, _FakeDelta, _FakeTC  # noqa: E402

_NoSpinner = type("NoSpinner", (), {
    "__init__": lambda self, **kw: None,
    "start": lambda self: None,
    "stop": lambda self: None,
    "_t": None,
})


def _quiet(monkeypatch):
    os.environ["MINION_SESSIONS_DIR"] = _tmp
    os.environ["MINION_HOME"] = _tmp
    monkeypatch.setattr(m, "_interrupt_watcher", lambda: None)
    monkeypatch.setattr(m, "LifeSpinner", _NoSpinner)
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)


def _raise_ki(*a, **k):
    raise KeyboardInterrupt


def test_native_ctrl_c_interrupted_then_skipped(monkeypatch):
    _quiet(monkeypatch)
    chunks = [
        _FakeChunk(_FakeDelta(tool_calls=[_FakeTC(0, "call_1", "run_bash", '{"command":"sleep 9"}')])),
        _FakeChunk(_FakeDelta(tool_calls=[_FakeTC(1, "call_2", "list_dir", '{}')])),
    ]
    monkeypatch.setattr(m, "open_stream", lambda messages, **kw: iter(chunks))
    monkeypatch.setattr(m, "run_tool", _raise_ki)
    messages = [{"role": "user", "content": "go"}]
    assert m.model_turn(messages) == m.TURN_ESC
    tools = [x for x in messages if x.get("role") == "tool"]
    assert [t["tool_call_id"] for t in tools] == ["call_1", "call_2"]
    assert tools[0]["content"].startswith("INTERRUPTED")
    assert "outcome unknown" in tools[0]["content"]
    assert tools[1]["content"].startswith("SKIPPED")
    assert messages[-1]["role"] == "user" and "Ctrl-C" in messages[-1]["content"]
    assert m._repair_tool_pairs(list(messages)) == 0  # history stays valid


def test_text_protocol_ctrl_c(monkeypatch):
    _quiet(monkeypatch)
    text = ('[minion_tool_call]{"name": "list_dir", "arguments": {}}[/minion_tool_call]')
    chunks = [_FakeChunk(_FakeDelta(content=text), finish_reason="stop")]
    monkeypatch.setattr(m, "open_stream", lambda messages, **kw: iter(chunks))
    monkeypatch.setattr(m, "run_tool", _raise_ki)
    messages = [{"role": "user", "content": "go"}]
    assert m.model_turn(messages) == m.TURN_ESC
    assert "INTERRUPTED" in messages[-1]["content"]
    assert m.TOOL_PENDING not in messages[-1]["content"]


def test_run_tool_reraises_keyboard_interrupt(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setitem(m.DISPATCH, "list_dir", _raise_ki)
    try:
        m.run_tool("list_dir", {})
    except KeyboardInterrupt:
        return
    raise AssertionError("run_tool swallowed KeyboardInterrupt")


def test_loop_maps_ctrl_c_to_cancelled(monkeypatch):
    _quiet(monkeypatch)
    chunks = [_FakeChunk(_FakeDelta(tool_calls=[_FakeTC(0, "c1", "list_dir", '{}')]))]
    monkeypatch.setattr(m, "open_stream", lambda messages, **kw: iter(chunks))
    monkeypatch.setattr(m, "run_tool", _raise_ki)
    res = m._run_model_turn_loop([{"role": "user", "content": "go"}])
    assert res["terminal"] == "cancelled"


def test_repl_saves_after_ctrl_c(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "_banner", lambda: "")
    monkeypatch.setattr(m, "_maybe_refresh_description", lambda *a, **k: None)
    monkeypatch.setattr(m, "_run_model_turn_loop", lambda *a, **k: _raise_ki())
    writes = []
    real_write = m._write_session
    monkeypatch.setattr(m, "_write_session",
                        lambda sid, msgs, meta=None: (writes.append(list(msgs)),
                                                      real_write(sid, msgs, meta)))
    prompts = iter(["hello", "/quit"])
    monkeypatch.setattr(m, "read_multiline", lambda history=None: next(prompts))
    m.main()  # must not propagate the KeyboardInterrupt
    assert writes, "session was not saved after Ctrl-C"
    assert any(msg.get("content") == "hello" for msg in writes[-1])
