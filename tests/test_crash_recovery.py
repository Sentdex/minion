#!/usr/bin/env python3
"""A crash mid-batch leaves a durable record: each tool result is saved as it
arrives, undone calls stay PENDING on disk, and resume turns them into
OUTCOME UNKNOWN without re-running anything."""
import builtins
import json
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


def test_crash_mid_batch_leaves_pending_then_outcome_unknown(monkeypatch):
    os.environ["MINION_SESSIONS_DIR"] = _tmp
    os.environ["MINION_HOME"] = _tmp
    monkeypatch.setattr(m, "_interrupt_watcher", lambda: None)
    monkeypatch.setattr(m, "LifeSpinner", _NoSpinner)
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)
    chunks = [_FakeChunk(_FakeDelta(tool_calls=[
        _FakeTC(0, "call_1", "write_file", '{"path":"a","content":"x"}'),
        _FakeTC(1, "call_2", "run_bash", '{"command":"make deploy"}'),
    ]))]
    monkeypatch.setattr(m, "open_stream", lambda messages, **kw: iter(chunks))
    runs = []

    def fake_run_tool(name, args):
        runs.append(name)
        if name == "run_bash":
            raise SystemExit(9)  # hard crash while call_2 is in flight
        return "wrote 1 bytes to a"
    monkeypatch.setattr(m, "run_tool", fake_run_tool)

    sid = "20250101-170000-crash0"
    messages = [{"role": "system", "content": m.SYSTEM},
                {"role": "user", "content": "deploy"}]
    try:
        m._run_model_turn_loop(messages, on_step=lambda: m._write_session(sid, messages))
    except SystemExit:
        pass
    else:
        raise AssertionError("SystemExit should propagate")
    assert m._TOOL_STEP_HOOK is None, "on_step hook leaked past the run"

    with open(m._session_path(sid)) as f:
        on_disk = json.load(f)["messages"]
    results = {x["tool_call_id"]: x["content"] for x in on_disk if x.get("role") == "tool"}
    assert results["call_1"] == "wrote 1 bytes to a"
    assert results["call_2"] == m.TOOL_PENDING

    resumed = m._session_messages(sid)
    results = {x["tool_call_id"]: x["content"] for x in resumed if x.get("role") == "tool"}
    assert results["call_1"] == "wrote 1 bytes to a"
    assert results["call_2"] == m.TOOL_OUTCOME_UNKNOWN
    assert runs.count("run_bash") == 1, "an in-doubt call was re-executed"
