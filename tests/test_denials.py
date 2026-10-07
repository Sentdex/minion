#!/usr/bin/env python3
"""Denials tell the model not to retry, and MINION_DENY_LIMIT consecutive
denials stop the turn as `blocked`."""
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


def _setup(monkeypatch):
    monkeypatch.setattr(m, "_interrupt_watcher", lambda: None)
    monkeypatch.setattr(m, "LifeSpinner", _NoSpinner)
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)
    monkeypatch.setattr(m, "YOLO", False)
    monkeypatch.setattr(m, "_confirm", lambda action, payload=None: False)


def test_denial_text_says_do_not_retry(monkeypatch, tmp_path):
    _setup(monkeypatch)
    out = m.write_file(str(tmp_path / "x"), "y")
    assert out.startswith("DENIED by user: write ")
    assert "Do not retry" in out
    assert not (tmp_path / "x").exists()


def test_third_consecutive_denial_stops_the_turn(monkeypatch, tmp_path):
    _setup(monkeypatch)
    monkeypatch.setattr(m, "DENY_LIMIT", 3)
    turns = {"n": 0}

    def stream(messages, **kw):
        turns["n"] += 1
        p = str(tmp_path / f"f{turns['n']}")
        return iter([_FakeChunk(_FakeDelta(tool_calls=[
            _FakeTC(0, f"c{turns['n']}", "write_file",
                    '{"path": "%s", "content": "x"}' % p)]))])
    monkeypatch.setattr(m, "open_stream", stream)
    messages = [{"role": "user", "content": "write files"}]
    res = m._run_model_turn_loop(messages)
    assert res["terminal"] == "blocked", res
    assert turns["n"] == 3, f"turn continued after the limit: {turns['n']} model calls"
    assert m._repair_tool_pairs(list(messages)) == 0


def test_limit_skips_rest_of_batch_and_resets_on_success(monkeypatch, tmp_path):
    _setup(monkeypatch)
    monkeypatch.setattr(m, "DENY_LIMIT", 2)
    calls = [_FakeTC(0, "a", "write_file", '{"path": "%s", "content": "x"}' % (tmp_path / "a")),
             _FakeTC(1, "b", "list_dir", '{"path": "%s"}' % tmp_path),  # not denied: resets
             _FakeTC(2, "c", "write_file", '{"path": "%s", "content": "x"}' % (tmp_path / "c")),
             _FakeTC(3, "d", "write_file", '{"path": "%s", "content": "x"}' % (tmp_path / "d")),
             _FakeTC(4, "e", "list_dir", '{"path": "%s"}' % tmp_path)]
    monkeypatch.setattr(m, "open_stream", lambda messages, **kw: iter(
        [_FakeChunk(_FakeDelta(tool_calls=calls))]))
    m._CONSECUTIVE_DENIALS = 0
    messages = [{"role": "user", "content": "go"}]
    assert m.model_turn(messages) == m.TURN_DONE
    assert m._LAST_TURN_REASON == "blocked"
    res = {x["tool_call_id"]: x["content"] for x in messages if x.get("role") == "tool"}
    assert res["b"] != m.TOOL_PENDING and not res["b"].startswith("DENIED")
    assert res["d"].startswith("DENIED")
    assert res["e"].startswith("SKIPPED")


def test_one_shot_deny_limit_exits_3(monkeypatch, tmp_path):
    _setup(monkeypatch)
    monkeypatch.setattr(m, "DENY_LIMIT", 1)
    monkeypatch.setattr(m, "_ensure_ctx_probe", lambda src: None)
    monkeypatch.setattr(m, "ACTIVE", object())
    pf = tmp_path / "task.txt"
    pf.write_text("do it")
    monkeypatch.setattr(m, "_PROMPT_FILE", str(pf))
    monkeypatch.setattr(m, "HEADLESS_APPROVAL", False)  # restored after the test
    monkeypatch.setattr(m.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(m, "open_stream", lambda messages, **kw: iter([_FakeChunk(_FakeDelta(
        tool_calls=[_FakeTC(0, "c", "write_file",
                            '{"path": "%s", "content": "x"}' % (tmp_path / "x"))]))]))
    try:
        m._run_one_shot()
    except SystemExit as e:
        assert e.code == 3, e.code
    else:
        raise AssertionError("one-shot did not exit")
