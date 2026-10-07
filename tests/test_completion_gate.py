#!/usr/bin/env python3
"""Completion gates: an answer cut at finish_reason=length is not "done", and
the opt-in MINION_VERIFY_CMD decides completed_verified vs. retry."""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


class _FakeDelta:
    def __init__(self, content=None):
        self.content = content
        self.tool_calls = None
        self.model_extra = {}
        self.reasoning_content = None


class _FakeChoice:
    def __init__(self, delta, finish_reason):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, text, finish_reason):
        self.choices = [_FakeChoice(_FakeDelta(text), finish_reason)]
        self.usage = None
        self.model_extra = {}


class _NoSpinner:
    _t = None

    def __init__(self, **kw):
        pass

    def start(self):
        pass

    def stop(self):
        pass


def _setup(monkeypatch, tmp_path, replies):
    monkeypatch.setattr(m, "_interrupt_watcher", lambda: None)
    monkeypatch.setattr(m, "LifeSpinner", _NoSpinner)
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path))
    monkeypatch.setattr(m, "VERIFY_CMD", "")
    it = iter(replies)
    monkeypatch.setattr(m, "open_stream",
                        lambda *a, **k: iter([_FakeChunk(*next(it))]))


# --- truncation ----------------------------------------------------------------
def test_length_cut_answer_is_continued_not_completed(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, [("The steps are: 1) ", "length"),
                                   ("2) done.", "stop")])
    messages = [{"role": "user", "content": "explain"}]
    res = m._run_model_turn_loop(messages)
    assert res["terminal"] == "completed_unverified"
    assert res["steps"] == 1
    assert any(x["role"] == "user" and "cut off by the output token limit" in x["content"]
               for x in messages)
    assert messages[-1] == {"role": "assistant", "content": "2) done."}


def test_length_cut_after_retries_is_truncated(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, [("part one", "length"), ("part two", "length")])
    monkeypatch.setattr(m, "TRUNCATION_RETRY_LIMIT", 1)
    messages = [{"role": "user", "content": "explain"}]
    res = m._run_model_turn_loop(messages)
    assert res["terminal"] == "truncated"
    assert "Truncated by token limit" in messages[-1]["content"]


def test_length_cut_with_retries_disabled_is_truncated(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, [("part one", "length")])
    monkeypatch.setattr(m, "TRUNCATION_RETRY_LIMIT", 0)
    res = m._run_model_turn_loop([{"role": "user", "content": "explain"}])
    assert res["terminal"] == "truncated"


# --- MINION_VERIFY_CMD -----------------------------------------------------------
def test_verify_cmd_true_is_completed_verified(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, [("done", "stop")])
    monkeypatch.setattr(m, "VERIFY_CMD", "true")
    res = m._run_model_turn_loop([{"role": "user", "content": "fix it"}])
    assert res["terminal"] == "completed_verified"


def test_verify_cmd_false_retries_then_budget(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, [("done", "stop")] * 10)
    monkeypatch.setattr(m, "VERIFY_CMD", "echo 3 tests failed; false")
    monkeypatch.setattr(m, "VERIFY_RETRY_LIMIT", 2)
    messages = [{"role": "user", "content": "fix it"}]
    res = m._run_model_turn_loop(messages)
    assert res["terminal"] == "budget_exhausted"
    assert res["budget"] == "verify"
    notes = [x["content"] for x in messages
             if x["role"] == "user" and "verification command" in x["content"]]
    assert len(notes) == 2
    assert "3 tests failed" in notes[0]
    # verify logs are cleaned up
    assert not [f for f in os.listdir(tmp_path / "bg-logs") if f.endswith(".log")]


def test_verify_cmd_timeout_counts_as_failure(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, [("done", "stop")])
    monkeypatch.setattr(m, "VERIFY_TIMEOUT", 1)
    ok, out = m._run_verify_cmd("sleep 30")
    assert ok is False
    assert "timed out" in out
