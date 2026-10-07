#!/usr/bin/env python3
"""--prompt-file maps the run's terminal state to an exit code, and headless
runs deny approvals instead of failing open (blocked → exit 3)."""
import io
import json
import os
import socket
import subprocess
import sys
import tempfile

import pytest

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
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
    def __init__(self, delta, finish_reason):
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


def _one_shot(monkeypatch, tmp_path):
    prompt = tmp_path / "task.txt"
    prompt.write_text("do the task\n")
    monkeypatch.setattr(m, "_PROMPT_FILE", str(prompt))
    monkeypatch.setattr(m, "ACTIVE", object())
    monkeypatch.setattr(m, "_ensure_ctx_probe", lambda src: None)
    monkeypatch.setattr(m, "_interrupt_watcher", lambda: None)
    monkeypatch.setattr(m, "LifeSpinner", _NoSpinner)
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path / "minion-home"))  # protected
    monkeypatch.setattr(m, "VERIFY_CMD", "")
    monkeypatch.setattr(m, "HEADLESS_APPROVAL", False)  # restored after the test
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # not a TTY
    monkeypatch.setattr(m, "_METRICS", None)
    monkeypatch.setenv("MINION_SESSIONS_DIR", str(tmp_path / "sessions"))
    with pytest.raises(SystemExit) as exc:
        m._run_one_shot()
    return exc.value.code


def test_turn_cap_exits_4(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "MAX_MODEL_TURNS", 2)
    monkeypatch.setattr(m, "model_turn", lambda *a, **k: m.TURN_TOOL)
    assert _one_shot(monkeypatch, tmp_path) == 4


def test_completed_exits_0(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "model_turn", lambda *a, **k: m._turn_done("completed_unverified"))
    assert _one_shot(monkeypatch, tmp_path) == 0


def _write_then_answer(target):
    replies = iter([
        [_FakeChunk(_FakeDelta(tool_calls=[_FakeTC(
            0, "c1", "write_file", '{"path": "%s", "content": "x"}' % target)]),
            "tool_calls")],
        [_FakeChunk(_FakeDelta(content="wrote it"), "stop")],
    ])
    return lambda *a, **k: iter(next(replies))


def test_headless_approval_denies_and_exits_3(monkeypatch, tmp_path):
    target = tmp_path / "out.txt"
    monkeypatch.setattr(m, "YOLO", False)
    monkeypatch.setattr(m, "APPROVE_LEVEL", None)
    monkeypatch.setattr(m, "_assess_risk", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no classifier call needed when nobody can approve")))
    monkeypatch.setattr(m, "open_stream", _write_then_answer(target))
    assert _one_shot(monkeypatch, tmp_path) == 3
    assert not target.exists()


def test_headless_with_explicit_approval_level_runs(monkeypatch, tmp_path):
    target = tmp_path / "out.txt"
    monkeypatch.setattr(m, "YOLO", False)
    monkeypatch.setattr(m, "APPROVE_LEVEL", "high")
    monkeypatch.setattr(m, "open_stream", _write_then_answer(target))
    assert _one_shot(monkeypatch, tmp_path) == 0
    assert target.read_text() == "x"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_unreachable_server_exits_5(tmp_path):
    prompt = tmp_path / "task.txt"
    prompt.write_text("hello\n")
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "MINION_HOME": str(tmp_path),
        "MINION_ENV_FILE": str(tmp_path / "no.env"),
        "MINION_BASE_URL": f"http://127.0.0.1:{_free_port()}/v1",
        "MINION_MODEL": "stub",
        "MINION_API_KEY": "sk-noop",
    }
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "minion.py"),
                           "--prompt-file", str(prompt)],
                          env=env, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 5, (proc.stdout[-2000:], proc.stderr[-2000:])
    assert "terminal: model_error" in proc.stderr


def test_help_documents_exit_codes(tmp_path):
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "minion.py"), "--help"],
                          env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path),
                               "MINION_HOME": str(tmp_path)},
                          stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 0
    assert "Exit codes" in proc.stdout and "3 blocked" in proc.stdout


# --- trial row (MINION_RESULT_FILE) and headless transcript -------------------

def test_result_file_row(monkeypatch, tmp_path):
    out = tmp_path / "rows" / "results.jsonl"
    monkeypatch.setattr(m, "RESULT_FILE", str(out))
    monkeypatch.setenv("MINION_TASK_ID", "t-1")
    monkeypatch.setenv("MINION_REASONING_ONLY_RETRIES", "0")  # an ablation knob
    monkeypatch.setenv("MINION_FAKE_API_KEY", "sk-should-not-leak-123")

    def fake(*a, **k):
        m._account_usage({"input_tokens": 100, "output_tokens": 20})
        return m._turn_done("completed_unverified")
    monkeypatch.setattr(m, "model_turn", fake)
    assert _one_shot(monkeypatch, tmp_path) == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["terminal"] == "completed_unverified" and row["exit_code"] == 0
    assert row["steps"] == 0 and row["input_tokens"] == 100
    assert row["output_tokens"] == 20 and row["tokens"] == 120
    assert isinstance(row["wall_s"], float)
    assert row["task_id"] == "t-1" and row["run_id"].startswith("run-")
    assert row["infra_error"] is False and row["passed"] is None
    assert set(row["counters"]) >= {"stalls", "empty_turns", "malformed",
                                    "collapses", "forced_finals"}
    knobs = row["manifest"]["knobs"]
    assert knobs["MINION_REASONING_ONLY_RETRIES"] == "0"
    assert "MINION_FAKE_API_KEY" not in knobs
    assert "sk-should-not-leak-123" not in out.read_text()
    assert row["manifest"]["prompt_sha"] and row["manifest"]["harness_sha"]


def test_result_file_row_on_budget(monkeypatch, tmp_path):
    out = tmp_path / "results.jsonl"
    monkeypatch.setattr(m, "RESULT_FILE", str(out))
    monkeypatch.setattr(m, "MAX_MODEL_TURNS", 2)
    monkeypatch.setattr(m, "model_turn", lambda *a, **k: m.TURN_TOOL)
    assert _one_shot(monkeypatch, tmp_path) == 4
    row = json.loads(out.read_text())
    assert row["terminal"] == "budget_exhausted" and row["budget"] == "turns"
    assert row["steps"] == 2 and row["counters"]["forced_finals"] == 1


def _only_session(tmp_path):
    files = list((tmp_path / "sessions").glob("*.json"))
    assert len(files) == 1, files
    return json.loads(files[0].read_text())


def test_headless_saves_transcript(monkeypatch, tmp_path, capsys):
    target = tmp_path / "out.txt"
    monkeypatch.setattr(m, "YOLO", False)
    monkeypatch.setattr(m, "APPROVE_LEVEL", "high")
    monkeypatch.setattr(m, "open_stream", _write_then_answer(target))
    assert _one_shot(monkeypatch, tmp_path) == 0
    data = _only_session(tmp_path)
    assert data["headless"] is True and data["status"] == "completed_unverified"
    assert f"session: {data['id']}" in capsys.readouterr().err
    roles = [x["role"] for x in data["messages"]]
    assert roles[:4] == ["system", "user", "assistant", "tool"]
    assert data["messages"][2]["tool_calls"][0]["id"] == "c1"
    assert data["messages"][3]["tool_call_id"] == "c1"


def test_headless_keyboard_interrupt(monkeypatch, tmp_path):
    out = tmp_path / "results.jsonl"
    monkeypatch.setattr(m, "RESULT_FILE", str(out))

    def boom(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(m, "model_turn", boom)
    assert _one_shot(monkeypatch, tmp_path) == 130
    data = _only_session(tmp_path)
    assert data["status"] == "cancelled" and data["status_detail"] == "interrupted"
    row = json.loads(out.read_text())
    assert row["terminal"] == "cancelled" and row["detail"] == "interrupted"
    assert row["exit_code"] == 130


def test_headless_save_off(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "HEADLESS_SAVE", 0)
    monkeypatch.setattr(m, "model_turn", lambda *a, **k: m._turn_done("completed_unverified"))
    assert _one_shot(monkeypatch, tmp_path) == 0
    assert not (tmp_path / "sessions").exists()
