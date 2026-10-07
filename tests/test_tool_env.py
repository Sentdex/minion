#!/usr/bin/env python3
"""Provider keys must not reach shell commands, tool results or the traffic log."""
import json
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


class _Src:
    def __init__(self, key):
        self.api_key = key


def _setup(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path))
    monkeypatch.setattr(m, "_confirm", lambda action, **_: True)
    monkeypatch.setenv("FOO_API_KEY", "sk-test-123")
    monkeypatch.setenv("MINION_SOURCE_X_API_KEY", "sk-test-456")
    # $name-resolved key whose env name doesn't look secret
    monkeypatch.setenv("ZAI_KEY_PLAIN", "zk-resolved-789")
    monkeypatch.setitem(m.SOURCES, "x", _Src("zk-resolved-789"))


def test_run_bash_env_is_scrubbed(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    out = m.run_bash("env", timeout=10)
    assert out.startswith("[exit 0]"), out
    for bad in ("FOO_API_KEY", "sk-test-123", "MINION_SOURCE_X", "sk-test-456",
                "ZAI_KEY_PLAIN", "zk-resolved-789"):
        assert bad not in out, bad
    assert "PATH=" in out and "HOME=" in out


def test_env_pass_opt_in(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-passme-000")
    monkeypatch.setenv("MINION_BASH_ENV_PASS", "GITHUB_TOKEN")
    env = m._tool_env()
    assert env["GITHUB_TOKEN"] == "ghp-passme-000"
    assert "FOO_API_KEY" not in env


def test_sanitize_redacts_short_and_long(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    assert "sk-test-123" not in m._sanitize_tool_result("leak sk-test-123")
    assert "<redacted>" in m._sanitize_tool_result("leak sk-test-123")
    big = ("line\n" * 400) + "zk-resolved-789"
    assert "zk-resolved-789" not in m._sanitize_tool_result(big)


def test_log_event_redacts(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    logf = open(tmp_path / "t.log", "w")
    monkeypatch.setattr(m, "_llog", logf)
    m._log_event("req", {"messages": [{"content": "key sk-test-456"}]})
    logf.close()
    line = (tmp_path / "t.log").read_text()
    assert "sk-test-456" not in line
    json.loads(line)  # still valid JSON
