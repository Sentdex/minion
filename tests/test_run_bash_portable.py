#!/usr/bin/env python3
"""run_bash must not depend on a `setsid` binary (macOS has none): it detaches
via start_new_session and still reports exit codes and output."""
import os
import sys
import tempfile
import time

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


def _setup(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path))
    monkeypatch.setattr(m, "_confirm", lambda action: True)
    monkeypatch.setenv("PATH", "/bin:/usr/bin")  # no setsid on macOS here


def test_quick_command_returns_output_without_setsid(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    assert m.run_bash("echo hi", timeout=10) == "[exit 0]\nhi"
    assert m.run_bash("exit 3", timeout=10).startswith("[exit 3]")


def test_background_job_gets_own_session_and_is_reaped(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    out = m.run_bash("sleep 2", timeout=0.3)
    assert out.startswith("[background] PID "), out
    pid = int(out.split("PID ", 1)[1].split()[0])
    try:
        assert os.getsid(pid) != os.getsid(0)
    except ProcessLookupError:
        raise AssertionError("background job died early")
    t0 = time.time()
    res = m.wait_background(pid=pid, timeout=10)
    # A zombie would keep kill(pid, 0) succeeding until the timeout.
    assert res.startswith("[exit 0]"), res
    assert time.time() - t0 < 8
