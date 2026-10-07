#!/usr/bin/env python3
"""Background jobs: MINION_BASH_MAX_SECONDS watchdog kills the process group,
kill_background only signals PIDs this minion started (after _confirm), and
survivors are reported at exit."""
import os
import sys
import tempfile
import time

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


def _setup(monkeypatch, tmp_path, confirm=True):
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path))
    monkeypatch.setattr(m, "_confirm", lambda action, payload=None: confirm)
    monkeypatch.setattr(m, "MAX_RUN_SECONDS", 0)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_watchdog_kills_long_job(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(m, "BASH_MAX_SECONDS", 1)
    t0 = time.time()
    out = m.run_bash("sleep 30")  # leading sleep → would poll for 40 s
    assert time.time() - t0 < 2.5, out
    assert "[killed by minion: exceeded MINION_BASH_MAX_SECONDS=1]" in out, out


def test_watchdog_kills_backgrounded_job(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(m, "BASH_MAX_SECONDS", 1)
    out = m.run_bash("sleep 30", timeout=0.2)
    assert out.startswith("[background] PID "), out
    pid = int(out.split("PID ", 1)[1].split()[0])
    time.sleep(2.5)
    assert m._BG_JOBS[pid]["proc"].poll() is not None
    assert not _alive(pid)


def test_kill_background_refuses_foreign_pid(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    res = m.kill_background(os.getpid())
    assert res.startswith("ERROR [not_found]"), res
    assert _alive(os.getpid())


def test_kill_background_kills_own_job_and_is_gated(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, confirm=False)
    monkeypatch.setattr(m, "BASH_MAX_SECONDS", 0)
    pid = int(m._run_bash("sleep 30")[0])
    try:
        assert m.kill_background(pid).startswith("DENIED by user")
        assert _alive(pid)
        monkeypatch.setattr(m, "_confirm", lambda action, payload=None: True)
        assert m.kill_background(pid) == f"[killed] PID {pid}"
        assert m._BG_JOBS[pid]["proc"].poll() is not None
        assert m.kill_background(pid).startswith("[already exited]")
        res = m.wait_background(pid=pid, timeout=5)
        assert "[killed by minion: killed]" in res, res
    finally:
        m._kill_bg_job(m._BG_JOBS[pid])


def test_map_file_records_job(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(m, "BASH_MAX_SECONDS", 0)
    import json
    pid, log = m._run_bash("true")
    data = json.loads((tmp_path / "bg-logs" / f"bg-pid-{pid}.map").read_text())
    assert data["log_path"] == log and data["command"] == "true"
    assert data["minion_pid"] == os.getpid()
    m._poll_pid(pid, 5)


def test_exit_survivors_listed_and_killed_on_yes(monkeypatch, tmp_path, capsys):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(m, "BASH_MAX_SECONDS", 0)
    monkeypatch.setattr(m, "_BG_JOBS", {})
    pid, _ = m._run_bash("sleep 30")
    pid = int(pid)
    m._handle_bg_survivors(interactive=False)  # headless: list only
    assert f"PID {pid}" in capsys.readouterr().err
    assert _alive(pid)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    m._handle_bg_survivors(interactive=True)
    assert m._BG_JOBS[pid]["proc"].poll() is not None
