#!/usr/bin/env python3
"""_delete_old_bg_logs never prunes the log (or map) of a job whose PID is
still alive, however old it is or however many newer logs exist."""
import json
import os
import subprocess
import sys
import tempfile
import time

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _make_logs(bg, n, now):
    paths = []
    for i in range(n):  # i=0 is the oldest
        p = bg / f"bg-{i:03d}.log"
        p.write_text("x")
        t = now - 3600 + i
        os.utime(p, (t, t))
        paths.append(p)
    return paths


def _map(bg, pid, log_path, mtime):
    mp = bg / f"bg-pid-{pid}.map"
    mp.write_text(json.dumps({"log_path": str(log_path), "command": "sleep"}))
    os.utime(mp, (mtime, mtime))
    return mp


def test_live_job_log_survives_count_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path))
    bg = tmp_path / "bg-logs"
    bg.mkdir()
    now = time.time()
    logs = _make_logs(bg, 52, now)
    live_map = _map(bg, os.getpid(), logs[0], now - 3600)
    dead_map = _map(bg, _dead_pid(), logs[1], now - 3600)
    m._delete_old_bg_logs(max_logs=50)
    assert logs[0].exists() and live_map.exists()   # oldest, but its PID is alive
    assert not logs[1].exists() and not dead_map.exists()  # dead job: pruned
    assert all(p.exists() for p in logs[2:])        # the 50 newest non-live logs


def test_live_job_log_survives_age_prune(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path))
    bg = tmp_path / "bg-logs"
    bg.mkdir()
    old = time.time() - 30 * 86400
    live_log, dead_log = bg / "bg-live.log", bg / "bg-dead.log"
    for p in (live_log, dead_log):
        p.write_text("x")
        os.utime(p, (old, old))
    live_map = _map(bg, os.getpid(), live_log, old)
    m._delete_old_bg_logs(max_age_days=7)
    assert live_log.exists() and live_map.exists()
    assert not dead_log.exists()


def test_pid_alive():
    assert m._pid_alive(os.getpid())
    assert not m._pid_alive(_dead_pid())
    assert not m._pid_alive("nope") and not m._pid_alive(0)
