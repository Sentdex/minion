#!/usr/bin/env python3
"""wait_background reads and deletes the log, so it must only accept minion's
own bg-*.log files under SESSION_HOME/bg-logs."""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402

DEAD_PID = 2**22 + 12345  # beyond default pid_max; never alive


def test_rejects_arbitrary_path(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path / "home"))
    victim = tmp_path / "important.txt"
    victim.write_text("keep me\n")
    res = m.wait_background(pid=DEAD_PID, log_path=str(victim))
    assert res.startswith("ERROR ["), res
    assert victim.read_text() == "keep me\n"


def test_rejects_traversal_and_symlink(monkeypatch, tmp_path):
    home = tmp_path / "home"
    bg = home / "bg-logs"
    bg.mkdir(parents=True)
    monkeypatch.setattr(m, "SESSION_HOME", str(home))
    victim = tmp_path / "secret.log"
    victim.write_text("x")
    link = bg / "bg-evil.log"
    link.symlink_to(victim)
    for p in (str(bg / ".." / ".." / "secret.log"), str(link)):
        assert m.wait_background(pid=DEAD_PID, log_path=p).startswith("ERROR [")
    assert victim.exists()


def test_map_pointing_outside_is_rejected(monkeypatch, tmp_path):
    home = tmp_path / "home"
    bg = home / "bg-logs"
    bg.mkdir(parents=True)
    monkeypatch.setattr(m, "SESSION_HOME", str(home))
    victim = tmp_path / "v.txt"
    victim.write_text("x")
    (bg / f"bg-pid-{DEAD_PID}.map").write_text('{"log_path": "%s"}' % victim)
    assert m.wait_background(pid=DEAD_PID).startswith("ERROR [")
    assert victim.exists()


def test_no_pid_substring_fallback(monkeypatch, tmp_path):
    home = tmp_path / "home"
    bg = home / "bg-logs"
    bg.mkdir(parents=True)
    monkeypatch.setattr(m, "SESSION_HOME", str(home))
    other = bg / "bg-unrelated.log"
    other.write_text(f"mentions {DEAD_PID}\n[exit: 0]\n")
    assert m.wait_background(pid=DEAD_PID).startswith("ERROR [")
    assert other.exists()


def test_real_bg_log_is_read_and_removed(monkeypatch, tmp_path):
    home = tmp_path / "home"
    bg = home / "bg-logs"
    bg.mkdir(parents=True)
    monkeypatch.setattr(m, "SESSION_HOME", str(home))
    log = bg / "bg-20260101-000000-abc123.log"
    log.write_text("done\n[exit: 0]\n")
    # 5.3: a PID needs a map file (run_bash wrote it); log_path alone is refused.
    assert m.wait_background(pid=DEAD_PID, log_path=str(log)).startswith("ERROR [not_found]")
    (bg / f"bg-pid-{DEAD_PID}.map").write_text('{"log_path": "%s"}' % log)
    assert m.wait_background(pid=DEAD_PID, log_path=str(log)) == "[exit 0]\ndone"
    assert not log.exists()
    # via the pid map alone too
    log.write_text("again\n[exit: 2]\n")
    assert m.wait_background(pid=DEAD_PID) == "[exit 2]\nagain"


def test_log_path_must_match_the_pid_map(monkeypatch, tmp_path):
    home = tmp_path / "home"
    bg = home / "bg-logs"
    bg.mkdir(parents=True)
    monkeypatch.setattr(m, "SESSION_HOME", str(home))
    mine = bg / "bg-20260101-000000-aaaaaa.log"
    other = bg / "bg-20260101-000000-bbbbbb.log"
    mine.write_text("x\n[exit: 0]\n")
    other.write_text("y\n[exit: 0]\n")
    (bg / f"bg-pid-{DEAD_PID}.map").write_text('{"log_path": "%s"}' % mine)
    assert m.wait_background(pid=DEAD_PID, log_path=str(other)).startswith("ERROR [bad_log_path]")
    assert other.exists()
