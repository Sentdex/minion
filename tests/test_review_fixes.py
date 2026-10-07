#!/usr/bin/env python3
"""Regression tests for the harness-hardening review findings."""
import json
import os
import subprocess
import sys
import tempfile
import time

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp
os.environ["MINION_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402

import pytest  # noqa: E402


def _quiet(monkeypatch):
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)


# --- spinner stopped when Ctrl-C lands inside open_stream ---------------------
def test_ctrl_c_in_open_stream_stops_spinner(monkeypatch):
    _quiet(monkeypatch)
    spinners = []

    class RecSpinner:
        def __init__(self, **kw):
            self.running = False
            self._t = None
            spinners.append(self)

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

    def boom(messages, **kw):
        raise KeyboardInterrupt
    monkeypatch.setattr(m, "LifeSpinner", RecSpinner)
    monkeypatch.setattr(m, "open_stream", boom)
    with pytest.raises(KeyboardInterrupt):
        m.model_turn([{"role": "user", "content": "hi"}])
    assert spinners and not any(s.running for s in spinners)


# --- secret-name matching: whole segments, not substrings ---------------------
_NOT_SECRETS = {"SSH_AUTH_SOCK": "/private/tmp/com.apple.launchd.abc/Listeners",
                "GIT_AUTHOR_NAME": "Adeeb Aldkheel",
                "GIT_COMMITTER_EMAIL": "someone@example.com",
                "XAUTHORITY": "/home/u/.Xauthority-x",
                "TOKENIZERS_PARALLELISM": "false-ish-value"}
_SECRETS = {"GITHUB_TOKEN": "ghp-abcdefgh-123", "AWS_SECRET_ACCESS_KEY": "aws-secret-xyz1",
            "NPM_AUTH_TOKEN": "npm-tok-12345678", "DB_PASSWORD": "hunter2-very-secret",
            "OPENAI_APIKEY": "sk-openai-0000000"}


def test_secret_names_are_whole_segments(monkeypatch):
    for k, v in {**_NOT_SECRETS, **_SECRETS}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("MINION_BASH_ENV_PASS", raising=False)
    env = m._tool_env()
    for k, v in _NOT_SECRETS.items():
        assert env.get(k) == v, k
        assert m._redact(f"x {v} y") == f"x {v} y", k
    for k, v in _SECRETS.items():
        assert k not in env, k
        assert v not in m._redact(f"x {v} y"), k


def test_source_model_and_url_survive_redaction(monkeypatch, tmp_path):
    monkeypatch.setenv("MINION_SOURCE_LOC_MODEL", "qwen3-coder-30b-a3b")
    monkeypatch.setenv("MINION_SOURCE_LOC_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("MINION_SOURCE_LOC_API_KEY", "sk-loc-secret-1")
    text = "model=qwen3-coder-30b-a3b url=http://127.0.0.1:9/v1 key=sk-loc-secret-1"
    assert m._redact(text) == ("model=qwen3-coder-30b-a3b url=http://127.0.0.1:9/v1 "
                               "key=<redacted>")
    env = m._tool_env()  # still kept out of shell commands
    assert not any(k.startswith("MINION_SOURCE_") for k in env)
    monkeypatch.setattr(m, "MODEL", "qwen3-coder-30b-a3b")
    out = tmp_path / "rows.jsonl"
    m._append_result_row(m._trial_row({"terminal": "completed_unverified"}, 0), str(out))
    row = json.loads(out.read_text())
    assert row["model"] == "qwen3-coder-30b-a3b"


# --- _repair_tool_pairs ---------------------------------------------------------
def _tc(id_, name):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": "{}"}}


def test_repair_pairs_duplicate_ids_by_position():
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": None,
             "tool_calls": [_tc("x", "read_file"), _tc("x", "list_dir")]},
            {"role": "tool", "tool_call_id": "x", "content": "R1"},
            {"role": "tool", "tool_call_id": "x", "content": "R2"}]
    assert m._repair_tool_pairs(msgs) == 0
    assert [t["content"] for t in msgs[2:]] == ["R1", "R2"]
    # a third result for the same id is a real duplicate and is dropped
    msgs.append({"role": "tool", "tool_call_id": "x", "content": "R3"})
    assert m._repair_tool_pairs(msgs) == 1
    assert [t["content"] for t in msgs[2:]] == ["R1", "R2"]


def test_repair_pending_only_rewrites_harness_lines():
    quoted = f'TOOL_PENDING = "{m.TOOL_PENDING}"'
    obs = (f"Observation (read_file): line1\n{quoted}\n"
           f"Observation (run_bash): {m.TOOL_PENDING}")
    msgs = [{"role": "user", "content": obs}]
    assert m._repair_tool_pairs(msgs) == 1
    content = msgs[0]["content"]
    assert quoted in content  # file content quoting the literal is untouched
    assert content.endswith(f"Observation (run_bash): {m.TOOL_OUTCOME_UNKNOWN}")


class _Delta:
    def __init__(self, tool_calls):
        self.content = None
        self.tool_calls = tool_calls
        self.model_extra = {}
        self.reasoning_content = None


class _Chunk:
    def __init__(self, delta=None):
        self.choices = ([type("C", (), {"delta": delta, "finish_reason": "tool_calls"})()]
                        if delta is not None else [])
        self.usage = None
        self.model_extra = {}


class _TC:
    def __init__(self, index, id_, name):
        self.index = index
        self.id = id_
        self.function = type("F", (), {"name": name, "arguments": "{}"})()


def test_model_turn_dedupes_tool_call_ids(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "_interrupt_watcher", lambda: None)
    monkeypatch.setattr(m, "LifeSpinner", type("S", (), {
        "__init__": lambda self, **kw: None, "start": lambda self: None,
        "stop": lambda self: None, "_t": None}))
    # index 0 reuses "call_1"; index 1 has none and must not collide with it;
    # index 2 repeats index 0's id.
    chunks = [_Chunk(_Delta([_TC(0, "call_1", "list_dir")])),
              _Chunk(_Delta([_TC(1, None, "list_dir")])),
              _Chunk(_Delta([_TC(2, "call_1", "list_dir")])), _Chunk()]
    monkeypatch.setattr(m, "open_stream", lambda messages, **kw: iter(chunks))
    results = iter(["A", "B", "C"])
    monkeypatch.setattr(m, "run_tool", lambda *a, **k: next(results))
    msgs = [{"role": "user", "content": "ls"}]
    monkeypatch.setattr(m, "_CONSECUTIVE_DENIALS", 0)  # earlier tests deny on purpose
    monkeypatch.setattr(m, "_LAST_TOOL_DENIED", False)
    assert m.model_turn(msgs) == m.TURN_TOOL
    ids = [c["id"] for c in msgs[1]["tool_calls"]]
    assert len(set(ids)) == 3 and ids[0] == "call_1", ids
    assert [(t["tool_call_id"], t["content"]) for t in msgs[2:5]] == list(zip(ids, "ABC"))
    assert m._repair_tool_pairs(msgs) == 0


# --- classifier copy keeps everything after quoting it can't model --------------
def test_strip_comments_fails_safe_on_ansi_c_quotes():
    cmd = "echo $'a\\' # '; python -c 'print(1)'"
    assert m._strip_shell_comments(cmd) == cmd
    assert m._strip_shell_comments("ls # low risk") == "ls"


# --- protected paths -----------------------------------------------------------
def _home_proj(monkeypatch, tmp_path):
    home, proj = tmp_path / "home", tmp_path / "proj"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("PRIVATE")
    (proj / ".git").mkdir(parents=True)
    (proj / "src").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(proj)
    monkeypatch.setattr(m, "PROJECT_ROOT", os.path.realpath(proj))
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path / "mh"))
    monkeypatch.setattr(m, "YOLO", False)
    return home, proj


@pytest.mark.parametrize("cmd", [
    "cd ~ && cat .ssh/id_rsa",
    "cd ~ && echo x >> .zshrc",
    "cd && cat .ssh/id_rsa",
    "(cd ~; cat .ssh/id_rsa)",
    "cat ~/.s*/id_rsa",
    "cat $(echo ~)/.ssh/id_rsa",
    "cat `echo ~`/.ssh/id_rsa",
    "cd .git && echo > config",
    "git config core.hooksPath /tmp/h",
    "git -C . config --global user.name x",
    "find ~ -name id_rsa -exec cat {} +",
    "grep -r BEGIN ~",
    "cat ~/.git-credentials",
])
def test_command_protection_bypasses_are_caught(monkeypatch, tmp_path, cmd):
    _home_proj(monkeypatch, tmp_path)
    assert m._action_protection("run: " + cmd), cmd


@pytest.mark.parametrize("cmd", [
    "ls -la", "cd src && ls", "git config --get user.name", "git config -l",
    "rg foo .", "find . -name '*.py'", "git status", "echo $(date) > out.txt",
    "ls ~",
])
def test_command_protection_leaves_ordinary_commands(monkeypatch, tmp_path, cmd):
    _home_proj(monkeypatch, tmp_path)
    assert m._action_protection("run: " + cmd) is None, cmd


def test_cd_home_cat_key_prompts_at_level_high(monkeypatch, tmp_path):
    _quiet(monkeypatch)
    _home_proj(monkeypatch, tmp_path)
    monkeypatch.setattr(m, "APPROVE_LEVEL", "high")
    monkeypatch.setattr(m, "HEADLESS_APPROVAL", False)
    prompts = []
    monkeypatch.setattr(m, "_ask_approval",
                        lambda p, default="y": prompts.append(p) or "n")
    assert m._confirm("run: cd ~ && cat .ssh/id_rsa") is False
    assert prompts and "[PROTECTED" in prompts[0]


@pytest.mark.parametrize("rel", [".git-credentials", ".docker/config.json",
                                 ".kube/config", ".npmrc", ".pypirc",
                                 ".cargo/credentials.toml"])
def test_more_credential_stores_are_read_protected(monkeypatch, tmp_path, rel):
    home, _ = _home_proj(monkeypatch, tmp_path)
    assert m._is_protected(str(home / rel), read=True), rel


def test_headless_protected_write_prints_no_diff(monkeypatch, tmp_path, capsys):
    _home_proj(monkeypatch, tmp_path)
    (tmp_path / "proj" / ".env").write_text("DB_PASSWORD=hunter2-very-secret\n")
    monkeypatch.setattr(m, "HEADLESS_APPROVAL", True)
    monkeypatch.setattr(m, "APPROVE_LEVEL", None)
    assert m.write_file(".env", "x").startswith("DENIED")
    out = capsys.readouterr()
    assert "hunter2" not in out.out + out.err


def test_interactive_protected_write_hides_old_lines(monkeypatch, tmp_path, capsys):
    _home_proj(monkeypatch, tmp_path)
    (tmp_path / "proj" / ".env").write_text("DB_PASSWORD=hunter2-very-secret\n")
    monkeypatch.setattr(m, "HEADLESS_APPROVAL", False)
    monkeypatch.setattr(m, "APPROVE_LEVEL", "high")
    monkeypatch.setattr(m, "_ask_approval", lambda p, default="y": "n")
    assert m.write_file(".env", "NEW=1\n").startswith("DENIED")
    out = capsys.readouterr().out
    assert "hunter2" not in out and "+NEW=1" in out and "hidden" in out


def test_edit_file_on_protected_file_asks_before_matching(monkeypatch, tmp_path):
    _home_proj(monkeypatch, tmp_path)
    (tmp_path / "proj" / ".env").write_text("OPENAI_API_KEY=sk-abc\n")
    asked = []
    monkeypatch.setattr(m, "_confirm", lambda action, **k: asked.append(action) or False)
    for guess in ("OPENAI_API_KEY=sk-z", "OPENAI_API_KEY=sk-a", "="):
        out = m.edit_file(".env", guess, "")
        assert out.startswith("DENIED"), out  # same answer whatever the match
    assert asked == ["read .env"] * 3


# --- risk floor spellings ---------------------------------------------------------
@pytest.mark.parametrize("cmd", [
    "git -C . push origin", "git -c a=b push", "cat f | /bin/sh", "x | /usr/bin/bash",
    "find . -delete", "find . -name x -exec rm {} +",
    "python3 -c 'import urllib.request; urllib.request.urlopen(\"http://x\")'",
    "ps eww -o command= -p $PPID", "ps -E", "cat /proc/1/environ",
])
def test_risk_floor_catches_more_spellings(cmd):
    assert m._risk_floor("run: " + cmd, facts={})[0] == "high", cmd


@pytest.mark.parametrize("cmd", ["git status", "git log --grep push", "ps aux | grep x",
                                 "find . -name '*.py'", "ls -la"])
def test_risk_floor_leaves_ordinary_commands(cmd):
    assert m._risk_floor("run: " + cmd, facts={})[0] is None, cmd


# --- background jobs that fork with & --------------------------------------------
def _group_alive(pgid):
    out = subprocess.run(["ps", "-A", "-o", "pid=,pgid="], capture_output=True,
                         text=True).stdout
    return any(line.split()[1:] == [str(pgid)] for line in out.splitlines()
               if len(line.split()) == 2)


def test_watchdog_kills_backgrounded_grandchild(monkeypatch, tmp_path):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path))
    monkeypatch.setattr(m, "_confirm", lambda action, **k: True)
    monkeypatch.setattr(m, "BASH_MAX_SECONDS", 1)
    monkeypatch.setattr(m, "BG_KILL_GRACE", 0.5)
    out = m.run_bash("sleep 47 > /dev/null 2>&1 &", timeout=5)
    assert out.startswith("[exit 0]"), out
    job = list(m._BG_JOBS.values())[-1]
    pgid = job["proc"].pid
    assert _group_alive(pgid)
    deadline = time.time() + 6
    while time.time() < deadline and _group_alive(pgid):
        time.sleep(0.2)
    assert not _group_alive(pgid), "leftover `sleep 47` survived the cap"
    assert job["killed"]


def test_kill_background_and_survivors_see_grandchild(monkeypatch, tmp_path):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path))
    monkeypatch.setattr(m, "_confirm", lambda action, **k: True)
    monkeypatch.setattr(m, "BASH_MAX_SECONDS", 0)
    monkeypatch.setattr(m, "BG_KILL_GRACE", 0.5)
    m.run_bash("sleep 48 > /dev/null 2>&1 &", timeout=5)
    job = list(m._BG_JOBS.values())[-1]
    pid = job["proc"].pid
    try:
        assert job in m._bg_survivors()
        assert m.kill_background(pid) == f"[killed] PID {pid}"
        assert not _group_alive(pid)
        assert job not in m._bg_survivors()
    finally:
        try:
            os.killpg(pid, 9)
        except OSError:
            pass


# --- wait_background on a map left by an earlier minion ---------------------------
def test_wait_background_refuses_stale_map(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path))
    bg = tmp_path / "bg-logs"
    bg.mkdir()
    log = bg / "bg-old-000000.log"
    log.write_text("old output\n")
    other = os.getppid()  # alive, but not a job of this process
    (bg / f"bg-pid-{other}.map").write_text(json.dumps(
        {"log_path": str(log), "minion_pid": 1}))
    t0 = time.time()
    out = m.wait_background(other)
    assert "stale_job" in out and time.time() - t0 < 2
    assert log.exists()


def test_prune_ignores_live_pid_when_log_finished(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path))
    bg = tmp_path / "bg-logs"
    bg.mkdir()
    log = bg / "bg-old-000001.log"
    log.write_text("done\n[exit: 0]\n")
    old = time.time() - 30 * 86400
    os.utime(log, (old, old))
    mp = bg / f"bg-pid-{os.getppid()}.map"
    mp.write_text(json.dumps({"log_path": str(log)}))
    m._delete_old_bg_logs()
    assert not log.exists()


# --- /recover archives what it folds ----------------------------------------------
def test_recover_passes_session_id(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "_banner", lambda: "")
    seen = []

    def fake_loop(messages, **kw):
        seen.append(kw)
        return {"terminal": "completed_unverified"}
    prompts = iter(["/recover", "/quit"])
    monkeypatch.setattr(m, "read_multiline", lambda history=None: next(prompts))
    monkeypatch.setattr(m, "_run_model_turn_loop", fake_loop)
    m.main()
    assert seen and seen[0].get("session_id")


# --- blocked runs say why ---------------------------------------------------------
def _drive(monkeypatch, turn):
    _quiet(monkeypatch)
    monkeypatch.setattr(m, "model_turn", turn)
    monkeypatch.setattr(m, "VERIFY_CMD", None)
    return m._run_model_turn_loop([{"role": "user", "content": "x"}])


def test_blocked_detail_deny_limit(monkeypatch):
    r = _drive(monkeypatch, lambda *a, **k: m._turn_done("blocked", "deny_limit"))
    assert (r["terminal"], r["detail"]) == ("blocked", "deny_limit")
    assert m._trial_row(r, 3)["detail"] == "deny_limit"


def test_blocked_detail_model(monkeypatch):
    r = _drive(monkeypatch, lambda *a, **k: m._turn_done("blocked", "model"))
    assert (r["terminal"], r["detail"]) == ("blocked", "model")


def test_blocked_detail_policy(monkeypatch):
    def turn(*a, **k):
        m._HEADLESS_DENIALS.append("write x")
        return m.TURN_TOOL
    r = _drive(monkeypatch, turn)
    assert (r["terminal"], r["detail"]) == ("blocked", "policy_blocked")
    del m._HEADLESS_DENIALS[:]


def test_completed_has_no_detail(monkeypatch):
    r = _drive(monkeypatch, lambda *a, **k: m._turn_done("completed_unverified"))
    assert r["detail"] is None
