#!/usr/bin/env python3
"""Protected paths (control plane + credential stores) always prompt, even at
approval level `high` with the classifier saying low."""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_ENV_FILE"] = "/dev/null"
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


def _setup(monkeypatch, tmp_path, level="high"):
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(proj)
    monkeypatch.setattr(m, "PROJECT_ROOT", os.path.realpath(proj))
    monkeypatch.setattr(m, "SESSION_HOME", str(tmp_path / "minion-home"))
    monkeypatch.setattr(m, "YOLO", False)
    monkeypatch.setattr(m, "APPROVE_LEVEL", level)
    monkeypatch.setattr(m, "_assess_risk", lambda action, payload=None: ("low", "stub"))
    prompts = []

    def ask(prompt, default="y"):
        prompts.append((prompt, default))
        return default  # user just hits Enter
    monkeypatch.setattr(m, "_ask_approval", ask)
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    return home, proj, prompts


def test_write_dotenv_prompts_at_level_high(monkeypatch, tmp_path):
    _, proj, prompts = _setup(monkeypatch, tmp_path)
    target = proj / ".env"
    assert m.write_file(str(target), "MINION_APPROVAL=yolo\n").startswith("DENIED by user")
    assert not target.exists()
    assert len(prompts) == 1
    prompt, default = prompts[0]
    assert "[PROTECTED" in prompt and default == "n"


def test_run_bash_append_to_home_env_prompts(monkeypatch, tmp_path):
    home, _, prompts = _setup(monkeypatch, tmp_path)
    assert m.run_bash("echo X >> ~/.env", timeout=5).startswith("DENIED by user")
    assert not (home / ".env").exists()
    assert prompts and "[PROTECTED" in prompts[0][0]
    # $HOME-expanded form and credential dirs are caught too
    assert m._action_protection("run: cat $HOME/.ssh/id_rsa")
    assert m._action_protection("run: cp x ~/.zshrc")


def test_symlink_to_dotenv_shows_realpath(monkeypatch, tmp_path):
    home, proj, prompts = _setup(monkeypatch, tmp_path)
    (home / ".env").write_text("A=1\n")
    os.symlink(home / ".env", proj / "cfg")
    assert m.write_file("cfg", "B=2\n").startswith("DENIED by user")
    assert (home / ".env").read_text() == "A=1\n"
    assert os.path.realpath(home / ".env") in prompts[0][0]


def test_control_plane_paths_are_protected(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    assert m._is_protected(m.__file__)
    assert m._is_protected(os.path.join(m.SESSION_HOME, "sessions", "x.json"))
    assert m._is_protected(os.path.join(m.PROJECT_ROOT, ".git", "hooks", "pre-commit"))
    extra = tmp_path / "secret-dir"
    monkeypatch.setenv("MINION_PROTECTED_EXTRA", f"/nonexistent:{extra}")
    assert m._is_protected(str(extra / "k"))


def test_ordinary_paths_and_bg_logs_not_protected(monkeypatch, tmp_path):
    _, proj, prompts = _setup(monkeypatch, tmp_path)
    assert m._is_protected(str(proj / "app.py")) is None
    assert m._is_protected(str(proj / ".env.example")) is None
    bg = os.path.join(m.SESSION_HOME, "bg-logs", "bg-1.log")
    assert m._is_protected(bg) is None
    assert m._action_protection(f"run: tail {bg}") is None
    assert m._action_protection("run: ls -la > /dev/null") is None
    # level high + unprotected → auto-allow, no prompt
    assert m.write_file(str(proj / "app.py"), "x = 1\n").startswith("wrote")
    assert prompts == []


def test_yolo_skips_protected_prompt(monkeypatch, tmp_path):
    _, proj, prompts = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(m, "YOLO", True)
    assert m._confirm(f"write {proj / '.env'} (3 bytes)") is True
    assert prompts == []
