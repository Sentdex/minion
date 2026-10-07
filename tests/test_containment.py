#!/usr/bin/env python3
"""Credential stores need approval to read; ordinary reads stay unprompted."""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_ENV_FILE"] = "/dev/null"
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


def _setup(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("PRIVATE KEY\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(m, "YOLO", False)
    monkeypatch.setattr(m, "APPROVE_LEVEL", "high")
    monkeypatch.setattr(m, "_assess_risk", lambda action, payload=None: ("low", "stub"))
    prompts = []

    def ask(prompt, default="y"):
        prompts.append(prompt)
        return "n"
    monkeypatch.setattr(m, "_ask_approval", ask)
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    return home, prompts


def test_read_ssh_key_denied(monkeypatch, tmp_path):
    home, prompts = _setup(monkeypatch, tmp_path)
    out = m.read_file(str(home / ".ssh" / "id_rsa"))
    assert out.startswith("DENIED by user")
    assert "PRIVATE KEY" not in out
    assert prompts and "[PROTECTED" in prompts[0]
    assert m.read_file("~/.ssh/id_rsa").startswith("DENIED by user")
    assert m.list_dir(str(home / ".ssh")).startswith("DENIED by user")


def test_ordinary_read_never_prompts(monkeypatch, tmp_path):
    _, prompts = _setup(monkeypatch, tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n")
    assert "hello" in m.read_file(str(tmp_path / "notes.txt"))
    assert "notes.txt" in m.list_dir(str(tmp_path))
    # control-plane files (not credentials) are readable without a prompt
    assert m._is_protected(m.__file__, read=True) is None
    assert prompts == []
