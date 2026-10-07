#!/usr/bin/env python3
"""Risk-classifier context for project-root path decisions."""
import json
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_ENV_FILE"] = "/dev/null"
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


def test_downloads_project_file_is_in_project(monkeypatch, tmp_path):
    home = tmp_path / "home"
    project = home / "Downloads" / "didenstuff"
    project.mkdir(parents=True)
    target = project / "pose_editor.py"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    monkeypatch.setattr(m, "PROJECT_ROOT", os.path.realpath(project))

    payload = json.loads(m._risk_user_message(f"edit {target}"))

    assert payload["project_root"] == os.path.realpath(project)
    assert payload["primary_path"] == os.path.realpath(target)
    assert payload["path_scope"] == "in_project"
    assert payload["path_in_downloads"] is True
    assert "Downloads" in payload["scope_guidance"]


def test_downloads_file_outside_project_is_outside(monkeypatch, tmp_path):
    home = tmp_path / "home"
    project = home / "code" / "app"
    downloads_project = home / "Downloads" / "didenstuff"
    project.mkdir(parents=True)
    downloads_project.mkdir(parents=True)
    target = downloads_project / "pose_editor.py"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    monkeypatch.setattr(m, "PROJECT_ROOT", os.path.realpath(project))

    payload = json.loads(m._risk_user_message(f"edit {target}"))

    assert payload["project_root"] == os.path.realpath(project)
    assert payload["primary_path"] == os.path.realpath(target)
    assert payload["path_scope"] == "outside_project"
    assert payload["path_in_downloads"] is True


def test_write_diff_goes_to_classifier_payload(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(m, "PROJECT_ROOT", os.path.realpath(tmp_path))
    monkeypatch.setattr(m, "YOLO", False)
    monkeypatch.setattr(m, "APPROVE_LEVEL", "medium")
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    target = tmp_path / "a.py"
    target.write_text("old line\n")
    seen = []

    def assess(action, payload=None):
        seen.append((action, payload))
        return ("low", "stub")
    monkeypatch.setattr(m, "_assess_risk", assess)

    assert m.write_file(str(target), "old line\nnew line\n").startswith("wrote")
    action, payload = seen[-1]
    assert "+new line" in payload
    assert action == f"write {target} (18 bytes)"  # diff is not in the action

    assert m.edit_file(str(target), "new line", "edited line") == f"edited {target}"
    assert "-new line" in seen[-1][1] and "+edited line" in seen[-1][1]

    msg = json.loads(m._risk_user_message(action, payload))
    assert msg["payload"] == payload and msg["action"] == action
    assert msg["primary_path"] == os.path.realpath(target)

    monkeypatch.setattr(m, "APPROVAL_DIFF_CHARS", 200)
    m.write_file(str(target), "x" * 50 + "\n" * 500)
    capped = seen[-1][1]
    assert len(capped) < 260 and "more diff chars" in capped
