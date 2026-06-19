#!/usr/bin/env python3
"""Approval default resolution: repo default, env file, env var, CLI overrides."""
import importlib
import os
import sys
import types

import httpx


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def reload_minion(argv=None, env=None, env_file="/dev/null"):
    argv = ["minion.py"] if argv is None else argv
    env = {} if env is None else env
    for k in list(os.environ):
        if k.startswith("MINION_"):
            del os.environ[k]
    os.environ["MINION_ENV_FILE"] = env_file
    os.environ.update(env)
    sys.argv = argv
    import minion
    return importlib.reload(minion)


def test_default_prompts_for_everything():
    m = reload_minion()
    assert m.YOLO is False
    assert m.APPROVE_LEVEL is None
    assert m.DEFAULT_APPROVE_LEVEL is None
    assert "prompt:all" in m._banner()


def test_minion_approval_env_sets_persistent_default():
    m = reload_minion(env={"MINION_APPROVAL": "medium"})
    assert m.YOLO is False
    assert m.APPROVE_LEVEL == "medium"
    assert m.DEFAULT_APPROVE_LEVEL == "medium"


def test_minion_approval_can_come_from_env_file(tmp_path):
    env_path = tmp_path / "minion.env"
    env_path.write_text("MINION_APPROVAL=low\n", encoding="utf-8")
    m = reload_minion(env_file=str(env_path))
    assert m.YOLO is False
    assert m.APPROVE_LEVEL == "low"
    assert m.DEFAULT_APPROVE_LEVEL == "low"


def test_cli_approval_overrides_env_yolo():
    m = reload_minion(
        argv=["minion.py", "--approval", "low"],
        env={"MINION_APPROVAL": "yolo"},
    )
    assert m.YOLO is False
    assert m.APPROVE_LEVEL == "low"
    assert m.DEFAULT_APPROVE_LEVEL == "low"


def test_cli_yolo_overrides_approval_level():
    m = reload_minion(
        argv=["minion.py", "--approval", "medium", "--yolo"],
        env={"MINION_APPROVAL": "low"},
    )
    assert m.YOLO is True
    assert m.APPROVE_LEVEL is None
    assert m.DEFAULT_APPROVE_LEVEL == "medium"


def test_prompt_all_aliases_are_accepted():
    m = reload_minion(argv=["minion.py", "--approval", "all"],
                      env={"MINION_APPROVAL": "medium"})
    assert m.YOLO is False
    assert m.APPROVE_LEVEL is None
    assert m.DEFAULT_APPROVE_LEVEL is None
    assert m._normalize_approval("strict") == ("prompt_all", None)


def test_risk_classifier_retries_connection_failures():
    m = reload_minion()
    saved = (
        m.client,
        m._log_event,
        m.RISK_CONNECTION_RETRIES,
        m.RISK_CONNECTION_RETRY_SECONDS,
    )
    calls = {"n": 0}
    request = httpx.Request("POST", "http://test/v1/chat/completions")

    class Resp:
        choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(
                content='{"level":"low","reason":"retry recovered"}'))]

        def model_dump(self):
            return {"ok": True}

    def create(**_):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise m.APIConnectionError(message="test connection failure",
                                       request=request)
        return Resp()

    try:
        m.RISK_CONNECTION_RETRIES = 3
        m.RISK_CONNECTION_RETRY_SECONDS = 0
        m._log_event = lambda *_: None
        m.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)))

        assert m._assess_risk("run: true") == ("low", "retry recovered")
        assert calls["n"] == 4
    finally:
        (
            m.client,
            m._log_event,
            m.RISK_CONNECTION_RETRIES,
            m.RISK_CONNECTION_RETRY_SECONDS,
        ) = saved
