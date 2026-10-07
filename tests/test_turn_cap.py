#!/usr/bin/env python3
"""The turn cap always runs exactly one forced-final turn (also when one was
already pending) and ends the run as budget_exhausted/turns."""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


def _drive(monkeypatch, script):
    calls = []

    def fake_model_turn(messages, rl, ms, empty_turn_count=0,
                        forced_final=False, recovery_sampling=False):
        calls.append(forced_final)
        if forced_final:
            return m.TURN_DONE
        return script[len(calls) - 1] if len(calls) <= len(script) else m.TURN_TOOL

    monkeypatch.setattr(m, "MAX_MODEL_TURNS", 3)
    monkeypatch.setattr(m, "model_turn", fake_model_turn)
    res = m._run_model_turn_loop([{"role": "user", "content": "go"}])
    return calls, res


def test_turn_cap_runs_one_forced_final(monkeypatch):
    calls, res = _drive(monkeypatch, [m.TURN_TOOL] * 3)
    assert calls == [False, False, False, True]
    assert res["terminal"] == "budget_exhausted"
    assert res["budget"] == "turns"


def test_turn_cap_runs_pending_forced_final(monkeypatch):
    # The 3rd turn asks for a forced final; the cap must still run it once.
    calls, res = _drive(monkeypatch, [m.TURN_TOOL, m.TURN_TOOL, m.TURN_FORCE_FINAL])
    assert calls == [False, False, False, True]
    assert res["terminal"] == "budget_exhausted"
