#!/usr/bin/env python3
"""Test the Esc-at-approval → back-to-chat behavior.

No live model or terminal needed — we monkeypatch the risk classifier and the
approval prompt so we can drive the control-flow deterministically, then assert
that:
  1. `_confirm` raises `_EscToChat` when the user picks Esc.
  2. `_confirm` returns True/False for Y/n as before.
  3. `run_tool` lets `_EscToChat` propagate (doesn't swallow it as a string).
  4. `model_turn` returns `TURN_ESC` when a tool is escaped, and fills in
     results for remaining tool calls so the message history stays valid.
  5. The native tool-call path: when Esc fires on the 1st of 2 tool calls,
     both calls get result messages (CANCELLED / SKIPPED).
"""
import os
import sys
import builtins
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m

# Capture the REAL print once at import time so tests can silence + restore it
# without the "builtins.print = print" self-reference bug (after reassigning,
# the bare name `print` resolves to the new value, not the original builtin).
_REAL_PRINT = builtins.print


def _use_test_sessions_dir():
    os.environ["MINION_SESSIONS_DIR"] = _tmp
    os.environ["MINION_HOME"] = _tmp


def setup_function(_function):
    _use_test_sessions_dir()


# ---------------------------------------------------------------------------
# 1. _confirm raises _EscToChat on "esc"
# ---------------------------------------------------------------------------
def test_confirm_esc_raises():
    saved_ask, saved_assess = m._ask_approval, m._assess_risk
    m._ask_approval = lambda prompt, default="y": "esc"
    m._assess_risk = lambda action, payload=None: ("high", "because test")
    try:
        try:
            m._confirm("write foo.py (10 bytes)")
        except m._EscToChat:
            print("PASS — _confirm raised _EscToChat on esc")
            return
        raise AssertionError("_confirm did NOT raise _EscToChat on esc")
    finally:
        m._ask_approval, m._assess_risk = saved_ask, saved_assess


# ---------------------------------------------------------------------------
# 2. _confirm still returns True/False for Y/n
# ---------------------------------------------------------------------------
def test_confirm_y_n():
    saved_ask, saved_assess = m._ask_approval, m._assess_risk
    m._assess_risk = lambda action, payload=None: ("high", "because test")
    builtins.print = lambda *a, **k: None  # hush
    try:
        m._ask_approval = lambda prompt, default="y": "y"
        assert m._confirm("write foo.py") is True
        m._ask_approval = lambda prompt, default="y": "n"
        assert m._confirm("write foo.py") is False
    finally:
        m._ask_approval, m._assess_risk = saved_ask, saved_assess
        builtins.print = _REAL_PRINT
    print("PASS — _confirm returns True/False for Y/n")


# ---------------------------------------------------------------------------
# 3. run_tool propagates _EscToChat (doesn't swallow it)
# ---------------------------------------------------------------------------
def test_run_tool_propagates_esc():
    saved_confirm = m._confirm
    # make write_file's _confirm raise
    m._confirm = lambda action, **_: (_ for _ in ()).throw(m._EscToChat(action))
    builtins.print = lambda *a, **k: None
    # suppress the spinner so we don't spawn a thread
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
    })
    raised = False
    try:
        try:
            m.run_tool("write_file", {"path": "x", "content": "y"})
        except m._EscToChat:
            raised = True
    finally:
        m._confirm = saved_confirm
        builtins.print = _REAL_PRINT
    if raised:
        print("PASS — run_tool propagated _EscToChat")
        return
    raise AssertionError("run_tool swallowed _EscToChat")


# ---------------------------------------------------------------------------
# 4. model_turn returns TURN_ESC + keeps history valid
#    Drive it with a fake stream that emits two tool calls; escape the first.
# ---------------------------------------------------------------------------
class _FakeDelta:
    def __init__(self, content=None, tool_calls=None, reasoning=None):
        self.content = content
        self.tool_calls = tool_calls
        self.model_extra = {}
        self._reasoning = reasoning
    @property
    def reasoning_content(self):
        return self._reasoning


class _FakeChoice:
    def __init__(self, delta, finish_reason="tool_calls"):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, delta=None, finish_reason="tool_calls"):
        self.choices = [_FakeChoice(delta, finish_reason)] if delta is not None else []
        self.usage = None
        self.model_extra = {}


class _FakeTC:
    """Minimal stand-in for a streamed tool-call delta."""
    def __init__(self, index, id_, name, args):
        self.index = index
        self.id = id_
        self.function = type("F", (), {"name": name, "arguments": args})()


def test_model_turn_native_esc():
    saved_open_stream = m.open_stream
    saved_confirm = m._confirm
    saved_watcher = m._interrupt_watcher
    m._interrupt_watcher = lambda: None  # don't spawn the thread
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None

    chunks = [
        _FakeChunk(_FakeDelta(tool_calls=[_FakeTC(0, "call_1", "write_file", '{"path":"a","content":"x"}')])),
        _FakeChunk(_FakeDelta(tool_calls=[_FakeTC(1, "call_2", "edit_file", '{"path":"b","old":"o","new":"n"}')])),
        _FakeChunk(),  # final empty-choices chunk
    ]
    m.open_stream = lambda messages, **kw: iter(chunks)
    # Esc on the FIRST tool; the second should be skipped, not re-prompted.
    m._confirm = lambda action, **_: (_ for _ in ()).throw(m._EscToChat(action))
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "do it"}]
        status = m.model_turn(messages)
        assert status == m.TURN_ESC, f"expected TURN_ESC, got {status!r}"
        # The assistant(tool_calls) turn must be present…
        asst = [msg for msg in messages if msg.get("role") == "assistant" and msg.get("tool_calls")]
        assert len(asst) == 1, f"expected 1 assistant(tool_calls), got {len(asst)}"
        ids = [tc["id"] for tc in asst[0]["tool_calls"]]
        assert ids == ["call_1", "call_2"], f"tool call ids wrong: {ids}"
        # …and EVERY tool_call must have a matching tool result, or the chat
        # template will reject the context on the next request.
        tool_msgs = [msg for msg in messages if msg.get("role") == "tool"]
        assert len(tool_msgs) == 2, f"expected 2 tool results, got {len(tool_msgs)}"
        result_ids = {tm["tool_call_id"] for tm in tool_msgs}
        assert result_ids == {"call_1", "call_2"}, f"missing results: {result_ids}"
        # first = CANCELLED, second = SKIPPED
        contents = {tm["tool_call_id"]: tm["content"] for tm in tool_msgs}
        assert "CANCELLED" in contents["call_1"], contents["call_1"]
        assert "SKIPPED" in contents["call_2"], contents["call_2"]
        # a trailing user note telling the model what happened
        assert messages[-1]["role"] == "user"
        assert "Esc" in messages[-1]["content"]
    finally:
        m.open_stream = saved_open_stream
        m._confirm = saved_confirm
        m._interrupt_watcher = saved_watcher
        builtins.print = _REAL_PRINT
    print("PASS — model_turn returns TURN_ESC, history stays valid (all calls have results)")


# ---------------------------------------------------------------------------
# 5. Malformed native tool-call args are retried, not executed as {}
# ---------------------------------------------------------------------------
def test_model_turn_malformed_native_tool_args_retries_cleanly():
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    saved_run_tool = m.run_tool
    m._interrupt_watcher = lambda: None
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    chunks = [
        _FakeChunk(_FakeDelta(tool_calls=[
            _FakeTC(0, "call_bad", "write_file",
                    '{"path":"big.html","content":"unterminated')
        ]), finish_reason="tool_calls"),  # length cuts have their own test below
    ]
    m.open_stream = lambda messages, **kw: iter(chunks)
    m.run_tool = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("malformed tool args must not execute"))
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "write a large file"}]
        status = m.model_turn(messages)
        assert status == m.TURN_STREAM_CUT, f"expected TURN_STREAM_CUT, got {status!r}"
        assert len(messages) == 2, f"partial assistant/tool history leaked: {messages!r}"
        assert messages[-1]["role"] == "user"
        assert "malformed or truncated JSON arguments" in messages[-1]["content"]
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        m.run_tool = saved_run_tool
        builtins.print = _REAL_PRINT
    print("PASS — malformed native tool args retry cleanly without executing {}")


# ---------------------------------------------------------------------------
# 5b. A response cut at finish_reason=length never runs its tool calls, even
#     when their JSON parses; a missing tool-call id is synthesized.
# ---------------------------------------------------------------------------
def test_model_turn_length_cut_valid_tool_call_not_executed():
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    saved_run_tool = m.run_tool
    m._interrupt_watcher = lambda: None
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    chunks = [
        _FakeChunk(_FakeDelta(tool_calls=[
            _FakeTC(0, None, "write_file", '{"path":"half.py","content":"def f():"}')
        ]), finish_reason="length"),
    ]
    m.open_stream = lambda messages, **kw: iter(chunks)
    m.run_tool = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("length-cut tool calls must not execute"))
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "write a file"}]
        status = m.model_turn(messages)
        assert status == m.TURN_STREAM_CUT, f"expected TURN_STREAM_CUT, got {status!r}"
        assert len(messages) == 2, f"assistant/tool history leaked: {messages!r}"
        assert "output token limit" in messages[-1]["content"]
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        m.run_tool = saved_run_tool
        builtins.print = _REAL_PRINT


def test_model_turn_synthesizes_missing_tool_call_ids():
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    saved_run_tool = m.run_tool
    m._interrupt_watcher = lambda: None
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    chunks = [_FakeChunk(_FakeDelta(tool_calls=[_FakeTC(0, None, "list_dir", '{}')]))]
    m.open_stream = lambda messages, **kw: iter(chunks)
    m.run_tool = lambda *a, **k: "ok"
    try:
        messages = [{"role": "user", "content": "ls"}]
        assert m.model_turn(messages) == m.TURN_TOOL
        assert messages[1]["tool_calls"][0]["id"] == "call_0"
        assert messages[2]["tool_call_id"] == "call_0"
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        m.run_tool = saved_run_tool
        builtins.print = _REAL_PRINT


# ---------------------------------------------------------------------------
# 6. REPL drops back to chat input on TURN_ESC (no extra model turn)
# ---------------------------------------------------------------------------
def test_repl_breaks_on_esc():
    saved_read_multiline = m.read_multiline
    saved_model_turn = m.model_turn
    saved_banner = m._banner
    m._banner = lambda: ""
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    call_count = {"n": 0}
    def fake_model_turn(messages, reasoning_loop_cut_count=0, malformed_stream_cut_count=0,
                        empty_turn_count=0, forced_final=False, recovery_sampling=False):
        call_count["n"] += 1
        # first turn: model emits a tool call, user escapes it
        if call_count["n"] == 1:
            return m.TURN_ESC
        return m.TURN_DONE
    prompts = iter(["hello", "again", "/quit"])
    m.read_multiline = lambda history=None: next(prompts)
    m.model_turn = fake_model_turn
    try:
        m.main()
        # model_turn should have been called exactly once per *real* user turn
        # (the TURN_ESC must break the inner loop, not trigger a retry). "hello"
        # → 1 call (esc, break), "again" → 1 call (done, break) = 2 total.
        # If TURN_ESC were mishandled (fell through to the steps++ branch) the
        # first turn would loop and call model_turn many more times.
        assert call_count["n"] == 2, f"expected 2 model_turn calls, got {call_count['n']}"
    finally:
        m.read_multiline = saved_read_multiline
        m.model_turn = saved_model_turn
        m._banner = saved_banner
        builtins.print = _REAL_PRINT
    print("PASS — REPL breaks inner loop on TURN_ESC, no extra model turn")


# ---------------------------------------------------------------------------
# 7. Reasoning-only stalls use the forced-finalizer path
# ---------------------------------------------------------------------------
def test_reasoning_only_stall_requests_forced_final():
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    saved_limit = m.REASONING_ONLY_CHAR_LIMIT
    saved_retries = m.REASONING_ONLY_RETRY_LIMIT
    m._interrupt_watcher = lambda: None
    m.REASONING_ONLY_CHAR_LIMIT = 3
    # Default is 3 (two-tier: early stalls nudge to act, the last forces a
    # final). Pin to 1 so this first stall IS the last retry.
    m.REASONING_ONLY_RETRY_LIMIT = 1
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    chunks = [_FakeChunk(_FakeDelta(reasoning="abcd"))]
    m.open_stream = lambda messages, **kw: iter(chunks)
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        status = m.model_turn(messages)
        assert status == m.TURN_FORCE_FINAL, f"expected TURN_FORCE_FINAL, got {status!r}"
        assert "complete visible answer" in messages[-1]["content"]
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        m.REASONING_ONLY_CHAR_LIMIT = saved_limit
        m.REASONING_ONLY_RETRY_LIMIT = saved_retries
        builtins.print = _REAL_PRINT
    print("PASS — reasoning-only stall requests forced final answer")


def test_reasoning_only_first_stall_nudges_to_act_generically():
    # With the default two-tier limit, the first stall nudges to ACT (full
    # tools) rather than forcing a final, and the nudge has no /app/ paths.
    saved = (m.open_stream, m._interrupt_watcher, m.LifeSpinner,
             m.REASONING_ONLY_CHAR_LIMIT, m.REASONING_ONLY_RETRY_LIMIT)
    m._interrupt_watcher = lambda: None
    m.REASONING_ONLY_CHAR_LIMIT = 3
    m.REASONING_ONLY_RETRY_LIMIT = 3
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    m.open_stream = lambda messages, **kw: iter([_FakeChunk(_FakeDelta(reasoning="abcd"))])
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        status = m.model_turn(messages)
        assert status == m.TURN_STALL, f"expected TURN_STALL, got {status!r}"
        assert messages[-1]["content"] == m.REASONING_STALL_CONTINUE_NOTE
        assert "/app/" not in messages[-1]["content"]
    finally:
        (m.open_stream, m._interrupt_watcher, m.LifeSpinner,
         m.REASONING_ONLY_CHAR_LIMIT, m.REASONING_ONLY_RETRY_LIMIT) = saved
        builtins.print = _REAL_PRINT


# ---------------------------------------------------------------------------
# 8. Forced finalizer tool call becomes visible assistant text
# ---------------------------------------------------------------------------
def test_forced_final_tool_call_becomes_assistant_text():
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    m._interrupt_watcher = lambda: None
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    captured = {}

    def fake_open_stream(messages, **kw):
        captured.update(kw)
        args = '{"answer":"forced answer works","status":"answered"}'
        return iter([_FakeChunk(_FakeDelta(
            tool_calls=[_FakeTC(0, "final_1", "final_answer", args)]
        ))])

    m.open_stream = fake_open_stream
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        status = m.model_turn(messages, forced_final=True)
        assert status == m.TURN_DONE, f"expected TURN_DONE, got {status!r}"
        assert captured["tools"] == [m.FINAL_ANSWER_TOOL]
        assert captured["tool_choice"] == m.FINAL_ANSWER_TOOL_CHOICE
        assert captured["max_tokens"] == m.FORCED_FINAL_MAX_TOKENS
        assert messages[-1] == {"role": "assistant", "content": "forced answer works"}
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        builtins.print = _REAL_PRINT
    print("PASS — forced final_answer tool call becomes assistant text")


def test_forced_final_length_marks_partial_answer():
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    m._interrupt_watcher = lambda: None
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None

    def fake_open_stream(messages, **kw):
        return iter([_FakeChunk(_FakeDelta(content="partial final"), finish_reason="length")])

    m.open_stream = fake_open_stream
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        status = m.model_turn(messages, forced_final=True)
        assert status == m.TURN_DONE, f"expected TURN_DONE, got {status!r}"
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].startswith("partial final")
        assert "Truncated by token limit" in messages[-1]["content"]
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        builtins.print = _REAL_PRINT
    print("PASS — forced final length stop is marked in history")


def test_reasoning_only_empty_completion_forces_final_without_empty_assistant():
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    saved_limit = m.REASONING_ONLY_CHAR_LIMIT
    saved_retries = m.REASONING_ONLY_RETRY_LIMIT
    m._interrupt_watcher = lambda: None
    m.REASONING_ONLY_CHAR_LIMIT = 9999
    m.REASONING_ONLY_RETRY_LIMIT = 1
    # Reasoning capture (default on) deliberately re-feeds reasoning as an
    # assistant message; turn it off to test the no-capture path.
    saved_capture = m.REASONING_CAPTURE_CHARS
    m.REASONING_CAPTURE_CHARS = 0
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    m.open_stream = lambda messages, **kw: iter([
        _FakeChunk(_FakeDelta(reasoning="I should answer soon but never emit content."))
    ])
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        status = m.model_turn(messages)
        assert status == m.TURN_FORCE_FINAL, f"expected forced final, got {status!r}"
        assert messages[-1]["role"] == "user"
        assert "complete visible answer" in messages[-1]["content"]
        assert not any(msg.get("role") == "assistant" for msg in messages)
        # With capture on, the only assistant message is the non-empty reasoning.
        m.REASONING_CAPTURE_CHARS = 8000
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        status = m.model_turn(messages)
        assert status == m.TURN_FORCE_FINAL
        assert "complete visible answer" in messages[-1]["content"]
        assert all(msg["content"].strip() for msg in messages
                   if msg.get("role") == "assistant")
    finally:
        m.REASONING_CAPTURE_CHARS = saved_capture
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        m.REASONING_ONLY_CHAR_LIMIT = saved_limit
        m.REASONING_ONLY_RETRY_LIMIT = saved_retries
        builtins.print = _REAL_PRINT
    print("PASS — empty reasoning-only turns force final without empty assistant")


def test_repl_recover_command_forces_visible_checkpoint():
    saved_read_multiline = m.read_multiline
    saved_model_turn = m.model_turn
    saved_banner = m._banner
    m._banner = lambda: ""
    builtins.print = lambda *a, **k: None
    calls = []

    def fake_model_turn(messages, reasoning_loop_cut_count=0, malformed_stream_cut_count=0,
                        empty_turn_count=0, forced_final=False, recovery_sampling=False):
        calls.append({
            "forced_final": forced_final,
            "recovery_sampling": recovery_sampling,
            "last_message": messages[-1]["content"],
        })
        return m.TURN_DONE

    prompts = iter(["/recover save the audio diagnosis", "/quit"])
    m.read_multiline = lambda history=None: next(prompts)
    m.model_turn = fake_model_turn
    try:
        m.main()
        assert len(calls) == 1, calls
        assert calls[0]["forced_final"] is True
        assert calls[0]["recovery_sampling"] is True
        assert "Manual recovery requested" in calls[0]["last_message"]
        assert "save the audio diagnosis" in calls[0]["last_message"]
    finally:
        m.read_multiline = saved_read_multiline
        m.model_turn = saved_model_turn
        m._banner = saved_banner
        builtins.print = _REAL_PRINT
    print("PASS — /recover forces a low-temp visible checkpoint")


# ---------------------------------------------------------------------------
# Approval prompt fails closed (non-TTY line mode, raw TTY mode, Ctrl-C)
# ---------------------------------------------------------------------------
def _with_input(fn, value):
    saved = builtins.input

    def fake_input(prompt=""):
        if isinstance(value, BaseException):
            raise value
        return value
    builtins.input = fake_input
    try:
        return fn()
    finally:
        builtins.input = saved


def test_ask_approval_non_tty_fails_closed():
    ask = m._ask_approval
    assert _with_input(lambda: ask("? ", default="n"), "") == "n"
    assert _with_input(lambda: ask("? ", default="n"), "maybe") == "n"
    assert _with_input(lambda: ask("? ", default="y"), "maybe") == "n"
    # a stray blank line from a pipe never approves, whatever the default
    assert _with_input(lambda: ask("? ", default="y"), "") == "n"
    assert _with_input(lambda: ask("? ", default="y"), "yes") == "y"
    assert _with_input(lambda: ask("? ", default="n"), EOFError()) == "n"
    assert _with_input(lambda: ask("? ", default="n"), "y") == "y"
    assert _with_input(lambda: ask("? ", default="n"), "yes") == "y"
    assert _with_input(lambda: ask("? ", default="n"), "esc") == "esc"
    try:
        _with_input(lambda: ask("? "), KeyboardInterrupt())
    except m._EscToChat:
        pass
    else:
        raise AssertionError("Ctrl-C at the prompt must raise _EscToChat")


def test_confirm_high_risk_defaults_to_deny():
    saved_ask, saved_assess = m._ask_approval, m._assess_risk
    seen = {}

    def ask(prompt, default="y"):
        seen["prompt"], seen["default"] = prompt, default
        return default  # user just hit Enter
    m._ask_approval = ask
    builtins.print = lambda *a, **k: None
    try:
        m._assess_risk = lambda action, payload=None: ("high", "rm -rf")
        assert m._confirm("run: rm -rf x") is False
        assert seen["default"] == "n" and "[y/N/esc]" in seen["prompt"]
        m._assess_risk = lambda action, payload=None: ("medium", "edit")
        assert m._confirm("write foo.py") is True
        assert seen["default"] == "y" and "[Y/n/esc]" in seen["prompt"]
    finally:
        m._ask_approval, m._assess_risk = saved_ask, saved_assess
        builtins.print = _REAL_PRINT


def _ask_on_pty(feed, default):
    """Drive the raw-mode prompt through a pty; `feed` is [(delay, bytes)]."""
    import io
    import pty
    import threading
    import time as _time
    import termios
    master, slave = pty.openpty()
    attrs = termios.tcgetattr(slave)
    attrs[3] &= ~termios.ECHO  # nobody reads master: echo would block TCSADRAIN
    termios.tcsetattr(slave, termios.TCSANOW, attrs)
    stdin = os.fdopen(slave, "r")

    class _TTYOut(io.StringIO):
        def isatty(self):
            return True

    def writer():
        for delay, data in feed:
            _time.sleep(delay)
            os.write(master, data)
    saved_in, saved_out = sys.stdin, sys.stdout
    sys.stdin, sys.stdout = stdin, _TTYOut()
    t = threading.Thread(target=writer, daemon=True)
    try:
        os.write(master, b"y")  # type-ahead before the prompt: must be flushed
        t.start()
        return m._ask_approval("? ", default=default)
    finally:
        sys.stdin, sys.stdout = saved_in, saved_out
        t.join(2)
        stdin.close()
        os.close(master)


def test_ask_approval_tty_ignores_typeahead_unknown_keys_and_sequences():
    # stray key and an arrow-key sequence re-prompt; Enter takes the default
    assert _ask_on_pty([(0.2, b"x"), (0.2, b"\x1b[A"), (0.3, b"\r")], "n") == "n"
    assert _ask_on_pty([(0.2, b"?"), (0.2, b"Y")], "n") == "y"


if __name__ == "__main__":
    test_confirm_esc_raises()
    test_confirm_y_n()
    test_run_tool_propagates_esc()
    test_model_turn_native_esc()
    test_model_turn_malformed_native_tool_args_retries_cleanly()
    test_repl_breaks_on_esc()
    test_reasoning_only_stall_requests_forced_final()
    test_forced_final_tool_call_becomes_assistant_text()
    test_forced_final_length_marks_partial_answer()
    test_reasoning_only_empty_completion_forces_final_without_empty_assistant()
    test_repl_recover_command_forces_visible_checkpoint()
    test_ask_approval_non_tty_fails_closed()
    test_confirm_high_risk_defaults_to_deny()
    test_ask_approval_tty_ignores_typeahead_unknown_keys_and_sequences()
    print("\nALL ESC-APPROVAL TESTS PASSED")
