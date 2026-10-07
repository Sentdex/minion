#!/usr/bin/env python3
"""Compaction hardening (plan items 6.2, 6.4, 6.7).

  6.2  compress() and /memory save share _render_history: tool calls are always
       rendered, tool results are named, the total is capped, the summary call
       passes max_tokens, and the header says it is a harness summary.
  6.4  With a session id, the folded turns are archived verbatim and the fold
       is logged in meta['compactions'].
  6.7  No two consecutive user turns after compress().

Offline: the client is a fake that records the request kwargs.
"""
import json
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


class _Resp:
    def __init__(self, text):
        msg = type("M", (), {"content": text})()
        self.choices = [type("C", (), {"message": msg})()]

    def model_dump(self):
        return {}


class _Client:
    base_url = "http://fake"

    def __init__(self, text="SUMMARY"):
        self.calls = []
        self._text = text
        self.chat = type("Ch", (), {})()
        self.chat.completions = self

    def create(self, **kw):
        self.calls.append(kw)
        return _Resp(self._text)


class _Src:
    name = "fake"

    def __init__(self, window):
        self._context_window = window

    def extra_request_kwargs(self):
        return {}


def _setup(monkeypatch, tmp_path, window=None):
    monkeypatch.setenv("MINION_SESSIONS_DIR", str(tmp_path))
    monkeypatch.setenv("MINION_HOME", str(tmp_path))
    fake = _Client()
    monkeypatch.setattr(m, "client", fake)
    monkeypatch.setattr(m, "MODEL", "test")
    monkeypatch.setattr(m, "ACTIVE", _Src(window))
    return fake


def _tool_history(n=4):
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "ORIGINAL GOAL: fix the parser"}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {
                "name": "edit_file",
                "arguments": json.dumps({"path": f"f{i}.py", "old": "a", "new": "b"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"edited f{i}.py"})
    msgs.append({"role": "user", "content": "now run the tests"})
    msgs.append({"role": "assistant", "content": "ok"})
    return msgs


def _no_consecutive_users(msgs):
    roles = [x["role"] for x in msgs]
    return all(not (a == b == "user") for a, b in zip(roles, roles[1:]))


# --- 6.2 -------------------------------------------------------------------
def test_render_history_shows_tool_calls_and_names_results():
    out = m._render_history(_tool_history(1))
    assert "edit_file(" in out
    assert "[tool result: edit_file] edited f0.py" in out
    assert "SYS" not in out  # system prompt is never rendered


def test_render_history_shows_calls_even_with_content():
    msgs = [{"role": "assistant", "content": "let me look", "tool_calls": [
        {"id": "x", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}]}]
    out = m._render_history(msgs)
    assert "let me look" in out and "read_file(" in out


def test_render_history_total_cap_drops_oldest_tool_bodies_first():
    msgs = []
    for i in range(500):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "function": {"name": "run_bash", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 5000})
    msgs.append({"role": "user", "content": "LATEST REQUEST"})
    out = m._render_history(msgs, per_msg_cap=2000, total_cap=50000)
    assert len(out) <= 50000
    assert "LATEST REQUEST" in out  # newest non-tool context survives
    # 500 x 5000-char messages, uncapped per message, still fit the budget.
    plain = [{"role": "user", "content": "y" * 5000} for _ in range(500)]
    assert len(m._render_history(plain, per_msg_cap=None, total_cap=40000)) <= 40000


def test_compress_prompt_has_tool_calls_max_tokens_and_budget(monkeypatch, tmp_path):
    fake = _setup(monkeypatch, tmp_path, window=10000)
    msgs = _tool_history()
    # Bloat the history: 500 x 5000 chars, far past the window.
    for i in range(500):
        msgs.insert(2, {"role": "user" if i % 2 else "assistant", "content": "z" * 5000})
    assert m.compress(msgs) is not None
    kw = fake.calls[0]
    prompt = kw["messages"][0]["content"]
    assert "edit_file(" in prompt
    assert kw["max_tokens"] == m.COMPRESS_MAX_TOKENS
    budget = int(10000 * 3.5 * 0.55)
    assert len(prompt) <= budget + len(m.COMPRESS_PROMPT) + 20
    assert "Harness summary" in msgs[1]["content"] or "harness summary" in msgs[1]["content"]
    assert "not new user instructions" in msgs[1]["content"]


def test_compress_max_tokens_env(monkeypatch, tmp_path):
    fake = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(m, "COMPRESS_MAX_TOKENS", 0)  # 0 = don't send the cap
    m.compress(_tool_history())
    assert "max_tokens" not in fake.calls[0]


def test_memory_save_uses_shared_renderer(monkeypatch, tmp_path):
    fake = _setup(monkeypatch, tmp_path)
    fake._text = "# Parser fix\nbody"
    monkeypatch.setattr(m, "_memories_dir", lambda: str(tmp_path / "mem"))
    m._cmd_memory_save(_tool_history(1))
    prompt = fake.calls[0]["messages"][0]["content"]
    assert "edit_file(" in prompt and "[tool result: edit_file]" in prompt


# --- 6.4 -------------------------------------------------------------------
def test_compress_archives_head_and_records_meta(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, window=10000)
    sid = "s-archive"
    msgs = _tool_history()
    big = "B" * 9000  # longer than the 2000-char render cap: archive is verbatim
    msgs[2]["tool_calls"][0]["function"]["arguments"] = json.dumps({"path": "x", "old": big, "new": "c"})
    m._write_session(sid, list(msgs))
    kept_n, summarized_n, _ = m.compress(msgs, session_id=sid)

    archive = os.path.join(str(tmp_path), f"{sid}.archive.txt")
    text = open(archive, encoding="utf-8").read()
    assert "ORIGINAL GOAL: fix the parser" in text
    assert big in text
    assert text.startswith("=== compaction 1 ")
    assert archive in msgs[1]["content"]  # header points at the archive

    meta = m._load_session(sid)
    entry = meta["compactions"][-1]
    assert entry["summarized_n"] == summarized_n
    assert entry["kept_n"] == kept_n
    assert entry["archive"] == archive and entry["auto"] is False
    assert {"ts", "prompt_tokens", "window"} <= set(entry)
    assert meta["messages"] == msgs  # compacted state persisted

    # A second fold appends (seq 2) and logs a second entry.
    msgs.extend([{"role": "user", "content": "more"}, {"role": "assistant", "content": "done"}] * 3)
    m.compress(msgs, session_id=sid)
    assert "=== compaction 2 " in open(archive, encoding="utf-8").read()
    assert len(m._load_session(sid)["compactions"]) == 2


def test_compress_without_session_writes_no_archive(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    msgs = _tool_history()
    m.compress(msgs)
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".archive.txt")]
    assert "archived at" not in msgs[1]["content"]


# --- 6.7 -------------------------------------------------------------------
def test_no_consecutive_user_turns_after_compress(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    # keep=2 → tail starts at the user turn "now run the tests".
    msgs = _tool_history()
    m.compress(msgs, keep=2)
    assert _no_consecutive_users(msgs)
    assert msgs[1]["role"] == "user"
    assert "SUMMARY" in msgs[1]["content"] and "now run the tests" in msgs[1]["content"]
    assert msgs[-1] == {"role": "assistant", "content": "ok"}


def test_alternation_with_alternating_history(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    msgs = [{"role": "system", "content": "S"}] + [
        {"role": "user" if j % 2 == 0 else "assistant", "content": f"t{j}"} for j in range(60)]
    kept_n, summarized_n, _ = m.compress(msgs, auto=True)
    assert (kept_n, summarized_n) == (20, 40)
    assert _no_consecutive_users(msgs)
    assert len(msgs) == 1 + 20  # summary folded into the first kept user turn


def test_alternation_when_tail_starts_with_assistant(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    msgs = _tool_history()
    m.compress(msgs, keep=1)  # tail = [assistant "ok"]
    assert [x["role"] for x in msgs] == ["system", "user", "assistant"]
    assert _no_consecutive_users(msgs)
