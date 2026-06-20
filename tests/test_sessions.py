#!/usr/bin/env python3
"""Tests for the chat-sessions persistence layer.

No live model or terminal needed — exercises the pure file-IO helpers:
write → load → list → resolve → delete, in a throwaway temp dir.
"""
import json
import os
import sys
import tempfile

# Point MINION_SESSIONS_DIR at a temp dir so we never touch ~/.minion.
_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp  # belt-and-suspenders

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m


def _use_test_sessions_dir():
    os.environ["MINION_SESSIONS_DIR"] = _tmp
    os.environ["MINION_HOME"] = _tmp
    return m._sessions_dir()


def setup_function(_function):
    _use_test_sessions_dir()


def _msg(role, content):
    return {"role": role, "content": content}


def _clear_test_sessions_dir():
    d = _use_test_sessions_dir()
    assert os.path.basename(d).startswith("minion-test-"), d
    for f in os.listdir(d):
        if f.endswith(".json"):
            os.remove(os.path.join(d, f))
    return d


def test_write_load_roundtrip():
    sid = "20250101-120000-abc123"
    messages = [
        {"role": "system", "content": m.SYSTEM},
        _msg("user", "hello there"),
        _msg("assistant", "hi!"),
    ]
    m._write_session(sid, messages, {"title": "greeting"})
    loaded = m._load_session(sid)
    assert loaded is not None, "load returned None after write"
    assert loaded["messages"] == messages
    assert loaded["title"] == "greeting"
    assert loaded["id"] == sid
    print("PASS — write → load round-trips messages + meta")


def test_write_session_prunes_empty_assistant_messages():
    sid = "20250101-120000-empty0"
    messages = [
        _msg("user", "hello"),
        _msg("assistant", ""),
        _msg("assistant", "   "),
        _msg("assistant", "real reply"),
    ]
    m._write_session(sid, messages)
    loaded = m._load_session(sid)
    assert loaded is not None
    assert loaded["messages"] == [
        _msg("user", "hello"),
        _msg("assistant", "real reply"),
    ]
    assert messages == loaded["messages"]
    print("PASS — empty assistant messages are pruned before session save")


def test_write_is_atomic_and_merges_meta():
    sid = "20250101-120000-def456"
    m._write_session(sid, [_msg("user", "first")], {"source": "local"})
    # second write should preserve created_at + source, update updated_at + messages
    m._write_session(sid, [_msg("user", "first"), _msg("assistant", "reply")])
    loaded = m._load_session(sid)
    assert loaded["source"] == "local", "source should survive a re-write"
    assert len(loaded["messages"]) == 2
    assert loaded["created_at"] <= loaded["updated_at"]
    print("PASS — re-write preserves existing meta + timestamps")


def test_load_missing_returns_none():
    assert m._load_session("does-not-exist-999") is None
    print("PASS — load of a missing id returns None")


def test_list_sessions_orders_newest_first_with_preview():
    # clear any sessions left over by earlier tests in this shared temp dir
    _clear_test_sessions_dir()
    texts = ["aaa", "bbb", "ccc"]
    sids = []
    for i, txt in enumerate(texts):
        sid = f"20250101-12000{i}-order{i}"
        m._write_session(sid, [_msg("user", txt)], {"updated_at": 100 + i})
        sids.append(sid)
    sessions = m._list_sessions()
    # written last → newest → should appear first
    assert sessions[0]["id"] == sids[-1], f"newest first; got {sessions[0]['id']}"
    # preview comes from the first user message
    previews = {s["preview"] for s in sessions}
    assert previews == {"aaa", "bbb", "ccc"}, previews
    print("PASS — list orders newest-first and previews first user message")


def test_resolve_session_supports_index_prefix_title():
    sid = m._new_session_id()
    m._write_session(sid, [_msg("user", "unique title here")],
                     {"title": "unique title here"})
    sessions = m._list_sessions(limit=50)
    assert m._resolve_session("1", sessions) == sessions[0]["id"]
    assert m._resolve_session(sessions[0]["id"], sessions) == sessions[0]["id"]
    # unique prefix (use most of the id including the random suffix)
    assert m._resolve_session(sessions[0]["id"][:18], sessions) == sessions[0]["id"]
    # exact title
    assert m._resolve_session("unique title here", sessions) == sid
    # ambiguous / unknown → None
    assert m._resolve_session("nope-no-such", sessions) is None
    print("PASS — resolve handles index / id / prefix / title")


def test_list_sessions_without_query_stops_at_limit():
    _clear_test_sessions_dir()
    for i in range(20):
        m._write_session(f"20250101-1300{i:02d}-limit{i:02d}",
                         [_msg("user", f"limited {i:02d}")],
                         {"updated_at": 200 + i})

    seen = []
    original = m._session_summary_from_file

    def wrapped(fname):
        seen.append(fname)
        return original(fname)

    m._session_summary_from_file = wrapped
    try:
        sessions = m._list_sessions(limit=5)
    finally:
        m._session_summary_from_file = original

    assert len(sessions) == 5
    assert len(seen) == 5, f"parsed {len(seen)} files instead of stopping at limit"
    print("PASS — recent listing stops after the requested page")


def test_delete_session():
    sid = m._new_session_id()
    m._write_session(sid, [_msg("user", "bye")])
    assert m._load_session(sid) is not None
    assert m._delete_session(sid) is True
    assert m._load_session(sid) is None
    assert m._delete_session(sid) is False  # already gone
    print("PASS — delete removes the file and is idempotent")


def test_new_session_id_is_unique_and_sortable():
    ids = {m._new_session_id() for _ in range(50)}
    assert len(ids) == 50, "session ids collided!"
    one = m._new_session_id()
    assert "-" in one and len(one) >= 15, one
    print("PASS — new session ids are unique + timestamp-prefixed")


def test_safe_title_collapses_and_clamps():
    assert m._safe_title("  hello\nworld  ") == "hello world"
    long = "x" * 200
    t = m._safe_title(long)
    assert len(t) == 60 and t.endswith("…")
    assert m._safe_title("") is None
    assert m._safe_title(None) is None
    print("PASS — title sanitization collapses whitespace + clamps length")


def test_bare_resume_picks_most_recent():
    """`minion --resume` (no target) should resume the newest session."""
    # clear the temp dir first
    _clear_test_sessions_dir()
    # no sessions yet → bare --resume resolves to None (clean fresh start)
    sys.argv = ["minion.py", "--resume"]
    assert m._session_id_from_args() is None, "should be None with no sessions"
    # now create two sessions; newest should win
    old = "20250101-120000-old123"
    m._write_session(old, [_msg("user", "old")], {"updated_at": 100})
    newest = "20250101-120001-new123"
    m._write_session(newest, [_msg("user", "newest")], {"updated_at": 101})
    sys.argv = ["minion.py", "--resume"]
    resolved = m._session_id_from_args()
    assert resolved == newest, f"bare --resume should pick newest {newest}, got {resolved}"
    # --resume <n> still works alongside it
    sys.argv = ["minion.py", "--resume", "1"]
    assert m._session_id_from_args() == m._list_sessions()[0]["id"]
    print("PASS — bare --resume resumes the most recent session")


def test_resume_flag_without_target_ignores_following_dash_flags():
    """`--resume --yolo` must not treat '--yolo' as the resume target."""
    sys.argv = ["minion.py", "--resume", "--yolo"]
    # newest session from the previous test still exists → resolves to it,
    # NOT to the literal string "--yolo"
    resolved = m._session_id_from_args()
    assert resolved != "--yolo", "should not eat a following flag as the target"
    assert resolved is None or resolved.startswith("20"), resolved
    print("PASS — --resume does not consume the next flag as its target")


def test_cli_sessions_list_and_filter():
    """`minion sessions` pages recent sessions; filters can search older ones."""
    import io, contextlib
    # seed a known set
    _clear_test_sessions_dir()
    for i in range(12):
        m._write_session(f"20250101-1200{i:02d}-page{i:02d}",
                         [_msg("user", f"topic {i:02d}")],
                         {"title": f"topic {i:02d}", "updated_at": 100 + i})
    m._write_session("20250101-115000-auth00",
                     [_msg("user", "refactor the auth module")],
                     {"title": "auth refactor", "updated_at": 50})
    m._write_session("20250101-114900-css000",
                     [_msg("user", "fix the css bug")],
                     {"title": "css bugfix", "updated_at": 49})

    # bare `sessions` → first page only
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        handled = m._cli_sessions(["sessions"])
    assert handled is True, "sessions should report it handled the request"
    out = buf.getvalue()
    assert "20250101-120011-page11" in out and "topic 02" in out, out
    assert "first: topic 11" in out, out
    assert "topic 01" not in out and "auth refactor" not in out, out
    assert "resume with" in out, "should show the resume hint"
    assert "next page" in out, "should show the pagination hint"

    # --sessions is the same paged listing path
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        handled = m._cli_sessions(["--sessions", "--limit", "5"])
    assert handled is True
    out = buf.getvalue()
    assert "topic 11" in out and "topic 07" in out, out
    assert "topic 06" not in out, out

    # page 2 shows the older tail of the list
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        handled = m._cli_sessions(["sessions", "--page", "2"])
    assert handled is True
    out = buf.getvalue()
    assert "topic 01" in out and "auth refactor" in out, out
    assert "topic 11" not in out, out

    # filter searches beyond the first page
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        handled = m._cli_sessions(["sessions", "auth"])
    assert handled is True
    out = buf.getvalue()
    assert "auth refactor" in out and "css bugfix" not in out, out

    # no match → graceful message, still handled
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        handled = m._cli_sessions(["sessions", "zzzznope"])
    assert handled is True
    assert "no sessions matching" in buf.getvalue()

    # not a sessions invocation → returns False
    assert m._cli_sessions(["--resume", "1"]) is False
    assert m._cli_sessions([]) is False
    print("PASS — `sessions` lists, filters, and exits cleanly")


if __name__ == "__main__":
    test_write_load_roundtrip()
    test_write_session_prunes_empty_assistant_messages()
    test_write_is_atomic_and_merges_meta()
    test_load_missing_returns_none()
    test_list_sessions_orders_newest_first_with_preview()
    test_resolve_session_supports_index_prefix_title()
    test_list_sessions_without_query_stops_at_limit()
    test_delete_session()
    test_new_session_id_is_unique_and_sortable()
    test_safe_title_collapses_and_clamps()
    test_bare_resume_picks_most_recent()
    test_resume_flag_without_target_ignores_following_dash_flags()
    test_cli_sessions_list_and_filter()
    print("\nALL SESSION TESTS PASSED")
