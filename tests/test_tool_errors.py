#!/usr/bin/env python3
"""Structured tool errors: `ERROR [reason] msg` + a hint line."""
import builtins
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402

_NoSpinner = type("NoSpinner", (), {
    "__init__": lambda self, **kw: None,
    "start": lambda self: None,
    "stop": lambda self: None,
    "_t": None,
})


def _quiet(monkeypatch):
    monkeypatch.setattr(m, "LifeSpinner", _NoSpinner)
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)


def test_unknown_tool_lists_valid_tools(monkeypatch):
    _quiet(monkeypatch)
    out = m.run_tool("read_fil", {"path": "x"})
    assert out.startswith("ERROR [unknown_tool]"), out
    assert "read_file" in out and "hint:" in out


def test_non_dict_args_name_expected_args(monkeypatch):
    _quiet(monkeypatch)
    out = m.run_tool("read_file", "x")
    assert out.startswith("ERROR [bad_args]"), out
    assert "path" in out and "offset" in out


def test_missing_required_arg(monkeypatch):
    _quiet(monkeypatch)
    out = m.run_tool("write_file", {"path": "x"})
    assert out.startswith("ERROR [bad_args]") and "content" in out, out


def test_binary_file(monkeypatch, tmp_path):
    _quiet(monkeypatch)
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\xff\xfe" * 10)
    out = m.run_tool("read_file", {"path": str(p)})
    assert out.startswith("ERROR [binary_file]"), out
    assert "UnicodeDecodeError" not in out


def test_invalid_utf8_text_does_not_crash(monkeypatch, tmp_path):
    _quiet(monkeypatch)
    p = tmp_path / "latin1.txt"
    p.write_bytes(b"caf\xe9\n")
    out = m.run_tool("read_file", {"path": str(p)})
    assert "caf" in out and "ERROR" not in out, out


def test_not_found(monkeypatch, tmp_path):
    _quiet(monkeypatch)
    out = m.run_tool("read_file", {"path": str(tmp_path / "nope.py")})
    assert out.startswith("ERROR [not_found]"), out


def test_edit_no_match_and_ambiguous(monkeypatch, tmp_path):
    _quiet(monkeypatch)
    p = tmp_path / "f.py"
    p.write_text("a = 1\na = 1\n")
    assert m.edit_file(str(p), "zzz", "y").startswith("ERROR [no_match]")
    out = m.edit_file(str(p), "a = 1", "a = 2")
    assert out.startswith("ERROR [ambiguous_match]") and "2 times" in out


def test_side_effects_unknown_only_for_mutating_tools(monkeypatch):
    _quiet(monkeypatch)

    def boom(**_):
        raise RuntimeError("half done")
    monkeypatch.setitem(m.DISPATCH, "write_file", boom)
    monkeypatch.setitem(m.DISPATCH, "list_dir", boom)
    w = m.run_tool("write_file", {"path": "x", "content": "y"})
    assert w.startswith("ERROR [RuntimeError] half done") and "side_effects: unknown" in w
    assert "side_effects" not in m.run_tool("list_dir", {})
