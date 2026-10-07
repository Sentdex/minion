#!/usr/bin/env python3
"""list_dir cap + dir markers (6.5), sanitizer cap and read_file char budget (6.6)."""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


# --- 6.5 list_dir ----------------------------------------------------------
def test_list_dir_caps_and_marks_dirs(tmp_path):
    for i in range(400):
        (tmp_path / f"f{i:03d}.txt").write_text("x")
    (tmp_path / "aaa_sub").mkdir()
    out = m.list_dir(str(tmp_path)).splitlines()
    assert out[0] == "aaa_sub/"
    assert len(out) == 301  # 300 entries + notice
    assert "300 of 401" in out[-1]


def test_list_dir_cap_env(tmp_path, monkeypatch):
    for i in range(5):
        (tmp_path / f"f{i}").write_text("x")
    monkeypatch.setenv("MINION_LIST_DIR_MAX", "0")  # 0 = no cap
    assert len(m.list_dir(str(tmp_path)).splitlines()) == 5
    monkeypatch.setenv("MINION_LIST_DIR_MAX", "2")
    out = m.list_dir(str(tmp_path)).splitlines()
    assert out[:2] == ["f0", "f1"] and "2 of 5" in out[2]


def test_list_dir_not_found(tmp_path):
    out = m.list_dir(str(tmp_path / "nope"))
    assert out.startswith("ERROR [not_found]")


# --- 6.6 sanitizer ---------------------------------------------------------
def test_sanitizer_collapses_repeated_lines():
    out = m._sanitize_tool_result("x\n" * 5000)
    assert len(out) < 200
    assert "identical lines elided" in out


def test_sanitizer_head_tail_cap(monkeypatch):
    monkeypatch.setattr(m, "TOOL_RESULT_CHARS", 3000)
    text = "".join(f"line {i:05d} " + "a" * 40 + "\n" for i in range(1000))[:50000]
    out = m._sanitize_tool_result(text)
    assert len(out) <= 3200
    assert out.startswith("line 00000") and out.endswith(text[-20:])
    assert "chars elided" in out


def test_sanitizer_short_passthrough():
    assert m._sanitize_tool_result("ok") == "ok"


# --- 6.6 read_file char budget --------------------------------------------
def test_read_file_char_budget_long_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_READ_FILE_CHARS", "3000")
    p = tmp_path / "wide.txt"
    p.write_text("".join(("L%03d" % i) + "w" * 996 + "\n" for i in range(50)))
    out = m.read_file(str(p))
    header, body = out.split("\n", 1)
    assert header.startswith(f"[{p}: lines 1-2 of 50;")
    assert f"offset=3)" in header and "next: read_file(" in header
    assert len(body) <= 3000
    # Paging on from the hint continues where it stopped.
    nxt = m.read_file(str(p), offset=3)
    assert nxt.split("\n", 1)[1].startswith("     3\tL002")


def test_read_file_single_huge_line(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_READ_FILE_CHARS", "3000")
    p = tmp_path / "min.js"
    p.write_text("q" * 50000 + "\nsecond\n")
    out = m.read_file(str(p))
    assert "line truncated, 50001 chars" in out  # the newline is part of the line
    header, body = out.split("\n", 1)
    assert "offset=2)" in header
    assert len(body) < 3100


def test_read_file_explicit_limit_not_char_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_READ_FILE_CHARS", "100")
    p = tmp_path / "x.txt"
    p.write_text("".join("y" * 80 + "\n" for _ in range(5)))
    out = m.read_file(str(p), limit=5)
    assert not out.startswith("[") and len(out.splitlines()) == 5


def test_read_file_binary_sniff(tmp_path):
    p = tmp_path / "b.bin"
    p.write_bytes(b"\x89PNG\r\n\x00\x00garbage")
    assert m.read_file(str(p)).startswith("ERROR [binary_file]")
