#!/usr/bin/env python3
"""Tests for paged, line-numbered read_file and edit_file's tolerance of pasted
line numbers.

No live model or terminal needed. We exercise read_file/edit_file directly and
monkeypatch _confirm so edits don't block on the approval prompt. Covered:
  1. A file that fits the window: numbered output, no header, faithful content.
  2. A large file: header announces "lines A-B of N", correct window + numbers.
  3. offset/limit paging: exact window, 1-based numbers, EOF clamping.
  4. offset past EOF and limit<=0 (read-to-end) escape hatch.
  5. Line numbers map to grep: the number on a line == its real line number.
  6. edit_file round-trip: pasting a numbered `old` block (and numbered `new`)
     still edits correctly — the <n>\\t prefixes are stripped.
  7. edit_file safety: numbering is only stripped when the WHOLE block carries
     it, and the normal (un-numbered) exact-match path is never altered, even
     for real content that itself looks like `digits<TAB>...`.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


def _write(name, text):
    path = os.path.join(_tmp, name)
    with open(path, "w") as f:
        f.write(text)
    return path


def _allow_edits():
    """Make _confirm auto-approve so edit_file writes without prompting."""
    m._confirm = lambda action: True


def test_small_file_numbered_no_header():
    path = _write("small.txt", "alpha\nbeta\ngamma\n")
    out = m.read_file(path)
    assert not out.startswith("["), "whole-file read must not emit a paging header"
    lines = out.splitlines()
    assert lines == ["     1\talpha", "     2\tbeta", "     3\tgamma"]


def test_no_trailing_newline_preserved_as_window():
    # last line without a newline still renders, exactly once
    path = _write("nonl.txt", "one\ntwo")
    out = m.read_file(path)
    assert out.splitlines() == ["     1\tone", "     2\ttwo"]


def test_large_file_default_window_has_header():
    path = _write("big.txt", "".join(f"line{i}\n" for i in range(1, 1001)))
    out = m.read_file(path)  # default limit 400
    assert out.startswith(f"[{path}: lines 1-400 of 1000;")
    body = out.split("\n", 1)[1]
    body_lines = body.splitlines()
    assert len(body_lines) == 400
    assert body_lines[0] == "     1\tline1"
    assert body_lines[-1] == "   400\tline400"


def test_offset_and_limit_window():
    path = _write("big2.txt", "".join(f"line{i}\n" for i in range(1, 1001)))
    out = m.read_file(path, offset=500, limit=3)
    assert out.startswith(f"[{path}: lines 500-502 of 1000;")
    body_lines = out.split("\n", 1)[1].splitlines()
    assert body_lines == ["   500\tline500", "   501\tline501", "   502\tline502"]


def test_limit_clamps_at_eof():
    path = _write("big3.txt", "".join(f"line{i}\n" for i in range(1, 11)))
    out = m.read_file(path, offset=8, limit=100)
    # window reaches the real end -> header shows ...-10 of 10
    assert out.startswith(f"[{path}: lines 8-10 of 10;")
    assert out.rstrip().endswith("    10\tline10")


def test_offset_past_eof():
    path = _write("short.txt", "a\nb\n")
    out = m.read_file(path, offset=99)
    assert out == f"[{path}: 2 lines; offset 99 is past end of file]"


def test_empty_file_clear_marker():
    # An empty file returns a clear marker, not "" (offset 1) or a nonsensical
    # "lines 5-0 of 0" header (offset > 1), so the model can tell an empty file
    # apart from a failed read.
    path = _write("empty.txt", "")
    assert m.read_file(path) == f"[{path}: empty file]"
    assert m.read_file(path, offset=5) == f"[{path}: empty file]"


def test_limit_zero_reads_to_end():
    path = _write("big4.txt", "".join(f"line{i}\n" for i in range(1, 51)))
    out = m.read_file(path, limit=0)
    # whole file -> no header, 50 numbered lines
    assert not out.startswith("[")
    assert len(out.splitlines()) == 50


def test_line_number_matches_real_position():
    # The number printed for a line equals its true 1-based file position, so a
    # grep hit at line N maps to read_file(path, offset=N).
    path = _write("map.txt", "".join(f"row{i}\n" for i in range(1, 31)))
    out = m.read_file(path, offset=17, limit=1)
    body = out.split("\n", 1)[1].rstrip("\n")
    num, _, content = body.partition("\t")
    assert int(num) == 17
    assert content == "row17"


def test_edit_file_strips_pasted_line_numbers():
    _allow_edits()
    path = _write("edit1.py", "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n")
    # Read a numbered window, then feed the numbered block straight back as `old`.
    win = m.read_file(path, offset=1, limit=2)
    old_block = win.split("\n", 1)[1].rstrip("\n")  # "     1\tdef foo():\n     2\t    return 1"
    assert "\t" in old_block and old_block.lstrip().startswith("1\t")
    # New content also pasted with numbers (worst case) — must not be written verbatim.
    new_block = "     1\tdef foo():\n     2\t    return 10"
    res = m.edit_file(path, old_block, new_block)
    assert res == f"edited {path}"
    with open(path) as f:
        src = f.read()
    assert "return 10" in src
    assert "\t" not in src, "line-number prefixes must not leak into the file"
    assert "return 2" in src, "the untouched function must remain"


def test_edit_file_plain_old_still_works():
    _allow_edits()
    path = _write("edit2.txt", "hello world\n")
    res = m.edit_file(path, "hello", "goodbye")
    assert res == f"edited {path}"
    with open(path) as f:
        assert f.read() == "goodbye world\n"


def test_edit_file_does_not_strip_real_numeric_content():
    # A file whose content genuinely is `<digits><TAB>...` (e.g. TSV). A normal
    # exact-match edit must apply verbatim and never lose the leading number.
    _allow_edits()
    path = _write("data.tsv", "1\tapple\n2\tbanana\n3\tcherry\n")
    res = m.edit_file(path, "2\tbanana", "2\tBANANA")
    assert res == f"edited {path}"
    with open(path) as f:
        assert f.read() == "1\tapple\n2\tBANANA\n3\tcherry\n"


def test_strip_only_when_whole_block_numbered():
    # Mixed block (one numbered line, one not) is NOT treated as numbered output.
    assert m._strip_line_numbers("     1\tfoo\nbar") == "     1\tfoo\nbar"
    # Fully numbered block IS stripped.
    assert m._strip_line_numbers("     1\tfoo\n    22\tbar") == "foo\nbar"
    # Plain text is untouched.
    assert m._strip_line_numbers("foo\nbar") == "foo\nbar"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
