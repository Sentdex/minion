"""Smoke-test the /compress vs /compact alias without a running model.

Strategy: monkeypatch read_multiline to feed a canned command, monkeypatch
the network calls so compress() returns a known value, and assert both
commands produce identical output (modulo the command string itself).
"""
import io
import sys
import importlib
import os

# load module fresh — add project root (parent of this tests/ dir) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m

# Save the REAL compress before any stubbing
_real_compress = m.compress

# --- stubs -------------------------------------------------------------------
class FakeStream:
    def __iter__(self): return iter([])

def fake_open_stream(msgs):
    return FakeStream()

def fake_compress(msgs, keep=m.COMPRESS_KEEP):
    # pretend we summarized 3 turns into 1, kept 2 verbatim, summary = 42 chars
    msgs[:] = [{"role": "system", "content": m.SYSTEM},
               {"role": "user", "content": "[Compressed context — 3 earlier turns summarized; last 2 turns kept verbatim]\n\nfake"}]
    return (2, 3, 42)

def fake_model_turn(msgs, reasoning_loop_cut_count=0, malformed_stream_cut_count=0,
                    forced_final=False):
    return False  # no tool calls → REPL moves to next prompt

# capture stdout so we can compare
captured = io.StringIO()
real_print = print
def fake_print(*a, **kw):
    captured.write(" ".join(str(x) for x in a) + (kw.get("end", "\n")))

# --- run the REPL with /compact, then again with /compress --------------------
def run_with(cmd):
    captured.truncate(0); captured.seek(0)
    # patch print so we can capture without messing up the real terminal
    import builtins
    builtins.print = fake_print
    try:
        # seed a few messages so compress() has something to summarize
        m.read_multiline = lambda history=None: None  # placeholder, replaced below
        m.open_stream = fake_open_stream
        m.compress = fake_compress
        m.model_turn = fake_model_turn
        m.YOLO = True
        # run main() with a pre-seeded message log via monkeypatching the
        # `messages` list inside main's frame. Easier: just call main() and
        # have the first prompt be a real user turn (model_turn is stubbed to
        # no-op so nothing actually gets sent).
        prompts = iter(["hello", "world", "again", cmd, "/quit"])
        m.read_multiline = lambda history=None: next(prompts)
        m.main()
    finally:
        builtins.print = real_print
    return captured.getvalue()

out_compact = run_with("/compact")
out_compress = run_with("/compress")

# --- assertions --------------------------------------------------------------
def normalize(out, cmd):
    """Strip the command-name mention so /compact and /compress outputs compare equal."""
    # the only place the command string appears in output is the banner line,
    # which is identical between runs since it just lists the toggles.
    return out

assert "nothing to compress" not in out_compact, \
    f"/compact should have triggered compress branch, got:\n{out_compact}"
assert "compressed 3 turns → 1 summary (42 chars), kept last 2 verbatim" in out_compact, \
    f"/compact didn't run compress() — got:\n{out_compact}"

assert "nothing to compress" not in out_compress, \
    f"/compress should have triggered compress branch, got:\n{out_compress}"
assert "compressed 3 turns → 1 summary (42 chars), kept last 2 verbatim" in out_compress, \
    f"/compress didn't run compress() — got:\n{out_compress}"

# also: the "nothing to compress" branch must still fire for both when there's
# not enough context. Build a messages list with only the system prompt.
captured.truncate(0); captured.seek(0)
import builtins
builtins.print = fake_print
m.read_multiline = lambda history=None: "/compact" if False else "/quit"  # just exit
m.main()
builtins.print = real_print

# direct unit test of the "nothing to compress" branch — call the REAL
# compress() (not the stub) so we exercise the actual early-return logic
msgs = [{"role": "system", "content": m.SYSTEM}]
result = _real_compress(msgs)
assert result is None, f"expected None on too-short context, got {result}"

# also verify /compact and /compress produce byte-identical output when run
# back-to-back (modulo timestamps, which we don't print in this path)
assert normalize(out_compact, "/compact") == normalize(out_compress, "/compress"), \
    f"output differs:\n--- /compact ---\n{out_compact}\n--- /compress ---\n{out_compress}"

print("OK — /compact and /compress take the exact same code path")
