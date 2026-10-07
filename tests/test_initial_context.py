#!/usr/bin/env python3
"""Guard the README's "~1,020 tokens on a bare hey" claim.

The README advertises the initial context a bare `hey` costs. That floor is
the whole reason minion exists — it's well over an order of magnitude under
the harnesses that spend 20K-50K before you've typed anything — so this test
watches it. If a new tool or a fatter SYSTEM bloats the payload, we want to
know before the README quietly becomes a lie.

WHAT WE MEASURE (the part we control and send to the API):
  - SYSTEM          the system prompt
  - TOOLS           the 7-function tool schema, JSON-encoded
  - "hey"           a minimal first user turn
That's ~973 today (chars/3.5: SYSTEM ~230 + TOOLS ~742 + 1). The server's
chat template then adds ~50 of framing that varies by backend and ISN'T
part of our payload, so we don't measure it here.

THRESHOLDS (on the content we control):
  - 985  ->  WARNING   any growth past today's payload; check the README
  - 1000 ->  FAIL      the plan's always-on budget; README claim no longer holds

History: commit a1a7526 grew this to ~1,930 (long run_bash guidance in both
SYSTEM and the tool description, plus Docker/CUDA principles). The
harness-hardening trim deduped SYSTEM against the tool descriptions and
dropped non-ASCII from TOOLS (json.dumps turns an em-dash into a 6-char
escape) to get back under 1000. The kill_background tool (plan 5.3) then
added ~62; the tool descriptions were trimmed to pay for it, keeping the
1000 cap. New prompt text must buy its own room.

TOKENIZER: prefers tiktoken (cl100k_base) when installed for a real count;
falls back to a conservative chars/3.5 estimate otherwise so the guard still
runs with zero extra dependencies. minion's only runtime dep is `openai`,
and we don't add one for a test. The fallback over-counts ~7% vs cl100k on
code/JSON, which is the safe direction for a guard — it trips slightly
early, never late.
"""
import json
import os
import sys
import tempfile
import warnings

# Keep minion from touching ~/.minion — same convention as the other tests,
# even though this one never writes a session.
_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402

# Thresholds on the content token count (SYSTEM + TOOLS JSON + "hey").
# Today this is ~973 (chars/3.5); the README's ~1,020 adds ~50 of framing.
WARN_TOKENS = 985   # soft: ratchet just above today; any growth warns
FAIL_TOKENS = 1000  # hard: the always-on budget (plan 6.1)


def _count_tokens(text):
    """Token count for a string.

    Returns (count, method). Uses tiktoken's cl100k_base when available
    (the closest thing to a shared standard across served models); falls
    back to a conservative chars/3.5 estimate so the test runs with no
    extra deps. The fallback over-counts on code/JSON, which is the safe
    direction for a bloat guard.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text)), "tiktoken"
    except Exception:
        return max(1, round(len(text) / 3.5)), "chars/3.5"


def test_initial_context_floor():
    # The exact payload a fresh "hey" turn sends: system prompt + 6 tool
    # schemas + a one-token user message. FINAL_ANSWER_TOOL is intentionally
    # excluded — it's only injected on a forced-final retry (minion.py:1847),
    # not on turn 1, so it's not part of the README's figure.
    system_tokens, method = _count_tokens(m.SYSTEM)
    tools_tokens, _ = _count_tokens(json.dumps(m.TOOLS))
    user_tokens, _ = _count_tokens("hey")
    total = system_tokens + tools_tokens + user_tokens

    breakdown = (
        f"initial context content = {total} tokens ({method}): "
        f"SYSTEM {system_tokens} + TOOLS {tools_tokens} "
        f"({len(m.TOOLS)} funcs) + 'hey' {user_tokens}"
    )
    print(breakdown)

    # Sanity: a broken payload (empty SYSTEM, TOOLS clobbered to []) would
    # make a token-floor test meaningless by passing on a near-zero count.
    assert m.SYSTEM.strip(), "SYSTEM prompt is empty"
    assert len(m.TOOLS) >= 1, "TOOLS is empty — nothing to guard"

    # Hard ceiling: past this the README's '~1,020' is fiction.
    assert total < FAIL_TOKENS, (
        f"initial context content is {total} tokens "
        f"(>= {FAIL_TOKENS}) — the README's ~1,020 claim no longer holds. "
        f"{breakdown}"
    )

    # Soft ceiling: warn (don't fail) past WARN_TOKENS. pytest surfaces
    # warnings in its end-of-run summary, so this is visible without
    # failing the suite.
    if total >= WARN_TOKENS:
        warnings.warn(
            f"initial context content is {total} tokens (>= {WARN_TOKENS}). "
            f"The README claims ~1,020; a new tool or fatter descriptions "
            f"may have pushed the floor up. {breakdown}",
            stacklevel=2,
        )



def test_system_frames_tool_output_as_data():
    # PROV-1: observations are data the model reads, never orders it follows.
    assert "not instructions" in m.SYSTEM


def test_no_benchmark_paths_in_always_on_prompts():
    # 6.1: the /app/... benchmark nudges were removed; prompts stay generic.
    texts = [m.SYSTEM, json.dumps(m.TOOLS), m.REASONING_STALL_NUDGE,
             m.REASONING_STALL_CONTINUE_NOTE, m.FORCED_FINAL_NUDGE,
             m.EMPTY_TURN_NUDGE]
    for t in texts:
        assert "/app/" not in t, t[:120]
    # SYSTEM no longer restates run_bash's own description.
    assert "wait_background(pid=" not in m.SYSTEM
    assert "sleep N" not in m.SYSTEM


def test_tool_descriptions_are_ascii():
    # Non-ASCII in TOOLS costs 6 chars per char once JSON-escaped.
    assert json.dumps(m.TOOLS).isascii()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
