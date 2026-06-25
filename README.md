# minion

![minion](minion.png)

A no-nonsense coding agent that doesn't use 50K tokens of context to say "hello."

Minion is a purpose built coding agent aimed at removing and keeping out context bloat. Many agent frameworks use 20K-50K+ tokens when you've just said "hey." This is caused by having a lot of features and tools that need to be loaded into the context of the LLM.

This presents real challenges to running coding agents on local models, where you're often hosting the best AI you can, and you don't have much room for context. 

Why do we care about this context?
1. That first 50K of context is your fastest context for your model. We want that speed.
2. That first 50K of context is where your model's attention mechanism is likely the best. We want that intelligence.
3. That first 50K of context riding along every singe message you send adds to cost over time, even if you're on some API.

On a bare `hey`, the entire prompt minion sends is about 625 tokens:

```
  system prompt (SYSTEM)               ~98
  tool schemas (5 functions, TOOLS)   ~475
  the word "hey"                         1
  chat-template framing                ~50
                                     ─────
                                     ~625 tokens
```

The variance is in the last line: every server's chat template wraps
the tool section differently (Qwen/Hermes add per-tool tags, llama.cpp
adds a functions header, OpenAI injects its own), so the real total
lands somewhere in the low 600s. The point is the floor — not the
exact figure. That's two orders of magnitude less than the harnesses
that spend the first 20K–50K of your context before you've said
anything, and it's paid on every single turn.

You don't have to take our word for it. Point any harness at a local
server and say `hey`. Most print a token footer; minion does. The
number you see is the number that rides along every message you send,
and it's the cheapest thing to compare.

Nothing against the more feature-rich agents, we need those too. But
we also need a very lightweight coding agent, and here it is.

Point it at any OpenAI-compatible endpoint — a local llama.cpp / vLLM / SGLang server, or a
remote API like Z.ai or OpenAI itself — and start chatting with an agent that
can read, write, edit, and run shell commands in your project.

The whole thing is one file (`minion.py`, ~3400 lines). No TUI framework, no
plugin system, no config file format. It reads from environment variables (and
`~/.env`), talks directly to the OpenAI SDK, and uses raw terminal escapes for
its interface. If you want to understand or modify how it works, you read one
file. That's the whole pitch.

It's built to survive the rough edges of self-hosted and open models: if the
server doesn't support native tool-calling, it falls back to parsing
standalone `[minion_tool_call]...[/minion_tool_call]` messages from the model's
text. Quoted examples or source code containing tool-call delimiters are treated
as normal text, and tool results have those delimiters escaped before they enter
the next model context. If the server streams a separate `reasoning_content`
field (MiniMax-M3, DeepSeek-R1, etc.), it renders that as a dim "thinking" block
above the answer. It degrades gracefully rather than demanding a perfect server.

## Quick start

```
pip install openai
export MINION_BASE_URL=http://localhost:8080/v1
export MINION_MODEL=your-model-name
export MINION_API_KEY=sk-noop        # any string; local servers ignore it
python minion.py
```

If `MINION_MODEL` is unset, minion asks the server what it's serving.

## Install as a command

If you'd rather have a `minion` command on your `$PATH`, install from this
repo:

```
pip install -e .
```

That registers a `minion` console script pointing at this checkout — edits you
make here are picked up immediately. Use `pip install .` (no `-e`) for a
non-editable install instead.

## Configuration

minion reads configuration from environment variables, and automatically loads
`~/.env` at startup (so you don't have to export things in every terminal).

### Single source (simple)

```
MINION_BASE_URL=http://localhost:8080/v1
MINION_MODEL=your-model-name
MINION_API_KEY=sk-noop
```

### Multiple sources

Define named endpoints and switch between them at runtime:

```
MINION_SOURCES=local,zai

MINION_SOURCE_LOCAL_BASE_URL=http://localhost:8080/v1
MINION_SOURCE_LOCAL_API_KEY=sk-noop

MINION_SOURCE_ZAI_BASE_URL=https://api.z.ai/api/paas/v4
MINION_SOURCE_ZAI_API_KEY=$zai_test         # $name = look up a key from env / ~/.env
MINION_SOURCE_ZAI_MODEL=glm-x-preview
```

See [`sources.example.env`](sources.example.env) for a full annotated example.
Switch at runtime with `/source [name]`. The conversation context is preserved
across switches (use `/reset` if you want a clean slate). Each source's maximum
context window (the server's, for the active model) is shown in the `/source`
list and on the switch line, and the per-turn stats footer shows it next to the
current context size — `ctx 12K/150K` — so you can see how much room is left.
The current half is colorized by how full the window is (green < 30%, yellow
30–60%, red 60%+), so a glance tells you whether you're nearing the limit. When
the conversation fills `MINION_AUTOCOMPRESS_PERCENT` (default 85%) of the
window, older turns are folded into a summary automatically, keeping the last
~⅓ verbatim — see `/autocompress`. The
max is probed once per source/model and cached: for **local** servers (llama.cpp)
it reads `/v1/models`/`/props` (sub-ms LAN round-trips); for **remote** hosts
(T Together, Z.ai, …) it leads with the over-`max_tokens` chat probe — a
deliberately over-sized request is rejected with the limit named in the error,
resolving in well under a second — so the `/<max>` shows up on the first turn
rather than after a multi-second `/v1/models` round-trip that returns nothing.

#### Built-in `together` source

If `TOGETHER_API_KEY` is set (in `~/.env` or your shell), minion auto-registers
a `together` source pointing at `https://api.together.xyz/v1`, defaulting to
the `zai-org/GLM-5.2` model. Together hosts many models, so you can override
the model per switch without editing any config:

```
/source together                    # → GLM-5.2 (the default)
/source together zai-org/GLM-4.6    # → any other Together model id
```

It's registered last, so it never displaces your default startup source — you
opt in with `/source together` (or `--source together`). Defining your own
`MINION_SOURCE_TOGETHER_*` vars overrides the built-in entirely.

### Flags

| flag                          | what it does                                              |
| ----------------------------- | -------------------------------------------------------- |
| `--yolo`                      | start in never-prompt mode (auto-approve everything)      |
| `--approval <all\|low\|medium\|high\|yolo>` | start with a non-default approval mode       |
| `--source <name>`             | start on a specific source                                |
| `--resume [target]`           | resume a saved session; bare = most recent                |
| `--session <id>`              | start a fresh run attached to a specific session id       |

### Environment variables

minion auto-loads `~/.env` at startup (override with `MINION_ENV_FILE`),
so per-user settings live in one place instead of being exported every shell.

| env var | what it does |
| --- | --- |
| `MINION_APPROVAL` | persistent default approval mode: `all`/`low`/`medium`/`high`/`yolo` (see below). CLI flags `--approval` / `--yolo` override it for a single run. |
| `MINION_BASE_URL` / `MINION_MODEL` / `MINION_API_KEY` | legacy single-source config (or the `local` fallback) |
| `MINION_SOURCES` / `MINION_SOURCE_*` | named multi-source endpoints |
| `MINION_ACTIVE` | name of the source to start on (same as `--source`, but persistent; defaults to the first in `MINION_SOURCES`) |
| `TOGETHER_API_KEY` | auto-registers a built-in `together` source (Together AI, default model `zai-org/GLM-5.2`); override with `MINION_SOURCE_TOGETHER_*` |
| `MINION_HOME` / `MINION_SESSIONS_DIR` | where session JSON files are stored |
| `MINION_MALFORMED_STREAM_RETRIES` | max clean retries for malformed/truncated tool-call args or SSE streams before waiting for user input (default 2) |
| `MINION_REASONING_ONLY_CHARS` | reasoning-only stall cutoff before forcing a visible answer (default 36000; `0` disables) |
| `MINION_REASONING_ONLY_RETRIES` | forced-final-answer rescue attempts after a reasoning-only stall (default 1) |
| `MINION_EMPTY_TURN_RETRIES` | auto-continue attempts when the model returns a completely empty turn (no text, no tool call) before dropping to the prompt (default 3; `0` disables) |
| `MINION_TOOL_RESULT_CHARS` | per-tool-result char cap before it enters message history, to starve context-copying repetition (default 20000; `0` disables the cap, dedup still runs) |
| `MINION_READ_FILE_LINES` | default number of lines the `read_file` tool returns when no explicit `limit` is given (default 400; `<=0` reads whole files) |
| `MINION_AUTOCOMPRESS_PERCENT` | auto-compress the conversation when it fills this % of the context window (default 85; `0` disables). Keeps the last ~⅓ of turns verbatim — more conservative than a manual `/compress` (which keeps 2). `/autocompress` adjusts at runtime |
| `MINION_RECOVERY_TEMPERATURE` / `MINION_RECOVERY_TOP_P` | standard sampler params used only for recovery retries (defaults `1.0` / `0.95`; negative values omit them) |
| `MINION_RECOVERY_MIN_P` | min-p floor for recovery retries (llama.cpp extension via `extra_body`; default `0.02`; negative omits it) |
| `MINION_RECOVERY_REPEAT_PENALTY` / `MINION_RECOVERY_REPEAT_LAST_N` | repeat penalty applied during recovery retries to lower a looping token's logit (defaults `1.2` / `512`; negative omits them) |
| `MINION_RECOVERY_DRY_MULTIPLIER` / `MINION_RECOVERY_DRY_BASE` / `MINION_RECOVERY_DRY_ALLOWED_LENGTH` | DRY (Don't Repeat Yourself) anti-repetition params for recovery retries (defaults `0.8` / `1.75` / `2`; set `MINION_RECOVERY_DRY_MULTIPLIER` to `0` to disable DRY) |
| `MINION_FORCED_FINAL_MAX_TOKENS` | token cap for the forced-final-answer rescue request (default 2048) |
| `MINION_MAX_TOKENS` | token cap for normal streaming requests (default 16000; `0` omits the cap) |
| `MINION_RISK_RETRIES` | connection retries for the command-risk classifier before prompting as high-risk (default 3) |
| `MINION_RISK_RETRY_SECONDS` | seconds to wait between command-risk classifier connection retries (default 1) |
| `MINION_SESSION_DESC_REFRESH` | refresh the model-generated session description every N turns (default 6; `0` disables) |
| `MINION_METRICS_URL` | optional endpoint to receive cumulative token-usage totals after each turn (default unset; see Metrics) |

## Subcommands

| subcommand          | what it does                                          |
| ------------------- | ---------------------------------------------------- |
| `minion`            | start the REPL                                        |
| `minion sessions [query]` | list saved sessions, 10 per page (prints + exits); optional substring filter |
| `minion --sessions [query]` | alias for `minion sessions [query]` |

## Commands

| command             | what it does                                            |
| ------------------- | ------------------------------------------------------ |
| `/source [name] [model]` | list sources, switch to one, or override its model for that switch (context preserved) |
| `/yolo`             | toggle auto-approve for writes and bash                 |
| `/approval [level]` | show or set risk threshold (`all`/`low`/`medium`/`high`/`yolo`) |
| `/sessions [n]`     | list recent sessions, or show one in full               |
| `/resume [target]`  | resume a past session (`n`/id/prefix/title)             |
| `/save [title]`     | save the current session (optional custom title)        |
| `/delete [target]`  | delete a saved session                                  |
| `/compress`         | summarize older turns into one, keep last 2 verbatim     |
| `/compact`          | alias for `/compress`                                    |
| `/autocompress [pct\|off\|on]` | show or set the auto-compress threshold (default 85%; `0`/`off` disables) |
| `/recover [note]`   | force a low-temp visible checkpoint after a bad stream   |
| `/reset`            | clear conversation, start a fresh session               |
| `/clear`            | alias for `/reset`                                       |
| `/new`              | alias for `/reset`                                       |
| `/quit`             | exit                                                     |

## Input

The prompt is a multi-line editor with a framed box:

- **Enter** submits; **Alt+Enter** or **Ctrl+J** inserts a newline
- **Paste** (bracketed-paste) inserts text verbatim, including newlines
- **Up/Down** navigate history; **Left/Right** move within the line
- **Home/End** jump to line start/end; **Ctrl+U** clears; **Ctrl+C** cancels
- Long lines word-wrap inside the box

Falls back to plain `input()` when stdin/stdout isn't a TTY.

## Interrupting the model

Press **Esc** at any point during generation to stop the model and drop back to
the prompt. The stream is closed, partial output is discarded, and a synthetic
"you were interrupted" note is appended to context so the model knows what
happened. In-flight tool calls (e.g. a running `run_bash`) are **not**
cancelled — they run to completion. Ctrl+C kills the whole process if you need
a hard stop.

## Approval modes

Every write / edit / bash call is risk-classified by a single cheap model call
before it runs. Levels: `low` (read-only or trivially reversible), `medium`
(modifies state but contained/reversible), `high` (destructive, hard to
reverse, or broad scope). The approval mode controls the maximum risk level
Minion may auto-allow:

| setting                 | prompts at          | auto-allows       |
| ----------------------- | ------------------- | ----------------- |
| _(default)_ / `all`     | low + medium + high | —                 |
| `--approval low`        | medium + high       | low               |
| `--approval medium`     | high                | low + medium      |
| `--approval high`       | —                   | low + medium + high |
| `--yolo` / `yolo`       | —                   | everything; skips classifier |

The risk assessment is shown in brackets next to the prompt, so you have
context for the decision:

```
allow rm -rf /tmp/foo? [risk: HIGH — recursive force delete in /tmp] [Y/n/esc]
```

For path-based write/edit risk, Minion sends the classifier its current working
directory, detected project root (git root when available, otherwise the launch
directory), and whether the target path is inside that root. A project under
`~/Downloads` is still treated as in-project; `~/Downloads` only raises concern
when the target is outside the active project root.

At the prompt, press:

- **Y** (or Enter) to approve
- **n** to deny — the model is told the action was refused and can adapt
- **Esc** to stop the turn and drop back to the chat input so you can add more
  guidance. The escaped action is recorded as cancelled; if the model emitted
  multiple tool calls, any remaining ones are marked skipped so the context
  stays valid. A note is left so the model knows you pulled it back.

Auto-allowed calls print a one-liner:

```
↳ auto-allow [low] ls -la (read-only listing)
```

YOLO mode skips the classifier entirely. If the classifier call fails or returns
garbage, the action defaults to `high` (always prompts) so it errs on the side
of asking.

## Sessions (save / resume)

Every chat is automatically saved to `~/.minion/sessions/` (override with
`MINION_HOME` or `MINION_SESSIONS_DIR`) — one JSON file per session holding
the exact message array the model sees plus a little metadata (id, title,
description, source, cwd, timestamps). Files are plain JSON and
human-readable/greppable.

- **Auto-save** happens after every model turn, so a crash or accidental close
  never loses your work. On Ctrl-D / Ctrl-C exit a grey
  `resume with: minion --resume <id>` hint is printed so you can pick right
  back up.
- The **title** is auto-derived from your first message; set a custom one with
  `/save <title>`.
- A **short id** (the 6-hex suffix) is shown in listings and accepted by
  `--resume` / `/resume`, so `minion --resume deadbe` works without typing the
  full timestamp.
- A **model-generated description** refreshes every `MINION_SESSION_DESC_REFRESH`
  turns (default **6**; `0` disables) and appears as a dim subtitle under each
  session in `minion sessions` / `/sessions` — it tracks the current task
  rather than freezing on the first message.
- **Resume** a session at startup with `minion --resume <target>` or mid-chat
  with `/resume <target>`. A `target` is a number from `/sessions`, a short id,
  a full session id, a unique id prefix, or an exact title. Bare
  `minion --resume` resumes your most recent session.
- On resume, the **full conversation history is printed** as a one-line-per-
  message recap (color-coded by role, tool calls shown as `→ name(...)`) so
  you immediately re-orient on what the chat was about.
- **Discover** saved sessions from the shell with `minion sessions` (prints
  and exits — no REPL). Add a substring query to filter:
  `minion sessions refactor` matches titles, descriptions, and ids.
  Listings show 10 sessions per page by default. Use `--page/-p` and
  `--limit/-n` to move through older sessions without loading every transcript.
- A resumed session **reselects the source** (endpoint + model) it was started
  on, so it lands on the same backend it was talking to.
- `/sessions <n>` shows the full transcript of a past session inline.
- Use `/sessions --page 2` for the next in-chat page. `/sessions 2` still
  opens session 2, so numeric selection keeps working.
- `/reset` starts a fresh session (it does not overwrite the old one).

```
$ minion sessions              # browse the 10 most recent sessions, then exit
$ minion sessions --page 2     # browse the next page
$ minion sessions -n 5 -p 3    # page 3, 5 sessions per page
$ minion sessions refactor     # filter sessions mentioning "refactor"
$ minion --resume 1            # resume the most recent session
$ minion --resume deadbe       # resume by short id
$ minion --resume implement    # resume the session titled "implement…"
```

This is a deliberately lightweight take on session persistence — inspired by
how Hermes (`hermes_state.py`) stores sessions, but flat JSON files instead of
SQLite, since minion is a single local agent rather than a multi-platform
gateway.

## Reasoning recovery guards

Reasoning models sometimes stream a long `reasoning_content` block and then stop
without ever emitting visible content or a tool call — a silent, empty turn that
burns tokens for nothing. minion counts reasoning-only chars, and once the stream
reaches `MINION_REASONING_ONLY_CHARS` (default **36000**) with no content or tool
call in sight, it cuts the stream and nudges the model to produce a visible answer
via the `final_answer` tool (`FORCE_FINAL_NUDGE`). After
`MINION_REASONING_ONLY_RETRIES` (default **1**) forced-final attempts also stall,
minion gives up and returns to the chat input for guidance. Set the char limit to
`0` to disable the cutoff. This is a plain char-count timeout — it makes no guess
about the *content* of the reasoning, only how much of it there is.

The same forced-final rescue path handles the no-signal case too: if a turn ends
with reasoning but zero content and zero tool calls (and wasn't already cut), the
char-count guard trips as if the limit had been hit, so a model that thinks in
silence still gets one visible-answer nudge instead of a dead turn.

A separate guard catches malformed or truncated tool-call args and SSE streams:
after `MINION_MALFORMED_STREAM_RETRIES` (default **2**) clean retries, minion
stops retrying and waits for input. Each retry re-opens the stream with recovery
sampler params — more entropy and anti-repetition than a normal turn — to escape
the low-entropy attractor that produced the corruption.

### Recovery samplers

Recovery retries (malformed-stream, reasoning-only-stall rescue, and `/recover`)
swap in a higher-entropy, anti-repetition sampler so the model doesn't collapse
back into the same broken output:

| knob | default | notes |
| --- | --- | --- |
| `MINION_RECOVERY_TEMPERATURE` | `1.0` | raised (not lowered) so a repetition collapse gets *more* entropy, not a sharper greedy pass |
| `MINION_RECOVERY_TOP_P` | `0.95` | |
| `MINION_RECOVERY_MIN_P` | `0.02` | llama.cpp extension, rides in `extra_body`; negative omits it |
| `MINION_RECOVERY_REPEAT_PENALTY` | `1.2` | lowers a looping token's logit |
| `MINION_RECOVERY_REPEAT_LAST_N` | `512` | window for the repeat penalty |
| `MINION_RECOVERY_DRY_MULTIPLIER` | `0.8` | DRY anti-repetition; set to `0` to disable |
| `MINION_RECOVERY_DRY_BASE` | `1.75` | |
| `MINION_RECOVERY_DRY_ALLOWED_LENGTH` | `2` | |

Path/code punctuation (`\n`, `:`, `"`, `*`, `/`, `\`, `` ` ``, `'`) are DRY
sequence breakers, so a long file path the model must emit verbatim is never
penalized as repetition. Normal turns pass no sampler params, so the server keeps
its own defaults unless a recovery path is in progress. Non-llama.cpp backends
ignore the unknown `extra_body` keys.

### Manual recovery

You can trigger the same checkpoint path manually with `/recover [optional note]`
after interrupting a bad stream or once the prompt returns. The command appends a
manual recovery note to the conversation and immediately forces a `final_answer`
checkpoint with recovery sampling, instead of letting the model continue free-form.

## Tools

| tool        | args                  | notes                           |
| ----------- | --------------------- | ------------------------------- |
| `read_file` | `path`                |                                 |
| `write_file`| `path`, `content`     | overwrites; requires confirmation |
| `edit_file` | `path`, `old`, `new`  | `old` must match exactly once   |
| `list_dir`  | `path`                |                                 |
| `run_bash`  | `command`             | requires confirmation           |

## Status bar

At startup (and after a `/source` / `/yolo` / `/approval` switch) minion
prints a one-line banner showing the model name, active source, approval
mode, and endpoint. The banner is printed into the normal scrollback —
there's no pinned/scroll-region status bar, so terminal scrollback works
normally and every line of output stays visible.

(An earlier version pinned a status bar at row 1 using a DECSTBM scroll
region, like tmux/vim. It was removed because it broke terminal scrollback —
lines scrolling off the top of the region never entered the scrollback
buffer, so the chat became unscrollable in a plain terminal.)

## Log

Every request and streamed SSE chunk is appended to `llamacpp.log` next to the
script (JSONL). Useful for debugging what the model actually saw and returned.

## Metrics

minion records token usage for every model call (input, output, cache-read,
and reasoning tokens) and writes it into the session JSON under
`~/.minion/sessions/`. This is always on — the numbers are already in hand
from the stats footer, so persisting them costs nothing and a local usage log
is useful whether or not you point it anywhere. A saved session carries:

```json
"input_tokens": 900,
"output_tokens": 230,
"cache_read_tokens": 600,
"reasoning_tokens": 120,
"api_calls": 2,
"started_at": 1700000000.0
```

The accounting matches the OpenAI usage convention (and what most dashboards
expect): `input_tokens` **excludes** cached prompt tokens (those are broken out
into `cache_read_tokens`), and `output_tokens` **excludes** reasoning tokens
(those are broken out into `reasoning_tokens`). Local llama.cpp servers report
via a `timings` object instead of streaming `usage`; both are normalized to the
same four fields. Totals accumulate across a session and are reloaded on
`--resume` / `/resume`, so they keep climbing across restarts rather than
resetting.

Optionally, set `MINION_METRICS_URL` to a POST endpoint and minion will also
push the same cumulative totals after each turn:

```
MINION_METRICS_URL=http://localhost:9121/api/tokens/push
```

The body is a small, dashboard-agnostic JSON blob:

```json
{
  "session_id": "20240101-120000-abcdef",
  "model": "glm-4.6",
  "source": "zai",
  "input_tokens": 900,
  "output_tokens": 230,
  "cache_read_tokens": 600,
  "reasoning_tokens": 120,
  "api_calls": 2,
  "started_at": 1700000000.0,
  "ended_at": 1700000123.0
}
```

Totals are cumulative per session, so the endpoint can compute its own deltas
between pushes. The push is fire-and-forget with a 1.5 s timeout and disables
itself after the first failure — if the endpoint is down or absent, minion
stops trying rather than adding latency to every turn. Unset the variable (the
default) and nothing leaves the machine; only the local session-JSON totals
remain.

## Built with

minion was developed using the following models:

- **minion** (eating its own dog food)
- [**GLM 5.2**](https://huggingface.co/zai-org/GLM-5.2) (Z.ai, open weights)
- [**MiniMax-M3**](https://huggingface.co/MiniMaxAI/MiniMax-M3) (MiniMax)

## License

MIT License. See [`LICENSE`](LICENSE).
