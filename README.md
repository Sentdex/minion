# minion

![minion](minion.png)

A single-file, single-dependency terminal coding agent. One Python file, one
`pip install openai`, no framework, no bloat. Point it at any
OpenAI-compatible endpoint — a local llama.cpp / vLLM / SGLang server, or a
remote API like Z.ai or OpenAI itself — and start chatting with an agent that
can read, write, edit, and run shell commands in your project.

The whole thing is one file (`minion.py`, ~2200 lines). No TUI framework, no
plugin system, no config file format. It reads from environment variables (and
`~/.env`), talks directly to the OpenAI SDK, and uses raw terminal escapes for
its interface. If you want to understand or modify how it works, you read one
file. That's the whole pitch.

It's built to survive the rough edges of self-hosted and open models: if the
server doesn't support native tool-calling, it falls back to parsing
`<tool_call>…</tool_call>` tags out of the model's text. If the server streams a
separate `reasoning_content` field (MiniMax-M3, DeepSeek-R1, etc.), it renders
that as a dim "thinking" block above the answer. It degrades gracefully rather
than demanding a perfect server.

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
across switches (use `/reset` if you want a clean slate).

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
| `MINION_HOME` / `MINION_SESSIONS_DIR` | where session JSON files are stored |
| `MINION_REASONING_LOOP_SIGNALS` | threshold for the reasoning-loop guard (default 10; `0` disables) |
| `MINION_REASONING_ONLY_CHARS` | reasoning-only stall cutoff before forcing a visible answer (default 12000; `0` disables) |
| `MINION_REASONING_ONLY_RETRIES` | forced-final-answer rescue attempts after a reasoning-only stall (default 1) |
| `MINION_FORCED_FINAL_MAX_TOKENS` | token cap for the forced-final-answer rescue request (default 1024) |
| `MINION_MAX_TOKENS` | token cap for normal streaming requests (default 8192; `0` omits the cap) |
| `MINION_RISK_RETRIES` | connection retries for the command-risk classifier before prompting as high-risk (default 3) |
| `MINION_RISK_RETRY_SECONDS` | seconds to wait between command-risk classifier connection retries (default 1) |
| `MINION_SESSION_DESC_REFRESH` | refresh the model-generated session description every N turns (default 6; `0` disables) |

## Subcommands

| subcommand          | what it does                                          |
| ------------------- | ---------------------------------------------------- |
| `minion`            | start the REPL                                        |
| `minion sessions [query]` | list saved sessions (prints + exits); optional substring filter |

## Commands

| command             | what it does                                            |
| ------------------- | ------------------------------------------------------ |
| `/source [name]`    | list sources or switch to one (context preserved)       |
| `/yolo`             | toggle auto-approve for writes and bash                 |
| `/approval [level]` | show or set risk threshold (`all`/`low`/`medium`/`high`/`yolo`) |
| `/sessions [n]`     | list recent sessions, or show one in full               |
| `/resume [target]`  | resume a past session (`n`/id/prefix/title)             |
| `/save [title]`     | save the current session (optional custom title)        |
| `/delete [target]`  | delete a saved session                                  |
| `/compress`         | summarize older turns into one, keep last 2 verbatim     |
| `/compact`          | alias for `/compress`                                    |
| `/reset`            | clear conversation, start a fresh session               |
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
- A resumed session **reselects the source** (endpoint + model) it was started
  on, so it lands on the same backend it was talking to.
- `/sessions <n>` shows the full transcript of a past session inline.
- `/reset` starts a fresh session (it does not overwrite the old one).

```
$ minion sessions              # browse recent sessions, then exit
$ minion sessions refactor     # filter to sessions mentioning "refactor"
$ minion --resume 1            # resume the most recent session
$ minion --resume deadbe       # resume by short id
$ minion --resume implement    # resume the session titled "implement…"
```

This is a deliberately lightweight take on session persistence — inspired by
how Hermes (`hermes_state.py`) stores sessions, but flat JSON files instead of
SQLite, since minion is a single local agent rather than a multi-platform
gateway.

## Reasoning-loop guard

Reasoning models sometimes spin in place — they keep saying "let me implement…"
without actually doing anything. minion counts those "ready to act" phrases
during the reasoning phase and, after `MINION_REASONING_LOOP_SIGNALS` (default
**10**) of them, cuts the stream and nudges the model to take a concrete action.
Set the env var to `0` to disable, or lower it (e.g. `5`) for a more aggressive
cut.

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

## Built with

minion was developed using the following models:

- **minion** (eating its own dog food)
- [**GLM 5.2**](https://huggingface.co/zai-org/GLM-5.2) (Z.ai, open weights)
- [**MiniMax-M3**](https://huggingface.co/MiniMaxAI/MiniMax-M3) (MiniMax)
