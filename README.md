# minion

![minion](minion.png)

A no-nonsense coding agent that doesn't use 50K tokens of context to say "hello."

Minion is a purpose built coding agent aimed at removing and keeping out context bloat. Many agent frameworks use 20K-50K+ tokens when you've just said "hey." This is caused by having a lot of features and tools that need to be loaded into the context of the LLM.

This presents real challenges to running coding agents on local models, where you're often hosting the best AI you can, and you don't have much room for context. 

Why do we care about this context?
1. That first 50K of context is your fastest context for your model. We want that speed.
2. That first 50K of context is where your model's attention mechanism is likely the best. We want that intelligence.
3. That first 50K of context riding along every single message you send adds to cost over time, even if you're on some API.

On a bare `hey`, the entire prompt minion sends is about 1,020 tokens:

```
  system prompt (SYSTEM)              ~230
  tool schemas (7 functions, TOOLS)   ~740
  the word "hey"                         1
  chat-template framing                ~50
                                     ─────
                                    ~1,020 tokens
```

(Measured by `tests/test_initial_context.py` with a conservative chars/3.5
estimate, which over-counts real tokenizers by a few percent; the test fails
if SYSTEM + TOOLS + `hey` reaches 1,060.)

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

The whole thing is one file (`minion.py`, ~4600 lines). No TUI framework, no
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
~⅓ verbatim — see `/autocompress`. This is also checked between tool turns of
a run (`MINION_AUTOCOMPRESS_IN_LOOP`), and a request the server rejects as too
long is compressed once and retried instead of falling back to no-tools. The
summary is labeled a harness summary (reference only, not new instructions);
with a session, the folded turns are appended verbatim to
`<sessions>/<id>.archive.txt` and each fold is logged in the session's
`compactions` metadata. The
max is probed once per source/model and cached: for **local** servers (llama.cpp)
it reads `/v1/models`/`/props` (sub-ms LAN round-trips); for **remote** hosts
(T Together, Z.ai, …) it leads with the over-`max_tokens` chat probe — a
deliberately over-sized request is rejected with the limit named in the error,
resolving in well under a second — so the `/<max>` shows up on the first turn
rather than after a multi-second `/v1/models` round-trip that returns nothing.

#### AWS Bedrock (via LiteLLM proxy)

minion works with any OpenAI-compatible endpoint, so you can use AWS Bedrock
models by running [LiteLLM](https://github.com/BerriAI/litellm) as a local
proxy:

```bash
pip install 'litellm[proxy]'
litellm --model bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0 --port 4000
```

Then point minion at it:

```
MINION_BASE_URL=http://localhost:4000/v1
MINION_MODEL=bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0
MINION_API_KEY=sk-noop
```

LiteLLM reads your standard AWS credentials (`~/.aws/credentials`,
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, or SSO/instance profiles).
Set `AWS_DEFAULT_REGION` if needed. Bedrock requires
[inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html)
for on-demand throughput — use the `us.` or `global.` prefixed model IDs
(e.g. `us.anthropic.claude-opus-4-6-v1`) rather than bare model IDs.

A convenience wrapper that auto-starts the proxy:

```bash
#!/bin/bash
# save as ~/bin/minion-bedrock and chmod +x
export AWS_DEFAULT_REGION=us-east-1
if ! lsof -i :4000 &>/dev/null; then
  litellm --model bedrock/us.anthropic.claude-opus-4-6-v1 --port 4000 &>/dev/null &
  sleep 3
fi
exec minion "$@"
```

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

#### Built-in `openrouter` source

If `OPENROUTER_API_KEY` is set (in `~/.env` or your shell), minion auto-registers
an `openrouter` source pointing at `https://openrouter.ai/api/v1`, defaulting to
the `z-ai/glm-5.2` model routed to the `parasail/fp8` provider (no fallback).
OpenRouter fronts many providers behind one model id, each with its own price,
precision, latency, and data-collection policy. Provider routing is controlled
per-session with the `/provider` command:

```
/source openrouter                       # → GLM-5.2 via parasail/fp8
/source openrouter z-ai/glm-4.6          # → any other OpenRouter model id
/provider Together,DeepInfra              # re-route to specific providers
/provider off                             # clear routing, let OpenRouter pick
```

It's registered after `together` (and after all user-defined sources), so it
never displaces your default startup source — opt in with `/source openrouter`
(or `--source openrouter`). Defining your own `MINION_SOURCE_OPENROUTER_*` vars
overrides the built-in entirely.

### Flags

| flag                          | what it does                                              |
| ----------------------------- | -------------------------------------------------------- |
| `--yolo`                      | start in never-prompt mode (auto-approve everything)      |
| `--approval <all\|low\|medium\|high\|yolo>` | start with a non-default approval mode       |
| `--source <name>`             | start on a specific source                                |
| `--resume [target]`           | resume a saved session; bare = most recent                |
| `--session <id>`              | start a fresh run attached to a specific session id       |
| `--prompt-file <path\|->` / `-f` | headless: run one task from a file (or stdin) and exit with a status code (below) |
| `--help` / `-h`               | print usage, env vars and exit codes                      |

#### Headless exit codes (`--prompt-file`)

| code | terminal state |
| --- | --- |
| `0` | completed (`completed_verified` when `MINION_VERIFY_CMD` passed, else `completed_unverified`) |
| `1` | internal error |
| `2` | usage error (missing/empty prompt file, no source) |
| `3` | `blocked`: the model reported `final_answer status=blocked`, `MINION_DENY_LIMIT` consecutive denials, or an action needed approval and stdin is not a TTY (headless runs deny instead of prompting — set `MINION_APPROVAL=<level>` or `--yolo` for unattended runs) |
| `4` | `budget_exhausted`: turn cap (`MINION_MAX_MODEL_TURNS`), a run budget (`MINION_MAX_RUN_SECONDS` / `_TOKENS` / `MINION_MAX_TOOL_CALLS`), or verify retries spent |
| `5` | `model_error`: server unreachable, API error, stream timeout, or `context_exhausted` (the request still overflowed the window after one compress-and-retry) |
| `6` | `recovery_failed`: malformed-stream / stall / forced-final recovery gave up |
| `7` | `truncated`: the answer kept hitting the token limit |
| `130` | `cancelled` (Ctrl-C / Esc) |

stderr ends with a `[minion] terminal: <state>[/<budget>] after N step(s), T tool call(s), K tokens, Ws`
line (budget is `turns`, `wall_time`, `tokens`, `tool_calls`, `verify` or `context_exhausted`); background
jobs still running at exit are listed on stderr,
and every run logs a `run_end` event to the traffic log. Every traffic-log line written during a
run carries its `"run": "<run_id>"`. Headless transcripts are saved as sessions (`MINION_HEADLESS_SAVE`).

#### Trial rows (`MINION_RESULT_FILE`)

With `MINION_RESULT_FILE=<path>`, each `--prompt-file` run appends one JSON line (also on Ctrl-C and
internal errors), shaped like an eval trial record so a grader can fill in `passed` / `reward` and
ablations are just env toggles plus a diff of rows:

```json
{"task_id": "t-1", "trial": "3", "run_id": "run-20261007-101500-a1b2c3",
 "session_id": "20261007-101500-a1b2c3", "passed": null, "reward": null,
 "terminal": "budget_exhausted", "budget": "turns", "detail": null, "exit_code": 4,
 "infra_error": false, "steps": 200, "tool_calls": 143, "api_calls": 201,
 "tokens": 912345, "input_tokens": 880000, "output_tokens": 30000,
 "cache_read_tokens": 0, "reasoning_tokens": 2345,
 "counters": {"stalls": 2, "empty_turns": 0, "malformed": 1, "collapses": 0,
              "forced_finals": 1, "truncations": 0, "context_full": 0, "verify_fails": 0},
 "wall_s": 1834.2, "model": "glm-x", "source": "local",
 "started_at": 1791367700.1, "ended_at": 1791369534.3,
 "manifest": {"harness_sha": "…", "prompt_sha": "…", "model": "glm-x", "source": "local",
              "base_url": "http://localhost:8080/v1", "python": "3.12.4", "platform": "darwin",
              "limits": {"max_model_turns": 200, "max_run_seconds": 0, "max_run_tokens": 0,
                         "max_tool_calls": 0, "max_tokens": 16000},
              "knobs": {"MINION_REASONING_ONLY_RETRIES": "0", "...": "..."}}}
```

`infra_error` is true for `internal_error` (exit 1) and for `model_error` other than `context_exhausted`
(endpoint/API failures), so they can be excluded from pass rates. `steps` is `null` when the run ended by
exception. `manifest.knobs` holds every `MINION_*` env var except credential-named ones and
`MINION_SOURCE_*`; compare only rows whose manifests match apart from the knob under test, e.g.
`MINION_REASONING_ONLY_RETRIES=0` or `MINION_AUTOCOMPRESS_IN_LOOP=0`.

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
| `OPENROUTER_API_KEY` | auto-registers a built-in `openrouter` source (OpenRouter, default model `z-ai/glm-5.2` routed to `parasail/fp8`); override with `MINION_SOURCE_OPENROUTER_*` |
| `MINION_BACKEND` | set to `vllm` to disable llama.cpp-only recovery knobs (`min_p`, `repeat_penalty`, DRY) that vLLM's speculative decoder rejects |
| `MINION_SOURCE_<NAME>_EXTRA_BODY` | JSON object merged into every chat request body for that source (used by OpenRouter's provider routing); invalid JSON is warned to stderr and ignored |
| `MINION_SOURCE_<NAME>_APP_NAME` / `MINION_SOURCE_<NAME>_APP_URL` | HTTP-Referer / X-Title headers for aggregator identification (OpenRouter dashboard) |
| `MINION_HOME` / `MINION_SESSIONS_DIR` | where session JSON files are stored |
| `MINION_BASH_ENV_PASS` | comma-separated env var names `run_bash` commands may still see. By default commands get an env without credentials (names with a whole `_`-separated segment `API_KEY`/`APIKEY`/`ACCESS_KEY`/`PRIVATE_KEY`/`TOKEN`/`SECRET(S)`/`PASSWORD`/`PASSWD`/`CREDENTIAL(S)`/`AUTH`, e.g. `GITHUB_TOKEN`, `NPM_AUTH_TOKEN`; every `MINION_SOURCE_*`; and any var holding a source key). `SSH_AUTH_SOCK`, `XAUTHORITY`, `GIT_AUTHOR_*` and `TOKENIZERS_PARALLELISM` are kept. Values of the credential names and resolved source keys (not model ids or base URLs) are masked as `<redacted>` in tool results and the traffic log. Keys exported in the shell that launched minion stay readable through `ps eww $PPID`; keys that live only in `MINION_ENV_FILE` do not show there (the `ps e`/`/proc/*/environ` risk-floor patterns cover the rest) |
| `MINION_MALFORMED_STREAM_RETRIES` | max clean retries for malformed/truncated tool-call args or SSE streams before waiting for user input (default 2) |
| `MINION_REASONING_ONLY_CHARS` | reasoning-only stall cutoff before forcing a visible answer (default 36000; `0` disables) |
| `MINION_REASONING_ONLY_RETRIES` | reasoning-only stall rescue attempts; the last one forces a final answer, 0 disables (default 3) |
| `MINION_EMPTY_TURN_RETRIES` | auto-continue attempts when the model returns a completely empty turn (no text, no tool call) before dropping to the prompt (default 3; `0` disables) |
| `MINION_TOOL_RESULT_CHARS` | per-tool-result char cap before it enters message history, to starve context-copying repetition (default 20000; `0` disables the cap, dedup still runs) |
| `MINION_READ_FILE_LINES` | default number of lines the `read_file` tool returns when no explicit `limit` is given (default 400; `<=0` reads to the end, still bounded by `MINION_READ_FILE_CHARS`) |
| `MINION_READ_FILE_CHARS` | char budget for a default (or `limit<=0`) `read_file` window; it stops early with a `next: read_file(path, offset=N)` header, and a single over-long line is shown truncated (default 16000; `0` = lines only). An explicit positive `limit` is not char-bounded |
| `MINION_LIST_DIR_MAX` | entries `list_dir` returns before an `[N of M entries shown; …]` notice (default 300; `0` = all). Directories end with `/` |
| `MINION_AUTOCOMPRESS_PERCENT` | auto-compress the conversation when it fills this % of the context window (default 85; `0` disables). Keeps the last ~⅓ of turns verbatim — more conservative than a manual `/compress` (which keeps 2). `/autocompress` adjusts at runtime |
| `MINION_AUTOCOMPRESS_IN_LOOP` | also check the threshold after every tool turn inside a run, so long and headless (`--prompt-file`) runs compact too (default 1; `0` = only after a REPL turn settles) |
| `MINION_COMPRESS_MAX_TOKENS` | `max_tokens` for the compress summary call (default 2048; `0` = don't send a cap) |
| `MINION_RECOVERY_TEMPERATURE` / `MINION_RECOVERY_TOP_P` | standard sampler params used only for recovery retries (defaults `1.0` / `0.95`; negative values omit them) |
| `MINION_RECOVERY_MIN_P` | min-p floor for recovery retries (llama.cpp extension via `extra_body`; default `0.02`; negative omits it) |
| `MINION_RECOVERY_REPEAT_PENALTY` / `MINION_RECOVERY_REPEAT_LAST_N` | repeat penalty applied during recovery retries to lower a looping token's logit (defaults `1.2` / `512`; negative omits them) |
| `MINION_RECOVERY_DRY_MULTIPLIER` / `MINION_RECOVERY_DRY_BASE` / `MINION_RECOVERY_DRY_ALLOWED_LENGTH` | DRY (Don't Repeat Yourself) anti-repetition params for recovery retries (defaults `0.8` / `1.75` / `2`; set `MINION_RECOVERY_DRY_MULTIPLIER` to `0` to disable DRY) |
| `MINION_FORCED_FINAL_MAX_TOKENS` | token cap for the forced-final-answer rescue request (default 2048) |
| `MINION_MAX_TOKENS` | token cap for normal streaming requests (default 16000; `0` omits the cap) |
| `MINION_RISK_RETRIES` | connection retries for the command-risk classifier before prompting as high-risk (default 3) |
| `MINION_PROTECTED_EXTRA` | colon-separated extra paths that always require approval to write, edit, or name in a command — and to read (see Approval modes) |
| `MINION_RISK_HIGH_PATTERNS` | extra regexes, separated by `;;`, that force a `run_bash` command to at least high risk |
| `MINION_APPROVAL_DIFF_CHARS` | max chars of the write/edit diff shown at the approval prompt and sent to the classifier (default 1500; `0` = no cap) |
| `MINION_MAX_MODEL_TURNS` | turn cap per run (default 200); at the cap minion runs one forced-final turn and ends the run as `budget_exhausted` |
| `MINION_TRUNCATION_RETRIES` | how many times an answer cut at `finish_reason=length` is nudged to continue where it stopped before the run ends as `truncated` (default 1; `0` = never). A response cut at the limit never has its tool calls executed |
| `MINION_VERIFY_CMD` | opt-in completion gate: shell command run (no approval, scrubbed env) after the model gives a final answer, e.g. `pytest -q`. Exit 0 → `completed_verified`; otherwise its output tail goes back to the model and the run continues (default unset) |
| `MINION_VERIFY_RETRIES` | verify failures fed back to the model before the run ends as `budget_exhausted/verify` (default 2) |
| `MINION_VERIFY_TIMEOUT` | seconds before the verify command is killed and counted as a failure (default 600) |
| `MINION_DENY_LIMIT` | consecutive denied tool calls after which the turn stops as `blocked` (default 3; `0` = off). Denials tell the model not to retry an equivalent action |
| `MINION_WAIT_MAX_SECONDS` | hard cap on any single `run_bash` poll / `wait_background` wait, even `timeout=0` (default 0 = no cap); the job keeps running in the background |
| `MINION_BASH_MAX_SECONDS` | watchdog for every `run_bash` job, foreground or backgrounded: after this many seconds its process group gets SIGTERM, then SIGKILL 2s later (default 0 = off). The model can also stop its own jobs with the `kill_background` tool (approval-gated; only PIDs this minion started); on `/quit` / Ctrl-D minion lists jobs still running and offers to kill them |
| `MINION_MAX_RUN_SECONDS` / `MINION_MAX_RUN_TOKENS` / `MINION_MAX_TOOL_CALLS` | per-run budgets (one user message / one `--prompt-file` task): wall-clock seconds, tokens (input + cached + output + reasoning over the run's API calls) and executed tool calls (default 0 = off). Checked before each model call; when one is spent the run gets one forced final answer and ends as `budget_exhausted/<wall_time\|tokens\|tool_calls>` (exit 4). `run_bash`/`wait_background` waits are clamped to the remaining wall time, and tool calls past the cap are refused |
| `MINION_RESULT_FILE` | `--prompt-file` only: append one JSONL trial row per run to this path (default unset; format below) |
| `MINION_TASK_ID` / `MINION_TRIAL` | optional labels copied into the trial row (`task_id`, `trial`) |
| `MINION_HEADLESS_SAVE` | `--prompt-file` only: save the run's transcript as a normal session file, after every step and tool call, and print its id to stderr (default 1; `0` = off). Its metadata has `headless: true` and `status` / `status_detail` = the terminal state (Ctrl-C → `cancelled` / `interrupted`) |
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
| `/provider [source] [a,b,…\|off]` | show or set OpenRouter provider-routing order; `/provider off` clears routing |
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
happened. Esc also works while `run_bash` / `wait_background` is waiting on a
command: the wait stops and the command keeps running in the background.

**Ctrl+C** while a tool runs stops the turn: that call is recorded as
`INTERRUPTED (Ctrl-C) … outcome unknown`, the rest of the batch as `SKIPPED`,
the session is saved, and you're back at the prompt (headless runs exit 130).
Detached `run_bash` jobs are not killed.

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
| `--approval high`       | protected paths     | everything else; skips classifier |
| `--yolo` / `yolo`       | —                   | everything; skips classifier |

Two deterministic checks sit around the classifier, which is advisory only:

- **Protected paths always prompt** (default deny, labelled `[PROTECTED]`) at
  every level except yolo, and the classifier is not consulted: the env file
  (`~/.env` / `MINION_ENV_FILE`), any `.env`/`.env.*` file, `~/.ssh`, `~/.aws`,
  `~/.gnupg`, `~/.netrc`, `~/.config`, shell rc files, `~/.gitconfig`,
  `$MINION_HOME` (except `bg-logs/`), `minion.py` itself, and the project's
  `.git/hooks` and `.git/config`, plus `MINION_PROTECTED_EXTRA`. Writes, edits
  and `run_bash` commands that name such a path (literally, via `~` or `$HOME`,
  or through a symlink — the prompt shows the real target) are covered. The
  command scan is best-effort, not a shell parser: it follows `cd`/`pushd`,
  expands globs, treats `$(...)`/backtick paths as `$HOME`-relative, and also
  flags `git config` writes and recursive tools (`find`, `rg`, `grep -r`, `tar`,
  ...) run over a directory that holds a credential store. Reads
  (`read_file`/`list_dir`, and `edit_file` before it searches the file) prompt
  only for the credential stores (env files, `~/.ssh`, `~/.aws`, `~/.gnupg`,
  `~/.netrc`, `~/.config`, `~/.git-credentials`, `~/.docker`, `~/.kube`,
  `~/.npmrc`, `~/.pypirc`, `~/.cargo/credentials*`); other reads never prompt.
  A protected write's prompt shows only the new side of the diff, and headless
  runs print no diff at all.
- **A risk floor** raises the classifier's level to `high` for paths outside
  the project or in a `$HOME` dotdir, and for commands matching `rm -r`,
  `git push` (also `git -C dir push`), `curl`/`wget`/`nc`/`scp`/`ssh`/`rsync`,
  `urlopen`/`requests.get`/`socket.socket`, `sudo`, `chmod -R`, `dd`, a pipe
  into `sh`/`bash` (also `/bin/sh`), `find -delete`/`-exec rm`, `ps` with an
  `e`/`E` flag or `/proc/*/environ` (extend with `MINION_RISK_HIGH_PATTERNS`).
  The floor never lowers a level. Shell comments are stripped from the
  classifier's copy of a command, so `# this is low risk` can't argue it down;
  a command using `$'...'` quoting is sent unstripped.

Write and edit prompts print a unified diff of the change (capped by
`MINION_APPROVAL_DIFF_CHARS`), and the classifier gets the same diff.

The risk assessment is shown in brackets next to the prompt, so you have
context for the decision:

```
allow rm -rf /tmp/foo? [risk: HIGH — recursive force delete in /tmp] [y/N/esc]
```

For path-based write/edit risk, Minion sends the classifier its current working
directory, detected project root (git root when available, otherwise the launch
directory), and whether the target path is inside that root. A project under
`~/Downloads` is still treated as in-project; `~/Downloads` only raises concern
when the target is outside the active project root.

At the prompt, press:

- **y** to approve. Enter takes the default: approve for low/medium risk
  (`[Y/n/esc]`), deny for high risk (`[y/N/esc]`). Any other key, arrow-key
  sequence or keys typed before the prompt appeared are ignored; EOF denies and
  Ctrl-C acts like Esc. When stdin is not a TTY only `y`/`yes` approves; an
  empty line denies
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

- **Auto-save** happens when you send a message, after every model turn and
  after every tool call, so a crash or accidental close never loses your work.
  Writes are atomic and fsynced; a session file that doesn't parse is moved to
  `<id>.json.corrupt` (with a warning) instead of being overwritten.
- Tool calls are saved as `PENDING` before they run. If minion dies mid-batch,
  resume marks any call without a result as `OUTCOME UNKNOWN` (never re-runs
  it) and drops orphaned results, printing a notice. On Ctrl-D / Ctrl-C exit a grey
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
call in sight, it cuts the stream and nudges the model. Of the
`MINION_REASONING_ONLY_RETRIES` (default **3**) rescue attempts, the earlier ones
nudge it to act with full tools (`REASONING_STALL_NUDGE`); the last one forces a
visible answer via the `final_answer` tool (`FORCED_FINAL_NUDGE`). If that also stalls,
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
| `read_file` | `path`, `offset`, `limit` | windowed, numbered; char-bounded (`MINION_READ_FILE_CHARS`) |
| `write_file`| `path`, `content`     | overwrites; requires confirmation |
| `edit_file` | `path`, `old`, `new`  | `old` must match exactly once   |
| `list_dir`  | `path`                | dirs end with `/`; capped at `MINION_LIST_DIR_MAX` |
| `run_bash`  | `command`             | requires confirmation           |
| `wait_background` | `pid`           | waits for a backgrounded `run_bash` job; only PIDs with a map file |
| `kill_background` | `pid`           | stops a `run_bash` job this minion started; requires confirmation |

Tool failures come back as `ERROR [reason] message` plus a `hint:` line
(`unknown_tool`, `bad_args`, `not_found`, `binary_file`, `no_match`,
`ambiguous_match`, …); a failed `write_file`/`edit_file`/`run_bash` adds
`side_effects: unknown`. Denied actions return `DENIED by user: <action>. Do
not retry this or an equivalent action; …`.

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

## Acknowledgements

The harness hardening work (containment and credential scrubbing, deterministic
approval floor, typed terminal states and exit codes, interruption and crash
recovery, run budgets, compaction archive, trial rows) is inspired by
*Understanding Harness Engineering: From Agent Loops and Tool Interfaces to
Context, Sandboxes, Verification, and Long-Running Work*, a short technical
handbook for AI engineers, developers, and serious learners by
[**@techNmak**](https://x.com/techNmak/status/2106765347793858563).

## License

MIT License. See [`LICENSE`](LICENSE).
