# Changelog

All notable changes to `minion.py` from this point forward.

### Fixed — `read_file` empty-file marker, `/autocompress on`, abbr + docs

- `read_file` on an empty file now returns a clear `[<path>: empty file]` marker
  instead of an empty string (offset 1) or a nonsensical `lines 2-0 of 0` header
  (offset > 1), so an empty file is distinguishable from a failed read.
- `/autocompress on` was resetting to a hardcoded `85`, silently throwing away a
  custom `MINION_AUTOCOMPRESS_PERCENT` (e.g. `30` → `off` → `on` jumped to 85).
  It now restores the configured default via the new `AUTOCOMPRESS_DEFAULT`
  constant — still 85% when unset, and falls back to 85 when configured off so
  "on" still turns something on.
- `_precise_abbr` floored the billions branch to an integer (`1B`) even though
  its docstring promised two decimals like K/M; now `1.50B`.
- README documents three env vars that were missing from the table:
  `MINION_ACTIVE`, `MINION_READ_FILE_LINES`, `MINION_EMPTY_TURN_RETRIES`.
- Tests: new `tests/test_abbr.py`; empty-file case in
  `tests/test_read_file_paging.py`; `tests/test_autocompress.py` now checks that
  `on` restores the configured (non-85) default.

### Added — automatic context compression (`MINION_AUTOCOMPRESS_PERCENT`, `/autocompress`)

minion can now fold older context automatically when the conversation gets
close to the context limit, instead of waiting for a manual `/compress`.
Triggered after each settled model turn: if the last request's
`prompt_tokens` filled at least `MINION_AUTOCOMPRESS_PERCENT` of the active
source's context window, it compresses in place and prints a subtle
`↻ auto-compressed N turns → 1 summary (M chars), kept last K verbatim`
line. Default **85%** (set `0` to disable).

- New env var `MINION_AUTOCOMPRESS_PERCENT` (default 85, clamped 0–100; `0`
  disables auto-compression entirely). E.g. `MINION_AUTOCOMPRESS_PERCENT=30`
  on a 170K window fires at ~51K tokens.
- New `/autocompress` command: bare shows the current threshold/status;
  `/autocompress <1-100>` sets it; `off`/`0`/`disable` disables; `on`/`enable`
  re-enables at the configured `MINION_AUTOCOMPRESS_PERCENT` (85% by default).
  Takes effect immediately for the next turn.
- `compress()` gained an `auto=False` param. When `auto=True`, the keep count
  is raised to `max(COMPRESS_KEEP, body_len // 3)` — roughly the last third of
  the conversation stays verbatim — so auto-compress is deliberately more
  conservative than a manual `/compress` (which keeps only the last 2 turns).
  A manual `/compress` is still the sharp instrument; auto-compress is the
  safety net that preserves recent work while reclaiming room.
- `_maybe_autocompress(messages, prompt_tokens)` is the guard: returns `False`
  (no-op) when disabled, when there's no prompt-token count, when no source is
  active, when the context window is unknown/≤ 0, or when usage is below the
  threshold. The new module global `_LAST_PROMPT_TOKENS` is set inside
  `model_turn` (from both the llama.cpp `timings.prompt_n` and the OpenAI
  `usage.prompt_tokens` code paths) and read by the REPL loop after each turn.
- `tests/test_autocompress.py` pins the keep-formula (auto keeps ~⅓ vs
  manual's flat 2), all the `_maybe_autocompress` guard conditions, the
  `/autocompress` command parser (show/set/off/on/invalid/out-of-range), and
  `MINION_AUTOCOMPRESS_PERCENT` env-var seeding + clamping.

### Added — max context window shown in the footer / `/source`
The per-turn stats footer now shows the server's maximum context window next
to the current context size — `ctx 12K/150K` instead of just `ctx 12K` — so
you can see at a glance how much room is left. The limit is also shown in the
`/source` listing (`· 150K ctx`) and on the `→ switched to …` line after a
`/source` switch. This works for both backends the project ships against:

- **llama.cpp (local):** read from `/v1/models` (`data[].meta.n_ctx`), with a
  `/props` (`default_generation_settings.n_ctx`) fallback — the same endpoints
  the model-name resolver already probes.
- **Together (and any OpenAI-compat host that mirrors OpenAI's over-limit
  wording):** when the models list is empty/unavailable, a one-token-prompt
  request with a deliberately over-large `max_tokens` is rejected with a `400`
  whose message names the limit ("This model's maximum context length is N
  tokens…"); that's parsed. No tokens are generated (the request is rejected
  before inference).

New `Source.resolve_context_window()` tries these paths cheapest-first and
caches the result per source (so each turn doesn't re-probe). The footer reads
the cache and is **non-blocking**: on a cold source the max is omitted on the
first turn and a background thread warms the cache, so it fills in on a later
turn — Together's probe takes ~4s, which would be a jarring stall after the
answer if it ran synchronously every turn. `switch_source()` clears the cache so
a `/source <name> <model>` override re-probes against the new model id. A total
miss degrades to the old `ctx N` footer (no `/<max>`).

The current-context half of the field is **colorized by utilization** — green
under 30%, yellow from 30–60%, red at 60%+ — so a glance tells you whether
you're getting close to the limit; the max stays cyan and the `/` … ` ctx`
separators are dimmed. When the max is unknown the field falls back to a plain
uncolored `NNN ctx` and kicks off the background probe.

New: `Source.resolve_context_window()` (+ `_ctx_from_models`,
`_ctx_from_props`, `_ctx_from_overrun_probe`), `_ctx_field()` (the colored
footer field), `_ensure_ctx_probe()`, `_CTX_PROBE_STARTED`, and a
`_context_window` cache field on `Source`; `switch_source()` invalidates it on
switch/override. `tests/test_context_window.py` pins all three probe paths,
caching, cache-invalidation on switch, and the `_ctx_field()` colorization.

### Fixed — context-window max missing on the first turn with remote hosts
The background probe resolved too slowly for remote providers (Together, Z.ai),
so the first turn's footer showed a bare `ctx 607` with no `/<max>` — the max
only filled in on a later turn. Root cause: the probe order was
`/v1/models` → `/props` → over-`max_tokens` chat probe, optimized for local
llama.cpp where `/v1/models` answers in sub-millisecond. But Together's
`/v1/models` returns an **empty list** (no model entries, no context info) and
the TLS round-trip alone costs ~2.8s; only after that wasted call did the
over-`max_tokens` chat probe run (and resolve in ~0.1s). Total probe time
(~3.1s) exceeded the first turn's wall time, so the cache was still empty when
the footer rendered.

`resolve_context_window()` now branches on host locality. `Source._is_local()`
classifies the base URL (localhost, 127/10/192.168/172.16-31/169.254, or blank
→ local; everything else → remote). **Local** sources keep the old order
(`/v1/models` → `/props` → chat probe) — llama.cpp's metadata endpoints are
cheap and carry `n_ctx`. **Remote** sources lead with the over-`max_tokens`
chat probe (one cheap request, ~0.1s, works for any OpenAI-compat host that
mirrors the "maximum context length is N tokens" 400 wording), falling back to
`/v1/models` then `/props` only if the chat probe misses. This drops Together's
probe from ~3.1s to ~0.45s, comfortably inside the first-turn window.

New: `Source._is_local()`. `tests/test_context_window.py` adds Test 7 (a remote
source resolves via the chat probe **without** hitting `/v1/models`) and Test 7b
(`_is_local()` classification for localhost, LAN, and remote hosts).

### Fixed — agent silently stopping on empty turns (auto-continue)
A model turn that produced **no content and no tool calls** used to return
`TURN_DONE`, silently dropping the user to the chat prompt mid-task with no
explanation. This happened most often right after a short or empty tool result
(the classic case: a `grep` that exits cleanly with no matches), where the model
"knows" the answer but emits zero tokens — a low-entropy attractor common with
quantized local models (e.g. GLM-5.2 IQ4) on dense 50K+ contexts. It looked like
the agent finished when it had actually gone mute.

The turn loop now detects the empty turn and auto-recovers instead of stopping:

- `model_turn` returns a new `TURN_EMPTY` status (when an assistant turn has no
  text and no tool calls). The loop nudges the model — a `[Runtime note: …]` is
  injected into the current user turn telling it to emit a tool call or a visible
  answer — and retries with **recovery sampling** (more entropy, anti-repetition
  knobs) to break out of the empty-output attractor. The dangling `tool` result is
  preserved, so the retry has full context.
- After `MINION_EMPTY_TURN_RETRIES` (default 3) consecutive empty turns, it
  escalates to a **forced final answer** (`TURN_FORCE_FINAL`, the existing path),
  so the agent either says something concrete or reports it's blocked rather than
  spinning forever. A successful tool call resets the empty-turn counter (mirrors
  the reasoning-loop / malformed-stream counters).
- The path is fully opt-out: `MINION_EMPTY_TURN_RETRIES=0` restores the old
  behavior (empty turn = done). Empty turns under `/recover` (forced-final) still
  stop, since an empty forced answer already means "give up."

New: `EMPTY_TURN_NUDGE`, `EMPTY_TURN_RETRY_LIMIT`, `TURN_EMPTY`, and
`_last_is_dangling_tool()`; `_run_model_turn_loop` threads an `empty_turn_cuts`
counter through `model_turn`. `tests/test_empty_turn_recovery.py` pins the
behavior.

### Fixed — `_interrupt_watcher` traceback under redirected stdin
The Esc-watcher daemon thread called `sys.stdin.fileno()` before its `os.isatty`
guard. Under a redirected/captured stdin (pytest, piped input) `fileno()` raises
`UnsupportedOperation`, which surfaced as a noisy `PytestUnhandledThreadException`
whenever `model_turn` ran in a test. The `fileno()` call is now wrapped so the
watcher exits cleanly when stdin isn't a real fd — purely cosmetic, no behavior
change in the REPL (where stdin is a tty).

### Added — built-in `together` source + per-switch model override
minion now ships a built-in `together` source for the Together AI API. When
`TOGETHER_API_KEY` is set (in `~/.env` or the shell), a `together` source is
auto-registered at `https://api.together.xyz/v1`, defaulting to the
`zai-org/GLM-5.2` model. It's appended last, so it never displaces your
default startup source — opt in with `/source together` or `--source together`.
Defining your own `MINION_SOURCE_TOGETHER_*` vars overrides the built-in
entirely.

Because Together hosts many models, `/source` now takes an optional model
argument so you can point a multi-model host at any of its models without a
config edit:

- `/source together` → GLM-5.2 (the source's configured default)
- `/source together zai-org/GLM-4.6` → that model on the same endpoint

The override is per-switch and non-sticky: a later bare `/source together`
returns to the default. No model validation is done up front — a bad id is
left for the server to reject. A model override is recorded in the session
file alongside the source, so a `/resume` / `--resume` lands on the same
endpoint **and** model.

- `switch_source(name, model_override=None)` — optional model pin.
- `restore_source(source_name, model=None)` — best-effort source + model
  restore used by both resume paths.
- `/source` listing line updated to advertise the `[model]` argument.

### Fixed — tool-call delimiter injection from file/tool output
Tool results now escape active tool-call protocol delimiters before they enter
the next model context. This treats file contents and command output as
untrusted input, so source code that happens to contain a tool-call-looking tag
cannot be echoed back into an executable tool request.

The preferred text fallback protocol is now
`[minion_tool_call]...[/minion_tool_call]`; legacy `<tool_call>...</tool_call>`
blocks are still parsed for compatibility, but only when the whole assistant
message is standalone protocol text. Normal answers that quote source code,
docs, or prompt text containing a valid-looking tool call remain plain text.

### Fixed — risk classifier respects projects under `~/Downloads`
Write/edit risk classification now sends structured cwd/project-root metadata
to the classifier. The project root is the git root when available, otherwise
the launch directory, and write/edit paths are tagged as `in_project` or
`outside_project` before the model judges risk.

This fixes the noisy approval case where editing
`~/Downloads/<project>/file.py` was classified high just because the project
lived under Downloads. Downloads still matters when the target is outside the
active project root.

### Added — `/clear` and `/new` aliases for `/reset`
`/clear` and `/new` now do the same thing as `/reset`: clear the in-memory
context down to the system prompt and fork a fresh session id (so the old
chat is preserved on disk rather than overwritten). Mirrors the
`/compact` → `/compress` alias pattern.

### Removed — in-stream reasoning heuristics (signal counter, gibberish detector, leak-token guard)
The model was fixed, so the heuristic guards that papered over its failure modes
were removed (~340 LOC, ~10%). They were content-quality guesses that misfired
on legitimate reasoning, so with the underlying problem gone they were net
negative.

Removed entirely:
- `_reasoning_gibberish_reason` + `_ReasoningGibberishDetector` — the
  numeric/markup/short-token/layout-loop noise scorer and its rolling-window
  wrapper. `TURN_GIBBERISH_CUT`, `REASONING_GIBBERISH_CHARS`,
  `REASONING_GIBBERISH_RETRY_LIMIT`, `GIBBERISH_RECOVERY_NUDGE`,
  `GIBBERISH_CHECKPOINT_NUDGE` all gone.
- `_ReasoningLoopSignalCounter` + `REASONING_LOOP_SIGNALS` /
  `REASONING_LOOP_SIGNAL_LIMIT` — the "ready to act" phrase counter and its
  milestone printing. `TURN_LOOP_CUT`, `REASONING_LOOP_NUDGES`,
  `REASONING_LOOP_RETRY_LIMIT` gone.
- `_leak_token_hit` + `_LEAK_TOKEN_STRICT_RE` / `_LEAK_TOKEN_BROAD_RE` +
  `LEAK_TOKEN_GUARD` — the GLM `<|…|>` / mask-token leak detector on reasoning,
  content, and tool-args tails. `content_tail` / `args_tail` rolling buffers gone.
- The `gibberish` / `signals` / `gibberish_cut_count` branches of the cut
  handler and `_run_model_turn_loop`. The cut handler is now a single linear
  `reasoning_only` branch.

Kept (non-heuristic safety nets):
- `REASONING_ONLY_CHAR_LIMIT` / `REASONING_ONLY_RETRY_LIMIT` — plain char-count
  timeout, not a content-quality guess.
- `FORCED_FINAL_*` / `FINAL_ANSWER_TOOL` — the forced-final-answer rescue path.
- `MALFORMED_STREAM_RETRY_LIMIT` / `TURN_STREAM_CUT` — truncated-JSON tool-args
  retry.
- `_recovery_sampling_opts` / `RECOVERY_*` — shared by malformed-stream retry
  and `/recover`. Recovery temperature raised to `1.0` (was `0.2`); min_p,
  repeat_penalty, and DRY params added to break repetition loops by raising
  entropy rather than sharpening toward greedy.
- `MANUAL_RECOVERY_NUDGE` / `/recover` command.

The `model_turn` reasoning stream now just streams + counts `reasoning_only_chars`
+ cuts on the char limit; the cut handler is a single linear `reasoning_only`
branch. Tests for the removed detectors were deleted; the remaining recovery-path
tests (forced-final, malformed-stream, `/recover`) keep coverage of what stayed.
README's env-var table and "Reasoning recovery guards" section updated to match
the surviving knobs (and the new `MINION_TOOL_RESULT_CHARS` cap + DRY/repeat
recovery params, which were previously undocumented).

### Added — `/recover` command (manual recovery checkpoint)
New in-session command `/recover [optional note]` forces a low-temperature
visible checkpoint via the `final_answer` tool. After a bad stream —
truncated output, reasoning spirals, gibberish — type `/recover` (optionally
with a note) and the model discards any corrupted reasoning and emits a
bounded visible answer instead of continuing free-form.

- New `MANUAL_RECOVERY_NUDGE` prompt; `/recover` appends it (plus the user
  note) as a `[Runtime note: …]` user turn and runs
  `_run_model_turn_loop(messages, force_final=True, recovery_sampling=True)`.
- The REPL turn loop was extracted into `_run_model_turn_loop(messages,
  force_final=False, recovery_sampling=False)` so both the main REPL path
  and `/recover` share one code path (the inline loop in `main` was replaced
  with a call to the new function).
- `final_answer` tool description sharpened to ask for a "complete, concise"
  answer that doesn't trail off, since forced-final answers are the recovery
  fallback and a truncated-feeling response defeats the point.

### Added — gibberish recovery escalates to a visible checkpoint
Previously, if the gibberish detector cut the stream and the recovery retry
also collapsed into noise, minion gave up and waited for user input —
leaving the user staring at a dead turn with no answer. Now, after
`MINION_REASONING_GIBBERISH_RETRIES` (default 1) recovery attempts fail,
minion escalates: it nudges the model with `GIBBERISH_CHECKPOINT_NUDGE`
(asking for a bounded visible checkpoint: last valid result, next step,
blockers) and returns `TURN_FORCE_FINAL`, which re-enters the turn with
`force_final=True` + `recovery_sampling=True` so the model emits a
`final_answer` tool call at a low temperature instead of more hidden
reasoning. The turn loop resets all cut counters on a successful tool turn,
so a clean recovery puts the agent back on track.

### Added — forced-final truncation marker
When a forced `final_answer` response hits the token limit (`finish_reason`
contains `"length"`), the partial text is now saved into the message
history with a `[Truncated by token limit before completion.]` marker and a
yellow `✂ FORCED FINAL ANSWER HIT TOKEN LIMIT` notice, instead of being
silently discarded as an empty turn. `FORCED_FINAL_MAX_TOKENS` raised from
`1024` to `2048` to give the forced-final rescue more room to complete.

### Added — expanded gibberish detector (low-info & layout loops)
`_reasoning_gibberish_reason` now catches two more failure modes beyond
dense numeric/markup noise:

- **Repeated short fragments** — high ratio of ≤4-char tokens with very low
  unique-token diversity and a single dominant token (the "your you the ` **
  your your to so it" loop).
- **Low-information repetition** — most tokens are stop-words / single
  letters / digits, with low unique diversity.
- **Dominant token loop** — top-5 tokens account for >55% of the stream
  alongside heavy numeric/symbol content.
- **Layout-token repetition** — words like `pane`, `split`, `tab`, `layout`,
  `start`, `name` dominate alongside numeric/symbol noise (the
  "pane pane // pane pane 01" loop a layout-obsessed model can get stuck in).

Each sub-detector contributes +2 to the score (threshold remains 3), so any
two firing is enough to trip the cut. The docstring was updated to reflect
that the detector now targets "low-information scaffolding" in addition to
numeric/markup noise. Tests expanded with new sludge patterns for both new
modes.

### Fixed — empty assistant messages pruned before save/stream
A reasoning-only turn that produced no visible content and no tool calls
could leave an empty assistant message (`{"role": "assistant", "content": ""}` or
`null`) in the message array, which some chat templates reject on the next
request. `_is_empty_assistant_message(msg)` / `_prune_empty_assistant_messages(messages)`
now strip those turns at three points: before writing a session file, after
loading a session, and before opening a stream. A reasoning-only stall that
triggers the forced-final path no longer appends an empty assistant turn
before returning `TURN_FORCE_FINAL`.

### Changed — reasoning-only stall now also catches the no-signal case
The `reasoning_only` cut (large reasoning output with zero content/tool calls)
was previously only reachable via the signal counter path. It's now also set
when the turn ends with no `loop_cut`, no text, no tool calls, but
`reasoning_only_chars > 0` — so a model that streams a long reasoning block
and then stops (without emitting "ready to act" signals) still gets the
forced-final rescue instead of a silent empty turn.

### Changed — recovery nudges extracted to named constants
The inline nudge strings for forced-final, gibberish-recovery, gibberish-
checkpoint, and manual-recovery are now module-level constants
(`FORCED_FINAL_NUDGE`, `GIBBERISH_RECOVERY_NUDGE`,
`GIBBERISH_CHECKPOINT_NUDGE`, `MANUAL_RECOVERY_NUDGE`) for readability and
so the tests can assert against stable wording.

> **Note:** the file was previously called `miniagent.py` and configured via
> `AGENT_BASE_URL` / `AGENT_MODEL` / `AGENT_API_KEY`. Renamed to `minion` /
> `MINION_*` for clarity. The old env vars are silently ignored — set the new
> ones.

### Added — transcript shown on resume
Resuming a session (`--resume` at startup, or `/resume` mid-chat) now prints
the full conversation history as a one-line-per-message recap before the
first prompt, so you immediately re-orient on what the chat was about.

- New `_print_transcript(messages, max_chars=120)` helper — collapses each
  message to a single line (newlines → spaces, truncated to ~120 chars),
  color-codes by role (cyan user / green assistant / dim tool), renders
  assistant tool-call turns as `→ name(...)` so you can see what ran, and
  skips system messages.
- Called from `main()`'s `--resume` startup path (after the "↻ resumed …"
  header, framed by `── transcript ──` / `──────────────` dividers) so the
  user sees the whole conversation before typing.
- `/sessions <id>` detail view refactored to use the same helper (was a
  near-duplicate of the same render loop; now shares one code path).

### Added — short ids + model-generated descriptions in session listings
Session listings (`minion sessions`, `/sessions`, bare `/resume`) now show
two things they didn't before: a scannable **short id** and a
**model-generated description** that evolves as the conversation progresses.

- `_short_id(id)` returns the 6-hex suffix of an id (`YYYYMMDD-HHMMSS-XXXXXX`
  → `XXXXXX`) since the date+time prefix is shared/redundant across sessions
  created in the same minute. Shown in magenta in every listing.
- `_resolve_session` now matches on that short id (the tail segment) in
  addition to full id / prefix / index / title — so `minion --resume deadbe`
  works.
- `_maybe_refresh_description(id, messages)` makes one cheap non-streaming
  call every `SESSION_DESC_REFRESH` (default **6**, override with
  `MINION_SESSION_DESC_REFRESH`; `0` disables) user turns, asking the model
  for a ≤70-char one-liner about what the session is working on. Stored in
  the session file's `description` field alongside a `desc_turns` counter so
  the next refresh is scheduled correctly. Surfaced as a dim subtitle line
  under each listing entry; falls back to the first-message preview when no
  description exists yet. Failures (server down, empty reply) leave the
  existing description untouched.
- New `DESC_SYSTEM` prompt and `_DESC_REFRESH_DEFAULT` sentinel (resolved
  lazily to `SESSION_DESC_REFRESH` after `_env_int()` is defined, since the
  sessions section sits above `_env_int` in the file).
- Listings updated in all three spots: `_cli_sessions` (the `minion sessions`
  subcommand), `_cmd_sessions` (the in-session `/sessions`), and the bare
  `/resume` picker — all show `index  short-id  title · N msg · when · source`
  with the description (or preview fallback) on a second dim line.

### Added — Ctrl+C exit shows a grey resume hint
On Ctrl-D / Ctrl-C exit, after flushing the session, minion now prints
`resume with: minion --resume <full-id>` in grey — so you can copy-paste
straight back into the session you just left without running `minion sessions`
first.

### Fixed — plain `minion` run never saved its session
A plain `minion` invocation (no `--resume`/`--session` flag) never minted a
session id: `_session_id_from_args()` returned `None` and nothing filled it
in, so `_save_current()`'s `if session_id is None: return` guard silently
discarded every turn. Every plain-`minion` session was lost. Fixed by minting
a fresh id immediately when no resume/session flag is present; also untangled
the `_resume_requested` logic so session-loading only happens when the user
explicitly asked to resume (a fresh run no longer pointlessly tries to load
a non-existent file for its newly-minted id).

### Fixed — docs reflected removed DECSTBM status-bar feature
The README and module docstring still described the pinned scroll-region
status bar (`--no-scroll-bottom`, DECSTBM) as if it were current, but the
working tree had already moved to a plain banner printed into scrollback
(the scroll-region bar was removed because it broke terminal scrollback).
- README **Status bar** section rewritten to describe the current banner
  behavior and note the removed scroll-region approach as history.
- Defunct `--no-scroll-bottom` flag dropped from the README flags table and
  the module docstring's flags line (it was already not parsed by the code).
- Intro line-count claim updated from `~1500` to `~2200` to match reality.

### Added — chat sessions (save / resume)

Every chat is now automatically saved and resumable. Sessions are stored as
plain JSON files under `~/.minion/sessions/` (override with `MINION_HOME` or
`MINION_SESSIONS_DIR`), one file per session, holding the exact `messages`
array the model sees plus light metadata (id, title, source, cwd, timestamps).
Greppable, human-readable, and trivially round-trippable. A deliberately
lightweight take on session persistence — inspired by how Hermes
(`hermes_state.py`) stores sessions, but flat JSON files instead of SQLite,
since minion is a single local agent rather than a multi-platform gateway.

**Persistence layer** (new section in `minion.py`):
- `_write_session(id, messages, meta)` — atomic write (temp file + rename)
  so a crash mid-write can't corrupt an existing session. Merges metadata
  into the stored file, preserving `created_at` / `source` / `title` across
  re-writes.
- `_load_session(id)` — read a session dict or `None`.
- `_list_sessions(limit)` — newest-first list with auto-derived title (from
  the first user message), first-message preview, message count, source.
- `_delete_session(id)` — idempotent file removal.
- `_resolve_session(target, sessions)` — resolve a user-typed target to an
  id via number, exact id, unique prefix, or exact title.
- `_new_session_id()` — timestamp + 6 hex chars (`YYYYMMDD-HHMMSS-XXXXXX`).
- New `import secrets`.

**Auto-save**: the REPL saves the in-memory `messages` to the session file
after every model turn (and on Ctrl-D / `/quit` exit), so a crash or
accidental close never loses work. A session that never receives a user
message is never written to disk. Mirrors Hermes's per-turn flush pattern.

**New in-session commands**:
- `/sessions [n]` — bare lists recent sessions with index, title, msg count,
  relative time, source badge, and a dim preview line. With an index/id/title
  it prints the full transcript of one session inline.
- `/resume [target]` — switch to a past session mid-chat. Bare shows the
  recent list; with a target (`n`/id/prefix/title) it loads it. Resuming
  reselects the recorded **source** (endpoint + model) so the chat lands on
  the same backend it was talking to.
- `/save [title]` — persist now (otherwise it's automatic); optional
  explicit title overrides the auto-derived one.
- `/delete [target]` — remove a saved session file (refuses the current one).

**New CLI flags / subcommand**:
- `--resume [target]` — resume a session at startup. Bare = most recent
  (the "I closed my laptop" case); with a target it resolves by
  index/id/prefix/title. The parser checks whether the next arg starts with
  `-`, so `minion --resume --yolo` doesn't swallow `--yolo` as the target.
  If there are no saved sessions, it degrades to a fresh start with an
  informative message instead of an error.
- `--session <id>` — start a fresh run pinned to a specific session id.
- `minion sessions [query]` — **subcommand** that prints recent sessions
  and exits (no REPL). Optional substring query filters across title,
  first-message preview, and id, so once the list grows you can narrow with
  e.g. `minion sessions refactor`. Deliberately a separate verb from
  `--resume` (discover vs. enter) rather than `--resume list`, since "list"
  looks like it could be a session title and a flag whose meaning flips on a
  magic word is a footgun.

**`/reset` now forks a new session id** (mirrors Hermes's "new session on
/new") instead of clobbering the current one in place — so resetting never
overwrites the conversation you were just in.

**Tests**: new `tests/test_sessions.py` (11 tests) covering write→load
round-trip, atomic merge, missing-file handling, newest-first ordering,
resolve-by-index/id/prefix/title, delete idempotency, unique ids, title
sanitization, bare-`--resume`-picks-most-recent, the `--resume`-vs-next-flag
guard, and the `sessions` list/filter subcommand.

### Changed — abbreviated token counts in stats footer
Context and cache token counts in the per-turn stats footer are now
abbreviated for readability: `832` stays as `832`, `1500` → `1.5K`,
`78825` → `78K`, `1234567` → `1.2M`. The footer (model-generated token
counts, prompt context, and cached tokens) is the noisiest line once a
session grows long — whole-word counts like `78825` are hard to scan at a
glance.

- New `_abbr(n)` helper. Cutoffs: raw under 1K, one-decimal `X.YK` from
  1K–10K, whole `NNK` up to 1M, then the same split at millions.
- Applied in both footer branches: the llama.cpp `timings` path
  (`ctx P+C cached`) and the standard OpenAI `usage` path.

### Added — Esc at approval prompts (back to chat)
The `Y/n` approval prompt for writes/edits/bash now also accepts **Esc**.
Pressing Esc stops the current turn and drops you back to the chat input so
you can add more guidance — useful when the model is about to do something
slightly off and you'd rather redirect it than approve/deny in isolation.

- New `TURN_ESC` turn status and new `_EscToChat` exception. The exception is
  a control-flow signal (not a real error): it's raised by `_confirm`, left
  to propagate through `run_tool`, caught in `model_turn`.
- New `_ask_approval(prompt)` helper. Reads a single keypress in raw terminal
  mode (rather than line-based `input()`, which can't detect a bare Esc),
  disambiguating bare Esc from arrow-key / bracketed-paste escape sequences
  via a 50ms wait for trailing bytes. Returns `y` / `n` / `esc`, echoes the
  choice, and falls back to line input (`esc`/`e`/`\x1b` accepted) when
  stdin isn't a TTY. The prompt now reads `[Y/n/esc]`.
- `_confirm` raises `_EscToChat(action)` on Esc; Y/n behavior is unchanged.
- `run_tool` re-raises `_EscToChat` instead of swallowing it as an error
  string, and closes the cyan tool box with a `(escaped)` marker first.
- `model_turn` catches `_EscToChat` in both the native tool-call path and the
  text-fallback path. The escaped call is recorded as `CANCELLED by user
  (Esc)`; if the assistant emitted multiple tool calls, the remaining ones
  are marked `SKIPPED` so every `tool_call` still has a matching `tool`
  result (otherwise the chat template rejects the context on the next
  request). A synthetic user note is appended and `TURN_ESC` is returned.
- `main`'s REPL breaks the inner turn loop on `TURN_ESC`, returning to the
  chat input. The `/compress` approval also honors Esc now.
- New `tests/test_esc_approval.py` (5 tests): `_confirm` raises on esc,
  Y/n passthrough, `run_tool` propagation, `model_turn` history-validity on
  a multi-tool escape, and the REPL loop break.

### Added — pinned status bar (DECSTBM scroll region)
A one-line status bar pinned at row 1 of the terminal (model name, source,
approval mode, endpoint, command hints) using DECSTBM — the same scroll-region
primitive tmux / vim / less use for their status lines. Chat output scrolls
in the region below it (rows 2..bottom) without disturbing the bar.

- New `_setup_status_bar()` — emits `\033[2;{rows}r` to set the scroll
  region, paints row 1 with `_build_status_bar()`, erases stale content
  below the bar (`\033[2;1H\033[J`), and parks the cursor at the bottom.
  Sets `_STATUS_BAR_ACTIVE` so `_paint_status_bar` knows it can repaint.
- New `_build_status_bar(cols)` — composes the status string left-to-right,
  dropping less-important pieces (log path, URL, commands) when the terminal
  is too narrow. Adds the source name in magenta when more than one source is
  configured. Shows the approval mode color-coded (green for yolo/high,
  yellow for medium, dim for low).
- New `_paint_status_bar()` — repaints row 1 after a `/source` or `/yolo` /
  `/approval` switch so the bar always reflects current state.
- New `--no-scroll-bottom` flag to disable the whole feature (useful for
  piped/redirected output or terminals that don't implement DECSTBM).
- `_banner()` simplified to a one-line welcome (model name + source) since
  the bar now carries the endpoint/commands. Rebuilt on each call so it
  reflects the active source after a switch.
- Stale terminal content (previous shell output, prior sessions) is now
  cleared on startup so it doesn't linger below the bar.

> **Known limitation:** the pinned bar relies on terminal scroll-region
> support. Works in most terminals and under tmux, but some multiplexers
> (notably Zellij) don't fully implement DECSTBM — in those the bar may
> scroll away after enough output.

### Added — multi-source system (`MINION_SOURCES`)
Define multiple named endpoints and switch between them at runtime without
restarting. Each "source" bundles a base URL, API key, and optional model
name. Conversation context is preserved across switches (use `/reset` for a
clean slate).

- New `MINION_SOURCES=local,zai` env var (comma-separated source names;
  the first is the default at startup).
- Per-source config via `MINION_SOURCE_<NAME>_BASE_URL`,
  `MINION_SOURCE_<NAME>_API_KEY`, `MINION_SOURCE_<NAME>_MODEL`.
- **`$key` indirection** — `MINION_SOURCE_ZAI_API_KEY=$zai_test` looks up
  the env var (or `~/.env` key) named `zai_test` rather than taking the
  literal string. Avoids duplicating keys that already live somewhere.
- **Auto-discovery** — if `MINION_SOURCES` is unset, minion scans for
  `MINION_SOURCE_*_BASE_URL` vars and builds sources from those.
- New `Source` class (`resolve_model()` queries `/v1/models` if no model is
  set; `display_model()` shows `auto` when unset). Each source has its own
  `OpenAI` client.
- New `switch_source(name)` — reassigns the global `client` / `MODEL` /
  `ACTIVE` so a mid-session swap is picked up instantly by every function
  that reads them.
- New `--source <name>` flag and `MINION_ACTIVE` env var to pick the
  starting source.
- New `/source` REPL command — bare lists all sources with model + URL;
  `/source <name>` switches and repaints the status bar.
- New `sources.example.env` with an annotated multi-source template.
- Backward compatible: if no `MINION_SOURCE_*` vars are present, a single
  `local` source is built from the legacy `MINION_BASE_URL` /
  `MINION_API_KEY` / `MINION_MODEL` vars.

### Added — `~/.env` auto-loading (`MINION_ENV_FILE`)
minion now reads `~/.env` at startup (before source discovery) and populates
`os.environ` from it, without clobbering vars already set in the shell.
Sources, API keys, and other config can live in one place instead of being
exported in every terminal.

- New `_load_env_file()` — parses `KEY=VALUE` lines (handles `export`
  prefix, quoted values, comments, blank lines). Points at
  `MINION_ENV_FILE` if set, otherwise `~/.env`.
- Existing `export`-based setups keep working unchanged.

### Added — escalating reasoning-loop nudges + retry limit
The reasoning-loop guard now escalates across multiple cuts instead of
firing the same nudge repeatedly.

- `REASONING_LOOP_NUDGES` — a tuple of 3 escalating nudges: first is a
  gentle "stop planning, take action"; second is stricter ("exactly one
  concrete action"); third is a hard stop ("emit only a tool call now").
  The nudge index is clamped to `len(nudges) - 1`.
- New `MINION_REASONING_LOOP_RETRIES` env var (default: number of nudges)
  — after this many cuts, minion stops retrying, prints a "max retries hit"
  message, and drops back to the prompt for user input.
- New `TURN_LOOP_CUT` return status from `model_turn`; the REPL increments
  `reasoning_loop_cuts` and retries with the next nudge (resets to 0 on a
  successful tool turn).
- New `_nudge_current_user_turn(messages, nudge)` — strips any prior
  `[Runtime note: …]` from the latest user turn and appends the new nudge.
- New `RUNTIME_NOTE_RE` to clean up stale runtime notes before re-nudging.

### Added — reasoning-loop milestone warnings
As "ready-to-act" signals accumulate during reasoning, minion now prints
visible warnings at 25 / 50 / 75 / 100% of the cut threshold — so you can
see the model spiraling before it gets cut, not just at the cut itself.

- `_ReasoningLoopSignalCounter.feed()` returns the running hit count.
- Milestones computed as clamped fractions of `REASONING_LOOP_SIGNAL_LIMIT`;
  warnings are yellow (`⚠ REASONING LOOP WARNING — M/N signals (XX%)`),
  the threshold-cross is red (`⚠ REASONING LOOP LIMIT HIT`).
- Fires at most ~5 lines per turn (first hit + 4 milestones) to keep noise
  down while remaining impossible to miss in the dim reasoning stream.

### Added — streaming usage + TTFT in stats footer
The stats footer at the end of each turn now works with servers that send
the standard OpenAI `usage` object (not just llama.cpp's `timings`).

- `stream_options={"include_usage": True}` sent on all streaming requests so
  the final SSE chunk carries token counts (OpenAI, Z.ai, vLLM, etc.).
- **TTFT** (time-to-first-token) measured client-side on the first chunk
  carrying real output; shown when available (`Tms ttft`).
- Footer now shows cached-token counts when the server reports them
  (`ctx P+C cached`).
- Three-tier fallback: llama.cpp `timings` → standard `usage` object →
  wall-clock only.

### Added — `pip install` support
- New `setup.py` / `pyproject.toml` — registers a `minion` console script
  pointing at `minion:main()`. `pip install -e .` for editable (picks up
  edits immediately), `pip install .` for non-editable.
- New `requirements.txt` (`openai>=1.0`).

### Changed — `model_turn` takes `reasoning_loop_cut_count`
Now accepts a second parameter so the REPL can track how many times the
reasoning loop has been cut in the current user turn and escalate nudges
accordingly. The stub signature change is what broke the old compact-alias
test.

### Changed — open_stream sends `stream_options`
Both the tools-enabled and fallback (no-tools) streaming requests now send
`stream_options={"include_usage": True}`.

---

### Added — Esc-to-interrupt during model generation
Press **Esc** while the model is streaming (or during the spinner wait
before the first token) to drop the current stream and return to the
prompt. Partial content is discarded and a synthetic user turn is appended
to context so the model knows what happened on its next turn.

- New `_interrupt_watcher()` daemon: puts stdin in raw mode (ISIG off so
  Ctrl+C still kills the process), polls for bare Esc with a 50ms wait
  for trailing bytes (so it doesn't fire on arrow-key / bracketed-paste
  escape sequences), debounces at 250ms, restores termios on exit.
  Started by `model_turn` before the spinner; signaled to exit in the
  `finally` block. Two events (`_INTERRUPT_EVENT`, `_USER_INTERRUPTED`)
  separate "watcher should exit" from "user actually pressed Esc" so
  cleanup doesn't get confused with a real interrupt.
- `model_turn` checks `_USER_INTERRUPTED` between chunks; on hit it closes
  the stream, prints `↳ interrupted by user (Esc) after N.Ns, N chars
  streamed`, appends a `[User interrupted your previous response with
  Esc. Acknowledge briefly and wait for their next message.]` user turn,
  and returns `TURN_DONE` so the REPL drops to the prompt instead of
  looping into another turn.
- Spinner label changed to `"thinking · esc to interrupt"` so the
  affordance is visible at the wait-for-first-token moment.
- In-flight tool calls (`run_bash`, `write_file`, etc.) are **not**
  cancelled — they run to completion. Hard-stop with Ctrl+C if you need
  one. (Cancel-a-running-tool is a separate follow-up.)

### Added — reasoning-loop guard (`MINION_REASONING_LOOP_SIGNALS`)
Reasoning models sometimes spin in place — they keep saying "let me
implement…" / "start coding…" / "now I'll write the code…" without ever
emitting content or a tool call, burning tokens and stalling the turn.
minion counts how many of those "ready to act" phrases appear during the
reasoning phase and, after the threshold, cuts the stream and appends a
one-shot nudge to the latest user turn telling the model to stop planning
and take a concrete action.

- `REASONING_LOOP_SIGNALS` (tuple of 9 phrases) and
  `REASONING_LOOP_SIGNALS_LIMIT` (default 10, override with
  `MINION_REASONING_LOOP_SIGNALS` env var; `0` disables) module constants.
- New `_ReasoningLoopSignalCounter` class — sliding-window phrase counter
  that scans each streamed `reasoning_content` chunk for new occurrences
  (only counts matches that extend past the previous boundary so we don't
  double-count a phrase split across two chunks).
- `model_turn` instantiates one per turn; on threshold hit it closes the
  stream and appends the `REASONING_LOOP_NUDGE` text to the most recent
  user turn (via `_nudge_current_user_turn`, which creates one if there
  isn't one yet). Prints `↳ cut reasoning loop after N ready-to-act
  signals; nudging implementation` so the cut is visible in the log.
- Only active during the reasoning phase (skipped once `content` /
  `tool_calls` start arriving), so a model that legitimately says "let me
  implement" once before doing it isn't tripped up.

### Added — tool-running spinner
The same `LifeSpinner` that animates during model streaming now also runs
between the cyan `┌─ name` / `│ args` lines and the cyan `└─ result`
lines, with label `"running"`. Without it the screen freezes for the
duration of a slow `run_bash` / `_assess_risk` round-trip / large
`write_file` — the user just sees the green model output end and then
nothing until the result box pops in.

- `LifeSpinner` gained a `label=` constructor arg (`"thinking"` by
  default; `model_turn` uses `"thinking · esc to interrupt"`; tool bodies
  use `"running"`).
- New `_ACTIVE_SPINNER` module-level pointer set by `run_tool()` around
  the tool body. `_confirm()` pauses/resumes it around its own I/O so the
  auto-allow line / `Y/n` prompt don't get clobbered by an animation tick.

### Added — risk-gated approval (`--approval` / `/approval`)
Sits between today's "ask for everything" default and `--yolo`'s "ask for
nothing". Every write / edit / bash call is risk-classified by a single
cheap non-streaming call to the same model before it runs. Levels:
`low` (read-only or trivially reversible), `medium` (modifies state but
contained/reversible), `high` (destructive, hard to reverse, or broad
scope). `APPROVE_LEVEL` is the minimum level that requires approval:

| flag                    | prompts at        | auto-allows       |
| ----------------------- | ----------------- | ----------------- |
| _(default)_             | low + medium + high | —               |
| `--approval medium`     | medium + high     | low               |
| `--approval high`       | high only         | low + medium      |
| `--yolo`                | _(never)_         | everything        |

In `--approval high` mode, `ls`, `cat`, single-file writes, `pip install`,
etc. run without asking; only `rm -rf`, `git push --force`, broad
destructive ops, etc. need a yes/no. The assessment is shown in brackets
next to the prompt (`[risk: HIGH — recursive force delete in /tmp]`) so
the user has context for the decision, and auto-allowed calls print a
one-liner (`↳ auto-allow [low] ls -la (read-only listing)`).

- New module-level `LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}` and
  `APPROVE_LEVEL` (string or `None` for yolo). CLI flag `--approval <level>`
  parsed in the config block; `--yolo` overrides it and sets `APPROVE_LEVEL
  = None` (no risk call, no prompt).
- New `RISK_SYSTEM` prompt + `_assess_risk(action)` function. Non-streaming
  call with a 15s timeout; logs to `llamacpp.log` under `_purpose: risk`.
  Defensive parse: tries JSON first, falls back to regex-matching a level
  word, falls back to `("high", "<error>")` on any failure — so a broken
  classifier can never silently auto-approve a dangerous command.
- `_confirm(action)` now takes the assessment, shows it inline at the
  prompt (color-coded: dim/yellow/red), and auto-allows with a one-liner
  when `LEVEL_ORDER[assessed] < LEVEL_ORDER[APPROVE_LEVEL]`. YOLO
  short-circuits before the call (no point paying for a call we won't
  act on).
- New REPL command `/approval [level]` — bare shows current setting,
  with arg sets it (`low`/`medium`/`high`/`yolo`; unknown values print a
  yellow warning and leave state unchanged). `/yolo` now prints both
  `yolo=` and `approval=` so the relationship is obvious.
- README updated with the approval-modes table; banner updated to list
  `/approval`.

### Added — base-level traffic log (`llamacpp.log`)
Append-only JSONL record of every byte shipped to / received from the llama.cpp
endpoint. Lives next to `minion.py`.
- `req` events: full outgoing request body (model, messages, tools, stream
  flag) logged before the HTTP call. Fallback (no-tools) requests are logged
  with a `_fallback` marker.
- `resp` events: every raw SSE chunk captured via `model_dump()` as it
  streams in — preserves `reasoning_content`, tool-call deltas, etc. at the
  literal ground-truth level, before any parsing/rendering.
- File opened once at module load with `buffering=1` (line-buffered) so each
  event is flushed immediately; survives crashes without losing recent turns.
- Stream wrapped in a `_LoggingStream` iterator; logging errors are swallowed
  so a disk-full / permission error can never break the agent's response.
- New `import time`; new `_log_event()` helper and `LOG_PATH` constant.

### Added — terminal UI polish
- `LifeSpinner` — a 1-row Conway's Game of Life that runs on a background
  thread while waiting for the first token. Gliders/blinkers actually evolve
  (rows above/below mirror the current row so each cell gets the standard
  8-neighbor count, otherwise a 1-row CA is degenerate). Uses `\033[2K\r` to
  overwrite its own line and `\033[?25l/h` to hide/show the cursor.
- Stats footer at the end of every turn, pulled from llama.cpp's `timings`
  object on the final SSE chunk: `N tok · X.X tok/s · ctx P+C cached · T.Ts wall`.
  Falls back to wall-clock only if the server doesn't send timings.
- Banner at startup with model name, endpoint, and command hints.
- Tool calls now render as a small box (`┌─ name` / `│ args` / `│ output` /
  `└─`) instead of a bare arrow. Output truncated to 800 chars in the display;
  the model still receives the full result via the messages array.
- New ANSI helpers: `MAGENTA`, `BOLD`, `CLEAR_LINE`, `HIDE_CURSOR`,
  `SHOW_CURSOR`.

### Changed
- `model_turn` now starts the spinner before opening the stream and stops it
  the moment the first chunk arrives (or in a `finally` if the stream errors).
- Tool output is line-wrapped into the box rather than dumped raw.
- `_log_event("resp", ...)` in `compress()` wraps the `model_dump()` call in
  `try/except` to match the streaming path's "never let logging break the
  agent" pattern — a non-pydantic response object won't crash the summary
  call anymore.

### Fixed — `/compress` could leave an orphan tool message in the kept tail
If the last `COMPRESS_KEEP` turns ended with a half-finished tool-call
sequence (e.g. the assistant called a tool but the result was the last turn,
or — more commonly — the assistant tool-call turn landed in `head` and only
its `tool` result made it into `tail`), llama.cpp's chat template would raise
`Message has tool role, but there was no previous assistant message with a
tool call!` on the very next request. `compress()` now walks the front of
the tail and drops any leading `tool` or unmatched `assistant(tool_calls)` turn
before splicing the summary in. The `summarized_n` count is bumped by the
number of extra turns absorbed so the user-visible footer stays honest.

### Added — multi-line chatbox input
Replaces the bare `input()` prompt with a framed, multi-line editor in the
terminal. Prompt, streamed model output, tool confirmations, and the next
prompt all stay in the normal terminal scrollback (no alternate screen) to
avoid garbling the REPL after submit.
- Enter submits; Alt+Enter / Ctrl+J insert newlines.
- Bracketed-paste mode preserves pasted newlines verbatim and strips a
  trailing newline so pasting never accidentally submits.
- Up/Down navigate past submissions; Left/Right move within the current
  line; Home/End jump to line start/end; Ctrl+U clears the line;
  Ctrl+C cancels.
- Long lines word-wrap visually inside the box; the buffer stays one logical
  string (newlines preserved) so the model sees the real text.
- Falls back to plain `input()` when stdin/stdout is not a TTY.
- New `read_multiline()` public entry point and `_chatbox_raw()` /
  `_chatbox_fallback()` helpers; new imports `select`, `shutil`, `termios`.

### Added — `/compress` context summarization
New REPL command. Asks the model to summarize everything except the system
prompt and the last `COMPRESS_KEEP=2` turns, then splices the summary in as a
single labeled user turn. Useful when the context window is filling up but
you want to keep working without `/reset`.
- Non-streaming summary call (spinner would be visual noise for a one-shot;
  one `model_dump` of the response is logged to `llamacpp.log` with a
  `_purpose: compress` marker so the summary is recoverable from the log).
- Confirmation prompt (skipped under `/yolo`).
- One-line stats footer: `compressed N turns → 1 summary (X chars), kept last K verbatim`.
- Header on the summary turn: `[Compressed context — N earlier turns
  summarized; last K turns kept verbatim]` so the model knows what it's
  reading on subsequent turns.
- Nothing-to-compress short-circuit: if the body has ≤ `COMPRESS_KEEP` turns,
  prints `nothing to compress (N turns in context)` and bails.
- Failure modes (APIConnectionError, generic API error, empty summary) leave
  `messages` untouched and print a one-line error — never half-compress.
- Tool-call turns and tool-result turns are rendered into the summary prompt
  with their content truncated to 2k chars each, so a giant `read_file`
  doesn't blow up the summarization call itself. Assistant tool-call turns
  are rendered as `[assistant] → tool_name(args)` for readability.
- New `compress(messages, keep=COMPRESS_KEEP)` function (returns
  `(kept_n, summarized_n, summary_chars)` or `None` on failure).
- New `COMPRESS_KEEP = 2` module constant.
- Banner and module docstring updated to mention `/compress`.
