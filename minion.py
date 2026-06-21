#!/usr/bin/env python3
"""minion — a deliberately tiny coding agent for self-hosted or remote models.

One file, one dep (`openai`), no TUI framework. Points at any OpenAI-compatible
endpoint (vLLM / llama.cpp / SGLang / Z.ai / OpenAI itself). Survives models
whose native tool-calling isn't wired up yet by falling back to parsing
<tool_call>...</tool_call> tags out of the text — the convention most open
models (Hermes/Qwen/Nemotron) emit.

  pip install openai
  export MINION_BASE_URL=http://localhost:8000/v1   # your served endpoint
  export MINION_MODEL=your-model-name
  export MINION_API_KEY=sk-noop                    # any string; local servers ignore it
  python minion.py

Multiple sources — define named endpoints and switch between them at runtime:

  MINION_SOURCES=local,zai
  MINION_SOURCE_LOCAL_BASE_URL=http://localhost:8080/v1
  MINION_SOURCE_ZAI_BASE_URL=https://api.z.ai/api/paas/v4
  MINION_SOURCE_ZAI_API_KEY=***                   # $name = key from env / ~/.env
  MINION_SOURCE_ZAI_MODEL=glm-x-preview

  python minion.py --source zai                     # start on Z.ai

Sessions — every chat is auto-saved to ~/.minion/sessions/ and resumable:

  python minion.py sessions           # list saved sessions (prints + exits)
  python minion.py sessions --page 2  # next page of saved sessions
  python minion.py sessions refactor  # …filtered by a substring query
  python minion.py --resume <id|short-id|prefix|title>  # resume a past session
  python minion.py --resume 1                        # resume the most recent
  /sessions                       # list recent sessions (with short ids)
  /sessions --page 2              # next page of recent sessions
  /resume <n|short-id|title>      # switch to another session mid-chat
  /save [title]                   # save the current session (title optional)

Toggles in-session: /source [name]  /yolo  /approval [level]  /compress  /compact  /reset  /recover  /sessions  /resume  /save  /delete  /quit
Flags: --yolo  --approval <all|low|medium|high|yolo>  --source <name>  --resume <target>  --session <id>
Env:   MINION_APPROVAL=<all|low|medium|high|yolo>  (persistent default approval mode; ~/.env or shell)
"""
import json
import os
import random
import re
import secrets
import select
import shlex
import shutil
import subprocess
import sys
import termios
import threading
import time

import httpx
from openai import OpenAI, APIConnectionError, APIError

# --- env file ---------------------------------------------------------------
# Load ~/.env (or MINION_ENV_FILE) into os.environ without clobbering vars
# already set in the shell. Lets source config / API keys live in one place
# instead of being exported in every terminal.
_ENV_FILE = os.path.expanduser(os.environ.get("MINION_ENV_FILE", "~/.env"))


def _load_env_file():
    try:
        with open(_ENV_FILE, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if k.startswith("export "):
                    k = k[len("export "):].strip()
                if not k or k in os.environ:
                    continue
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                    v = v[1:-1]
                os.environ[k] = v
    except (OSError, IOError):
        pass


_load_env_file()


# --- sessions ---------------------------------------------------------------
# Chat history persistence. Each session is one JSON file under
# ~/.minion/sessions/ (override with MINION_HOME / MINION_SESSIONS_DIR).
# The file stores the exact `messages` array the model sees (system prompt
# + every turn + tool calls/results), plus a little metadata (id, title,
# created_at, updated_at, cwd, source). Greppable, human-readable, and it
# round-trips trivially — load it back in and you have a resumable chat.
#
# Design cribbed from Hermes (hermes_state.py / SessionDB), which uses a
# SQLite store + FTS5 search because it's a multi-platform gateway (web,
# CLI, Telegram, …) with billing and compression chains. minion is a single
# local agent, so a flat directory of JSON files gives the same UX
# (auto-save per turn, /sessions, /resume, /save) without the weight.

SESSION_HOME = os.path.expanduser(
    os.environ.get("MINION_HOME", "~/.minion")
)
SESSION_LIST_DEFAULT_LIMIT = 10
SESSION_LIST_MAX_LIMIT = 100


def _sessions_dir():
    """Where session files live. Honors MINION_SESSIONS_DIR, then MINION_HOME/sessions."""
    return os.path.expanduser(
        os.environ.get("MINION_SESSIONS_DIR",
                       os.path.join(SESSION_HOME, "sessions"))
    )


def _new_session_id():
    """Short, unguessable, sortable-ish: YYYYMMDD-HHMMSS-<6 hex>."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def _safe_title(text, maxlen=60):
    """Turn the first user message into a filesystem-safe-ish title.

    Collapses whitespace, strips control chars, clamps length. We don't
    scrub for path-separators beyond replacing them with spaces — the id
    (not the title) is the filename, so a weird title can't break lookup.
    """
    if not text:
        return None
    text = " ".join(str(text).split())
    text = "".join(c for c in text if c.isprintable())
    if len(text) > maxlen:
        text = text[:maxlen - 1] + "…"
    return text or None


def _is_empty_assistant_message(msg):
    if msg.get("role") != "assistant" or msg.get("tool_calls"):
        return False
    content = msg.get("content")
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str) and part.strip():
                return False
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return False
        return True
    return False


def _prune_empty_assistant_messages(messages):
    """Drop assistant turns that have neither visible content nor tool calls."""
    if not isinstance(messages, list):
        return 0
    kept = [m for m in messages if not _is_empty_assistant_message(m)]
    removed = len(messages) - len(kept)
    if removed:
        messages[:] = kept
    return removed


def _session_path(session_id):
    return os.path.join(_sessions_dir(), f"{session_id}.json")


def _write_session(session_id, messages, meta=None):
    """Persist `messages` to the session file. Creates the dir if needed.

    Writes atomically (temp file + rename) so a crash mid-write can't
    corrupt the existing session. `meta` (title, source, cwd, …) is
    merged into the stored metadata.
    """
    d = _sessions_dir()
    os.makedirs(d, exist_ok=True)
    _prune_empty_assistant_messages(messages)
    path = _session_path(session_id)
    now = time.time()
    existing = {}
    try:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, dict):
            existing = {}
    except (OSError, IOError, json.JSONDecodeError):
        pass
    data = {
        "id": session_id,
        "messages": messages,
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    for k in ("title", "source", "cwd", "model"):
        if k in existing:
            data[k] = existing[k]
    if meta:
        for k, v in meta.items():
            if v is not None:
                data[k] = v
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
    try:
        os.utime(path, (data["updated_at"], data["updated_at"]))
    except (OSError, TypeError, ValueError):
        pass


def _load_session(session_id):
    """Read a session file. Returns the dict (id, messages, meta…) or None."""
    path = _session_path(session_id)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "messages" in data:
            _prune_empty_assistant_messages(data.get("messages"))
            return data
    except (OSError, IOError, json.JSONDecodeError):
        pass
    return None


def _session_files_newest():
    """Return session JSON filenames newest-first using filesystem mtimes.

    Listing many sessions should not require parsing every old transcript.
    The session writer keeps mtime aligned with updated_at, so this is a
    cheap index for recent-session browsing.
    """
    d = _sessions_dir()
    files = []
    try:
        with os.scandir(d) as entries:
            for entry in entries:
                if not entry.name.endswith(".json"):
                    continue
                try:
                    if not entry.is_file():
                        continue
                    files.append((entry.stat().st_mtime, entry.name))
                except OSError:
                    continue
    except OSError:
        return []
    files.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [name for _, name in files]


def _session_summary_from_file(fname):
    d = _sessions_dir()
    path = os.path.join(d, fname)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "messages" not in data:
            return None
    except (OSError, IOError, json.JSONDecodeError):
        return None

    sid = data.get("id") or fname[:-5]
    msgs = data.get("messages", [])
    preview = ""
    for m in msgs:
        if m.get("role") == "user" and m.get("content"):
            preview = _safe_title(m["content"]) or ""
            break
    return {
        "id": sid,
        "short": _short_id(sid),
        "title": data.get("title") or preview or "(empty)",
        "description": data.get("description"),
        "preview": preview,
        "updated_at": data.get("updated_at", 0),
        "n": len([m for m in msgs if m.get("role") != "system"]),
        "model": data.get("model"),
        "source": data.get("source"),
        "cwd": data.get("cwd"),
    }


def _session_matches_query(summary, query):
    if not query:
        return True
    q = query.lower()
    return (
        q in (summary.get("title") or "").lower()
        or q in (summary.get("description") or "").lower()
        or q in (summary.get("preview") or "").lower()
        or q in (summary.get("id") or "").lower()
        or q in (summary.get("short") or "").lower()
    )


def _list_sessions(limit=20, offset=0, query=None):
    """Return sessions newest-first as dicts: id, title, description, preview, n.

    `preview` is the first ~60 chars of the first user message; `description`
    is an optional model-generated one-liner refreshed every N turns (richer
    than the static first-message title). `n` is the turn count (non-system
    messages)."""
    limit = max(0, int(limit)) if limit is not None else None
    offset = max(0, int(offset or 0))
    files = _session_files_newest()
    out = []
    for fname in files:
        summary = _session_summary_from_file(fname)
        if not summary or not _session_matches_query(summary, query):
            continue
        out.append(summary)
        if query:
            continue
        if limit is not None and len(out) >= offset + limit:
            break
    if query:
        out.sort(key=lambda s: s["updated_at"], reverse=True)
    end = None if limit is None else offset + limit
    return out[offset:end]


def _delete_session(session_id):
    """Remove a session file. Returns True if something was deleted."""
    path = _session_path(session_id)
    try:
        os.remove(path)
        return True
    except OSError:
        return False


# After this many user turns, minion asks the model for a short (≤70 char)
# description of the whole conversation so far and stores it in the session
# file's `description` field. The description refreshes on every Nth turn
# thereafter, so it tracks what the chat is actually about as it evolves —
# far more useful in `minion sessions` than a static first-message title.
# 0 disables the refresh entirely (the auto-derived title is used as-is).
# The actual value is resolved after _env_int() is defined (below), so this
# is just a sentinel; see SESSION_DESC_REFRESH.
_DESC_REFRESH_DEFAULT = 6


def _short_id(session_id):
    """The scannable tail of a session id (the 6-hex suffix), for listings.

    The full id is `YYYYMMDD-HHMMSS-XXXXXX`; in a list of recent sessions the
    date+time prefix is shared/redundant, so we show just the 6 hex chars as a
    quick tag the user can grep or pass to --resume."""
    if session_id and "-" in session_id:
        return session_id.rsplit("-", 1)[-1]
    return session_id or ""


def _maybe_refresh_description(session_id, messages):
    """Ask the model for a one-line session description, if enough turns have
    passed since the last refresh.

    Refreshes at the configured interval (every SESSION_DESC_REFRESH user
    turns). The description is stored in the session file's `description`
    field and surfaced in `minion sessions` / `/sessions` listings. A failure
    (server down, empty response) leaves the existing description untouched.

    Returns the new description string, or None if no refresh happened.
    """
    if SESSION_DESC_REFRESH <= 0:
        return None
    # Count user turns (exclude synthetic runtime-note-only turns).
    user_turns = sum(
        1 for m in messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and not m["content"].lstrip().startswith("[")
    )
    existing = _load_session(session_id) or {}
    last_desc_turns = existing.get("desc_turns", 0)
    # Refresh when we've crossed a multiple of SESSION_DESC_REFRESH since the
    # last recorded refresh. Don't refresh before the first threshold (so a
    # 1-turn "hello" session doesn't burn a model call).
    if user_turns < SESSION_DESC_REFRESH:
        return None
    if user_turns - last_desc_turns < SESSION_DESC_REFRESH:
        return None
    # Build a compact transcript for the summarizer — truncate tool outputs
    # and skip the system prompt to keep the call cheap.
    def _trim(msgs, per_msg=500, cap=20):
        out = []
        for m in msgs:
            if m.get("role") == "system":
                continue
            c = m.get("content")
            if c is None and m.get("tool_calls"):
                calls = ", ".join(
                    f"{tc['function']['name']}(...)" for tc in m["tool_calls"])
                c = f"→ {calls}"
            elif isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            c = (c or "").replace("\n", " ").strip()
            if len(c) > per_msg:
                c = c[:per_msg - 1] + "…"
            out.append(f"[{m.get('role', '?')}] {c}")
            if len(out) >= cap:
                break
        return "\n".join(out)
    prompt = _trim(messages[-30:])  # last ~30 messages is plenty of context
    payload = [
        {"role": "system", "content": DESC_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    try:
        _log_event("req", {"model": MODEL, "messages": payload,
                           "stream": False, "_purpose": "session_desc"})
        resp = client.chat.completions.create(
            model=MODEL, messages=payload, stream=False, timeout=20)
        try:
            _log_event("resp", {"_purpose": "session_desc", "data": resp.model_dump()})
        except Exception:
            pass
    except APIConnectionError:
        return None  # server down — leave existing description as-is
    except Exception:
        return None
    desc = (resp.choices[0].message.content or "").strip().splitlines()
    desc = desc[0].strip() if desc else ""
    if not desc:
        return None
    desc = _safe_title(desc, maxlen=70) or desc[:70]
    # Persist the description + the turn count it was generated at so we know
    # when to refresh next. Merge into existing meta (don't clobber messages).
    _write_session(session_id, messages, {"description": desc,
                                          "desc_turns": user_turns})
    return desc


def _resolve_session(target, sessions=None):
    """Resolve a user-typed target to a session id.

    Accepts: a full id, a numeric index into the recent-sessions list,
    a unique id prefix, or an exact title. Returns the id or None.
    """
    if not target:
        return None
    target = target.strip()
    sessions = sessions if sessions is not None else _list_sessions(limit=50)
    ids = [s["id"] for s in sessions]
    # numeric index → recent-sessions slot
    if target.isdigit():
        idx = int(target)
        if 1 <= idx <= len(sessions):
            return sessions[idx - 1]["id"]
    # exact id
    if target in ids:
        return target
    # unique prefix
    prefixed = [i for i in ids if i.startswith(target)]
    if len(prefixed) == 1:
        return prefixed[0]
    # short id (the 6-hex suffix shown in listings) — match the tail segment
    # since the date+time prefix is shared/redundant across sessions created
    # in the same minute.
    suffixed = [i for i in ids if i.endswith("-" + target) or _short_id(i) == target]
    if len(suffixed) == 1:
        return suffixed[0]
    # exact title
    titled = [s["id"] for s in sessions if s["title"] == target]
    if len(titled) == 1:
        return titled[0]
    return None


# --- model sources ----------------------------------------------------------
# minion talks to any OpenAI-compatible endpoint. A "source" bundles a
# base_url, api_key, and model name. Define sources with env vars:
#
#   MINION_SOURCES=local,zai
#   MINION_SOURCE_LOCAL_BASE_URL=http://localhost:8080/v1
#   MINION_SOURCE_ZAI_BASE_URL=https://api.z.ai/api/paas/v4
#   MINION_SOURCE_ZAI_API_KEY=$zai_test        ← $name = look up env/file key
#   MINION_SOURCE_ZAI_MODEL=glm-x-preview
#
# If no MINION_SOURCE_* vars are present, a single "local" source is built
# from the legacy MINION_BASE_URL / MINION_API_KEY / MINION_MODEL vars
# (same defaults as before, so existing setups keep working).
# Switch at runtime with /source.

class Source:
    def __init__(self, name, base_url, api_key, model=None):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key or "sk-noop"
        self.model = model or None  # None → ask the server at resolve time
        self.client = OpenAI(base_url=base_url, api_key=self.api_key)

    def resolve_model(self):
        if self.model:
            return self.model
        try:
            return self.client.models.list(timeout=10).data[0].id
        except Exception:
            return "local-model"

    def display_model(self):
        return self.model or "auto"


def _resolve_api_key(val):
    """$name → look up env var (populated from ~/.env if present); else literal."""
    if val and val.startswith("$"):
        return os.environ.get(val[1:], "")
    return val


def _discover_sources():
    """Build SOURCES + SOURCE_ORDER from MINION_SOURCE_* env vars, falling
    back to a single 'local' source from the legacy MINION_* vars."""
    names = []
    raw = os.environ.get("MINION_SOURCES", "")
    if raw:
        names = [n.strip() for n in raw.split(",") if n.strip()]
    # auto-discover from MINION_SOURCE_<NAME>_BASE_URL if MINION_SOURCES absent
    if not names:
        prefix = "MINION_SOURCE_"
        found = []
        for k in os.environ:
            if k.startswith(prefix) and k.endswith("_BASE_URL"):
                found.append(k[len(prefix):-len("_BASE_URL")].lower())
        names = sorted(found)
    for name in names:
        p = f"MINION_SOURCE_{name.upper()}_"
        base_url = os.environ.get(p + "BASE_URL")
        if not base_url:
            continue
        api_key = _resolve_api_key(os.environ.get(p + "API_KEY"))
        model = os.environ.get(p + "MODEL")
        src = Source(name, base_url, api_key, model)
        SOURCES[name] = src
        SOURCE_ORDER.append(name)
    if not SOURCES:
        # legacy fallback: one source from MINION_BASE_URL etc.
        src = Source(
            "local",
            os.environ.get("MINION_BASE_URL", "http://localhost:8080/v1"),
            os.environ.get("MINION_API_KEY", "sk-noop"),
            os.environ.get("MINION_MODEL"),
        )
        SOURCES["local"] = src
        SOURCE_ORDER.append("local")


SOURCES = {}        # name → Source
SOURCE_ORDER = []   # preserve definition order for /source listing
ACTIVE = None       # current Source

# `client` and `MODEL` are bare globals read throughout the file. They always
# mirror the active source; switch_source() reassigns both. Every function that
# needs them (open_stream, _assess_risk, compress, …) does a call-time global
# lookup, so a mid-session swap is picked up instantly — same pattern /yolo
# already uses for its own globals.
client = None
MODEL = None


def switch_source(name):
    """Swap the active source. Reassigns client + MODEL globals. Returns True
    on success, False (with a message) if the name is unknown."""
    global ACTIVE, client, MODEL
    src = SOURCES.get(name)
    if not src:
        print(f"{RED}  ✗ unknown source {name!r}{RESET}")
        return False
    ACTIVE = src
    client = src.client
    MODEL = src.resolve_model()
    return True


_discover_sources()

# Pick the starting source: --source flag, then MINION_ACTIVE env, then first.
_start = None
for _i, _arg in enumerate(sys.argv):
    if _arg == "--source" and _i + 1 < len(sys.argv):
        _start = sys.argv[_i + 1]
        break
_start = _start or os.environ.get("MINION_ACTIVE") or (SOURCE_ORDER[0] if SOURCE_ORDER else None)
if not _start or _start not in SOURCES:
    _start = SOURCE_ORDER[0] if SOURCE_ORDER else None
if _start:
    switch_source(_start)


# --- approval gating --------------------------------------------------------
# Three risk levels (low < medium < high) plus an explicit "prompt all" state
# and YOLO mode. APPROVE_LEVEL is the maximum risk level to AUTO-APPROVE:
# actions classified at ≤ APPROVE_LEVEL run without prompting; anything
# strictly above prompts. APPROVE_LEVEL=None means prompt for every classified
# action. YOLO=True short-circuits entirely and skips the classifier call.
#   None     → prompt low + medium + high
#   "low"    → auto-approve low (reads, grep, wc); prompt medium + high
#   "medium" → auto-approve low + medium (edits, cp, mv, tests); prompt high
#   "high"   → auto-approve low + medium + high
LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}

# Accept full words and common abbreviations (med, hi, lo, m, h, l …).
_LEVEL_ALIASES = {
    "l": "low", "lo": "low", "low": "low",
    "m": "medium", "med": "medium", "mid": "medium", "medium": "medium",
    "h": "high", "hi": "high", "high": "high",
}


def _normalize_level(arg):
    """Resolve 'med' → 'medium', 'hi' → 'high', etc. Returns the canonical
    level name or None if the input isn't recognised."""
    return _LEVEL_ALIASES.get(arg.lower().strip())


def _normalize_approval(arg):
    """Resolve approval setting aliases.

    Returns (kind, value):
      ("level", "low"|"medium"|"high") for auto-approval thresholds
      ("prompt_all", None) for strict approval of every classified action
      ("yolo", None) for never-prompt mode
      (None, None) when unrecognized
    """
    raw = arg.lower().strip()
    if raw in ("all", "prompt", "prompt-all", "strict", "none"):
        return "prompt_all", None
    if raw == "yolo":
        return "yolo", None
    lvl = _normalize_level(raw)
    if lvl:
        return "level", lvl
    return None, None


def _approval_display():
    if YOLO:
        return "off (yolo)"
    if APPROVE_LEVEL is None:
        return "all"
    return APPROVE_LEVEL


def _apply_approval(kind, level=None, *, update_default=True):
    global YOLO, APPROVE_LEVEL, DEFAULT_APPROVE_LEVEL
    if kind == "yolo":
        YOLO = True
        APPROVE_LEVEL = None
        if update_default:
            DEFAULT_APPROVE_LEVEL = None
    elif kind == "prompt_all":
        YOLO = False
        APPROVE_LEVEL = None
        if update_default:
            DEFAULT_APPROVE_LEVEL = None
    elif kind == "level":
        YOLO = False
        APPROVE_LEVEL = level
        if update_default:
            DEFAULT_APPROVE_LEVEL = level
    else:
        return False
    return True


# Approval default resolution (highest precedence wins):
#   1. --yolo / --approval CLI flags  (explicit, per-invocation)
#   2. MINION_APPROVAL env var         (persistent, per-user — set in ~/.env)
#   3. prompt-all built-in default     (safest: prompts on every action)
#
# This lets each developer keep their own comfort level in ~/.env (e.g.
# MINION_APPROVAL=medium) without typing --approval every time, while the
# repo ships with the safest default (prompt everything) so anyone
# cloning it starts in cautious mode and can raise it with /approval or the
# env var. Accepts all|low|medium|high|yolo.
YOLO = False
APPROVE_LEVEL = None
DEFAULT_APPROVE_LEVEL = None
_env_approval = os.environ.get("MINION_APPROVAL", "").strip().lower()
if _env_approval:
    _kind, _lvl = _normalize_approval(_env_approval)
    if not _apply_approval(_kind, _lvl):
        print(f"  ✗ unknown MINION_APPROVAL={_env_approval!r} "
              f"(want all|low|medium|high|yolo); prompting for all actions")
for _i, _arg in enumerate(sys.argv):
    if _arg == "--approval" and _i + 1 < len(sys.argv):
        _kind, _lvl = _normalize_approval(sys.argv[_i + 1])
        if not _apply_approval(_kind, _lvl):
            print(f"  ✗ unknown --approval level {sys.argv[_i + 1]!r} "
                  f"(want all|low|medium|high|yolo); keeping {_approval_display()}")
if "--yolo" in sys.argv:
    YOLO = True
    APPROVE_LEVEL = None  # --yolo overrides everything; never prompt

# --- base-level traffic log -------------------------------------------------
# Append-only JSONL record of every byte we ship to / receive from the server.
# Lives next to this script so it's easy to find; rotate by hand if it gets big.
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "llamacpp.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
_llog = open(LOG_PATH, "a", buffering=1)  # line-buffered; flushes per write


def _log_event(direction, payload):
    """direction: 'req' (outgoing) or 'resp' (incoming SSE chunk)."""
    _llog.write(json.dumps({"ts": time.time(), "dir": direction, "data": payload}) + "\n")

# --- ANSI -------------------------------------------------------------------
DIM, CYAN, GREEN, YELLOW, RED, MAGENTA, BOLD, RESET = (
    "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[35m",
    "\033[1m", "\033[0m",
)
CLEAR_LINE = "\033[2K\r"   # erase entire line, return cursor to col 0
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


# --- waiting animation (tiny Conway's Game of Life) -------------------------
# A spinner is boring. A 1-row toroidal Game of Life is the same shape on screen
# (one line of cells) but actually does something — patterns glide, blinkers
# flash, gliders crawl. Runs in a background thread; the main loop kills it
# the instant the first token arrives.
_GOL_W = 24
_GOL_ALIVE = "█"
_GOL_DEAD = "·"
_GOL_GLIDER = {(0, 0), (1, 1), (2, 1), (0, 2), (1, 0)}  # 5-cell, period-4


class LifeSpinner:
    def __init__(self, width=_GOL_W, tick_ms=90, label="thinking"):
        self.w = width
        self.tick = tick_ms / 1000
        self.label = label  # shown before the cells ("thinking" / "running" / ...)
        self._stop = threading.Event()
        self._t = None

    def _seed(self):
        row = [0] * self.w
        x = random.randrange(self.w)
        for dx, _ in _GOL_GLIDER:
            row[(x + dx) % self.w] = 1
        for _ in range(2):
            x = random.randrange(self.w)
            row[x] = row[(x + 1) % self.w] = row[(x + 2) % self.w] = 1
        for _ in range(self.w // 6):
            row[random.randrange(self.w)] = 1
        return row

    def _step(self, row):
        # A 1-row GoL is degenerate (cells have only 2 neighbors). Cheat: treat
        # the row as the middle of a 3-row toroidal world where the rows above
        # and below mirror the current one. Gives every cell the standard 8
        # neighbors, so gliders/blinkers/etc. actually work.
        w, above, below, nxt = self.w, row, row, [0] * self.w
        for x in range(w):
            n = (above[(x - 1) % w] + above[x] + above[(x + 1) % w] +
                 row[(x - 1) % w]                   + row[(x + 1) % w] +
                 below[(x - 1) % w] + below[x] + below[(x + 1) % w])
            cur = row[x]
            nxt[x] = 1 if (cur and n in (2, 3)) or (not cur and n == 3) else 0
        return nxt

    def _run(self):
        sys.stdout.write(HIDE_CURSOR)
        try:
            row = self._seed()
            # initial render — also reserve the line so subsequent prints don't shift things
            sys.stdout.write(CLEAR_LINE + "  " + DIM + f"{self.label} " + RESET +
                             "".join(_GOL_ALIVE if c else _GOL_DEAD for c in row))
            sys.stdout.flush()
            while not self._stop.is_set():
                time.sleep(self.tick)
                if self._stop.is_set():
                    break
                row = self._step(row)
                sys.stdout.write(CLEAR_LINE + "  " + DIM + f"{self.label} " + RESET +
                                 "".join(_GOL_ALIVE if c else _GOL_DEAD for c in row))
                sys.stdout.flush()
        finally:
            # wipe the spinner line and restore cursor
            sys.stdout.write(CLEAR_LINE + SHOW_CURSOR)
            sys.stdout.flush()

    def start(self):
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=0.5)
            self._t = None


# --- interrupt watcher ------------------------------------------------------
# Lets the user press Esc during model generation to stop the stream and drop
# back to the prompt. Runs in a daemon thread for the lifetime of model_turn;
# the main loop checks _INTERRUPT_EVENT between chunks and closes the stream
# on interrupt. Tools are NOT cancelled — run_bash etc. run to completion.
# (Hard-cancelling a tool mid-flight is a separate follow-up.)
#
# Two events, two purposes:
#   _INTERRUPT_EVENT    — "watcher should exit / main loop should check"
#                          set by main on cleanup, set by watcher on user Esc
#   _USER_INTERRUPTED   — "the user actually pressed Esc" (not just cleanup)
#                          only the watcher sets this; main reads it after join
_INTERRUPT_EVENT = threading.Event()
_USER_INTERRUPTED = threading.Event()


def _interrupt_watcher():
    """Daemon: watch stdin for bare Esc during model generation.

    Puts stdin into raw mode (ISIG off so Ctrl+C doesn't kill the process)
    so we can read without echo. A bare Esc (not the start of an arrow-key /
    bracketed-paste / etc. CSI sequence) sets _USER_INTERRUPTED and
    _INTERRUPT_EVENT, then returns. Exits when _INTERRUPT_EVENT is set by
    main's cleanup. Restores termios on exit.
    """
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        return
    try:
        old = termios.tcgetattr(fd)
    except Exception:
        return
    new = old[:]
    new[3] &= ~(termios.ECHO | termios.ICANON | termios.ISIG)
    new[0] &= ~termios.ICRNL
    new[6][termios.VMIN] = 0
    new[6][termios.VTIME] = 0
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
    except Exception:
        return
    try:
        last_fire = 0.0
        while not _INTERRUPT_EVENT.is_set():
            r, _, _ = select.select([fd], [], [], 0.1)
            if not r:
                continue
            try:
                c = os.read(fd, 1)
            except OSError:
                return
            if c != b"\x1b":
                continue  # discard anything that isn't Esc
            # Could be bare Esc OR the lead byte of an escape sequence
            # (arrow keys, Home/End, bracketed paste, etc.). Wait up to 50ms
            # for more bytes; if none arrive, it's a bare Esc.
            r2, _, _ = select.select([fd], [], [], 0.05)
            if r2:
                try:
                    os.read(fd, 1)  # swallow the rest of the sequence
                except OSError:
                    pass
                continue
            now = time.time()
            if now - last_fire < 0.25:  # debounce — don't fire twice in a row
                continue
            last_fire = now
            _USER_INTERRUPTED.set()
            _INTERRUPT_EVENT.set()
            return
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


# --- tools ------------------------------------------------------------------
def read_file(path, **_):
    with open(path) as f:
        return f.read()


def write_file(path, content, **_):
    if not _confirm(f"write {path} ({len(content)} bytes)"):
        return "DENIED by user"
    with open(path, "w") as f:
        f.write(content)
    return f"wrote {len(content)} bytes to {path}"


def edit_file(path, old, new, **_):
    with open(path) as f:
        src = f.read()
    if src.count(old) != 1:
        return f"ERROR: `old` matched {src.count(old)} times (need exactly 1)"
    if not _confirm(f"edit {path}"):
        return "DENIED by user"
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    return f"edited {path}"


def list_dir(path=".", **_):
    return "\n".join(sorted(os.listdir(path)))


def run_bash(command, **_):
    if not _confirm(f"run: {command}"):
        return "DENIED by user"
    r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
    out = (r.stdout or "") + (r.stderr or "")
    # Big safety slice to bound memory; _sanitize_tool_result applies the real
    # per-result context budget (TOOL_RESULT_CHARS) with dedup.
    return f"[exit {r.returncode}]\n{out[:200000]}"


DISPATCH = {
    "read_file": read_file, "write_file": write_file, "edit_file": edit_file,
    "list_dir": list_dir, "run_bash": run_bash,
}

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file's contents",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write (overwrite) a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace one exact occurrence of `old` with `new` in a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List a directory",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "run_bash", "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

FINAL_ANSWER_TOOL = {"type": "function", "function": {
    "name": "final_answer",
    "description": "Return a complete, concise visible answer to the user. Use this when no more tool calls are needed.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The complete concise answer to show to the user. Keep it short and do not trail off.",
            },
            "status": {"type": "string", "enum": ["answered", "blocked"]},
        },
        "required": ["answer"],
    },
}}
FINAL_ANSWER_TOOL_CHOICE = {
    "type": "function",
    "function": {"name": "final_answer"},
}

SYSTEM = """You are a terminal coding agent working in the user's current directory.
Use the provided tools to inspect and modify code. Take one concrete step at a time.

If your runtime does NOT support native tool calls, emit a call as text exactly like:
<tool_call>{"name": "read_file", "arguments": {"path": "foo.py"}}</tool_call>
Emit nothing after a tool call; wait for the Observation. When the task is done, reply in plain prose."""


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _abbr(n):
    """Compact token count for the stats footer: 832 → '832', 1500 → '1.5K',
    78825 → '78K', 1234567 → '1.2M'. Keeps the footer tidy once context grows."""
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}K"
    if n < 1_000_000:
        return f"{n // 1000}K"
    if n < 10_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n // 1_000_000}M"


def _precise_abbr(n):
    """Like _abbr but keeps two decimals in the K/M/B ranges.

    Used for the live tool-call-args counter ("generating tool call write_file
    args 25.15K chars"), where watching the number tick up as args stream in
    is part of the feedback — so we don't want to round 25,152 down to '25K'.
    832 → '832', 25152 → '25.15K', 1234567 → '1.23M'."""
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.2f}K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.2f}M"
    return f"{n // 1_000_000_000}B"

REASONING_LOOP_SIGNALS = (
    "start coding",
    "let me implement",
    "let's implement",
    "now implement",
    "i'll implement",
    "i will implement",
    "write the code",
    "let me write",
    "start with the code",
)
REASONING_LOOP_SIGNAL_LIMIT = _env_int("MINION_REASONING_LOOP_SIGNALS", 10)
REASONING_ONLY_CHAR_LIMIT = _env_int("MINION_REASONING_ONLY_CHARS", 36000)
REASONING_GIBBERISH_CHARS = _env_int("MINION_REASONING_GIBBERISH_CHARS", 600)
MAX_COMPLETION_TOKENS = _env_int("MINION_MAX_TOKENS", 16000)
# Cap + dedup tool results before they enter the message history. Large,
# near-duplicate tool outputs (find/ls/grep path dumps) are the fuel for the
# context-copying repetition collapse some local models fall into — bounding
# them is the cheapest prevention. 0 disables the size cap (dedup still runs).
TOOL_RESULT_CHARS = _env_int("MINION_TOOL_RESULT_CHARS", 20000)
# Recovery sampler. A gibberish/repetition collapse is a LOW-ENTROPY attractor:
# the model is copying long n-grams already in context (paths, flags). Escaping
# it needs MORE entropy (raise temperature, lower min_p) plus anti-repetition
# penalties that lower the looping token's logit — the OPPOSITE of sharpening
# toward greedy. So recovery now RAISES temperature (was 0.2, which made the
# loop more deterministic) and adds DRY + repeat_penalty. The non-OpenAI knobs
# ride in extra_body (llama.cpp reads them; other backends ignore unknown keys).
RECOVERY_TEMPERATURE = _env_float("MINION_RECOVERY_TEMPERATURE", 1.0)
RECOVERY_TOP_P = _env_float("MINION_RECOVERY_TOP_P", 0.95)
RECOVERY_MIN_P = _env_float("MINION_RECOVERY_MIN_P", 0.02)
RECOVERY_REPEAT_PENALTY = _env_float("MINION_RECOVERY_REPEAT_PENALTY", 1.2)
RECOVERY_REPEAT_LAST_N = _env_int("MINION_RECOVERY_REPEAT_LAST_N", 512)
RECOVERY_DRY_MULTIPLIER = _env_float("MINION_RECOVERY_DRY_MULTIPLIER", 0.8)
RECOVERY_DRY_BASE = _env_float("MINION_RECOVERY_DRY_BASE", 1.75)
RECOVERY_DRY_ALLOWED_LENGTH = _env_int("MINION_RECOVERY_DRY_ALLOWED_LENGTH", 2)
# Treat path/code punctuation as DRY sequence breakers so a long file path the
# agent must emit verbatim is never penalized as "repetition".
RECOVERY_DRY_SEQUENCE_BREAKERS = ["\n", ":", "\"", "*", "/", "\\", "`", "'"]
# Cut the stream when the model leaks GLM-native control markers it should never
# emit (the </arg_value> / <arg_key> salad in the gibberish reports). 0 disables.
LEAK_TOKEN_GUARD = _env_int("MINION_LEAK_TOKEN_GUARD", 1)
RISK_CONNECTION_RETRIES = _env_int("MINION_RISK_RETRIES", 3)
RISK_CONNECTION_RETRY_SECONDS = _env_int("MINION_RISK_RETRY_SECONDS", 1)
# Resolved here (not at the sessions-section top) because _env_int() is
# defined just above. See the comment at _DESC_REFRESH_DEFAULT for behavior.
SESSION_DESC_REFRESH = _env_int("MINION_SESSION_DESC_REFRESH", _DESC_REFRESH_DEFAULT)
DESC_SYSTEM = (
    "You summarize coding-agent conversations into one short line. "
    "Given the transcript, reply with ONLY a single line of at most 70 "
    "characters describing what this session is about / working on — the "
    "current task, key files, or goal. No preamble, no quotes, no trailing "
    "punctuation. Write in the same language as the conversation."
)
REASONING_LOOP_NUDGES = (
    "You are looping in reasoning after repeatedly deciding to start implementation. "
    "Stop planning now. Take the next concrete action: either call the appropriate tool "
    "or give the final answer. Do not continue private reasoning.",
    "The previous runtime nudge did not work. Your next assistant turn must contain "
    "exactly one concrete action: either a tool call or the final answer. Do not "
    "explain, plan, or continue reasoning. If you need file context, call read_file "
    "or list_dir now.",
    "Hard stop. Emit only a tool call now. If native tool calls are unavailable, "
    "emit exactly one <tool_call>{...}</tool_call> block and nothing else. For code "
    "edits, read the target file first unless you already know the exact replacement.",
)
REASONING_LOOP_RETRY_LIMIT = _env_int(
    "MINION_REASONING_LOOP_RETRIES", len(REASONING_LOOP_NUDGES))
REASONING_ONLY_RETRY_LIMIT = _env_int("MINION_REASONING_ONLY_RETRIES", 1)
REASONING_GIBBERISH_RETRY_LIMIT = _env_int("MINION_REASONING_GIBBERISH_RETRIES", 1)
MALFORMED_STREAM_RETRY_LIMIT = _env_int("MINION_MALFORMED_STREAM_RETRIES", 2)
FORCED_FINAL_MAX_TOKENS = _env_int("MINION_FORCED_FINAL_MAX_TOKENS", 2048)
RUNTIME_NOTE_RE = re.compile(r"\n\n\[Runtime note: .*?\]\s*$", re.DOTALL)

FORCED_FINAL_NUDGE = (
    "Your previous streamed response produced reasoning only. Do not continue "
    "private reasoning. Use the final_answer tool if available. Return a complete "
    "visible answer now in at most six short bullets or paragraphs. If you are "
    "blocked, say exactly what is blocking you and what input is needed."
)
GIBBERISH_RECOVERY_NUDGE = (
    "Your previous reasoning stream became unreadable token noise. Discard that "
    "broken reasoning completely. Do not rephrase it or continue from any corrupted "
    "tokens. Continue from the last valid tool result or visible user instruction "
    "with exactly one concrete action: a valid tool call or a concise visible answer."
)
GIBBERISH_CHECKPOINT_NUDGE = (
    "Repeated reasoning recovery failed. Do not continue private reasoning and do "
    "not attempt another tool call except final_answer. Return a bounded visible checkpoint for the "
    "user with: (1) the last valid result you can rely on, (2) the next concrete "
    "action you would take, and (3) any blocker or uncertainty. Keep it to at most "
    "six short bullets or paragraphs."
)
MANUAL_RECOVERY_NUDGE = (
    "Manual recovery requested by the user because the previous response appeared "
    "off the rails or corrupted. Discard any corrupted reasoning/output and do not "
    "continue private reasoning. Use the final_answer tool if available. Return a "
    "bounded visible checkpoint with: (1) the last valid result you can rely on, "
    "(2) the next concrete action you would take, and (3) any blocker or "
    "uncertainty. Keep it concise."
)


def _nudge_current_user_turn(messages, nudge):
    note = f"[Runtime note: {nudge}]"
    for msg in reversed(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), str):
            continue
        content = RUNTIME_NOTE_RE.sub("", msg["content"]).rstrip()
        msg["content"] = f"{content}\n\n{note}" if content else note
        return
    messages.append({"role": "user", "content": note})


# --- risk classifier --------------------------------------------------------
# One cheap non-streaming call per write/bash action. Same model, tiny prompt,
# expects {"level": "low|medium|high", "reason": "<short>"}. Defensive parse —
# if the model rambles or returns garbage we fall back to "high" so we err on
# the side of asking. Connection failures get a few short retries first because
# Minion normally talks to reliable or local endpoints. Skipped entirely in YOLO
# mode (no point paying for a call we won't act on) and for read-only tools.

RISK_SYSTEM = (
    "You are a risk classifier for a coding agent's tool calls. "
    "Given one tool action, respond with ONLY a JSON object of the form "
    '{"level": "low"|"medium"|"high", "reason": "<one short sentence>"}.\n'
    "Levels:\n"
    '- low: read-only or trivially reversible (ls, cat, grep, git status, mkdir, touch, file reads).\n'
    '- medium: modifies state but contained/reversible (writing a single file, editing a file, cp, mv, '
    'pip install in a venv, running tests, git commit).\n'
    '- high: destructive, hard to reverse, or broad scope (rm -rf, git push --force, git reset --hard, '
    'dd, chmod -R, writing outside the project, network sends to external hosts, killing processes, '
    'system-level changes, anything touching dotfiles in $HOME).\n'
    "When in doubt, classify higher. Output ONLY the JSON, no preamble."
)


def _assess_risk(action):
    """Return (level, reason). level is one of LEVEL_ORDER; reason is a short
    string. On any failure (server down, bad JSON, unknown level) returns
    ("high", "<error>") so the caller falls through to the prompt path."""
    payload = [
        {"role": "system", "content": RISK_SYSTEM},
        {"role": "user", "content": action},
    ]
    attempts = max(0, RISK_CONNECTION_RETRIES) + 1
    try:
        for attempt in range(attempts):
            try:
                _log_event("req", {"model": MODEL, "messages": payload,
                                   "stream": False, "_purpose": "risk",
                                   "_attempt": attempt + 1})
                resp = client.chat.completions.create(
                    model=MODEL, messages=payload, stream=False, timeout=15)
                break
            except APIConnectionError:
                if attempt >= attempts - 1:
                    return ("high", f"server unreachable after {attempts} attempts; defaulting to high")
                time.sleep(max(0, RISK_CONNECTION_RETRY_SECONDS))
        try:
            _log_event("resp", {"_purpose": "risk", "data": resp.model_dump()})
        except Exception:
            pass
        text = (resp.choices[0].message.content or "").strip()
        # Try JSON first; fall back to scanning for a level word.
        level, reason = None, ""
        try:
            obj = json.loads(text)
            level = (obj.get("level") or "").strip().lower()
            reason = (obj.get("reason") or "").strip()
        except (json.JSONDecodeError, AttributeError, TypeError):
            m = re.search(r'\b(low|medium|high)\b', text, re.IGNORECASE)
            if m:
                level = m.group(1).lower()
            reason = text[:120]
        if level not in LEVEL_ORDER:
            return ("high", f"unparseable risk response: {text[:80]!r}")
        return (level, reason or level)
    except Exception as e:
        return ("high", f"risk call failed: {type(e).__name__}")


# A special exception that propagates up from _confirm (via the tool function
# body → run_tool → model_turn) when the user presses Esc at an approval prompt.
# It's not a real error — it's a control-flow signal meaning "stop this turn and
# drop the user back to the chat input so they can add more guidance."
class _EscToChat(Exception):
    pass


_ACTIVE_SPINNER = None  # set by run_tool() while a tool body is executing


def _ask_approval(prompt):
    """Read a single keypress for a Y/n/esc approval prompt. Returns one of
    'y', 'n', 'esc'. Esc is accepted both as the bare Esc key (read in raw
    mode) and as the letter 'e'/'E' for terminals where raw mode isn't
    usable. Falls back to line-based `input()` when stdin isn't a TTY."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        ans = input(prompt).strip().lower()
        if ans in ("esc", "e", "\x1b"):
            return "esc"
        if ans == "n":
            return "n"
        return "y"
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = old[:]
    new[3] &= ~(termios.ECHO | termios.ICANON)
    new[6][termios.VMIN] = 1
    new[6][termios.VTIME] = 0
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        sys.stdout.write(prompt)
        sys.stdout.flush()
        c = os.read(fd, 1).decode("utf-8", "replace")
        if c == "\x1b":
            # Could be bare Esc OR the lead byte of an escape sequence
            # (arrow keys, Home/End, bracketed paste, …). Wait briefly; if
            # no more bytes arrive, it's a bare Esc.
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                try:
                    os.read(fd, 1)  # swallow the rest of the sequence
                except OSError:
                    pass
                # Any non-bare-Esc sequence (arrow keys etc.) → default proceed.
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "y"
            sys.stdout.write(f"{DIM}esc{RESET}\n")
            sys.stdout.flush()
            return "esc"
        # Echo the choice for feedback
        if c in ("\r", "\n"):
            shown, result = "Y (default)", "y"
        elif c in ("y", "Y"):
            shown, result = "Y", "y"
        elif c in ("n", "N"):
            shown, result = "n", "n"
        elif c in ("e", "E"):
            shown, result = "esc", "esc"
        else:
            shown, result = f"{c} → Y (default)", "y"
        sys.stdout.write(f"{DIM}{shown}{RESET}\n")
        sys.stdout.flush()
        return result
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


def _confirm(action):
    """Decide whether to run `action`. Returns True to proceed, False to deny.
    Raises `_EscToChat` if the user presses Esc — propagating up so the turn
    stops and the REPL drops back to the chat input.

    Flow:
      1. YOLO → True (no call, no prompt).
      2. Ask the model for a risk level (skipped in step 1).
      3. If APPROVE_LEVEL is set and level ≤ APPROVE_LEVEL → auto-allow.
      4. Otherwise prompt (Y/n/esc), showing the level + reason so the user
         has context.

    Reads module globals YOLO and APPROVE_LEVEL at call time — /yolo and
    /approval reassign them mid-session, and we must always see the latest
    value, not a snapshot from when the tool function was defined.
    """
    if YOLO:
        return True
    # If a tool spinner is running, keep it alive during the risk-classifier
    # request and pause it only around our own terminal I/O. Otherwise a slow
    # classifier call looks exactly like a frozen tool.
    sp = _ACTIVE_SPINNER
    paused = False

    def pause_spinner():
        nonlocal paused
        if sp is not None and not paused:
            sp.stop()
            paused = True

    try:
        level, reason = _assess_risk(action)
        # APPROVE_LEVEL is the max level to AUTO-APPROVE. level ≤ threshold → run.
        if APPROVE_LEVEL is not None and LEVEL_ORDER[level] <= LEVEL_ORDER[APPROVE_LEVEL]:
            # Auto-allow. Show the assessment so the user has a paper trail.
            short = reason if len(reason) <= 80 else reason[:77] + "..."
            pause_spinner()
            print(f"{DIM}  ↳ auto-allow [{level}] {action}  ({short}){RESET}")
            return True
        short = reason if len(reason) <= 80 else reason[:77] + "..."
        lvl_color = {"low": DIM, "medium": YELLOW, "high": RED}[level]
        pause_spinner()
        choice = _ask_approval(
            f"{YELLOW}  allow {action}? {lvl_color}[risk: {level.upper()} — {short}]{RESET} "
            f"{YELLOW}[Y/n/esc] {RESET}")
        if choice == "esc":
            raise _EscToChat(action)
        return choice != "n"
    finally:
        if sp is not None and paused:
            sp.start()


# --- text-fallback parsing --------------------------------------------------
TOOL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_text_calls(content):
    """Pull <tool_call>{...}</tool_call> blocks out of model text."""
    calls = []
    for m in TOOL_TAG.finditer(content or ""):
        try:
            obj = json.loads(m.group(1))
            calls.append((obj["name"], obj.get("arguments", {})))
        except (json.JSONDecodeError, KeyError):
            pass
    return calls


def _sanitize_tool_result(text):
    """Collapse repetitive runs and cap size before a tool result enters the
    message history. Large, near-duplicate tool outputs (find/ls/grep path
    dumps) are the fuel for context-copying repetition collapse, so we (1)
    collapse runs of >=3 identical consecutive lines, then (2) head/tail-cap to
    TOOL_RESULT_CHARS with a visible marker. Short, non-repetitive results pass
    through untouched, so normal file reads and command output are unaffected."""
    if not isinstance(text, str) or len(text) <= 1000:
        return text
    lines = text.split("\n")
    out_lines = []
    i, n = 0, len(lines)
    while i < n:
        j = i + 1
        while j < n and lines[j] == lines[i]:
            j += 1
        run = j - i
        if run >= 3:
            out_lines.append(lines[i])
            out_lines.append(f"... [+{run - 1} identical lines elided]")
        else:
            out_lines.extend(lines[i:j])
        i = j
    result = "\n".join(out_lines)
    budget = TOOL_RESULT_CHARS
    if budget > 0 and len(result) > budget:
        head = budget * 2 // 3
        tail = budget - head
        elided = len(result) - head - tail
        result = (result[:head]
                  + f"\n... [{elided} chars elided to bound context; "
                    f"re-run more narrowly if you need the rest]\n"
                  + result[-tail:])
    return result


def run_tool(name, args):
    fn = DISPATCH.get(name)
    if not fn:
        return f"ERROR: unknown tool {name}"
    # newline so the tool arrow gets its own line — streamed text uses end=""
    # and would otherwise run straight into the indicator
    arg_preview = json.dumps(args)
    if len(arg_preview) > 120:
        arg_preview = arg_preview[:117] + "..."
    print(f"\n{CYAN}  ┌─ {name}{RESET}")
    print(f"{CYAN}  │ {RESET}{DIM}{arg_preview}{RESET}")
    # Animate the gap between "cyan args line" and "cyan result line". Tool
    # bodies can take a while — _confirm makes a network round-trip to the
    # risk classifier, run_bash can run for tens of seconds, write_file on a
    # big payload takes a beat — and without this the user just sees a frozen
    # screen after the green model output finishes. _confirm pauses/resumes
    # us around its own I/O so the auto-allow line / Y/n prompt aren't clobbered.
    spinner = LifeSpinner(label=f"running {name}")
    spinner.start()
    global _ACTIVE_SPINNER
    _ACTIVE_SPINNER = spinner
    try:
        result = fn(**args)
    except _EscToChat:
        # Close the tool box before propagating so it isn't left visually open.
        print(f"{CYAN}  └─{RESET} {DIM}(escaped){RESET}")
        raise  # not an error — control-flow signal back to model_turn/REPL
    except Exception as e:  # noqa: BLE001 — surface any tool error back to the model
        result = f"ERROR: {type(e).__name__}: {e}"
    finally:
        _ACTIVE_SPINNER = None
        spinner.stop()
    # Dedup + cap the result BEFORE it enters the message history (the model and
    # the on-screen preview both get this sanitized form). Bounding repetitive
    # tool output is the cheapest defense against context-copying collapse.
    result = _sanitize_tool_result(result)
    # box the result; the preview is a further display-only trim of the (already
    # context-bounded) result
    preview = result if len(result) < 800 else result[:800] + f"\n... [{len(result) - 800} more chars]"
    for line in preview.splitlines():
        print(f"{CYAN}  │ {RESET}{line}")
    print(f"{CYAN}  └─{RESET}")
    return result


def _tool_args_parse_error(c, err, finish_reasons=None):
    name = c.get("name") or "tool"
    args_text = c.get("args") or ""
    detail = f"{name} arguments are not valid JSON"
    if args_text:
        detail += f" ({_abbr(len(args_text))} chars"
        if finish_reasons:
            detail += f"; finish_reason={','.join(finish_reasons)}"
        detail += f"; {err.msg} at char {err.pos})"
    else:
        detail += f" ({err.msg} at char {err.pos})"
    return detail


def _recovery_sampling_opts():
    """Loop-breaking sampler for recovery retries (see RECOVERY_* constants).

    temperature/top_p are standard OpenAI params and go top-level. min_p,
    repeat_penalty, repeat_last_n and the DRY family are llama.cpp extensions the
    OpenAI SDK won't accept as kwargs, so they ride in extra_body — llama.cpp's
    server merges that into the request body; non-llama.cpp endpoints ignore
    unknown keys. Set any RECOVERY_* value negative to omit it."""
    opts = {}
    if RECOVERY_TEMPERATURE >= 0:
        opts["temperature"] = RECOVERY_TEMPERATURE
    if RECOVERY_TOP_P >= 0:
        opts["top_p"] = RECOVERY_TOP_P
    extra = {}
    if RECOVERY_MIN_P >= 0:
        extra["min_p"] = RECOVERY_MIN_P
    if RECOVERY_REPEAT_PENALTY >= 0:
        extra["repeat_penalty"] = RECOVERY_REPEAT_PENALTY
        extra["repeat_last_n"] = RECOVERY_REPEAT_LAST_N
    if RECOVERY_DRY_MULTIPLIER > 0:
        extra["dry_multiplier"] = RECOVERY_DRY_MULTIPLIER
        extra["dry_base"] = RECOVERY_DRY_BASE
        extra["dry_allowed_length"] = RECOVERY_DRY_ALLOWED_LENGTH
        extra["dry_sequence_breakers"] = RECOVERY_DRY_SEQUENCE_BREAKERS
    if extra:
        opts["extra_body"] = extra
    return opts


def open_stream(messages, tools=TOOLS, tool_choice=None, max_tokens=None,
                recovery_sampling=False):
    """Open a streaming completion. Retries without tools= if the server rejects
    that param; returns None (after a friendly message) on connection/API failure."""
    try:
        _prune_empty_assistant_messages(messages)
        token_limit = MAX_COMPLETION_TOKENS if max_tokens is None else max_tokens
        request_opts = {}
        if token_limit > 0:
            request_opts["max_tokens"] = token_limit
        if recovery_sampling:
            request_opts.update(_recovery_sampling_opts())
        if tool_choice is not None:
            request_opts["tool_choice"] = tool_choice
        try:
            _log_event("req", {"model": MODEL, "messages": messages, "tools": tools, "stream": True, **request_opts})
            stream = client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools, stream=True,
                stream_options={"include_usage": True}, **request_opts)
        except APIConnectionError:
            raise  # server unreachable — don't bother retrying without tools
        except httpx.HTTPError:
            raise  # transport failure — retrying without tools won't help
        except Exception:  # reachable but rejected tools= → text-protocol fallback
            fallback_opts = {k: v for k, v in request_opts.items() if k != "tool_choice"}
            _log_event("req", {"model": MODEL, "messages": messages, "stream": True, "_fallback": "no-tools", **fallback_opts})
            stream = client.chat.completions.create(
                model=MODEL, messages=messages, stream=True,
                stream_options={"include_usage": True}, **fallback_opts)
        # Wrap the stream so every chunk is captured to the log on its way out.
        return _LoggingStream(stream, _llog)
    except (APIConnectionError, httpx.HTTPError):
        print(f"{RED}  ✗ can't reach {client.base_url} — is the server up? "
              f"Set MINION_BASE_URL (and MINION_MODEL) to point at it.{RESET}")
    except Exception as e:
        print(f"{RED}  ✗ API error: {type(e).__name__}: {e}{RESET}")
    return None


# --- context compression ----------------------------------------------------
# Summarize the older turns of `messages` into a single user-role turn, keeping
# the system prompt and the last K turns verbatim. Frees context without losing
# the model's grip on what it was just doing.
COMPRESS_KEEP = 2  # how many recent turns to leave untouched


def compress(messages, keep=COMPRESS_KEEP):
    """Ask the model to summarize everything except system + last `keep` turns.

    Mutates `messages` in place on success: replaces the middle slice with a
    single user-role summary turn. Returns (kept_n, summarized_n, summary_chars)
    or None on failure (in which case `messages` is untouched).

    Non-streaming on purpose — we want the whole summary before splicing it in,
    and a spinner for a one-shot summary would be visual noise.
    """
    # Layout: [system?, ..., user, assistant, tool, ..., user, assistant(tool_calls)?, ...]
    # We assume messages[0] is the system prompt (matches how main() builds it).
    # Anything before the "tail" we want to summarize; the tail stays verbatim.
    if len(messages) <= 1 + keep:
        return None  # nothing to compress

    sys_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    body = messages[1:] if sys_msg else messages
    if len(body) <= keep:
        return None

    head, tail = body[:-keep], body[-keep:]
    summarized_n = len(head)

    # The tail must start on a turn the chat template can render. A `tool` turn
    # with no preceding assistant(tool_calls) parent — or an assistant(tool_calls)
    # turn whose result got cut off into `head` — makes llama.cpp's Jinja template
    # raise "Message has tool role, but there was no previous assistant message
    # with a tool call!". Walk from the front of the tail and drop any leading
    # tool/half-tool-call turns until we land on something safe (user, plain
    # assistant, or system). Bump `summarized_n` so the user-visible count stays
    # honest about how many turns actually got folded into the summary.
    while tail and tail[0].get("role") in ("tool", "assistant"):
        first = tail[0]
        if first.get("role") == "tool":
            tail = tail[1:]
            summarized_n += 1
            continue
        # assistant: only safe if it has NO tool_calls, OR every tool_call has
        # its matching tool result later in the tail
        if first.get("tool_calls"):
            ids = {tc["id"] for tc in first["tool_calls"]}
            seen = set()
            for m in tail[1:]:
                tcid = m.get("tool_call_id")
                if m.get("role") == "tool" and tcid:
                    seen.add(tcid)
            if ids - seen:
                tail = tail[1:]
                summarized_n += 1
                continue
        break

    # Render the head as plain text the model can summarize. Tool outputs are
    # the bulkiest part of a real session — include them but truncate each one
    # so a single huge read_file doesn't blow up the summary prompt itself.
    def _render(msgs):
        out = []
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content")
            if content is None and m.get("tool_calls"):
                # assistant tool-call turn — show the calls so the summary knows what ran
                calls = ", ".join(
                    f"{c['function']['name']}({c['function']['arguments']})"
                    for c in m["tool_calls"]
                )
                out.append(f"[{role}] → {calls}")
            elif isinstance(content, list):
                # some servers return content as a list of parts; flatten it
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                out.append(f"[{role}] {content[:2000]}")
            else:
                out.append(f"[{role}] {(content or '')[:2000]}")
        return "\n\n".join(out)

    summary_prompt = (
        "Summarize the following conversation history for context retention. "
        "Preserve: the original user goal/task, key decisions made, file paths "
        "and identifiers touched, current state of any in-progress work, and "
        "any unresolved questions. Drop: raw tool outputs, full file contents, "
        "and verbose back-and-forth — keep it dense and information-rich. "
        "Write in the same language as the conversation. Output ONLY the "
        "summary, no preamble.\n\n"
        f"---\n{_render(head)}\n---"
    )

    payload = [{"role": "user", "content": summary_prompt}]
    try:
        _log_event("req", {"model": MODEL, "messages": payload, "stream": False, "_purpose": "compress"})
        resp = client.chat.completions.create(model=MODEL, messages=payload, stream=False)
        try:
            _log_event("resp", {"_purpose": "compress", "data": resp.model_dump()})
        except Exception:
            pass  # never let logging break the summary call
    except APIConnectionError:
        print(f"{RED}  ✗ can't reach {client.base_url} — context unchanged{RESET}")
        return None
    except Exception as e:
        print(f"{RED}  ✗ compress failed: {type(e).__name__}: {e}{RESET}")
        return None

    summary = (resp.choices[0].message.content or "").strip()
    if not summary:
        print(f"{RED}  ✗ compress returned empty summary — context unchanged{RESET}")
        return None

    header = f"[Compressed context — {summarized_n} earlier turns summarized; last {keep} turns kept verbatim]"
    new_mid = [{"role": "user", "content": f"{header}\n\n{summary}"}]
    messages[:] = ([sys_msg] if sys_msg else []) + new_mid + tail
    return len(tail), summarized_n, len(summary)


class _LoggingStream:
    """Iterator wrapper that tees each SSE chunk to llamacpp.log before yielding.
    Uses model_dump so we capture the chunk's full structure (incl. reasoning_content)."""
    def __init__(self, inner, log_file):
        self._inner = inner
        self._log = log_file

    def __iter__(self):
        for chunk in self._inner:
            try:
                self._log.write(json.dumps({"ts": time.time(), "dir": "resp",
                                             "data": chunk.model_dump()}) + "\n")
            except Exception:
                pass  # never let logging break the stream
            yield chunk

    def close(self):
        close = getattr(self._inner, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass


class _ReasoningLoopSignalCounter:
    """Counts repeated "ready to act" phrases across streamed reasoning chunks."""
    def __init__(self, phrases):
        self.phrases = tuple(p.lower() for p in phrases)
        self.tail = ""
        self.hits = 0
        self.max_phrase_len = max((len(p) for p in self.phrases), default=1)

    def feed(self, chunk):
        if not self.phrases:
            return self.hits
        old_len = len(self.tail)
        text = self.tail + chunk.lower()
        scan_start = max(0, old_len - self.max_phrase_len + 1)
        for phrase in self.phrases:
            start = text.find(phrase, scan_start)
            while start != -1:
                if start + len(phrase) > old_len:
                    self.hits += 1
                start = text.find(phrase, start + 1)
        self.tail = text[-(self.max_phrase_len - 1):]
        return self.hits


# GLM-native chat-template control markers that minion's own protocol never
# emits. When the model leaks these as literal output the decode has almost
# certainly collapsed — they were prominent in the observed gibberish.
# Deliberately excludes <think>/<tool_call> (legit reasoning / text tool-call
# protocol). Two tiers, because file *content* travels through tool args:
#   STRICT = model-internal specials that never appear in real code/text — safe
#            to trip on a single hit, even inside tool-call args.
#   BROAD  = STRICT plus GLM's <arg_key>/<arg_value> XML, which CAN legitimately
#            appear in a chat-template file someone is editing; only used on the
#            reasoning/content channels (which never carry written file bodies),
#            and only with min_hits>=2 to spare incidental mentions in prose.
_LEAK_TOKEN_STRICT_RE = re.compile(r"<\|[A-Za-z0-9_]{1,32}\|>|\[s?g?MASK\]|</?sop>|</?eop>")
_LEAK_TOKEN_BROAD_RE = re.compile(
    r"</?arg_(?:key|value)\s*>|<\|[A-Za-z0-9_]{1,32}\|>|\[s?g?MASK\]|</?sop>|</?eop>")


def _leak_token_hit(text, min_hits=1, strict=False):
    """Return the leaked marker string if `text` contains >= min_hits control
    markers, else None. strict=True restricts to model-internal specials that are
    never valid in file content (use on tool-call args)."""
    if not LEAK_TOKEN_GUARD or not text:
        return None
    rx = _LEAK_TOKEN_STRICT_RE if strict else _LEAK_TOKEN_BROAD_RE
    hits = rx.findall(text)
    if len(hits) >= min_hits:
        return hits[0]
    return None


def _reasoning_gibberish_reason(sample):
    """Return a short reason when reasoning text has collapsed into junk.

    This intentionally targets observed failure modes: dense numeric/markup
    noise, or repetitive low-information scaffolding in reasoning_content
    before the model emits content or a tool call. It is not a general
    "weird text" filter.
    """
    sample = sample.strip()
    n = len(sample)
    if n < 180:
        return None

    whitespace_ratio = sum(c.isspace() for c in sample) / n
    digit_ratio = sum(c.isdigit() for c in sample) / n
    alpha_ratio = sum(c.isalpha() for c in sample) / n
    punct_ratio = sum((not c.isalnum() and not c.isspace()) for c in sample) / n
    markup_hits = len(re.findall(
        r"(?:</|<a\b|div|span|href|class|svg|var)\w*", sample, re.I))

    score = 0
    reasons = []
    if digit_ratio > 0.65 and whitespace_ratio < 0.10:
        score += 3
        reasons.append("mostly digits")
    elif digit_ratio > 0.42 and whitespace_ratio < 0.08:
        score += 2
        reasons.append("dense digits")
    if re.search(r"\d{28,}", sample):
        score += 1
        reasons.append("long numeric run")
    if markup_hits >= 10 and whitespace_ratio < 0.12:
        score += 2
        reasons.append("markup fragments")
    elif markup_hits >= 16:
        score += 1
        reasons.append("many markup fragments")
    if re.search(r"\b([A-Za-z]{2,8})\1{2,}\b", sample):
        score += 1
        reasons.append("repeated fragments")
    if whitespace_ratio < 0.06 and punct_ratio > 0.10 and alpha_ratio < 0.38:
        score += 1
        reasons.append("compact symbol soup")

    tokens = re.findall(r"[A-Za-z']+|\d+|//|[`*_{}\"'-]{1,3}", sample.lower())
    if len(tokens) >= 40:
        counts = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        ranked = sorted(counts.values(), reverse=True)
        unique_ratio = len(counts) / len(tokens)
        top_ratio = max(counts.values()) / len(tokens)
        top5_ratio = sum(ranked[:5]) / len(tokens)
        short_ratio = sum(len(tok) <= 4 for tok in tokens) / len(tokens)
        numeric_token_ratio = sum(tok.isdigit() for tok in tokens) / len(tokens)
        symbol_token_ratio = sum(
            bool(re.fullmatch(r"//|[`*_{}\"'-]{1,3}", tok))
            for tok in tokens
        ) / len(tokens)
        low_info = {
            "`", "``", "```", "*", "**", "***", "_", "__", "___", "//",
            "a", "i", "it", "me", "my", "so", "the", "to", "you",
            "your", "you're", "as", "and", "or", "of", "in", "on",
        }
        layoutish = {
            "arg", "args", "bottom", "command", "cwd", "focus", "home",
            "layout", "left", "name", "pane", "right", "size", "split",
            "start", "tab", "top",
        }
        low_info_ratio = sum(
            tok in low_info or tok.isdigit()
            for tok in tokens
        ) / len(tokens)
        layoutish_ratio = sum(tok in layoutish for tok in tokens) / len(tokens)
        if unique_ratio < 0.24 and short_ratio > 0.72 and top_ratio > 0.10:
            score += 2
            reasons.append("repeated short fragments")
        if unique_ratio < 0.36 and low_info_ratio > 0.62:
            score += 2
            reasons.append("low-information repetition")
        if (unique_ratio < 0.34 and top5_ratio > 0.55
                and numeric_token_ratio + symbol_token_ratio > 0.28):
            score += 2
            reasons.append("dominant token loop")
        if (layoutish_ratio > 0.28 and top5_ratio > 0.48
                and numeric_token_ratio + symbol_token_ratio > 0.22):
            score += 2
            reasons.append("layout-token repetition")

    if score >= 3:
        return ", ".join(reasons[:2]) or "unreadable reasoning"
    return None


class _ReasoningGibberishDetector:
    def __init__(self, window):
        self.window = max(0, int(window))
        self.buf = ""
        self.min_chars = min(240, self.window) if self.window else 0

    def feed(self, chunk):
        if not self.window:
            return None
        self.buf = (self.buf + chunk)[-self.window:]
        if len(self.buf) < self.min_chars:
            return None
        return _reasoning_gibberish_reason(self.buf)


def _tool_call_progress(tcs):
    parts = []
    for i in sorted(tcs):
        c = tcs[i]
        name = c["name"] or "tool"
        args_n = len(c["args"])
        parts.append(f"{name} args {_precise_abbr(args_n)} chars" if args_n else f"{name} ...")
    return ", ".join(parts) if parts else "tool call ..."


TURN_DONE = "done"
TURN_TOOL = "tool"
TURN_LOOP_CUT = "loop_cut"
TURN_STREAM_CUT = "stream_cut"
TURN_GIBBERISH_CUT = "gibberish_cut"
TURN_FORCE_FINAL = "force_final"
TURN_ESC = "esc"  # user pressed Esc at an approval prompt → drop to chat input


# --- one model turn (streamed), returns TURN_* status -----------------------
def model_turn(messages, reasoning_loop_cut_count=0, malformed_stream_cut_count=0,
               gibberish_cut_count=0, forced_final=False, recovery_sampling=False):
    # Start the spinner BEFORE open_stream() so the HTTP-handshake + interrupt-
    # watcher-setup window (which can be tens of ms on a warm local server but
    # seconds on a cold/wake-from-sleep one) isn't a frozen green-text gap.
    # The spinner is still killed by the first SSE chunk in the loop below.
    # t0 must be set BEFORE open_stream() — the HTTP request is sent inside
    # open_stream() (TCP/TLS handshake, request bytes, etc.), and all of that
    # latency is part of TTFT. If we set t0 after, the server has already been
    # processing and the first token may be sitting in the buffer by the time
    # we start the clock, making TTFT look implausibly tiny.
    t0 = time.time()
    spinner_label = (
        "forcing final answer · esc to interrupt"
        if forced_final else
        "recovering · esc to interrupt" if recovery_sampling else
        "thinking · esc to interrupt"
    )
    spinner = LifeSpinner(label=spinner_label)
    spinner.start()
    try:
        if forced_final:
            stream = open_stream(
                messages,
                tools=[FINAL_ANSWER_TOOL],
                tool_choice=FINAL_ANSWER_TOOL_CHOICE,
                max_tokens=FORCED_FINAL_MAX_TOKENS,
                recovery_sampling=recovery_sampling,
            )
        else:
            stream = open_stream(messages, recovery_sampling=recovery_sampling)
    except Exception:
        spinner.stop()
        raise
    if stream is None:
        spinner.stop()
        return TURN_DONE  # error already reported; REPL continues

    # Interrupt watcher: a daemon thread watches stdin for bare Esc. On hit,
    # it sets _USER_INTERRUPTED and closes the stream; the main loop breaks
    # out of the chunk loop on its next iteration. We start it BEFORE the
    # spinner so the user always has a moment to interrupt even if the first
    # token takes a while to arrive.
    _INTERRUPT_EVENT.clear()
    _USER_INTERRUPTED.clear()
    watcher = threading.Thread(target=_interrupt_watcher, daemon=True)
    watcher.start()
    content, tcs, mode = [], {}, None
    timings = None
    usage = None
    t_first = None   # time of first output token (for TTFT)
    loop_signals = _ReasoningLoopSignalCounter(REASONING_LOOP_SIGNALS)
    gibberish = _ReasoningGibberishDetector(REASONING_GIBBERISH_CHARS)
    content_tail = ""  # rolling tail of streamed content, for leak-token detection
    args_tail = ""     # rolling tail of streamed tool-call args, same purpose
    loop_cut = False
    loop_cut_reason = "signals"
    gibberish_reason = None
    reasoning_only_chars = 0
    stream_error = None
    interrupted = False
    finish_reasons = []
    tool_status_active = False
    last_tool_status = 0.0

    def show_tool_status(force=False):
        nonlocal tool_status_active, last_tool_status
        now = time.time()
        if not force and tool_status_active and now - last_tool_status < 0.2:
            return
        sys.stdout.write(CLEAR_LINE + f"{CYAN}  ↳ generating tool call{RESET} "
                         f"{DIM}{_tool_call_progress(tcs)}{RESET}")
        sys.stdout.flush()
        tool_status_active = True
        last_tool_status = now

    def clear_tool_status():
        nonlocal tool_status_active
        if tool_status_active:
            sys.stdout.write(CLEAR_LINE)
            sys.stdout.flush()
            tool_status_active = False

    try:
        for chunk in stream:
            if _USER_INTERRUPTED.is_set():
                interrupted = True
                # close() makes the next iteration raise StopIteration / a
                # connection error; we're breaking anyway, but be tidy
                close = getattr(stream, "close", None)
                if close:
                    try:
                        close()
                    except Exception:
                        pass
                break
            # first byte in: kill the spinner, let the real output take this line
            if spinner._t is not None:
                spinner.stop()
            # Capture streaming usage (OpenAI/Z.ai send a final chunk with
            # usage populated when stream_options.include_usage is True).
            # This chunk has an empty choices array, so check before the
            # choices guard below.
            if chunk.usage:
                usage = chunk.usage
            # The final usage-only chunk has an empty choices array — nothing
            # else to do with it.
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reasons.append(str(choice.finish_reason))
            d = choice.delta
            # Capture TTFT on the first chunk carrying real output (reasoning,
            # content, or tool calls).
            if t_first is None:
                rc_peek = getattr(d, "reasoning_content", None) or \
                          (getattr(d, "model_extra", None) or {}).get("reasoning_content")
                if d.content or d.tool_calls or rc_peek:
                    t_first = time.time() - t0
            # llama.cpp attaches a `timings` object to the final chunk — grab it
            # for the stats footer. It's the only place we get real tok/s numbers
            # (streaming `usage` is always null on llama.cpp).
            extra = getattr(chunk, "model_extra", None) or {}
            if "timings" in extra:
                timings = extra["timings"]
            # reasoning models (e.g. MiniMax-M3) stream a separate reasoning_content
            # field before content/tool_calls. Header + dim text, then a blank line
            # so the green "actual response" always lands on its own row (reasoning
            # from the model often doesn't end in \n — without the gap it would
            # run straight into the answer).
            rc = getattr(d, "reasoning_content", None) or (d.model_extra or {}).get("reasoning_content")
            if rc:
                if not content and not tcs and REASONING_GIBBERISH_CHARS > 0:
                    gibberish_reason = gibberish.feed(rc)
                    # leaked GLM control markers in the reasoning window are a
                    # high-signal collapse tell the ratio heuristic can miss
                    if not gibberish_reason:
                        leak = _leak_token_hit(gibberish.buf, min_hits=2)
                        if leak:
                            gibberish_reason = f"leaked control token {leak!r}"
                    if gibberish_reason:
                        if mode == "think":
                            print()
                        print(f"{RED}  ⚠ REASONING GIBBERISH DETECTED — "
                              f"{gibberish_reason}; cutting stream now{RESET}")
                        loop_cut = True
                        loop_cut_reason = "gibberish"
                        close = getattr(stream, "close", None)
                        if close:
                            close()
                        break
                if mode != "think":
                    print(f"{DIM}  ── reasoning ──{RESET}")
                    mode = "think"
                print(f"{DIM}{rc}{RESET}", end="", flush=True)
                if not content and not tcs:
                    reasoning_only_chars += len(rc)
                if REASONING_LOOP_SIGNAL_LIMIT > 0 and not content and not tcs:
                    prev_hits = loop_signals.hits
                    hits = loop_signals.feed(rc)
                    # Print a loud, obvious counter on threshold crossings so the
                    # user can see the model spiraling before it gets cut. We
                    # fire at the first hit, then at 25/50/75/100% of the limit
                    # (clamped so milestones never exceed the limit). Keeps the
                    # noise down to ≤5 lines per turn while still being impossible
                    # to miss in the dim reasoning stream.
                    _limit = REASONING_LOOP_SIGNAL_LIMIT
                    _milestones = sorted(set(min(_limit, v) for v in (
                        1,                                       # first hit
                        (_limit + 3) // 4,                       # 25%
                        (_limit + 1) // 2,                       # 50%
                        (3 * _limit + 3) // 4,                   # 75%
                        _limit,                                  # 100% — the cut itself
                    ) if 0 < v <= _limit))
                    crossed_milestones = [m for m in _milestones if prev_hits < m <= hits]
                    for milestone in crossed_milestones:
                        # Break out of the dim inline stream so the warning
                        # lands on its own line, then reopen reasoning mode
                        # below so subsequent chunks (if any) keep streaming
                        # cleanly. The end-of-loop divider in the finally-ish
                        # block below will close it out properly when we break.
                        print()
                        if milestone >= _limit:
                            print(f"{RED}  ⚠ REASONING LOOP LIMIT HIT — {hits}/{_limit} ready-to-act signals "
                                  f"(“{loop_signals.phrases[0]}”, etc.) — cutting stream now{RESET}")
                        else:
                            pct = (milestone * 100) // _limit if _limit else 0
                            print(f"{YELLOW}  ⚠ REASONING LOOP WARNING — {milestone}/{_limit} ready-to-act signals "
                                  f"({pct}% of cut threshold); model keeps re-deciding to start coding{RESET}")
                        print(f"{DIM}  ── reasoning ──{RESET}")
                    if hits >= REASONING_LOOP_SIGNAL_LIMIT:
                        loop_cut = True
                        loop_cut_reason = "signals"
                        close = getattr(stream, "close", None)
                        if close:
                            close()
                        break
                if (REASONING_ONLY_CHAR_LIMIT > 0 and not content and not tcs
                        and reasoning_only_chars >= REASONING_ONLY_CHAR_LIMIT):
                    print()
                    print(f"{RED}  ⚠ REASONING-ONLY LIMIT HIT — "
                          f"{_abbr(reasoning_only_chars)} reasoning chars with no "
                          f"answer/tool call — cutting stream now{RESET}")
                    loop_cut = True
                    loop_cut_reason = "reasoning_only"
                    close = getattr(stream, "close", None)
                    if close:
                        close()
                    break
            if d.content:
                clear_tool_status()
                if mode == "think":
                    # close out the reasoning block; newline guarantees the green
                    # answer starts on a fresh line below the dim text
                    print()  # end the current reasoning line
                    print(f"{DIM}  ──────────────{RESET}")
                print(f"{GREEN}", end="")
                mode = "say"
                print(d.content, end="", flush=True)
                content.append(d.content)
                # leaked GLM control markers in visible output → collapse (the
                # ratio heuristic only watches the reasoning channel)
                content_tail = (content_tail + d.content)[-512:]
                leak = _leak_token_hit(content_tail, min_hits=2)
                if leak:
                    print()  # close the green content line
                    print(f"{RED}  ⚠ OUTPUT GIBBERISH DETECTED — leaked control "
                          f"token {leak!r}; cutting stream now{RESET}")
                    loop_cut = True
                    loop_cut_reason = "gibberish"
                    gibberish_reason = f"leaked control token {leak!r}"
                    close = getattr(stream, "close", None)
                    if close:
                        close()
                    break
            for tc in (d.tool_calls or []):
                # if we were mid-reasoning when tools kicked in, close it out so
                # the cyan tool box (which starts with its own \n) gets a clean line
                if mode == "think":
                    print()
                    print(f"{DIM}  ──────────────{RESET}")
                    mode = None
                elif mode == "say":
                    print(RESET)
                    mode = None
                s = tcs.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    s["id"] = tc.id
                if tc.function and tc.function.name:
                    s["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    s["args"] += tc.function.arguments
                    # GLM <arg_key>/<arg_value>/pipe-specials are never valid
                    # JSON tool args — one occurrence is enough to call collapse
                    if not loop_cut:
                        args_tail = (args_tail + tc.function.arguments)[-512:]
                        leak = _leak_token_hit(args_tail, min_hits=1, strict=True)
                        if leak:
                            loop_cut = True
                            loop_cut_reason = "gibberish"
                            gibberish_reason = f"leaked control token {leak!r} in tool args"
                show_tool_status()
            if loop_cut:  # tool-args collapse detected above
                clear_tool_status()
                print(f"\n{RED}  ⚠ TOOL-CALL GIBBERISH DETECTED — "
                      f"{gibberish_reason}; cutting stream now{RESET}")
                close = getattr(stream, "close", None)
                if close:
                    close()
                break
    except (APIError, APIConnectionError, httpx.HTTPError) as e:
        stream_error = e
        close = getattr(stream, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass
    finally:
        spinner.stop()
        clear_tool_status()
        # signal the watcher to exit (it'll restore termios in its own finally)
        _INTERRUPT_EVENT.set()
        watcher.join(timeout=0.5)
        _INTERRUPT_EVENT.clear()
    if stream_error is not None:
        if mode in ("think", "say"):
            print()
        if mode == "think":
            print(f"{DIM}  ──────────────{RESET}")
        print(RESET)
        if isinstance(stream_error, httpx.TimeoutException):
            print(f"{YELLOW}  ✂ STREAM TIMEOUT — server stopped delivering chunks; "
                  f"discarded partial output and returned to chat input{RESET}")
            print(f"{DIM}    {type(stream_error).__name__}: {stream_error}{RESET}")
            return TURN_DONE
        retry_limit = max(0, MALFORMED_STREAM_RETRY_LIMIT)
        if malformed_stream_cut_count >= retry_limit:
            print(f"{RED}  ✗ malformed stream from model/API after "
                  f"{malformed_stream_cut_count} recoveries — waiting for user input{RESET}")
            print(f"{DIM}    {type(stream_error).__name__}: {stream_error}{RESET}")
            return TURN_DONE
        print(f"{YELLOW}  ✂ MALFORMED STREAM — discarded partial output/tool args; "
              f"retrying cleanly ({malformed_stream_cut_count + 1}/{retry_limit}){RESET}")
        _nudge_current_user_turn(
            messages,
            "Your previous streamed response became malformed before it completed. "
            "Discard it. Retry the same task from the current conversation state, "
            "but emit either valid tool calls or a concise final answer only.")
        return TURN_STREAM_CUT
    # reasoning-only turn (no content, no tool_calls) — close out the block so
    # the stats footer doesn't run straight into the dim reasoning text
    if mode == "think":
        print()
        print(f"{DIM}  ──────────────{RESET}")
    print(RESET)
    text = "".join(content)
    elapsed = time.time() - t0

    if interrupted:
        print(f"{YELLOW}  ↳ interrupted by user (Esc) after {elapsed:4.1f}s, "
              f"{len(content)} chars streamed{RESET}")
        # Discard partial content — it's almost certainly a half-formed
        # sentence / tool-call args. Append a synthetic user turn so the
        # model has context for what just happened, then return False so the
        # REPL drops to the prompt instead of looping into another turn.
        messages.append({"role": "user", "content":
            "[User interrupted your previous response with Esc. "
            "Acknowledge briefly and wait for their next message.]"})
        return TURN_DONE

    if not loop_cut and not text.strip() and not tcs and reasoning_only_chars > 0:
        loop_cut = True
        loop_cut_reason = "reasoning_only"

    if loop_cut:
        if loop_cut_reason == "reasoning_only":
            retry_limit = max(0, REASONING_ONLY_RETRY_LIMIT)
            cut_count = reasoning_loop_cut_count
            loop_detail = (
                f"{_abbr(reasoning_only_chars)} reasoning chars without content/tool calls")
        elif loop_cut_reason == "gibberish":
            retry_limit = max(0, REASONING_GIBBERISH_RETRY_LIMIT)
            cut_count = gibberish_cut_count
            loop_detail = f"unreadable reasoning noise ({gibberish_reason or 'detected'})"
        else:
            retry_limit = max(0, REASONING_LOOP_RETRY_LIMIT)
            cut_count = reasoning_loop_cut_count
            loop_detail = f"{loop_signals.hits} ready-to-act signals"
        if forced_final:
            print(f"{RED}  ✂ FORCED FINAL ANSWER FAILED — got {loop_detail}; "
                  f"returning to chat input{RESET}")
            return TURN_DONE
        if cut_count >= retry_limit:
            if loop_cut_reason == "reasoning_only":
                print(f"{RED}  ✂ REASONING-ONLY RESCUE FAILED — gave up after "
                      f"{cut_count} forced final "
                      f"attempt{'s' if cut_count != 1 else ''} "
                      f"× {loop_detail}; waiting for user input{RESET}")
            elif loop_cut_reason == "gibberish":
                print(f"{YELLOW}  ✂ REASONING GIBBERISH RECOVERY FAILED — "
                      f"{cut_count} recover"
                      f"{'ies' if cut_count != 1 else 'y'} × {loop_detail}; "
                      f"forcing visible checkpoint{RESET}")
                _nudge_current_user_turn(messages, GIBBERISH_CHECKPOINT_NUDGE)
                return TURN_FORCE_FINAL
            else:
                print(f"{RED}  ✂ REASONING LOOP MAX RETRIES HIT — gave up after "
                      f"{cut_count} cut{'s' if cut_count != 1 else ''} "
                      f"× {loop_detail}; "
                      f"waiting for user input{RESET}")
            return TURN_DONE
        if loop_cut_reason == "signals":
            nudge = REASONING_LOOP_NUDGES[
                min(cut_count, len(REASONING_LOOP_NUDGES) - 1)]
            cut_msg = (f"{loop_signals.hits} ready-to-act signals "
                       f"(limit {REASONING_LOOP_SIGNAL_LIMIT})")
            print(f"{YELLOW}  ✂ REASONING LOOP CUT — {cut_msg}; nudging implementation "
                  f"(retry {cut_count + 1}/{retry_limit}){RESET}")
            _nudge_current_user_turn(messages, nudge)
            return TURN_LOOP_CUT
        if loop_cut_reason == "reasoning_only":
            cut_msg = (f"{_abbr(reasoning_only_chars)} reasoning chars with no "
                       f"answer/tool call (limit {_abbr(REASONING_ONLY_CHAR_LIMIT)})")
            print(f"{YELLOW}  ✂ REASONING-ONLY STALL — {cut_msg}; forcing final answer "
                  f"(attempt {cut_count + 1}/{retry_limit}){RESET}")
            _nudge_current_user_turn(messages, FORCED_FINAL_NUDGE)
            return TURN_FORCE_FINAL
        print(f"{YELLOW}  ✂ REASONING GIBBERISH CUT — {loop_detail}; "
              f"retrying with recovery sampling "
              f"(attempt {cut_count + 1}/{retry_limit}){RESET}")
        _nudge_current_user_turn(messages, GIBBERISH_RECOVERY_NUDGE)
        return TURN_GIBBERISH_CUT

    # stats footer — prefer llama.cpp timings if present; otherwise fall back
    # to the standard streaming `usage` object (OpenAI, Z.ai, etc.); otherwise
    # fall back to wall-clock only. TTFT (time to first token) is shown when
    # available — for local it comes from llama.cpp timings, for remote we
    # measure it client-side.
    if timings and timings.get("predicted_n"):
        prompt_n = timings.get("prompt_n", 0)
        cache_n = timings.get("cache_n", 0)
        gen_n = timings["predicted_n"]
        tps = timings.get("predicted_per_second", 0)
        ctx = f"ctx {_abbr(prompt_n)}+{_abbr(cache_n)} cached" if cache_n else f"ctx {_abbr(prompt_n)}"
        prompt_ms = timings.get("prompt_ms", 0)
        ttft = (prompt_ms / 1000.0) if prompt_ms else t_first
        parts = [f"{gen_n} tok", f"{tps:5.1f} tok/s", ctx]
        if ttft:
            parts.append(f"{ttft*1000:4.0f}ms ttft")
        parts.append(f"{elapsed:4.1f}s wall")
        print(f"{DIM}  └ {' · '.join(parts)}{RESET}")
    elif usage and (usage.completion_tokens or 0):
        gen_n = usage.completion_tokens or 0
        prompt_n = usage.prompt_tokens or 0
        cache_n = getattr(usage, "prompt_tokens_details", None)
        cached = None
        if cache_n is not None:
            cached = getattr(cache_n, "cached_tokens", None)
        tps = gen_n / elapsed if elapsed > 0 else 0
        if cached:
            ctx = f"ctx {_abbr(prompt_n)}+{_abbr(cached)} cached"
        else:
            ctx = f"ctx {_abbr(prompt_n)}"
        parts = [f"{gen_n} tok", f"{tps:5.1f} tok/s", ctx]
        if t_first:
            parts.append(f"{t_first*1000:4.0f}ms ttft")
        parts.append(f"{elapsed:4.1f}s wall")
        print(f"{DIM}  └ {' · '.join(parts)}{RESET}")
    elif text or tcs:
        print(f"{DIM}  └ {elapsed:4.1f}s wall{RESET}")

    if forced_final and tcs:
        ordered = [tcs[i] for i in sorted(tcs)]
        for c in ordered:
            if c["name"] != "final_answer":
                continue
            try:
                args = json.loads(c["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            answer = str(args.get("answer") or "").strip()
            status = str(args.get("status") or "answered").strip()
            if answer:
                print(f"{GREEN}{answer}{RESET}")
                messages.append({"role": "assistant", "content": answer})
                return TURN_DONE
            print(f"{RED}  ✂ FORCED FINAL ANSWER EMPTY — status={status or 'unknown'}; "
                  f"returning to chat input{RESET}")
            return TURN_DONE
        names = ", ".join(c["name"] or "tool" for c in ordered)
        print(f"{RED}  ✂ FORCED FINAL ANSWER FAILED — model emitted {names}; "
              f"returning to chat input{RESET}")
        return TURN_DONE

    if tcs:  # native tool-calling path
        ordered = [tcs[i] for i in sorted(tcs)]
        parsed_args = []
        parse_error = None
        for c in ordered:
            try:
                parsed_args.append(json.loads(c["args"] or "{}"))
            except json.JSONDecodeError as e:
                parse_error = (c, e)
                break
        if parse_error is not None:
            c, err = parse_error
            detail = _tool_args_parse_error(c, err, finish_reasons)
            retry_limit = max(0, MALFORMED_STREAM_RETRY_LIMIT)
            if malformed_stream_cut_count >= retry_limit:
                print(f"{RED}  ✗ malformed tool call after "
                      f"{malformed_stream_cut_count} recoveries — waiting for user input{RESET}")
                print(f"{DIM}    {detail}{RESET}")
                return TURN_DONE
            print(f"{YELLOW}  ✂ MALFORMED TOOL CALL — {detail}; "
                  f"discarded partial tool args and retrying cleanly "
                  f"({malformed_stream_cut_count + 1}/{retry_limit}){RESET}")
            _nudge_current_user_turn(
                messages,
                "Your previous native tool call had malformed or truncated JSON "
                "arguments, so it was discarded before execution. Retry the same "
                "task from the current conversation state, but emit a complete, "
                "valid tool call with all required arguments. For large file "
                "writes, emit only the tool call and skip any explanatory prose.")
            return TURN_STREAM_CUT
        messages.append({"role": "assistant", "content": text or None, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
            for c in ordered]})
        esc_action = None
        for idx, (c, args) in enumerate(zip(ordered, parsed_args)):
            try:
                result = run_tool(c["name"], args)
            except _EscToChat as e:
                esc_action = e.args[0] if e.args else c["name"]
                # Record this call as cancelled, and fill in results for any
                # remaining tool_calls so the message history stays valid
                # (every assistant tool_call needs a matching tool result,
                # or the chat template rejects the context on the next turn).
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": "CANCELLED by user (Esc) — returned to chat input"})
                for c2 in ordered[idx + 1:]:
                    messages.append({"role": "tool", "tool_call_id": c2["id"],
                                     "content": "SKIPPED (user pressed Esc at an earlier approval)"})
                break
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        if esc_action is not None:
            print(f"{YELLOW}  ↳ escaped approval of {esc_action!r} — back to chat input{RESET}")
            messages.append({"role": "user", "content":
                "[User pressed Esc at a tool approval prompt and returned to chat to "
                "add more input. Acknowledge briefly and wait for their next message.]"})
            return TURN_ESC
        return TURN_TOOL

    calls = parse_text_calls(text)  # text-fallback path
    if calls:
        messages.append({"role": "assistant", "content": text})
        obs = []
        esc_action = None
        for n, a in calls:
            try:
                obs.append(f"Observation ({n}): {run_tool(n, a)}")
            except _EscToChat as e:
                esc_action = e.args[0] if e.args else n
                obs.append(f"Observation ({n}): CANCELLED by user (Esc)")
                break
        if esc_action is not None:
            obs.append("[User pressed Esc at a tool approval prompt and returned to chat "
                       "to add more input. Acknowledge briefly and wait for their next message.]")
            print(f"{YELLOW}  ↳ escaped approval of {esc_action!r} — back to chat input{RESET}")
            messages.append({"role": "user", "content": "\n".join(obs)})
            return TURN_ESC
        messages.append({"role": "user", "content": "\n".join(obs)})
        return TURN_TOOL

    if forced_final and text.strip() and "length" in finish_reasons:
        print(f"{YELLOW}  ✂ FORCED FINAL ANSWER HIT TOKEN LIMIT — saved partial answer "
              f"with truncation marker{RESET}")
        messages.append({
            "role": "assistant",
            "content": text.rstrip() + "\n\n[Truncated by token limit before completion.]",
        })
        return TURN_DONE

    if not text.strip():
        return TURN_DONE
    messages.append({"role": "assistant", "content": text})
    return TURN_DONE


def _run_model_turn_loop(messages, force_final=False, recovery_sampling=False):
    steps = 0
    reasoning_loop_cuts = 0
    malformed_stream_cuts = 0
    gibberish_cuts = 0
    while steps < 25:  # cap runaway tool/retry loops
        status = model_turn(messages, reasoning_loop_cuts, malformed_stream_cuts,
                            gibberish_cuts, forced_final=force_final,
                            recovery_sampling=recovery_sampling)
        force_final = False
        recovery_sampling = False
        if status == TURN_DONE:
            break
        if status == TURN_ESC:
            break  # user pressed Esc at an approval → drop to chat input
        steps += 1
        if status == TURN_LOOP_CUT:
            reasoning_loop_cuts += 1
            continue
        if status == TURN_STREAM_CUT:
            malformed_stream_cuts += 1
            recovery_sampling = True
            continue
        if status == TURN_GIBBERISH_CUT:
            gibberish_cuts += 1
            recovery_sampling = True
            continue
        if status == TURN_FORCE_FINAL:
            reasoning_loop_cuts += 1
            force_final = True
            recovery_sampling = True
            continue
        if status == TURN_TOOL:
            reasoning_loop_cuts = 0
            malformed_stream_cuts = 0
            gibberish_cuts = 0
            continue


# --- multi-line chatbox input ---------------------------------------------
# Replaces the bare `input()` prompt with a framed, multi-line editor:
#   • Enter submits, Alt+Enter (or Ctrl+J) inserts a newline
#   • Paste (bracketed-paste mode) inserts its text verbatim — newlines stay,
#     a trailing newline at the end of paste is stripped so pasting never
#     accidentally submits
#   • Up/Down navigate history, Left/Right move within the current line,
#     Home/End jump to line start/end, Ctrl+U clears the line, Ctrl+C cancels
#   • Long lines word-wrap visually inside the box; the buffer stays one
#     logical string (newlines preserved) so the model sees the real text
# Falls back to plain `input()` when stdin/stdout is not a TTY.

def _chatbox_fallback(prompt):
    """Plain `input()` fallback used when raw terminal mode isn't usable. Single-line only;
    newlines must be typed as the literal `\\n` (rare in practice)."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        raise


def _raw_read_key(fd):
    return os.read(fd, 1).decode("utf-8", "replace")


def _raw_read_available(fd, timeout=0.02):
    parts = []
    while True:
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            break
        parts.append(os.read(fd, 1).decode("utf-8", "replace"))
        timeout = 0
    return "".join(parts)


def _chatbox_raw(initial="", history=None):
    """Normal-scrollback multi-line editor.

    This does not enter the alternate screen. The prompt, streamed model output,
    tool confirmations, and the next prompt all stay in one terminal mode, which
    avoids garbling the REPL after submit.
    """
    history = history or []
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = old[:]
    new[3] &= ~(termios.ECHO | termios.ICANON)
    new[0] &= ~termios.ICRNL
    new[6][termios.VMIN] = 1
    new[6][termios.VTIME] = 0

    buf = initial.split("\n") if initial else [""]
    row, col = len(buf) - 1, len(buf[-1])
    h_idx = len(history)
    rendered_lines = 0
    cursor_line = 0

    def move_up(n):
        return f"\x1b[{n}A" if n > 0 else ""

    def move_down(n):
        return f"\x1b[{n}B" if n > 0 else ""

    def move_right(n):
        return f"\x1b[{n}C" if n > 0 else ""

    def build_visual(inner_w):
        visual = []
        for bi, line in enumerate(buf):
            if not line:
                visual.append((bi, 0, ""))
                continue
            for start in range(0, len(line), inner_w):
                visual.append((bi, start, line[start:start + inner_w]))
            if col == len(line) and row == bi and len(line) % inner_w == 0:
                visual.append((bi, len(line), ""))
        return visual

    def render():
        nonlocal rendered_lines, cursor_line
        width = shutil.get_terminal_size((80, 24)).columns
        box_w = max(20, min(width - 2, 100))
        inner_w = max(1, box_w - 2)
        visual = build_visual(inner_w)

        cur_vrow = 0
        cur_vcol = 0
        for i, (bi, start, seg) in enumerate(visual):
            if bi != row:
                continue
            if start <= col <= start + len(seg):
                cur_vrow = i
                cur_vcol = col - start
                break

        hints = "Enter send · Alt+Enter / ^J newline · ^C cancel"
        max_hints = max(0, box_w - 11)
        if len(hints) > max_hints:
            hints = hints[:max_hints - 1] + "…" if max_hints > 1 else "…"
        top_fill = max(0, box_w - 11 - len(hints))
        stats = f"{len(buf)} line{'s' if len(buf) != 1 else ''} · {sum(len(l) for l in buf)} chars"
        max_stats = max(0, box_w - 6)
        if len(stats) > max_stats:
            stats = stats[:max_stats - 1] + "…" if max_stats > 1 else "…"
        bot_fill = max(0, box_w - 4 - len(stats))

        lines = [
            f"{DIM}╭─ {RESET}{CYAN}you{RESET}{DIM} · {hints} {'─' * top_fill}╮{RESET}",
            *[f"{DIM}│{RESET}{seg:<{inner_w}}{DIM}│{RESET}" for _, _, seg in visual],
            f"{DIM}╰─ {stats}{' ' * bot_fill}╯{RESET}",
        ]

        redraw_rows = max(rendered_lines, len(lines))
        out = "\r" + move_up(cursor_line)
        for i in range(redraw_rows):
            out += "\x1b[2K"
            if i < len(lines):
                out += lines[i]
            if i != redraw_rows - 1:
                out += "\n"
        out += "\r" + move_up((redraw_rows - 1) - (cur_vrow + 1)) + move_right(cur_vcol + 1)
        sys.stdout.write(out)
        sys.stdout.flush()
        rendered_lines = len(lines)
        cursor_line = cur_vrow + 1

    def finish():
        sys.stdout.write("\r" + move_down((rendered_lines - 1) - cursor_line) + "\n")
        sys.stdout.write("\x1b[?25h\x1b[?2004l")
        sys.stdout.flush()

    def insert_text(s):
        nonlocal row, col
        if not s:
            return
        parts = s.split("\n")
        cur = buf[row]
        tail = cur[col:]
        if len(parts) == 1:
            buf[row] = cur[:col] + parts[0] + tail
            col += len(parts[0])
            return
        buf[row] = cur[:col] + parts[0]
        new_lines = list(parts[1:-1]) + [parts[-1] + tail]
        buf[row + 1:row + 1] = new_lines
        row += len(new_lines)
        col = len(parts[-1])

    def backspace():
        nonlocal row, col
        if col > 0:
            buf[row] = buf[row][:col - 1] + buf[row][col:]
            col -= 1
        elif row > 0:
            prev = buf[row - 1]
            col = len(prev)
            buf[row - 1] = prev + buf[row]
            del buf[row]
            row -= 1

    def delete_forward():
        line = buf[row]
        if col < len(line):
            buf[row] = line[:col] + line[col + 1:]
        elif row < len(buf) - 1:
            buf[row] = line + buf[row + 1]
            del buf[row + 1]

    def load_from_history(hist_text):
        nonlocal row, col
        buf[:] = hist_text.split("\n") if hist_text else [""]
        row = len(buf) - 1
        col = len(buf[-1])

    def handle_escape(seq):
        nonlocal row, col, h_idx
        if seq.startswith("[200~"):
            paste = seq[5:]
            while "\x1b[201~" not in paste:
                paste += _raw_read_key(fd)
            paste = paste.split("\x1b[201~", 1)[0]
            if paste.endswith("\n") or paste.endswith("\r"):
                paste = paste[:-1]
            insert_text(paste)
            return
        if seq in ("[A", "OA"):
            if history and h_idx > 0:
                h_idx -= 1
                load_from_history(history[h_idx])
            return
        if seq in ("[B", "OB"):
            if history and h_idx < len(history):
                h_idx += 1
                load_from_history("" if h_idx == len(history) else history[h_idx])
            return
        if seq in ("[C", "OC"):
            if col < len(buf[row]):
                col += 1
            elif row < len(buf) - 1:
                row += 1
                col = 0
            return
        if seq in ("[D", "OD"):
            if col > 0:
                col -= 1
            elif row > 0:
                row -= 1
                col = len(buf[row])
            return
        if seq in ("[H", "OH"):
            col = 0
            return
        if seq in ("[F", "OF"):
            col = len(buf[row])
            return
        if seq == "[3~":
            delete_forward()
            return
        if seq in ("\r", "\n"):
            insert_text("\n")

    termios.tcsetattr(fd, termios.TCSADRAIN, new)
    sys.stdout.write("\x1b[?25h\x1b[?2004h\n")
    sys.stdout.flush()
    try:
        render()
        while True:
            c = _raw_read_key(fd)
            if c == "\x1b":
                handle_escape(_raw_read_available(fd))
            elif c == "\r":
                finish()
                return "\n".join(buf)
            elif c == "\n":
                insert_text("\n")
            elif c == "\x03":
                raise KeyboardInterrupt
            elif c == "\x04":
                if not any(buf):
                    raise EOFError
            elif c in ("\x7f", "\x08"):
                backspace()
            elif c == "\x15":
                buf[row] = ""
                col = 0
            elif c >= " " or c == "\t":
                insert_text(c)
            render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\x1b[?25h\x1b[?2004l")
        sys.stdout.flush()


def read_multiline(initial="", history=None):
    """Public entry point. Returns the entered text, or '' on empty submit.
    Raises EOFError / KeyboardInterrupt to match `input()`'s contract."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        # Fallback path — single-line input. Same UX as before this feature.
        return _chatbox_fallback(f"{CYAN}you ›{RESET} ")
    try:
        return _chatbox_raw(initial, history)
    except (EOFError, KeyboardInterrupt):
        raise
    except (OSError, termios.error):
        return _chatbox_fallback(f"{CYAN}you ›{RESET} ")


# --- repl -------------------------------------------------------------------
def _banner():
    """Banner shown at startup and after source/approval changes."""
    sep = f"{DIM} · {RESET}"
    parts = [f"{BOLD}minion{RESET}", f"{CYAN}{MODEL}{RESET}"]
    if len(SOURCES) > 1:
        parts.append(f"{MAGENTA}{ACTIVE.name}{RESET}")
    if YOLO:
        parts.append(f"{GREEN}yolo{RESET}")
    elif APPROVE_LEVEL is None:
        parts.append(f"{YELLOW}prompt:all{RESET}")
    elif APPROVE_LEVEL == "high":
        parts.append(f"{GREEN}auto:high{RESET}")
    elif APPROVE_LEVEL == "medium":
        parts.append(f"{YELLOW}auto:medium{RESET}")
    else:
        parts.append(f"{DIM}auto:low{RESET}")
    parts.append(f"{DIM}{client.base_url}{RESET}")
    return sep.join(parts)


def main():
    global YOLO, APPROVE_LEVEL, DEFAULT_APPROVE_LEVEL
    # `minion sessions [query]` — discover + exit, no REPL. Checked first so
    # it short-circuits before the banner / network source resolution that the
    # interactive loop depends on.
    if _cli_sessions(sys.argv[1:]):
        return
    # The banner is printed at startup and after source/approval changes.
    print(_banner())
    print()
    messages = [{"role": "system", "content": SYSTEM}]
    history = []  # past user submissions, newest last; Up/Down navigates

    # --- session state ------------------------------------------------------
    # Every minion run has a session id. If the user passed --resume / -r
    # (with an id, prefix, index, or title) or --session <id>, we load that;
    # otherwise a fresh id is minted. The session is auto-saved after each
    # model turn (the messages array is the source of truth — we just write
    # it to ~/.minion/sessions/<id>.json atomically). A session that never
    # got any user input is never written to disk.
    session_id = _session_id_from_args()
    _resume_requested = any(a in ("--resume", "-r") for a in sys.argv[1:])
    if session_id is None:
        if _resume_requested:
            # bare `minion --resume` but no saved sessions exist — start fresh.
            print(f"{DIM}  no saved sessions to resume — starting fresh{RESET}")
        session_id = _new_session_id()
    session_dirty = False  # has the in-memory context diverged from disk?
    if _resume_requested or any(a == "--session" for a in sys.argv[1:]):
        # Only attempt to load when the user explicitly asked to resume a
        # specific (or most-recent) session. A plain `minion` run mints a
        # fresh id above and starts with an empty context.
        data = _load_session(session_id)
        if data and isinstance(data.get("messages"), list):
            messages = data["messages"]
            # Drop the old system prompt and inject the current one so a
            # resumed session picks up any SYSTEM edits / tool changes.
            messages = [m for m in messages if m.get("role") != "system"]
            messages.insert(0, {"role": "system", "content": SYSTEM})
            n_msgs = len([m for m in messages if m.get("role") != "system"])
            title = data.get("title") or "(untitled)"
            print(f"{DIM}  ↻ resumed session {session_id} ({title!r}, "
                  f"{n_msgs} messages){RESET}")
            print()
            # Show the conversation history so the user immediately remembers
            # what the session was about before the first prompt.
            if n_msgs:
                print(f"{DIM}  ── transcript ──{RESET}")
                _print_transcript(messages)
                print(f"{DIM}  ──────────────{RESET}")
                print()
            # Restore the active source if the session recorded one.
            src_name = data.get("source")
            if src_name and src_name in SOURCES and src_name != ACTIVE.name:
                switch_source(src_name)
            session_dirty = False  # just loaded — in sync with disk
        elif _resume_requested:
            print(f"{YELLOW}  ✗ no saved session found for {session_id!r}; "
                  f"starting fresh{RESET}")

    def _save_current(meta=None):
        """Persist the in-memory `messages` to the current session file.

        Title is auto-derived from the first user message (unless the user
        has set one explicitly with /save <title>). Source and cwd are
        recorded so a resume can re-select the right endpoint.
        """
        if session_id is None:
            return
        m = dict(meta or {})
        if "title" not in m:
            first_user = ""
            for msg in messages:
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    first_user = msg["content"]
                    break
            t = _safe_title(first_user)
            if t:
                m["title"] = t
        m.setdefault("source", ACTIVE.name if ACTIVE else None)
        m.setdefault("cwd", os.getcwd())
        m.setdefault("model", MODEL)
        try:
            _write_session(session_id, messages, m)
        except OSError as e:
            print(f"{RED}  ✗ couldn't save session: {type(e).__name__}: {e}{RESET}")

    while True:
        try:
            user = read_multiline(history=history)
        except (EOFError, KeyboardInterrupt):
            print()
            # Flush the current session on Ctrl-D / Ctrl-C exit so the last
            # turn isn't lost (matches Hermes's exit-path flush).
            if session_dirty:
                _save_current()
            # Show a grey resume hint so the user can pick up right where they
            # left off without having to run `minion sessions` first.
            if session_id:
                print(f"{DIM}  resume with: minion --resume {session_id}{RESET}")
            break
        user = user.strip()
        if not user:
            continue
        if user == "/quit":
            if session_dirty:
                _save_current()
            break
        if user == "/source" or user.startswith("/source "):
            sp = user.split()
            if len(sp) == 1:
                print(f"{DIM}  sources:{RESET}")
                for sname in SOURCE_ORDER:
                    src = SOURCES[sname]
                    mark = f"{GREEN}★{RESET}" if sname == ACTIVE.name else f"{DIM} ·{RESET}"
                    m = src.display_model()
                    print(f"  {mark} {CYAN}{sname:<12}{RESET} {DIM}{m} @ {src.base_url}{RESET}")
                print(f"{DIM}  /source <name> to switch · context preserved (use /reset to clear){RESET}")
            else:
                target = sp[1]
                if target not in SOURCES:
                    avail = ", ".join(SOURCE_ORDER)
                    print(f"{RED}  ✗ unknown source {target!r} — available: {avail}{RESET}")
                    continue
                if target == ACTIVE.name:
                    print(f"{DIM}  already on {target}{RESET}")
                    continue
                switch_source(target)
                src = SOURCES[target]
                print(f"{BOLD}{YELLOW}  → switched to {target}{RESET} {DIM}({MODEL} @ {src.base_url}){RESET}")
                print(_banner())
                session_dirty = True  # source change is worth persisting
            continue
        if user == "/yolo":
            YOLO = not YOLO
            if YOLO:
                APPROVE_LEVEL = None  # never prompt
            else:
                APPROVE_LEVEL = DEFAULT_APPROVE_LEVEL
            print(f"{DIM}  yolo={YOLO}  approval={_approval_display()}{RESET}")
            print(_banner())
            continue
        if user.startswith("/approval"):
            parts = user.split()
            if len(parts) == 1:
                # /approval with no arg → show current setting
                print(f"{DIM}  approval={_approval_display()}  (all|low|medium|high|yolo){RESET}")
                continue
            arg = parts[1]
            kind, resolved = _normalize_approval(arg)
            if kind:
                _apply_approval(kind, resolved, update_default=(kind != "yolo"))
                if kind == "prompt_all":
                    print(f"{DIM}  approval=all (prompt every classified action){RESET}")
                elif kind == "level":
                    print(f"{DIM}  approval={resolved} (auto-allow ≤ {resolved}){RESET}")
                else:
                    print(f"{DIM}  approval=off (yolo — never prompt){RESET}")
                print(_banner())
            else:
                print(f"{YELLOW}  unknown level {arg!r} — want all|low|medium|high|yolo{RESET}")
            continue
        if user == "/reset":
            messages = [{"role": "system", "content": SYSTEM}]
            print(f"{DIM}  context cleared{RESET}")
            # A reset forks a new session id so the fresh chat isn't written
            # over the old one (mirrors Hermes's "new session on /new").
            session_id = _new_session_id()
            session_dirty = False
            print(f"{DIM}  new session {session_id}{RESET}")
            continue
        if user == "/recover" or user.startswith("/recover "):
            note = user.split(None, 1)[1].strip() if " " in user else ""
            content = f"[Runtime note: {MANUAL_RECOVERY_NUDGE}"
            if note:
                content += f" User note: {note}"
            content += "]"
            messages.append({"role": "user", "content": content})
            print(f"{YELLOW}  ↳ manual recovery requested — forcing visible checkpoint{RESET}")
            _run_model_turn_loop(messages, force_final=True, recovery_sampling=True)
            session_dirty = True
            continue
        if user in ("/compress", "/compact"):
            # nothing to compress if we're under (system + KEEP) turns
            body_len = len(messages) - (1 if messages and messages[0].get("role") == "system" else 0)
            if body_len <= COMPRESS_KEEP:
                print(f"{DIM}  nothing to compress ({body_len} turn{'s' if body_len != 1 else ''} in context){RESET}")
                continue
            try:
                ok = _confirm(f"compress {body_len - COMPRESS_KEEP} older turns (keep last {COMPRESS_KEEP})")
            except _EscToChat:
                print(f"{YELLOW}  ↳ escaped compress approval — back to chat input{RESET}")
                continue
            if not ok:
                print(f"{DIM}  cancelled{RESET}")
                continue
            print(f"{DIM}  compressing…{RESET}")
            result = compress(messages)
            if result is None:
                continue  # error already printed
            kept_n, summarized_n, summary_chars = result
            print(f"{DIM}  └ compressed {summarized_n} turns → 1 summary "
                  f"({summary_chars} chars), kept last {kept_n} verbatim{RESET}")
            session_dirty = True
            continue
        # --- session commands -------------------------------------------------
        if user == "/sessions" or user.startswith("/sessions "):
            _cmd_sessions(user, session_id)
            continue
        if user == "/resume" or user.startswith("/resume "):
            new_sid = _cmd_resume(user, session_id)
            if new_sid:
                # Persist the outgoing session before switching away.
                if session_dirty:
                    _save_current()
                session_id = new_sid
                messages[:] = _session_messages(session_id)
                session_dirty = False
            continue
        if user == "/save" or user.startswith("/save "):
            title = user.split(None, 1)[1].strip() if " " in user else None
            _save_current({"title": title} if title else {})
            session_dirty = False
            shown = f" as {title!r}" if title else ""
            print(f"{DIM}  ✓ saved session {session_id}{shown}{RESET}")
            continue
        if user == "/delete" or user.startswith("/delete "):
            target = user.split(None, 1)[1].strip() if " " in user else None
            _cmd_delete(target, session_id)
            continue
        # record for history (skip duplicates of the very last entry so
        # Up doesn't immediately re-show what was just submitted)
        if not history or history[-1] != user:
            history.append(user)
        print()  # breathing room before the spinner/text starts
        messages.append({"role": "user", "content": user})
        _run_model_turn_loop(messages)
        # Auto-save after the turn settles (whether it ended cleanly, hit the
        # step cap, or was escaped). This is the Hermes pattern: persist every
        # turn so a crash / accidental close never loses work.
        session_dirty = True
        _save_current()
        # Maybe refresh the model-generated description (a cheap non-streaming
        # call every SESSION_DESC_REFRESH turns). Done after the save so the
        # turn's messages are already on disk; the description lands as a
        # second write. Failures are silent — the title is always present.
        _maybe_refresh_description(session_id, messages)
        session_dirty = False


def _session_id_from_args():
    """Resolve a starting session id from CLI flags: --resume/-r and --session.

    `--resume` with no following target resumes the most recent session
    (the natural "pick up where I left off" intent). With a target it resolves
    by index/id/prefix/title. `--session <id>` forces an exact id (creates if
    absent — useful for scripts that want a stable id)."""
    args = sys.argv[1:]
    out = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--resume", "-r"):
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                # --resume <target>
                target = args[i + 1]
                sessions = _list_sessions(limit=50)
                out = _resolve_session(target, sessions) or target
                i += 2
            else:
                # bare --resume → most recent session, if any
                sessions = _list_sessions(limit=1)
                out = sessions[0]["id"] if sessions else None
                i += 1
            continue
        if a == "--session" and i + 1 < len(args):
            out = args[i + 1]
            i += 2
            continue
        i += 1
    return out


def _session_messages(session_id):
    """Load just the messages list for a session, with the live SYSTEM prepended."""
    data = _load_session(session_id) or {}
    msgs = [m for m in data.get("messages", []) if m.get("role") != "system"]
    msgs.insert(0, {"role": "system", "content": SYSTEM})
    return msgs


def _fmt_when(ts):
    """Compact relative timestamp for the session list."""
    if not ts:
        return "?"
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    if delta < 86400 * 7:
        return f"{int(delta // 86400)}d ago"
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _print_transcript(messages, max_chars=120):
    """Render a session's message history as a one-line-per-message recap.

    Used on resume so the user immediately sees what the conversation was
    about, and by `/sessions <id>` for its detail view. Each message is
    collapsed to a single line (newlines → spaces) and truncated so the whole
    history reads as a scannable timeline rather than a wall of text. Tool
    calls render as `→ name(...)` so you can see what ran. System messages
    are skipped. Returns the number of lines printed."""
    printed = 0
    for m in messages:
        role = m.get("role", "?")
        if role == "system":
            continue
        content = m.get("content")
        if content is None and m.get("tool_calls"):
            calls = ", ".join(
                f"{tc['function']['name']}(...)" for tc in m["tool_calls"])
            content = f"→ {calls}"
        elif isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = (content or "").strip().replace("\n", " ")
        if len(content) > max_chars:
            content = content[:max_chars - 1] + "…"
        color = {"user": CYAN, "assistant": GREEN, "tool": DIM}.get(role, "")
        print(f"  {color}{role:>9}{RESET}  {content}")
        printed += 1
    return printed


def _session_list_page_options(args, default_limit=SESSION_LIST_DEFAULT_LIMIT):
    """Parse paging flags and leave everything else as the search query."""
    page = 1
    limit = default_limit
    query_parts = []
    warnings = []
    i = 0

    def positive_int(value, fallback, name):
        try:
            n = int(value)
        except (TypeError, ValueError):
            warnings.append(f"ignored invalid {name}: {value!r}")
            return fallback
        if n < 1:
            warnings.append(f"ignored invalid {name}: {value!r}")
            return fallback
        return n

    while i < len(args):
        a = args[i]
        if a in ("--page", "-p"):
            if i + 1 >= len(args):
                warnings.append(f"ignored missing value for {a}")
                i += 1
                continue
            page = positive_int(args[i + 1], page, "page")
            i += 2
            continue
        if a.startswith("--page="):
            page = positive_int(a.split("=", 1)[1], page, "page")
            i += 1
            continue
        if a in ("--limit", "-n"):
            if i + 1 >= len(args):
                warnings.append(f"ignored missing value for {a}")
                i += 1
                continue
            limit = positive_int(args[i + 1], limit, "limit")
            i += 2
            continue
        if a.startswith("--limit="):
            limit = positive_int(a.split("=", 1)[1], limit, "limit")
            i += 1
            continue
        query_parts.append(a)
        i += 1

    limit = max(1, min(limit, SESSION_LIST_MAX_LIMIT))
    query = " ".join(query_parts).strip() or None
    return query, page, limit, warnings


def _print_session_list(sessions, start_index=1, current_id=None):
    for offset, s in enumerate(sessions):
        i = start_index + offset
        title = s["title"] or "(empty)"
        if current_id is not None:
            mark = f"{GREEN}●{RESET}" if s["id"] == current_id else f"{DIM}{i:>3}{RESET}"
            prefix = f"  {mark} "
        else:
            prefix = f"  {DIM}{i:>3}{RESET}  "
        sid = s.get("id") or ""
        print(f"{prefix}{MAGENTA}{sid}{RESET}  {title}")

        meta = [f"{s['n']} msg", _fmt_when(s["updated_at"])]
        if s.get("source"):
            meta.append(f"source {s['source']}")
        if s.get("model"):
            meta.append(f"model {s['model']}")
        cwd = _display_path(s.get("cwd"))
        if cwd:
            meta.append(f"cwd {cwd}")
        print(f"       {DIM}{' · '.join(meta)}{RESET}")

        detail = s.get("description")
        label = "desc"
        if not detail and s.get("preview"):
            detail = s["preview"]
            label = "first"
        if detail:
            print(f"       {DIM}{label}: {detail}{RESET}")


def _session_next_hint(command, page, limit, query=None):
    quoted_query = f" {shlex.quote(query)}" if query else ""
    return f"{command}{quoted_query} --page {page + 1} --limit {limit}"


def _display_path(path):
    if not path:
        return None
    path = os.path.expanduser(str(path))
    home = os.path.expanduser("~")
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def _cli_sessions(args):
    """`minion sessions [query]` — list saved sessions and exit (no REPL).

    With no query: prints the first page of recent sessions. With a query:
    substring-filters (case-insensitive) across title, description, preview,
    and id, so once
    you have dozens of sessions you can narrow with e.g.
    `minion sessions refactor`. Add `--page/-p` and `--limit/-n` to page
    through the list. Returns True if it handled a listing request (caller
    should then exit); False if this wasn't a sessions invocation.

    Deliberately a top-level verb rather than `--resume list`: "list" looks
    like it could be a session title, and a "resume" flag doing two different
    things (list vs. switch) depending on whether it has an arg is the kind
    of inconsistency that confuses. `sessions` = discover, `--resume` = enter.
    """
    if not args:
        return False
    if args[0] not in ("sessions", "--sessions", "ls", "list"):
        return False
    query, page, limit, warnings = _session_list_page_options(args[1:])
    offset = (page - 1) * limit
    fetched = _list_sessions(limit=limit + 1, offset=offset, query=query)
    sessions = fetched[:limit]
    has_next = len(fetched) > limit
    for warning in warnings:
        print(f"{YELLOW}  {warning}{RESET}")
    if not sessions:
        if query:
            print(f"  no sessions matching {query!r} on page {page}")
        else:
            if page > 1:
                print(f"  no sessions on page {page}  ({_sessions_dir()})")
            else:
                print(f"  no saved sessions yet  ({_sessions_dir()})")
        return True
    where = f" matching {query!r}" if query else ""
    print(f"{DIM}  sessions{where} · page {page} · {_sessions_dir()}{RESET}")
    _print_session_list(sessions, start_index=offset + 1)
    print(f"{DIM}  resume with: minion --resume <n|short-id|title>{RESET}")
    if has_next:
        print(f"{DIM}  next page: {_session_next_hint('minion sessions', page, limit, query)}{RESET}")
    return True


def _cmd_sessions(user, current_id):
    """Handle /sessions [n] — list recent sessions, or show one in detail."""
    parts = user.split()
    args = parts[1:]
    list_requested = not args or args[0] in ("--page", "-p", "--limit", "-n")
    list_requested = list_requested or (args and (
        args[0].startswith("--page=")
        or args[0].startswith("--limit=")
        or args[0] in ("page", "p")
    ))
    if list_requested:
        page_args = args
        if page_args and page_args[0] in ("page", "p"):
            page_args = ["--page", *page_args[1:]]
        query, page, limit, warnings = _session_list_page_options(page_args)
        offset = (page - 1) * limit
        fetched = _list_sessions(limit=limit + 1, offset=offset, query=query)
        sessions = fetched[:limit]
        has_next = len(fetched) > limit
        for warning in warnings:
            print(f"{YELLOW}  {warning}{RESET}")
        if not sessions:
            if query:
                print(f"{DIM}  no sessions matching {query!r} on page {page}{RESET}")
            elif page > 1:
                print(f"{DIM}  no sessions on page {page} ({_sessions_dir()}){RESET}")
            else:
                print(f"{DIM}  no saved sessions yet (saved to {SESSION_HOME}/sessions){RESET}")
            return
        where = f" matching {query!r}" if query else ""
        print(f"{DIM}  recent sessions{where} · page {page} ({_sessions_dir()}):{RESET}")
        _print_session_list(sessions, start_index=offset + 1, current_id=current_id)
        print(f"{DIM}  /resume <n|short-id|title> to switch · /delete <n|id> to remove{RESET}")
        if has_next:
            query_tail = f" {shlex.quote(query)}" if query else ""
            print(f"{DIM}  next page: /sessions --page {page + 1} --limit {limit}{query_tail}{RESET}")
        return
    # /sessions <target> — show detail for one session
    arg = " ".join(args).strip()
    sessions = _list_sessions(limit=50)
    sid = _resolve_session(arg, sessions)
    if not sid:
        print(f"{YELLOW}  no session matching {arg!r}{RESET}")
        return
    data = _load_session(sid)
    if not data:
        print(f"{YELLOW}  ✗ couldn't read session {sid}{RESET}")
        return
    msgs = data.get("messages", [])
    title = data.get("title") or "(untitled)"
    print(f"{DIM}  session {sid}{RESET}")
    print(f"{DIM}  title: {title}{RESET}")
    print(f"{DIM}  messages: {len(msgs)} · updated {_fmt_when(data.get('updated_at', 0))}{RESET}")
    print(f"{DIM}  ── transcript ──{RESET}")
    _print_transcript(msgs)


def _cmd_resume(user, current_id):
    """Handle /resume [target] — pick a session to resume. Returns the new id
    or None if nothing changed."""
    parts = user.split(None, 1)
    target = parts[1].strip() if len(parts) > 1 else ""
    sessions = _list_sessions(limit=15)
    if not target:
        if not sessions:
            print(f"{DIM}  no saved sessions to resume{RESET}")
            return None
        print(f"{DIM}  recent sessions:{RESET}")
        for i, s in enumerate(sessions, 1):
            mark = f"{GREEN}●{RESET}" if s["id"] == current_id else f"{DIM}{i:>2}{RESET}"
            title = s["title"] or "(empty)"
            print(f"  {mark} {MAGENTA}{s.get('short', _short_id(s['id']))}{RESET}  "
                  f"{title}{DIM} · {s['n']} msg · {_fmt_when(s['updated_at'])}{RESET}")
            desc = s.get("description") or (s.get("preview") if s.get("preview") != title else None)
            if desc:
                print(f"        {DIM}{desc}{RESET}")
        print(f"{DIM}  /resume <n|short-id|prefix|title> to switch{RESET}")
        return None
    target = target.strip()
    # Let the user strip surrounding <> [] "" they may have copied from help.
    if len(target) >= 2 and target[0] in "<[\"'" and target[-1] in ">]\"'":
        target = target[1:-1].strip()
    sessions = _list_sessions(limit=50)
    sid = _resolve_session(target, sessions)
    if not sid:
        print(f"{YELLOW}  ✗ no session matching {target!r}{RESET}")
        return None
    if sid == current_id:
        print(f"{DIM}  already on that session{RESET}")
        return None
    data = _load_session(sid)
    if not data:
        print(f"{YELLOW}  ✗ couldn't read session {sid}{RESET}")
        return None
    # Reselect the recorded source if it's configured, so a resume lands on
    # the same endpoint the session started on.
    src = data.get("source")
    if src and src in SOURCES and src != ACTIVE.name:
        switch_source(src)
        print(f"{BOLD}{YELLOW}  → source {src}{RESET} {DIM}({MODEL}){RESET}")
    n_msgs = len([m for m in data.get("messages", []) if m.get("role") != "system"])
    title = data.get("title") or "(untitled)"
    print(f"{DIM}  ↻ resumed {sid} ({title!r}, {n_msgs} messages){RESET}")
    print()
    return sid


def _cmd_delete(target, current_id):
    """Handle /delete [target] — remove a session file."""
    if not target:
        print(f"{YELLOW}  usage: /delete <n|id|prefix>  (use /sessions to list){RESET}")
        return
    sessions = _list_sessions(limit=50)
    sid = _resolve_session(target, sessions)
    if not sid:
        print(f"{YELLOW}  ✗ no session matching {target!r}{RESET}")
        return
    if sid == current_id:
        print(f"{YELLOW}  can't delete the session you're currently in — /reset first{RESET}")
        return
    if _delete_session(sid):
        print(f"{DIM}  ✓ deleted session {sid}{RESET}")
    else:
        print(f"{YELLOW}  ✗ couldn't delete session {sid}{RESET}")


if __name__ == "__main__":
    main()
