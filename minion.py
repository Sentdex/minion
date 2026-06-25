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

Toggles in-session: /source [name] [model]  /yolo  /approval [level]  /compress  /compact  /autocompress [pct|off|on]  /reset  /clear  /new  /recover  /sessions  /resume  /save  /delete  /quit
Flags: --yolo  --approval <all|low|medium|high|yolo>  --source <name>  --resume <target>  --session <id>
Env:   MINION_APPROVAL=<all|low|medium|high|yolo>  (persistent default approval mode; ~/.env or shell)
       MINION_AUTOCOMPRESS_PERCENT=<0-100>  (auto-compress threshold; 0=off; default 85)
       TOGETHER_API_KEY  (auto-registers a built-in `together` source; default model zai-org/GLM-5.2)
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
import urllib.error
import urllib.request

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
    # Start from the existing metadata so partial writes (e.g. the session
    # description refresh, which only passes {"description", "desc_turns"})
    # don't clobber fields a prior _save_current wrote — token totals,
    # started_at, etc.  Then let `meta` override on top.
    data = dict(existing)
    data["id"] = session_id
    data["messages"] = messages
    data["created_at"] = existing.get("created_at", now)
    data["updated_at"] = now
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
        # Lazily-resolved max context window (in tokens) for the active model,
        # or None when no probe could determine it. Cached after first success
        # so the footer + /source list don't re-hit the network each turn.
        # Call resolve_context_window(force=True) to re-probe (e.g. after a
        # /source <name> <model> override pins a different model id).
        self._context_window = None

    # Server-advertised ids that carry no real identity — when /v1/models
    # returns one of these (bare llama.cpp with no --alias, or a generic
    # OpenAI-compat stub), fall through to a /props probe before giving up so
    # saved sessions don't all collapse into a meaningless "local-model".
    _GENERIC_MODEL_IDS = {"local-model", "model", "auto", "unknown", "gpt-3.5-turbo"}

    def resolve_model(self):
        if self.model:
            return self.model
        advertised = None
        try:
            advertised = self.client.models.list(timeout=10).data[0].id
        except Exception:
            advertised = None
        if advertised and advertised.strip().lower() not in self._GENERIC_MODEL_IDS:
            return advertised
        # The endpoint gave us nothing useful. Many local servers (llama.cpp's
        # llama-server) expose the actually-loaded model file at /props even
        # when /v1/models only advertises a generic id — recover the real name
        # (incl. quant) from there. Best-effort; any failure keeps the old value.
        probed = self._probe_loaded_model()
        return probed or advertised or "local-model"

    def _probe_loaded_model(self):
        """Best-effort GET <server>/props for the loaded model path/alias.
        llama.cpp serves /props at the root (not under /v1). Returns None on
        any failure — never raises, never required, makes no assumption about
        what (if anything) consumes minion's session metrics."""
        try:
            root = re.sub(r"/v\d+/?$", "", self.base_url.rstrip("/"))
            with urllib.request.urlopen(root + "/props", timeout=5) as r:
                props = json.loads(r.read().decode("utf-8"))
        except Exception:
            return None
        for key in ("model_alias", "model_path", "model"):
            val = props.get(key)
            if isinstance(val, str) and val.strip() and \
                    val.strip().lower() not in self._GENERIC_MODEL_IDS:
                return val.strip()
        return None

    def display_model(self):
        return self.model or "auto"

    # Regex for the provider-agnostic context-limit probe: Together (and any
    # OpenAI-compat backend that mirrors OpenAI's error wording) rejects an
    # over-large request with "This model's maximum context length is N tokens,
    # but the request …". We send a deliberately over-sized max_tokens with a
    # one-token prompt so the server tells us N without us ever generating.
    _CTX_LIMIT_RE = re.compile(r"maximum context length is (\d+) tokens", re.IGNORECASE)

    def _is_local(self):
        """True if this source points at a host on the local machine or LAN
        (llama.cpp, Ollama, etc.) rather than a remote API (Together, Z.ai,
        OpenAI, …). Local servers expose /v1/models meta and /props cheaply
        (sub-millisecond LAN round-trips); remote hosts return empty lists or
        404 there and waste a multi-second TLS round-trip, so the probe order
        is flipped for them (over-max_tokens first)."""
        host = (self.base_url or "").lower()
        # strip the scheme
        if "://" in host:
            host = host.split("://", 1)[1]
        host = host.split("/", 1)[0].split(":", 1)[0]
        if not host:
            return True  # blank → assume local
        if host in ("localhost", "0.0.0.0", "::1"):
            return True
        # 127.x.x.x / 10.x / 192.168.x / 169.254.x / 172.16-31.x are LAN
        if re.match(r"^(127\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2[0-9]|3[01])\.)", host):
            return True
        return False

    def resolve_context_window(self, model=None, force=False):
        """Best-effort max context window (tokens) for `model` on this source.

        Returns the cached value if present (unless force=True), else probes
        and caches. Never raises — any failure returns None and the footer
        just omits the "/<max>" suffix. Probe order depends on host locality:

          • Local (llama.cpp etc.) — /v1/models meta, then /props, then the
            over-max_tokens chat probe. Local metadata endpoints are cheap
            (sub-ms LAN) and llama.cpp stashes n_ctx in data[0].meta, so they
            win on the first try.

          • Remote (Together, Z.ai, OpenAI, …) — over-max_tokens chat probe
            FIRST (one cheap request, ~0.1s, works for any OpenAI-compat
            host that mirrors the "maximum context length is N tokens" 400),
            then /v1/models as a fallback. Remote /v1/models often returns an
            empty list or lacks context_length, and the TLS round-trip alone
            costs 2-3s — leading with it added multi-second latency to the
            background probe, so the first turn's footer rendered before the
            max resolved. The chat probe resolves in well under a second.

        `model` defaults to the source's configured model, falling back to a
        live resolve_model() so an auto source is probed against the model it
        will actually use. The result is keyed to that model id, so a
        /source <name> <other-model> override should call with force=True.

        A single probe attempt can hit a transient error — most notably
        llama.cpp's 503 "Loading model" while the weights are still being
        mmapped, which blocks /v1/models AND /props even though a chat slot
        may already be serving. So a failed attempt does NOT cache None;
        the caller (footer/listing) re-arms the probe on a later turn until
        it resolves. Use force=True to retry immediately (e.g. an explicit
        /source switch the user is waiting on)."""
        if not force and self._context_window is not None:
            return self._context_window
        mid = model or self.model or self.resolve_model()
        if self._is_local():
            n = self._ctx_from_models(mid) or self._ctx_from_props()
            if n is None:
                n = self._ctx_from_overrun_probe(mid)
        else:
            # Remote: the over-max_tokens chat probe is cheap and universal;
            # lead with it so the max resolves on the first turn instead of
            # after a multi-second /v1/models round-trip that returns nothing.
            n = self._ctx_from_overrun_probe(mid)
            if n is None:
                n = self._ctx_from_models(mid) or self._ctx_from_props()
        if isinstance(n, int) and n > 0:
            self._context_window = n
        # NOTE: a miss does NOT set _context_window — stays None so the
        # footer re-probes next turn (the server may have finished loading).
        return self._context_window

    def _ctx_from_models(self, mid):
        """Pull n_ctx / context_length from GET /v1/models. llama.cpp stashes
        it under data[0].meta.n_ctx; some generic hosts expose context_length
        as a top-level model field. Returns int or None."""
        try:
            data = self.client.models.list(timeout=10).data
        except Exception:
            return None
        # Prefer the entry whose id matches the active model (multi-model
        # hosts), else the first. Match loosely on a trailing token so a full
        # path id ("…/GLM-5.2-…gguf") still hits a short alias ("GLM-5.2").
        pick = None
        if mid:
            mtail = mid.rsplit("/", 1)[-1].lower()
            for m in data:
                if (m.id or "").lower() == mid.lower() or \
                   (m.id or "").rsplit("/", 1)[-1].lower() == mtail:
                    pick = m
                    break
        if pick is None and data:
            pick = data[0]
        if pick is None:
            return None
        # OpenAI SDK preserves unknown fields in __pydantic_extra__.
        extra = getattr(pick, "__pydantic_extra__", None) or {}
        meta = extra.get("meta") if isinstance(extra, dict) else None
        if isinstance(meta, dict):
            for k in ("n_ctx", "context_length", "context_window",
                      "max_context_length", "max_input_tokens"):
                v = meta.get(k)
                if isinstance(v, int) and v > 0:
                    return v
        # Some hosts put it directly on the model object (OpenAI-compat extra).
        for k in ("context_length", "context_window", "max_context_length",
                  "max_input_tokens", "max_tokens"):
            v = extra.get(k) if isinstance(extra, dict) else None
            if v is None:
                v = getattr(pick, k, None)
            if isinstance(v, int) and v > 0:
                return v
        return None

    def _ctx_from_props(self):
        """llama.cpp /props fallback: default_generation_settings.n_ctx."""
        try:
            root = re.sub(r"/v\d+/?$", "", self.base_url.rstrip("/"))
            with urllib.request.urlopen(root + "/props", timeout=5) as r:
                props = json.loads(r.read().decode("utf-8"))
        except Exception:
            return None
        dgs = props.get("default_generation_settings")
        if isinstance(dgs, dict):
            for k in ("n_ctx", "context_length", "context_window"):
                v = dgs.get(k)
                if isinstance(v, int) and v > 0:
                    return v
        # Some builds put n_ctx at the top level of /props.
        for k in ("n_ctx", "context_length", "context_window"):
            v = props.get(k)
            if isinstance(v, int) and v > 0:
                return v
        return None

    def _ctx_from_overrun_probe(self, mid):
        """Send a chat completion with max_tokens far beyond any real context
        and a 1-token prompt; parse the 400's "maximum context length is N
        tokens" message. Works for Together and any host mirroring OpenAI's
        over-limit wording. No tokens are generated (the request is rejected
        before inference). Returns int or None."""
        if not mid:
            return None
        try:
            self.client.chat.completions.create(
                model=mid,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10_000_000,
                stream=False,
                timeout=30,
            )
        except Exception as e:
            msg = str(e)
            body = getattr(e, "body", None)
            if isinstance(body, dict):
                # Together nests it under body["message"]["message"]; OpenAI
                # under body["error"]["message"]. Flatten to a string search.
                msg = json.dumps(body) + " " + msg
            m = self._CTX_LIMIT_RE.search(msg)
            if m:
                return int(m.group(1))
        return None


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

    # Built-in convenience source: Together AI. It's a multi-model host (you
    # point it at any model by id), so unlike the local fallback it only
    # appears when a key is actually available — and only when the user
    # hasn't already defined a "together" source themselves (so an explicit
    # MINION_SOURCE_TOGETHER_* config always wins). Registered last, so the
    # user's first/source-listed source stays the default at startup and
    # `/source together` (or `--source together`) is purely opt-in.
    if "together" not in SOURCES:
        together_key = _resolve_api_key(os.environ.get("TOGETHER_API_KEY", ""))
        if together_key:
            SOURCES["together"] = Source(
                "together",
                "https://api.together.xyz/v1",
                together_key,
                "zai-org/GLM-5.2",  # default model; override per-switch via /source together <model>
            )
            SOURCE_ORDER.append("together")


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

# Updated by model_turn() on every turn that reports a prompt-token count
# (llama.cpp timings.prompt_n or OpenAI/Z.ai usage.prompt_tokens). Read by
# _maybe_autocompress() after a turn settles to decide whether the context
# window is full enough to warrant a silent auto-compress.
_LAST_PROMPT_TOKENS = 0


def switch_source(name, model_override=None):
    """Swap the active source. Reassigns client + MODEL globals. Returns True
    on success, False (with a message) if the name is unknown.

    `model_override` (optional) pins MODEL to a specific model id for this
    switch instead of resolving the source's default — used by
    `/source <name> <model>` so a multi-model host (e.g. Together) can be
    pointed at any of its models without a config edit. A bare switch (no
    override) always falls back to the source's configured/default model,
    so `/source together` returns to GLM-5.2 even after an override."""
    global ACTIVE, client, MODEL
    src = SOURCES.get(name)
    if not src:
        print(f"{RED}  ✗ unknown source {name!r}{RESET}")
        return False
    ACTIVE = src
    client = src.client
    MODEL = model_override if model_override else src.resolve_model()
    # A model change (different source, or a per-switch model override on a
    # multi-model host like Together) means the cached max-context may no longer
    # apply — drop it so the footer re-probes against the new model next turn.
    src._context_window = None
    return True


def restore_source(source_name, model=None):
    """Best-effort restore of the source (and optional model) a session was
    started on, used when resuming. Switches source if it's configured and
    differs from the active one; if `model` is set and not equal to the
    source's configured default, pins MODEL to it so a `/source <name>
    <model>` override survives a resume. Returns True if the source is now
    the requested one."""
    if not source_name or source_name not in SOURCES:
        return False
    src = SOURCES[source_name]
    pin = model if model and model != src.model else None
    if source_name == ACTIVE.name and MODEL == (pin or src.model):
        return True
    switch_source(source_name, model_override=pin)
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
    try:
        fd = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
        return  # stdin has no fileno (pytest capture, redirect, etc.)
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
def read_file(path, offset=1, limit=None, **_):
    """Read a file as numbered lines (1-based, `cat -n` style: right-aligned
    line number, a tab, then the line). Large files return only a window — pass
    offset (start line) and limit (max lines) to page through the rest. A header
    announces the visible range + total line count so the model can ask for more
    instead of being head/tail-capped by _sanitize_tool_result."""
    with open(path) as f:
        lines = f.readlines()
    total = len(lines)
    try:
        start = max(1, int(offset)) - 1
    except (TypeError, ValueError):
        start = 0
    if total == 0:
        return f"[{path}: empty file]"
    if start >= total:
        return f"[{path}: {total} lines; offset {start + 1} is past end of file]"
    if limit is None:
        limit = _env_int("MINION_READ_FILE_LINES", 400)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 400
    end = total if limit <= 0 else min(total, start + limit)
    body = "".join(f"{i + 1:6d}\t{lines[i]}" for i in range(start, end))
    if body and not body.endswith("\n"):
        body += "\n"
    if start > 0 or end < total:
        return (f"[{path}: lines {start + 1}-{end} of {total}; "
                f"call read_file with offset/limit to page]\n" + body)
    return body


def write_file(path, content, **_):
    if not _confirm(f"write {path} ({len(content)} bytes)"):
        return "DENIED by user"
    with open(path, "w") as f:
        f.write(content)
    return f"wrote {len(content)} bytes to {path}"


_LINE_NUM_PREFIX = re.compile(r"^ *\d+\t")


def _strip_line_numbers(text):
    """Remove read_file's `<n>\\t` line-number prefixes from a pasted block.
    Only strips when EVERY non-empty line carries one (i.e. the whole block came
    from numbered read_file output) — so ordinary edits, and code that merely
    starts a line with digits, are never altered."""
    lines = text.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if nonempty and all(_LINE_NUM_PREFIX.match(ln) for ln in nonempty):
        return "\n".join(_LINE_NUM_PREFIX.sub("", ln) for ln in lines)
    return text


def edit_file(path, old, new, **_):
    with open(path) as f:
        src = f.read()
    if src.count(old) != 1:
        # The model may have pasted read_file's numbered output into `old`.
        # Only as a fallback (exact match already failed), retry with the
        # numbering stripped — and strip `new` the same way so we never write
        # line numbers into the file. Normal edits hit the exact match above
        # and never reach here, so genuine content is never touched.
        stripped_old = _strip_line_numbers(old)
        if stripped_old != old and src.count(stripped_old) == 1:
            old, new = stripped_old, _strip_line_numbers(new)
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
    {"type": "function", "function": {"name": "read_file", "description": "Read a file's contents. Returns lines numbered (1-based, like `cat -n`: a right-aligned number, a tab, then the line). Large files return only a window — pass `offset` (1-based start line) and `limit` (max lines, default 400) to page through the rest; a header shows the visible range and total line count.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer", "description": "1-based line to start from (default 1)"}, "limit": {"type": "integer", "description": "max lines to return (default 400; <=0 reads to end)"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write (overwrite) a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace one exact occurrence of `old` with `new` in a file. Prefer raw file text; if you paste read_file's numbered lines, the `<n>\\t` line-number prefixes are stripped automatically.",
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

If your runtime does NOT support native tool calls, emit a standalone text-protocol call exactly like:
[minion_tool_call]{"name": "read_file", "arguments": {"path": "foo.py"}}[/minion_tool_call]
Emit nothing before or after a tool call; wait for the Observation. When showing a tool-call example, use prose or a code block so it is not a standalone tool call. When the task is done, reply in plain prose."""


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


# --- usage metrics (default off) --------------------------------------------
# minion already captures every model call's token usage (the streaming
# `usage` object from OpenAI/Z.ai, plus llama.cpp's `timings`) to print the
# stats footer — and then throws it away. This module retains those numbers
# so they can be written into the session JSON (always — it's free, the data
# is already in hand) and optionally pushed to an external metrics endpoint
# (only when MINION_METRICS_URL is set). The endpoint is generic: minion
# posts a small JSON blob of cumulative session totals after each turn. One
# such consumer is a personal dashboard; the format is dashboard-agnostic.
#
# Accounting matches the OpenAI usage convention (and Hermes's canonical
# usage): `input_tokens` EXCLUDES cached tokens (cache is broken out into
# `cache_read_tokens`), and `output_tokens` EXCLUDES reasoning tokens
# (reasoning is broken out into `reasoning_tokens`). Dashboards that want a
# "total in" sum input + cache_read; a "total out" sum output + reasoning.

def _normalize_usage(usage, timings):
    """Pull a {input, output, cache_read, reasoning} token dict out of a
    finished model call, from whichever the server gave us: the standard
    streaming `usage` object (OpenAI/Z.ai) or the llama.cpp `timings` extra
    (where `usage` is null). Returns None if neither has anything to say.

    Numbers are the per-call totals reported by the server, not deltas —
    callers accumulate them. The split is:

      input_tokens        prompt tokens, MINUS cached (matches OpenAI's
                          prompt_tokens minus prompt_tokens_details.cached)
      cache_read_tokens   cached prompt tokens (prompt_tokens_details.cached
                          on OpenAI; cache_n on llama.cpp)
      output_tokens       completion tokens, MINUS reasoning
      reasoning_tokens    completion_tokens_details.reasoning_tokens, if any
                          (reasoning models: MiniMax-M3, DeepSeek-R1, …)
    """
    # llama.cpp: no `usage`, but a `timings` object on the final chunk with
    # prompt_n / predicted_n / cache_n. There's no reasoning split here —
    # predicted_n is all generated tokens.
    if (not usage) and timings and timings.get("predicted_n"):
        prompt_n = int(timings.get("prompt_n") or 0)
        cache_n = int(timings.get("cache_n") or 0)
        gen_n = int(timings.get("predicted_n") or 0)
        return {
            "input_tokens": max(0, prompt_n - cache_n),
            "output_tokens": gen_n,
            "cache_read_tokens": cache_n,
            "reasoning_tokens": 0,
        }
    if not usage:
        return None
    prompt_total = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_total = int(getattr(usage, "completion_tokens", 0) or 0)
    # OpenAI/Z.ai: prompt_tokens_details.cached_tokens is the cache portion.
    ptd = getattr(usage, "prompt_tokens_details", None)
    cache_n = int(getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0
    # Reasoning models break completion into content + reasoning.
    otd = getattr(usage, "output_tokens_details", None)
    reasoning_n = int(getattr(otd, "reasoning_tokens", 0) or 0) if otd else 0
    return {
        "input_tokens": max(0, prompt_total - cache_n),
        "output_tokens": max(0, completion_total - reasoning_n),
        "cache_read_tokens": cache_n,
        "reasoning_tokens": reasoning_n,
    }


class _Metrics:
    """Cumulative token usage for the current session, plus an optional push.

    A single module-global instance, `_METRICS`, is created in main() when a
    session starts (and recreated on /reset, reloaded on /resume). When
    MINION_METRICS_URL is unset it's just an in-memory accumulator that feeds
    the session-JSON totals — no network, no threads, no dependency. When the
    URL is set, each flush() also fires off a fire-and-forget POST so a live
    dashboard can plot tok/min. The push is best-effort: it disables itself
    after the first failure so a missing dashboard never slows the REPL.
    """

    def __init__(self, session_id, model=None, source=None, started_at=None,
                 push_url=None):
        self.session_id = session_id
        self.model = model
        self.source = source
        self.started_at = started_at or time.time()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.reasoning_tokens = 0
        self.api_calls = 0
        self.push_url = push_url
        self._push_disabled = False   # latched off after the first failure

    def add(self, delta):
        """Accumulate a per-call usage dict from _normalize_usage()."""
        if not delta:
            return
        self.input_tokens += int(delta.get("input_tokens") or 0)
        self.output_tokens += int(delta.get("output_tokens") or 0)
        self.cache_read_tokens += int(delta.get("cache_read_tokens") or 0)
        self.reasoning_tokens += int(delta.get("reasoning_tokens") or 0)
        self.api_calls += 1

    def as_meta(self):
        """The fields merged into the session JSON on save."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "api_calls": self.api_calls,
            "started_at": self.started_at,
        }

    def flush(self):
        """Fire the optional HTTP push. Cumulative totals, so the endpoint
        can compute its own deltas (agentdash's /api/tokens/push does)."""
        if not self.push_url or self._push_disabled:
            return
        payload = {
            "session_id": self.session_id,
            "model": self.model,
            "source": self.source,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "api_calls": self.api_calls,
            "started_at": self.started_at,
            "ended_at": time.time(),
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.push_url, data=data,
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=1.5)
        except Exception:
            # Dashboard down / wrong port / not running — stop trying so we
            # never add latency to the REPL on every turn.
            self._push_disabled = True


# Module-global recorder. None until main() builds one for the session.
# It's threaded down into the model-turn path so each API call's usage is
# accumulated; the REPL merges its .as_meta() into the session JSON on save.
_METRICS = None


def _env_metrics_url():
    """MINION_METRICS_URL — optional endpoint to receive cumulative token
    totals after each turn. Default unset (metrics stay local to the session
    JSON, nothing leaves the machine). No scheme/host validation beyond
    "looks like a URL" — a bad value just latches the push off on first try."""
    url = os.environ.get("MINION_METRICS_URL", "").strip()
    return url or None


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
    return f"{n / 1_000_000_000:.2f}B"

REASONING_ONLY_CHAR_LIMIT = _env_int("MINION_REASONING_ONLY_CHARS", 36000)
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
REASONING_ONLY_RETRY_LIMIT = _env_int("MINION_REASONING_ONLY_RETRIES", 1)
MALFORMED_STREAM_RETRY_LIMIT = _env_int("MINION_MALFORMED_STREAM_RETRIES", 2)
FORCED_FINAL_MAX_TOKENS = _env_int("MINION_FORCED_FINAL_MAX_TOKENS", 2048)
# An empty assistant turn (no content, no tool calls) is a silent stall: the model
# produced nothing and the loop would otherwise return TURN_DONE, dropping the user
# to the chat prompt mid-task. That's confusing — it looks like the agent finished
# when it actually just went mute (common with quantized models on a short/empty
# tool result, where the model "knows" the answer but emits no tokens). Auto-nudge
# it back into producing either a tool call or a visible answer. 0 disables the
# auto-continue (an empty turn then drops to chat as before).
EMPTY_TURN_RETRY_LIMIT = _env_int("MINION_EMPTY_TURN_RETRIES", 3)
RUNTIME_NOTE_RE = re.compile(r"\n\n\[Runtime note: .*?\]\s*$", re.DOTALL)

# --- auto context-compression -----------------------------------------------
# When the context window fills past this fraction (default 85%), the next
# settled model turn silently compresses the older history so the chat can
# keep going instead of running out of room. Unlike manual /compress (which
# keeps only COMPRESS_KEEP=2 turns), auto-compress is conservative: it keeps
# more recent turns so in-progress work isn't disturbed. Set to 0 to disable.
AUTOCOMPRESS_PERCENT = _env_int("MINION_AUTOCOMPRESS_PERCENT", 85)
AUTOCOMPRESS_PERCENT = max(0, min(100, AUTOCOMPRESS_PERCENT))  # clamp 0–100
# What `/autocompress on` restores; falls back to 85 when configured off.
AUTOCOMPRESS_DEFAULT = AUTOCOMPRESS_PERCENT or 85

FORCED_FINAL_NUDGE = (
    "Your previous streamed response produced reasoning only. Do not continue "
    "private reasoning. Use the final_answer tool if available. Return a complete "
    "visible answer now in at most six short bullets or paragraphs. If you are "
    "blocked, say exactly what is blocking you and what input is needed."
)
EMPTY_TURN_NUDGE = (
    "Your previous response was empty — you produced no visible text and no tool "
    "call, so nothing happened. Do not stop. If the task is not finished, emit the "
    "next tool call now. If you have everything you need, use the final_answer tool "
    "(or reply with a concise visible answer). Do not repeat the empty turn."
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


def _last_is_dangling_tool(messages):
    """True if the last message is a `tool` result with no assistant turn after it.

    This is the layout right after a tool ran and before the model follows up: an
    empty assistant turn here is most suspicious — the model produced nothing in
    response to a fresh tool result, so the user sees the agent "finish" with no
    answer. Auto-continue is most warranted in exactly this case.
    """
    return bool(messages) and messages[-1].get("role") == "tool"


# --- risk classifier --------------------------------------------------------
# One cheap non-streaming call per write/bash action. Same model, tiny prompt,
# expects {"level": "low|medium|high", "reason": "<short>"}. Defensive parse —
# if the model rambles or returns garbage we fall back to "high" so we err on
# the side of asking. Connection failures get a few short retries first because
# Minion normally talks to reliable or local endpoints. Skipped entirely in YOLO
# mode (no point paying for a call we won't act on) and for read-only tools.

def _detect_project_root(cwd=None):
    """Return the git root for cwd when available, otherwise cwd itself."""
    cwd = os.path.realpath(cwd or os.getcwd())
    if shutil.which("git"):
        try:
            r = subprocess.run(
                ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=2)
            root = r.stdout.strip()
            if r.returncode == 0 and root:
                return os.path.realpath(root)
        except (OSError, subprocess.SubprocessError):
            pass
    return cwd


PROJECT_ROOT = _detect_project_root()


def _is_under_path(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except (OSError, ValueError):
        return False


def _extract_action_path(action):
    """Pull the primary file path from Minion's short approval action string."""
    if not isinstance(action, str):
        return None
    action = action.strip()
    if action.startswith("edit "):
        return action[len("edit "):].strip() or None
    m = re.match(r"^write\s+(.+)\s+\(\d+\s+bytes\)$", action)
    if m:
        return m.group(1).strip() or None
    return None


def _classify_action_path(path, cwd=None, project_root=None):
    cwd = os.path.realpath(cwd or os.getcwd())
    project_root = os.path.realpath(project_root or PROJECT_ROOT)
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        expanded = os.path.join(cwd, expanded)
    abs_path = os.path.realpath(expanded)
    home = os.path.realpath(os.path.expanduser("~"))
    downloads = os.path.join(home, "Downloads")
    rel_home = None
    if _is_under_path(abs_path, home):
        try:
            rel_home = os.path.relpath(abs_path, home)
        except ValueError:
            rel_home = None
    home_parts = [] if rel_home in (None, ".") else rel_home.split(os.sep)
    touches_home_dotdir = bool(home_parts and home_parts[0].startswith("."))
    in_project = _is_under_path(abs_path, project_root)
    return {
        "primary_path": abs_path,
        "path_scope": "in_project" if in_project else "outside_project",
        "path_in_cwd": _is_under_path(abs_path, cwd),
        "path_in_downloads": _is_under_path(abs_path, downloads),
        "path_touches_home_dotdir": touches_home_dotdir,
    }


def _risk_user_message(action):
    cwd = os.path.realpath(os.getcwd())
    msg = {
        "action": action,
        "cwd": cwd,
        "project_root": os.path.realpath(PROJECT_ROOT),
        "primary_path": None,
        "path_scope": "unknown",
        "scope_guidance": (
            "Use project_root/path_scope for outside-project decisions. "
            "A file under project_root is in-project even when project_root "
            "is inside ~/Downloads."
        ),
    }
    path = _extract_action_path(action)
    if path:
        msg.update(_classify_action_path(path, cwd=cwd))
    return json.dumps(msg, sort_keys=True)


RISK_SYSTEM = (
    "You are a risk classifier for a coding agent's tool calls. "
    "Given one tool action as JSON, respond with ONLY a JSON object of the form "
    '{"level": "low"|"medium"|"high", "reason": "<one short sentence>"}.\n'
    "The user JSON includes cwd, project_root, and when possible path_scope. "
    "For outside-project decisions, trust project_root/path_scope over folder-name heuristics. "
    "If path_scope is in_project, do not classify high merely because the path is under ~/Downloads; "
    "projects can live in Downloads. If path_scope is outside_project, treat that as outside the project.\n"
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
        {"role": "user", "content": _risk_user_message(action)},
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
TOOL_TAG_PATTERN = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
MINION_TOOL_TAG_PATTERN = r"\[minion_tool_call\]\s*(\{.*?\})\s*\[/minion_tool_call\]"
TOOL_TAG = re.compile(TOOL_TAG_PATTERN, re.DOTALL)
MINION_TOOL_TAG = re.compile(MINION_TOOL_TAG_PATTERN, re.DOTALL)
TOOL_TAGS = (MINION_TOOL_TAG, TOOL_TAG)  # prefer Minion's bracketed protocol; keep legacy support
TOOL_PROTOCOL_TAG_RE = re.compile(r"</?tool_call\b[^>]*>")
TOOL_RESULT_PROTOCOL_NOTE = (
    "[minion note: escaped tool-call protocol delimiters from this tool result "
    "before sending it back to the model]\n"
)


def parse_text_calls(content):
    """Pull standalone text-protocol tool-call messages.

    Literal protocol tags can appear in files, docs, and code examples. Treat
    them as executable only when the whole assistant message is protocol text.
    """
    text = content or ""
    pos = 0
    calls = []
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        match = None
        for tag in TOOL_TAGS:
            match = tag.match(text, pos)
            if match:
                break
        if not match:
            return []
        try:
            obj = json.loads(match.group(1))
            calls.append((obj["name"], obj.get("arguments", {})))
        except (json.JSONDecodeError, KeyError):
            return []
        pos = match.end()
    return calls


def _escape_tool_protocol_delimiters(text):
    """Neutralize active tool-call delimiters in untrusted tool output."""
    if not isinstance(text, str):
        return text
    safe = TOOL_PROTOCOL_TAG_RE.sub(
        lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"),
        text)
    safe = safe.replace("[minion_tool_call]", "&#91;minion_tool_call&#93;")
    safe = safe.replace("[/minion_tool_call]", "&#91;/minion_tool_call&#93;")
    if safe != text:
        return TOOL_RESULT_PROTOCOL_NOTE + safe
    return text


def _sanitize_tool_result(text):
    """Neutralize active protocol delimiters, dedup, and cap tool results.

    Large, near-duplicate tool outputs (find/ls/grep path dumps) are the fuel
    for context-copying repetition collapse, so we collapse runs of >=3
    identical consecutive lines, then head/tail-cap to TOOL_RESULT_CHARS with a
    visible marker. Short, non-repetitive results pass through except for
    protocol-delimiter escaping.
    """
    if not isinstance(text, str):
        return text
    text = _escape_tool_protocol_delimiters(text)
    if len(text) <= 1000:
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


def compress(messages, keep=COMPRESS_KEEP, auto=False):
    """Ask the model to summarize everything except system + last `keep` turns.

    Mutates `messages` in place on success: replaces the middle slice with a
    single user-role summary turn. Returns (kept_n, summarized_n, summary_chars)
    or None on failure (in which case `messages` is untouched).

    Non-streaming on purpose — we want the whole summary before splicing it in,
    and a spinner for a one-shot summary would be visual noise.

    When auto=True the keep count is raised above COMPRESS_KEEP so
    auto-compression (triggered when the context window fills past
    AUTOCOMPRESS_PERCENT) is more conservative than a manual /compress —
    it keeps roughly the last third of the conversation verbatim so
    in-progress work and recent tool results survive the fold.
    """
    # Layout: [system?, ..., user, assistant, tool, ..., user, assistant(tool_calls)?, ...]
    # We assume messages[0] is the system prompt (matches how main() builds it).
    # Anything before the "tail" we want to summarize; the tail stays verbatim.
    sys_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    if auto:
        body_len = len(messages) - (1 if sys_msg else 0)
        keep = max(keep, body_len // 3)
    if len(messages) <= 1 + keep:
        return None  # nothing to compress

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


def _maybe_autocompress(messages, prompt_tokens):
    """Silently compress when context usage crosses the AUTOCOMPRESS_PERCENT
    threshold. Called after each settled model turn with the last turn's
    `prompt_tokens` (the size of the context the model just saw).

    Returns True if a compression happened, False otherwise. Guards:
      - AUTOCOMPRESS_PERCENT == 0 → disabled
      - ACTIVE context window unknown / ≤ 0 → can't compute a ratio
      - ratio < threshold → nothing to do
      - compress() itself bails when there's too little to fold
    """
    if AUTOCOMPRESS_PERCENT <= 0:
        return False
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        return False
    if ACTIVE is None:
        return False
    mx = ACTIVE._context_window
    if not isinstance(mx, int) or mx <= 0:
        return False
    ratio = prompt_tokens / mx
    if ratio * 100 < AUTOCOMPRESS_PERCENT:
        return False
    result = compress(messages, auto=True)
    if result is None:
        return False
    kept_n, summarized_n, summary_chars = result
    pct = int(ratio * 100)
    print(f"{DIM}  ↻ auto-compressed {summarized_n} turns → 1 summary "
          f"({summary_chars} chars), kept last {kept_n} verbatim "
          f"(context was at {pct}% of {_abbr(mx)}){RESET}")
    return True


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
TURN_STREAM_CUT = "stream_cut"
TURN_FORCE_FINAL = "force_final"
TURN_EMPTY = "empty"  # assistant emitted no content and no tool calls → silent stall
TURN_ESC = "esc"  # user pressed Esc at an approval prompt → drop to chat input


def _ctx_field(prompt_n):
    """The colored `ctx` field for the stats footer: `<cur>/<max> ctx`.
    The max context window is cyan; the current context is colored by
    utilization — green under 30%, yellow under 60%, red at or above.
    When the max isn't known yet (probe still pending or unreachable)
    the field degrades to `<cur> ctx` with no color, and a background
    probe is kicked off so a later turn can fill in the max."""
    cur = _abbr(prompt_n)
    if ACTIVE is None:
        return f"{cur} ctx"
    mx = ACTIVE._context_window
    if not isinstance(mx, int) or mx <= 0:
        _ensure_ctx_probe(ACTIVE)
        return f"{cur} ctx"
    mx_s = _abbr(mx)
    ratio = prompt_n / mx if mx > 0 else 0.0
    if ratio < 0.30:
        color = GREEN
    elif ratio < 0.60:
        color = YELLOW
    else:
        color = RED
    # RESET after each colored run, then re-apply DIM so the rest of the
    # footer (joined around this field) stays dim.
    return f"{color}{cur}{RESET}{DIM}/{CYAN}{mx_s}{RESET}{DIM} ctx"


_CTX_PROBE_STARTED = set()  # id(Source) currently being probed (avoid dupe threads)


def _ensure_ctx_probe(src):
    """Start a background thread to resolve a Source's max context window if
    it isn't already known and no probe is in flight for it. The result lands
    in src._context_window; the footer picks it up on a later turn. Idempotent
    via _CTX_PROBE_STARTED (cleared when the thread finishes).

    A local server's metadata endpoints can 503 "Loading model" for tens of
    seconds at cold start (the weights are still mmapping) even while a chat
    slot is already serving — so a single attempt often misses. The thread
    retries with backoff until it resolves (or gives up after ~2 min), so the
    suffix fills in the first turn AFTER the server is ready rather than
    depending on the user to keep typing."""
    sid = id(src)
    if sid in _CTX_PROBE_STARTED:
        return
    _CTX_PROBE_STARTED.add(sid)

    def _work():
        # Backoff schedule (seconds between attempts): a handful of quick
        # tries for a fast local server, then wider spacing for a slow cold
        # start. Total ~2 min of probing before giving up (the footer will
        # re-arm on the next user turn anyway, so this is just a best effort
        # to resolve the max before the user has to wait for another turn).
        delays = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 30]
        try:
            for delay in delays:
                if isinstance(src._context_window, int) and src._context_window > 0:
                    return  # resolved by another path (e.g. a /source switch)
                src.resolve_context_window()
                if isinstance(src._context_window, int) and src._context_window > 0:
                    return
                time.sleep(delay)
        except Exception:
            pass  # never raise from a background probe
        finally:
            _CTX_PROBE_STARTED.discard(sid)

    t = threading.Thread(target=_work, daemon=True, name="minion-ctx-probe")
    t.start()


# --- one model turn (streamed), returns TURN_* status -----------------------
def model_turn(messages, reasoning_loop_cut_count=0, malformed_stream_cut_count=0,
               empty_turn_count=0, forced_final=False, recovery_sampling=False):
    global _LAST_PROMPT_TOKENS
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
    loop_cut = False
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
                if mode != "think":
                    print(f"{DIM}  ── reasoning ──{RESET}")
                    mode = "think"
                print(f"{DIM}{rc}{RESET}", end="", flush=True)
                if not content and not tcs:
                    reasoning_only_chars += len(rc)
                if (REASONING_ONLY_CHAR_LIMIT > 0 and not content and not tcs
                        and reasoning_only_chars >= REASONING_ONLY_CHAR_LIMIT):
                    print()
                    print(f"{RED}  ⚠ REASONING-ONLY LIMIT HIT — "
                          f"{_abbr(reasoning_only_chars)} reasoning chars with no "
                          f"answer/tool call — cutting stream now{RESET}")
                    loop_cut = True
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
                show_tool_status()
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

    if loop_cut:
        retry_limit = max(0, REASONING_ONLY_RETRY_LIMIT)
        cut_count = reasoning_loop_cut_count
        loop_detail = (
            f"{_abbr(reasoning_only_chars)} reasoning chars without content/tool calls")
        if forced_final:
            print(f"{RED}  ✂ FORCED FINAL ANSWER FAILED — got {loop_detail}; "
                  f"returning to chat input{RESET}")
            return TURN_DONE
        if cut_count >= retry_limit:
            print(f"{RED}  ✂ REASONING-ONLY RESCUE FAILED — gave up after "
                  f"{cut_count} forced final "
                  f"attempt{'s' if cut_count != 1 else ''} "
                  f"× {loop_detail}; waiting for user input{RESET}")
            return TURN_DONE
        cut_msg = (f"{_abbr(reasoning_only_chars)} reasoning chars with no "
                   f"answer/tool call (limit {_abbr(REASONING_ONLY_CHAR_LIMIT)})")
        print(f"{YELLOW}  ✂ REASONING-ONLY STALL — {cut_msg}; forcing final answer "
              f"(attempt {cut_count + 1}/{retry_limit}){RESET}")
        _nudge_current_user_turn(messages, FORCED_FINAL_NUDGE)
        return TURN_FORCE_FINAL

    # stats footer — prefer llama.cpp timings if present; otherwise fall back
    # to the standard streaming `usage` object (OpenAI, Z.ai, etc.); otherwise
    # fall back to wall-clock only. TTFT (time to first token) is shown when
    # available — for local it comes from llama.cpp timings, for remote we
    # measure it client-side.
    if _METRICS is not None:
        _METRICS.add(_normalize_usage(usage, timings))
    if timings and timings.get("predicted_n"):
        prompt_n = timings.get("prompt_n", 0)
        if prompt_n:
            _LAST_PROMPT_TOKENS = prompt_n
        gen_n = timings["predicted_n"]
        tps = timings.get("predicted_per_second", 0)
        ctx = _ctx_field(prompt_n)
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
        if prompt_n:
            _LAST_PROMPT_TOKENS = prompt_n
        tps = gen_n / elapsed if elapsed > 0 else 0
        ctx = _ctx_field(prompt_n)
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
        # Silent stall: the model emitted nothing. In a real agentic loop this is
        # almost never a legitimate "done" — it usually follows a short/empty
        # tool result (the model "knows" the answer but produces no tokens), and
        # returning TURN_DONE here drops the user to the chat prompt mid-task with
        # no explanation. Surface it so the loop can nudge + auto-continue instead.
        #
        # Never auto-continue when:
        #   • forced_final — an empty forced answer already means "give up" (the
        #     forced-final block above printed its own message); stopping is right.
        #   • EMPTY_TURN_RETRY_LIMIT <= 0 — the user opted out; empty = done.
        #   • recovery already exhausted the budget — escalate to a forced answer
        #     (mirrors the reasoning-loop path) instead of spinning forever.
        if (not forced_final and EMPTY_TURN_RETRY_LIMIT > 0
                and empty_turn_count < EMPTY_TURN_RETRY_LIMIT):
            dangling = _last_is_dangling_tool(messages)
            print(f"{YELLOW}  ✂ EMPTY TURN — no content and no tool call"
                  f"{'; dangling tool result awaits a follow-up' if dangling else ''}"
                  f"; nudging the model to continue "
                  f"({empty_turn_count + 1}/{EMPTY_TURN_RETRY_LIMIT}){RESET}")
            _nudge_current_user_turn(messages, EMPTY_TURN_NUDGE)
            return TURN_EMPTY
        if not forced_final and EMPTY_TURN_RETRY_LIMIT > 0:
            print(f"{YELLOW}  ✂ EMPTY TURN — no content after {empty_turn_count} nudge"
                  f"{'s' if empty_turn_count != 1 else ''}; forcing a final answer{RESET}")
            _nudge_current_user_turn(messages, FORCED_FINAL_NUDGE)
            return TURN_FORCE_FINAL
        return TURN_DONE
    messages.append({"role": "assistant", "content": text})
    return TURN_DONE


def _run_model_turn_loop(messages, force_final=False, recovery_sampling=False,
                         on_step=None):
    steps = 0
    reasoning_loop_cuts = 0
    malformed_stream_cuts = 0
    empty_turn_cuts = 0
    while steps < 25:  # cap runaway tool/retry loops
        status = model_turn(messages, reasoning_loop_cuts, malformed_stream_cuts,
                            empty_turn_count=empty_turn_cuts,
                            forced_final=force_final,
                            recovery_sampling=recovery_sampling)
        if on_step is not None:
            on_step()
        force_final = False
        recovery_sampling = False
        if status == TURN_DONE:
            break
        if status == TURN_ESC:
            break  # user pressed Esc at an approval → drop to chat input
        steps += 1
        if status == TURN_STREAM_CUT:
            malformed_stream_cuts += 1
            recovery_sampling = True
            continue
        if status == TURN_EMPTY:
            empty_turn_cuts += 1
            recovery_sampling = True  # more entropy to escape the empty-output attractor
            continue
        if status == TURN_FORCE_FINAL:
            reasoning_loop_cuts += 1
            empty_turn_cuts = 0  # handed off to the forced-final path; reset
            force_final = True
            recovery_sampling = True
            continue
        if status == TURN_TOOL:
            reasoning_loop_cuts = 0
            malformed_stream_cuts = 0
            empty_turn_cuts = 0
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
    global YOLO, APPROVE_LEVEL, DEFAULT_APPROVE_LEVEL, AUTOCOMPRESS_PERCENT
    # `minion sessions [query]` — discover + exit, no REPL. Checked first so
    # it short-circuits before the banner / network source resolution that the
    # interactive loop depends on.
    if _cli_sessions(sys.argv[1:]):
        return
    # The banner is printed at startup and after source/approval changes.
    print(_banner())
    print()
    # Warm the active source's max-context-window cache before the first
    # prompt: the probe (esp. on a still-loading local server, whose /v1/models
    # + /props 503 while mmapping) can take seconds-to-tens-of-seconds, but the
    # user typically spends that long typing their first message — so by the
    # time turn 1 finishes the cache is often already filled and the footer
    # shows the "/<max>" suffix immediately instead of a turn later. Background,
    # non-blocking; degrades to "no suffix" if it never resolves.
    if ACTIVE is not None:
        _ensure_ctx_probe(ACTIVE)
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
            # Restore the active source (and any model override) the session
            # was started on, so a resume lands on the same endpoint + model.
            src_name = data.get("source")
            if src_name:
                restore_source(src_name, model=data.get("model"))
            session_dirty = False  # just loaded — in sync with disk
        elif _resume_requested:
            print(f"{YELLOW}  ✗ no saved session found for {session_id!r}; "
                  f"starting fresh{RESET}")

    # --- usage metrics -------------------------------------------------------
    # Build the cumulative-token recorder for this session. When resuming, we
    # reload its running totals from the saved session JSON so they keep
    # climbing across restarts instead of resetting to zero. The optional
    # HTTP push (MINION_METRICS_URL) is off by default — see _Metrics.
    global _METRICS
    push_url = _env_metrics_url()
    _METRICS = _Metrics(
        session_id, model=MODEL,
        source=ACTIVE.name if ACTIVE else None,
        push_url=push_url)
    if _resume_requested or any(a == "--session" for a in sys.argv[1:]):
        saved = _load_session(session_id) or {}
        if isinstance(saved.get("input_tokens"), int):
            _METRICS.input_tokens = saved.get("input_tokens") or 0
            _METRICS.output_tokens = saved.get("output_tokens") or 0
            _METRICS.cache_read_tokens = saved.get("cache_read_tokens") or 0
            _METRICS.reasoning_tokens = saved.get("reasoning_tokens") or 0
            _METRICS.api_calls = saved.get("api_calls") or 0
            if saved.get("started_at"):
                _METRICS.started_at = saved["started_at"]

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
        # Merge cumulative token totals into the session JSON so they persist
        # across restarts and are available to any metrics consumer that reads
        # the session files (e.g. a dashboard). This is always-on — the numbers
        # are already in hand from the stats footer, so storing them costs
        # nothing and a local session log is useful regardless of a push sink.
        if _METRICS is not None:
            m.update(_METRICS.as_meta())
        try:
            _write_session(session_id, messages, m)
        except OSError as e:
            print(f"{RED}  ✗ couldn't save session: {type(e).__name__}: {e}{RESET}")
        # Fire-and-forget push to the optional metrics endpoint. Only does
        # anything if MINION_METRICS_URL is set; self-disables on first
        # failure so a down/absent dashboard never slows the REPL.
        if _METRICS is not None:
            _METRICS.flush()

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
                    ctx = src._context_window
                    if isinstance(ctx, int) and ctx > 0:
                        ctx_s = f" · {_abbr(ctx)} ctx"
                    else:
                        ctx_s = ""
                        _ensure_ctx_probe(src)  # warm the cache for next time
                    print(f"  {mark} {CYAN}{sname:<12}{RESET} {DIM}{m} @ {src.base_url}{ctx_s}{RESET}")
                print(f"{DIM}  /source <name> [model] to switch · "
                      "context preserved (use /reset to clear){RESET}")
            else:
                target = sp[1]
                # Optional second token: a model id to pin for this switch.
                # Useful for multi-model hosts (e.g. Together) so you can do
                #   /source together                    → default (GLM-5.2)
                #   /source together zai-org/GLM-4.6     → that model, same host
                # The override is non-sticky: a later bare /source <name>
                # always falls back to the source's configured default.
                model_override = sp[2] if len(sp) >= 3 else None
                if target not in SOURCES:
                    avail = ", ".join(SOURCE_ORDER)
                    print(f"{RED}  ✗ unknown source {target!r} — available: {avail}{RESET}")
                    continue
                if target == ACTIVE.name and not model_override:
                    print(f"{DIM}  already on {target}{RESET}")
                    continue
                switch_source(target, model_override=model_override)
                src = SOURCES[target]
                # An explicit /source switch is user-initiated, so a synchronous
                # probe here is the right trade: the user asked for this source
                # and will want to see its context limit. Caches after the first
                # switch, so later switches to a known source are instant.
                ctx = src.resolve_context_window()
                ctx_s = f" · {_abbr(ctx)} ctx" if isinstance(ctx, int) and ctx > 0 else ""
                print(f"{BOLD}{YELLOW}  → switched to {target}{RESET} {DIM}({MODEL} @ {src.base_url}{ctx_s}){RESET}")
                print(_banner())
                session_dirty = True  # source/model change is worth persisting
                if _METRICS is not None:
                    _METRICS.model = MODEL
                    _METRICS.source = target
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
        if user in ("/reset", "/clear", "/new"):
            messages = [{"role": "system", "content": SYSTEM}]
            print(f"{DIM}  context cleared{RESET}")
            # A reset forks a new session id so the fresh chat isn't written
            # over the old one (mirrors Hermes's "new session on /new").
            session_id = _new_session_id()
            session_dirty = False
            # Fresh recorder for the fresh session.
            _METRICS = _Metrics(
                session_id, model=MODEL,
                source=ACTIVE.name if ACTIVE else None,
                push_url=_env_metrics_url())
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
            _run_model_turn_loop(messages, force_final=True, recovery_sampling=True,
                                 on_step=_save_current)
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
        if user == "/autocompress" or user.startswith("/autocompress "):
            arg = user[len("/autocompress"):].strip().lower()
            if not arg:
                if AUTOCOMPRESS_PERCENT <= 0:
                    print(f"{DIM}  auto-compress: off{RESET}")
                else:
                    print(f"{DIM}  auto-compress: {AUTOCOMPRESS_PERCENT}% "
                          f"(auto-compress when context ≥ {AUTOCOMPRESS_PERCENT}% full){RESET}")
                continue
            if arg in ("off", "disable", "0"):
                AUTOCOMPRESS_PERCENT = 0
                print(f"{DIM}  auto-compress: off (disabled){RESET}")
                continue
            if arg in ("on", "enable"):
                AUTOCOMPRESS_PERCENT = AUTOCOMPRESS_DEFAULT
                print(f"{DIM}  auto-compress: {AUTOCOMPRESS_PERCENT}% "
                      f"(auto-compress when context ≥ {AUTOCOMPRESS_PERCENT}% full){RESET}")
                continue
            try:
                val = int(arg)
            except ValueError:
                print(f"{YELLOW}  usage: /autocompress [<1-100|off|on>] — "
                      f"current: {AUTOCOMPRESS_PERCENT}%{RESET}")
                continue
            if not (1 <= val <= 100):
                print(f"{YELLOW}  {val} is out of range — want 1–100 (or off / on){RESET}")
                continue
            AUTOCOMPRESS_PERCENT = val
            print(f"{DIM}  auto-compress: {val}% "
                  f"(auto-compress when context ≥ {val}% full){RESET}")
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
                # Reload the cumulative-token recorder from the resumed
                # session's saved totals so they keep climbing from here.
                _METRICS = _Metrics(
                    session_id, model=MODEL,
                    source=ACTIVE.name if ACTIVE else None,
                    push_url=_env_metrics_url())
                saved = _load_session(session_id) or {}
                if isinstance(saved.get("input_tokens"), int):
                    _METRICS.input_tokens = saved.get("input_tokens") or 0
                    _METRICS.output_tokens = saved.get("output_tokens") or 0
                    _METRICS.cache_read_tokens = saved.get("cache_read_tokens") or 0
                    _METRICS.reasoning_tokens = saved.get("reasoning_tokens") or 0
                    _METRICS.api_calls = saved.get("api_calls") or 0
                    if saved.get("started_at"):
                        _METRICS.started_at = saved["started_at"]
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
        _run_model_turn_loop(messages, on_step=_save_current)
        # Auto-save after the turn settles (whether it ended cleanly, hit the
        # step cap, or was escaped). This is the Hermes pattern: persist every
        # turn so a crash / accidental close never loses work.
        session_dirty = True
        _save_current()
        # Maybe auto-compress: if the context window is over the configured
        # fill threshold, silently fold the older turns so the chat can keep
        # going. Done after the save so the pre-compression state is on disk;
        # the compressed result is persisted by the _save_current() below.
        if _maybe_autocompress(messages, _LAST_PROMPT_TOKENS):
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
    # Reselect the recorded source (and model override) if configured, so a
    # resume lands on the same endpoint + model the session started on.
    src = data.get("source")
    if src and restore_source(src, model=data.get("model")):
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
