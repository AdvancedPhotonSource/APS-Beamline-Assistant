"""APEXA agent timing / instrumentation.

Records structured LLM-call and per-query telemetry to JSONL for later analysis
by `scripts/analyze_timing.py` (benchmark table generator). Companion to the
existing console print in ArgoProvider.chat() — console lines are human-facing,
JSONL rows are machine-readable for latency/throughput analysis and comparison
against external benchmarks (e.g. NVIDIA DGX Spark agentic workload numbers).

Three row types in the same JSONL:
    endpoint = "argo-chat"      Tier 1 — one row per blocking /chat/ HTTP call
    endpoint = "argo-messages"  Tier 3 — one row per streaming /messages/ call
                                 (adds real TTFT and TPOT)
    endpoint = "query"          Tier 2 — one row per user query wrapping the
                                 above; groups them via query_id

Gated on APEXA_SHOW_TIMING=1. Log path overridable via APEXA_TIMING_LOG
(default: ~/.apexa/timing.jsonl).

Token counts come from tiktoken (cl100k_base) when available and fall back
to len(text)//4. Argo model names (gpt55, claudeopus48, gemini35flash) don't
map to tiktoken encodings, so counts are approximate — accurate enough for
order-of-magnitude latency/throughput/cost analysis, not for billing.
"""

from __future__ import annotations
import json
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

try:
    import tiktoken
    _ENC_CACHE: dict[str, Any] = {}
except ImportError:
    tiktoken = None
    _ENC_CACHE = {}


# ─── Token counting ────────────────────────────────────────────────────────

def _encoding_for(_model: str):
    if tiktoken is None:
        return None
    enc = _ENC_CACHE.get("default")
    if enc is None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = None
        _ENC_CACHE["default"] = enc
    return enc


def count_tokens(text: str, model: str = "") -> int:
    """Count tokens for `text`. Returns len(text)//4 when tiktoken is absent."""
    if not text:
        return 0
    enc = _encoding_for(model)
    if enc is None:
        return len(text) // 4
    try:
        return len(enc.encode(text, disallowed_special=()))
    except Exception:
        return len(text) // 4


def count_message_tokens(messages: Iterable[dict], model: str = "") -> int:
    """Sum tokens across role+content of each message, with a small per-message
    overhead (~4 tok) to approximate role/format framing."""
    total = 0
    for m in messages:
        total += count_tokens(str(m.get("content", "")), model)
        total += 4
    return total


# ─── JSONL sink ────────────────────────────────────────────────────────────

def timing_enabled() -> bool:
    return bool(os.environ.get("APEXA_SHOW_TIMING"))


def _log_path() -> Path:
    env = os.environ.get("APEXA_TIMING_LOG")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".apexa" / "timing.jsonl"


def log_llm_call(record: dict) -> None:
    """Append one row to the JSONL timing log.

    No-op when APEXA_SHOW_TIMING is not set. Never raises — telemetry must
    not break the caller.

    Named ``log_llm_call`` for historical Tier-1 reasons; also used for
    Tier-2 ``endpoint="query"`` and Tier-3 ``endpoint="argo-messages"`` rows.
    """
    if not timing_enabled():
        return
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        stamped = {
            "iso_ts": datetime.now(timezone.utc).isoformat(),
            "ts": time.time(),
            **record,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(stamped, default=str) + "\n")
    except Exception:
        pass


# ─── Tier 2: per-query context ─────────────────────────────────────────────

@dataclass
class QueryContext:
    """Per-user-query state carried across the orchestrator loop.

    Instrumentation only — never mutated by user code. Populated by:
      - ArgoProvider.chat()               → n_llm_calls, sum_llm_elapsed_s
      - StreamingAnthropicProvider.chat() → n_llm_calls, sum_llm_elapsed_s
      - APEXAClient.execute_tool_call()   → n_tools, tool_elapsed_s
      - OrchestratorAgent (caller)        → agent (specialist name)
    """
    query_id:          str
    started_ts:        float
    query_chars:       int   = 0
    agent:             str   = ""
    n_llm_calls:       int   = 0
    sum_llm_elapsed_s: float = 0.0
    n_tools:           int   = 0
    tool_elapsed_s:    float = 0.0
    llm_prompt_tok:    int   = 0
    llm_response_tok:  int   = 0


_current_query: ContextVar[Optional[QueryContext]] = ContextVar(
    "apexa_query", default=None
)


def current_query_id() -> Optional[str]:
    ctx = _current_query.get()
    return ctx.query_id if ctx else None


def record_llm_call(elapsed_s: float, prompt_tok: int = 0,
                    response_tok: int = 0) -> None:
    """Called from within a provider after a completed LLM HTTP round-trip."""
    ctx = _current_query.get()
    if ctx is None:
        return
    ctx.n_llm_calls       += 1
    ctx.sum_llm_elapsed_s += elapsed_s
    ctx.llm_prompt_tok    += prompt_tok
    ctx.llm_response_tok  += response_tok


def record_tool_call(elapsed_s: float) -> None:
    """Called from within the client after a completed tool execution."""
    ctx = _current_query.get()
    if ctx is None:
        return
    ctx.n_tools        += 1
    ctx.tool_elapsed_s += elapsed_s


@contextmanager
def query_scope(query: str = "", agent: str = "") -> Iterator[QueryContext]:
    """Bracket a user query. Emits an ``endpoint="query"`` row on exit.

    Usage::

        with query_scope(query=query) as ctx:
            result = await orchestrator.process(...)
            ctx.agent = orchestrator._last_agent.name  # optional refinement

    Nested scopes are supported (inner scope is a no-op summary — outer wins).
    Exception during the scope still emits the summary with what was collected.
    """
    if _current_query.get() is not None:
        # Nested scope — don't double-count. Yield an empty context so callers
        # can still set .agent without effect.
        yield QueryContext(query_id="", started_ts=time.time())
        return
    ctx = QueryContext(
        query_id=uuid.uuid4().hex[:12],
        started_ts=time.time(),
        query_chars=len(query or ""),
        agent=agent,
    )
    token = _current_query.set(ctx)
    try:
        yield ctx
    finally:
        _current_query.reset(token)
        wall = time.time() - ctx.started_ts
        log_llm_call({
            "endpoint":          "query",
            "query_id":          ctx.query_id,
            "agent":             ctx.agent,
            "query_chars":       ctx.query_chars,
            "n_llm_calls":       ctx.n_llm_calls,
            "sum_llm_elapsed_s": round(ctx.sum_llm_elapsed_s, 3),
            "n_tools":           ctx.n_tools,
            "tool_elapsed_s":    round(ctx.tool_elapsed_s, 3),
            "wall_clock_s":      round(wall, 3),
            "llm_prompt_tok":    ctx.llm_prompt_tok,
            "llm_response_tok":  ctx.llm_response_tok,
        })
