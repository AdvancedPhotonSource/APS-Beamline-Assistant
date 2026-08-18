"""OpenAI-compatible LLM provider for APEXA (via argo-proxy).

Why this exists
---------------
``ArgoProvider`` talks to Argo's ``/chat/`` endpoint, whose payload sanitizer
flattens every message to ``{role, content:<string>}``. That makes it impossible
to send a *structured* tool result back, so APEXA's follow-up turns fall back to a
text ``TOOL_CALL:`` / ``ARGUMENTS:`` protocol — which frontier models drift off,
which is why a cluster of regex anti-fabrication guards grew around the loop.

``argo-proxy`` (MIT, on-prem, https://github.com/Oaklight/argo-proxy) fronts the
same Argo Gateway with a standard OpenAI ``/v1/chat/completions`` surface and
handles per-vendor tool translation itself. Speaking that protocol gives APEXA:

  * real ``{"role":"tool","tool_call_id":...}`` results on EVERY turn, not just the
    first — so tool calling is structured end-to-end and format drift is impossible;
  * one code path for Anthropic / OpenAI / Gemini — no ``_to_vendor_tools``, no
    per-model parameter table, no three-shape ``_parse_tool_calls``;
  * a stable, cacheable prompt prefix.

This module is a drop-in for ``ArgoProvider`` at the ``select_provider()`` seam:
same duck-typed ``chat(messages, temperature, tools) -> AgentResponse`` contract,
same ``close()``, same ``~/.apexa/timing.jsonl`` instrumentation.

Setup (ANL internal network or VPN required):

    pip install argo-proxy
    argo-proxy config init
    argo-proxy serve                      # http://localhost:44497

    export APEXA_LLM_MODE=proxy
    export APEXA_LLM_BASE_URL=http://localhost:44497/v1

Verify multi-turn support first with ``scripts/gate0_argo_proxy_smoke.py``.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from apexa_llm_endpoints import EndpointRejected, active_endpoint
from apexa_timing import (
    count_message_tokens,
    count_tokens,
    current_query_id,
    log_llm_call,
    record_llm_call,
)

DEFAULT_BASE_URL = "http://localhost:44497/v1"
DEFAULT_MAX_TOKENS = 16000


def proxy_mode_enabled() -> bool:
    """True when APEXA should route through the OpenAI-compatible proxy.

    ``APEXA_LLM_MODE`` is tri-state: ``proxy`` → on, ``argo`` → off, unset →
    off (the conservative default while the proxy soaks at the beamline).
    """
    return (os.environ.get("APEXA_LLM_MODE") or "argo").strip().lower() == "proxy"


def base_url() -> str:
    """Endpoint URL from the active preset (``APEXA_LLM_PRESET``), overridable by
    ``APEXA_LLM_BASE_URL``. See apexa_llm_endpoints."""
    try:
        return active_endpoint().base_url
    except EndpointRejected:
        return (os.environ.get("APEXA_LLM_BASE_URL") or DEFAULT_BASE_URL).strip()


def api_key_for(username: str) -> str:
    """Credential for the configured endpoint, resolved FRESH on each call.

    Precedence lives in ``OpenAIEndpoint.resolve_key``: ``APEXA_LLM_API_KEY`` →
    preset key → ``APEXA_LLM_TOKEN_CMD`` output (cached ~10 min) → ANL username
    (the argo-proxy convention).

    Resolution is deliberately lazy. ALCF Globus access tokens expire after 48 h, so
    a key captured once at construction would go stale mid-experiment; delegating
    refresh to ALCF's own helper keeps a long-running beamline session alive.

    This is also what lets each user bring their own credential: a shared APEXA
    install on the beamline host, with every user's own environment carrying their
    own token, so inference is attributed to them and not to a shared beamline
    account. Never put a key in a shared or committed file.
    """
    try:
        return active_endpoint().resolve_key(username)
    except EndpointRejected:
        return (os.environ.get("APEXA_LLM_API_KEY") or username or "apexa").strip()


def strict_mode() -> bool:
    """Refuse to silently downgrade to the legacy Argo transport.

    At a beamline deployment a proxy that failed to start must be an operator-
    visible failure, not a quiet fall back to the text ``TOOL_CALL:`` path — that
    path is the fabrication-prone one, and nobody would notice APEXA had been
    running on it for weeks. Same philosophy as the deletion permission gate: fail
    closed on the integrity-relevant path.

    Defaults ON whenever ``APEXA_LLM_MODE=proxy`` is explicitly requested; set
    ``APEXA_LLM_STRICT=0`` to allow the fallback (reasonable on a dev laptop).
    """
    raw = (os.environ.get("APEXA_LLM_STRICT") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return proxy_mode_enabled()


async def preflight(username: str, model: str) -> tuple[bool, str]:
    """Check at STARTUP that the proxy is reachable and the model resolves.

    Without this the first failure surfaces mid-experiment as a failed query.
    Returns ``(ok, human_readable_detail)``; never raises.
    """
    if not proxy_mode_enabled():
        return True, "APEXA_LLM_MODE=argo — legacy Argo /chat/ transport"
    try:
        p = OpenAICompatProvider(username, model)
    except ProviderUnavailable as e:
        return False, str(e)
    try:
        resolved = await p._resolve_model()
        return True, f"argo-proxy {p.url} — model {model!r} → {resolved!r}"
    except ProviderUnavailable as e:
        return False, str(e)
    except Exception as e:
        return False, (f"cannot reach argo-proxy at {p.url}: {type(e).__name__}: {e}. "
                       f"Is the sidecar running (`argo-proxy serve`) and is "
                       f"APEXA_LLM_BASE_URL pointing at its port?")
    finally:
        try:
            await p.close()
        except Exception:
            pass


def _norm(name: str) -> str:
    """Fold a model id for tolerant matching: strip punctuation, case, ``argo:``.

    ``argo:gpt-5.6-sol`` → ``gpt56sol`` (== Argo's own compact id).
    """
    return re.sub(r"[^a-z0-9]", "", (name or "").lower()).removeprefix("argo")


def _anagram_key(name: str) -> str:
    """Order-insensitive fold, for vendors whose id components are permuted.

    Argo names Anthropic models ``claude<tier><version>`` (``claudeopus5``) while
    argo-proxy serves them ``claude-<version>-<tier>`` (``argo:claude-5-opus``).
    Those are the same characters in a different order, so a sorted-character key
    matches them where ``_norm`` alone cannot. OpenAI and Gemini ids already agree
    under ``_norm``; this is the Anthropic-shaped fallback.

    Used only as a LAST resort and only when it yields exactly one candidate — a
    permutation fold is inherently lossy (a hypothetical ``claude-4.6-opus`` and
    ``claude-6.4-opus`` would collide), so an ambiguous result is reported rather
    than guessed.
    """
    return "".join(sorted(_norm(name)))


class ProviderUnavailable(RuntimeError):
    """argo-proxy is not reachable — caller may fall back to ArgoProvider."""


class OpenAICompatProvider:
    """Argo via an OpenAI-compatible endpoint. Structured tool calling throughout.

    Deliberately has no per-model parameter table. Where a model rejects a
    sampling parameter, we learn that from the 400 and retry without it
    (``_ADAPTIVE_PARAMS``) rather than hand-maintaining a matrix against a model
    list that changes every few months. The learned drops are remembered for the
    lifetime of the provider instance.
    """

    # Marks this provider as able to carry native tool_calls / role:"tool" turns.
    # AgentRunner keys the structured loop off this (see provider_is_structured).
    structured_tools = True

    # Parameters we will surrender, in order, when the gateway 400s about them.
    _ADAPTIVE_PARAMS = ("temperature", "top_p", "max_completion_tokens")

    def __init__(self, username: str, model: str,
                 base_url_override: Optional[str] = None):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover - dependency guard
            raise ProviderUnavailable(
                "openai SDK not installed — `uv sync` or `pip install openai`"
            ) from e

        self.username = username
        self.model = model
        self.url = base_url_override or base_url()
        # argo-proxy authenticates with the ANL username in the api_key slot;
        # ALCF / vLLM / commercial APIs need a real key or a Globus bearer token.
        # Resolved fresh per request (see _client_for_request) so an expiring ALCF
        # token can be refreshed without restarting a long beamline session.
        self._key = api_key_for(username)
        self._client = AsyncOpenAI(base_url=self.url, api_key=self._key,
                                   timeout=120.0, max_retries=0)
        self._dropped: set[str] = set()
        self._resolved_model: Optional[str] = None

    # ── model resolution ────────────────────────────────────────────────────

    async def _resolve_model(self) -> str:
        """Map APEXA's model id onto whatever the proxy actually serves.

        Argo ids are compact (``claudeopus5``); a proxy may expose dashed ids
        (``claude-opus-5``). Match case/punctuation-insensitively so operators
        don't have to care, and fail with the near-misses listed rather than a
        bare 404.
        """
        if self._resolved_model:
            return self._resolved_model
        try:
            served = [m.id for m in (await self._client.models.list()).data]
        except Exception as e:
            # A proxy that simply doesn't implement /models is tolerable — fall
            # through and try the raw id. An UNREACHABLE proxy is not: swallowing
            # that here would make preflight report success against a dead
            # endpoint and render APEXA_LLM_STRICT useless (observed: pointing at
            # the wrong port resolved "fine" and only failed at first query).
            if type(e).__name__ in ("APIConnectionError", "APITimeoutError",
                                    "AuthenticationError", "PermissionDeniedError"):
                raise ProviderUnavailable(
                    f"cannot reach argo-proxy at {self.url}: {type(e).__name__}: {e}. "
                    f"Is the sidecar running, and does APEXA_LLM_BASE_URL point at "
                    f"its port?"
                ) from e
            print(f"  \033[33m⚠ {self.url} does not support /models "
                  f"({type(e).__name__}) — using model id verbatim\033[0m",
                  file=sys.stderr)
            self._resolved_model = self.model
            return self._resolved_model

        if self.model in set(served):
            self._resolved_model = self.model
            return self._resolved_model

        # Tier 1: punctuation/case/prefix fold (covers OpenAI + Gemini).
        hit = {_norm(m): m for m in served}.get(_norm(self.model))

        # Tier 2: permutation fold (covers Anthropic's reordered ids). Only
        # accepted when unambiguous — see _anagram_key.
        if not hit:
            key = _anagram_key(self.model)
            cands = sorted({m for m in served if _anagram_key(m) == key})
            if len(cands) == 1:
                hit = cands[0]
            elif len(cands) > 1:
                raise ProviderUnavailable(
                    f"model {self.model!r} matches {len(cands)} served models "
                    f"ambiguously: {cands}. Set ARGO_MODEL to the exact served id."
                )

        if not hit:
            stem = _norm(self.model)[:6]
            near = [m for m in served if stem and stem in _norm(m)][:5]
            raise ProviderUnavailable(
                f"model {self.model!r} is not served by {self.url}. "
                f"Similar: {near or 'none'}. `argo-proxy models` lists all."
            )

        self._resolved_model = hit
        print(f"  \033[2m↪ model {self.model!r} → {hit!r} (as served by proxy)\033[0m",
              file=sys.stderr)
        return self._resolved_model

    # ── request construction ────────────────────────────────────────────────

    def _kwargs(self, messages: List[Dict], temperature: float,
                tools: Optional[List[Dict]], model: str) -> Dict[str, Any]:
        """Build create() kwargs. Messages pass through STRUCTURED — no flattening.

        This is the whole point: ``assistant`` turns keep their ``tool_calls`` and
        ``tool`` turns keep their ``tool_call_id``.
        """
        kw: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": DEFAULT_MAX_TOKENS,
            "temperature": temperature,
        }
        if tools:
            # Plain OpenAI function format — exactly what APEXA already stores in
            # _available_tools. The proxy translates per vendor.
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
        for p in self._dropped:
            kw.pop(p, None)
        return kw

    @staticmethod
    def _unsupported_param(err: Exception) -> Optional[str]:
        """Name the sampling parameter a 400 is complaining about, if any."""
        msg = str(err).lower()
        if "400" not in msg and "unsupported" not in msg and "invalid" not in msg:
            return None
        for p in ("temperature", "top_p", "max_completion_tokens", "tool_choice"):
            if p in msg:
                return p
        return None

    # ── response parsing ────────────────────────────────────────────────────

    @staticmethod
    def _to_agent_response(completion) -> "Any":
        from apexa_agents import AgentResponse, ToolCall  # local: avoid circular import

        msg = completion.choices[0].message
        calls: List[ToolCall] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            fn = getattr(tc, "function", None)
            raw = getattr(fn, "arguments", "") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
            except (json.JSONDecodeError, TypeError, ValueError):
                # A malformed argument blob is a real tool call we cannot execute.
                # Surface it as such rather than silently dropping the call.
                args = {"__unparsed_arguments__": raw}
            if not isinstance(args, dict):
                args = {"value": args}
            calls.append(ToolCall(
                id=getattr(tc, "id", "") or f"call_{len(calls)}",
                name=getattr(fn, "name", "") or "",
                arguments=args,
            ))
        return AgentResponse(
            content=(msg.content or ""),
            tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
        )

    # ── public API ──────────────────────────────────────────────────────────

    def _client_for_request(self):
        """Client with an up-to-date credential.

        Re-resolves the key and swaps it only when it actually changed, so an ALCF
        Globus token refreshed on disk is picked up by a running session without a
        restart, and the common case costs nothing.
        """
        try:
            key = api_key_for(self.username)
        except Exception as e:      # a broken token helper must not kill the turn
            print(f"  \033[33m⚠ credential refresh failed ({e}) — reusing last key\033[0m",
                  file=sys.stderr)
            return self._client
        if key and key != self._key:
            self._key = key
            self._client = self._client.with_options(api_key=key)
        return self._client

    async def chat(self, messages: List[Dict],
                   temperature: float = 0.7,
                   tools: Optional[List[Dict]] = None):
        model = await self._resolve_model()
        prompt_tok = count_message_tokens(messages, self.model)
        n_messages = len(messages)

        if os.environ.get("APEXA_DEBUG"):
            print(f"  [debug] proxy {self.url} model={model} "
                  f"msgs={n_messages} tools={len(tools or [])} dropped={sorted(self._dropped)}",
                  file=sys.stderr)

        retries = 3
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            t0 = time.monotonic()
            try:
                completion = await self._client_for_request().chat.completions.create(
                    **self._kwargs(messages, temperature, tools, model))
                elapsed = time.monotonic() - t0
                parsed = self._to_agent_response(completion)
                if os.environ.get("APEXA_SHOW_TIMING"):
                    print(f"  \033[2m⏱ {self.model} responded in {elapsed:.1f}s\033[0m", flush=True)
                if not parsed.content and not parsed.tool_calls:
                    print(f"  \033[33m⚠ empty completion\033[0m (model={model}, "
                          f"user={self.username!r}) via {self.url}", file=sys.stderr)
                log_llm_call(self._timing_record(
                    attempt=attempt, status=200, elapsed=elapsed,
                    prompt_tok=prompt_tok, n_messages=n_messages,
                    temperature=temperature, parsed=parsed,
                    empty=not (parsed.content or parsed.tool_calls)))
                return parsed

            except Exception as e:  # noqa: BLE001 — normalize SDK's exception zoo
                elapsed = time.monotonic() - t0
                last_exc = e
                name = type(e).__name__

                # A rejected sampling parameter is not a failure — learn and retry
                # immediately, without consuming a backoff slot.
                bad = self._unsupported_param(e)
                if bad and bad not in self._dropped:
                    self._dropped.add(bad)
                    print(f"  \033[33m⚠ {model} rejected {bad!r} — dropping it and retrying\033[0m",
                          file=sys.stderr)
                    log_llm_call(self._timing_record(
                        attempt=attempt, status=400, elapsed=elapsed,
                        prompt_tok=prompt_tok, n_messages=n_messages,
                        temperature=temperature, error=f"drop_{bad}"))
                    continue

                transient = name in ("APITimeoutError", "APIConnectionError",
                                     "RateLimitError", "InternalServerError")
                log_llm_call(self._timing_record(
                    attempt=attempt, status=0, elapsed=elapsed,
                    prompt_tok=prompt_tok, n_messages=n_messages,
                    temperature=temperature,
                    error=f"retry_{name}" if (transient and attempt < retries - 1) else name))

                if name == "APIConnectionError" and attempt == 0:
                    print(f"  \033[33m⚠ cannot reach argo-proxy at {self.url} — "
                          f"is `argo-proxy serve` running, and are you on the ANL "
                          f"network/VPN?\033[0m", file=sys.stderr)
                if transient and attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f"  \033[33m⚠ {name}, retrying in {wait}s "
                          f"({attempt + 1}/{retries})\033[0m", file=sys.stderr)
                    await asyncio.sleep(wait)
                    continue
                raise

        if last_exc:
            raise last_exc
        raise ProviderUnavailable(f"no response from {self.url} after {retries} attempts")

    # ── instrumentation ─────────────────────────────────────────────────────

    def _timing_record(self, *, attempt: int, status: int, elapsed: float,
                       prompt_tok: int, n_messages: int, temperature: float,
                       parsed: Optional[Any] = None, empty: bool = False,
                       error: str = "") -> Dict[str, Any]:
        """One JSONL row for ~/.apexa/timing.jsonl. Mirrors ArgoProvider's schema
        so existing analysis in docs/INSTRUMENTATION.md keeps working; only
        ``endpoint`` differs."""
        resp_tok = count_tokens(parsed.content or "", self.model) if parsed else 0
        n_tool_calls = len(parsed.tool_calls) if parsed else 0
        gen_tps = (resp_tok / elapsed) if (elapsed > 0 and resp_tok) else 0.0
        if not error.startswith(("retry_", "drop_")):
            record_llm_call(elapsed_s=elapsed, prompt_tok=prompt_tok,
                            response_tok=resp_tok)
        return {
            "endpoint":     "argo-proxy",
            "query_id":     current_query_id() or "",
            "model":        self.model,
            "attempt":      attempt,
            "http_status":  status,
            "elapsed_s":    round(elapsed, 3),
            "prompt_tok":   prompt_tok,
            "response_tok": resp_tok,
            "gen_tps":      round(gen_tps, 2),
            "n_messages":   n_messages,
            "n_tool_calls": n_tool_calls,
            "temperature":  temperature,
            "empty":        empty,
            "error":        error,
        }

    async def close(self):
        try:
            await self._client.close()
        except Exception:
            pass
