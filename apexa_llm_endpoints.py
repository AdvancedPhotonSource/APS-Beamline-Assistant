"""Typed LLM endpoint presets for APEXA.

Follows the pattern used by the sister APS framework
[EAA](https://github.com/AdvancedPhotonSource/EAA)
(`packages/eaa-core/src/eaa_core/api/llm_config.py`): a base OpenAI-compatible
config that vendor presets subclass and merely override defaults on. Typed
dataclasses beat env-var soup — the valid combinations are discoverable, and a
preset can carry the operational facts that matter (does it support tool calling?
how is it authenticated? does the token expire?).

APEXA extends the pattern in two ways EAA does not currently cover:

1. **An ALCF Inference Service preset.** ALCF is the answer to the multi-user
   question: it is OpenAI-compatible, on-prem DOE hardware, and authenticated with
   a **per-user Globus identity** — so each beamline user runs under their own
   credential rather than a shared beamline account.
2. **Lazy credential resolution.** ALCF access tokens expire (48 h, and a 30-day
   re-auth policy), so a key frozen at construction goes stale mid-experiment.
   Credentials resolve per request, optionally by shelling out to ALCF's own
   refreshing helper, with a short cache so it isn't invoked on every call.

A preset only sets *defaults*; any field can still be overridden by the matching
``APEXA_LLM_*`` environment variable, so an operator is never boxed in.

Selected with ``APEXA_LLM_PRESET``:

    argo-proxy    local argo-proxy sidecar (default)      ANL username as key
    alcf-sophia   ALCF Inference Service, Sophia cluster  Globus token, per-user
    alcf-metis    ALCF Metis cluster                      REJECTED: no tool calling
    openai        api.openai.com                          personal API key
    anthropic     api.anthropic.com                       personal API key
    custom        anything OpenAI-compatible (vLLM, …)    APEXA_LLM_API_KEY
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# Cache shell-fetched tokens briefly: ALCF's helper does a Globus round trip, far
# too slow to run on every request, but tokens are short-lived so we must not
# cache them for long either.
_TOKEN_TTL_S = 600
_token_cache: dict[str, tuple[float, str]] = {}


class EndpointRejected(RuntimeError):
    """The endpoint cannot support APEXA (e.g. no tool calling)."""


@dataclass
class OpenAIEndpoint:
    """Base: any OpenAI-compatible ``/v1/chat/completions`` service."""

    name: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    #: Static credential. When None, resolution falls back to token_cmd → ANL username.
    api_key: Optional[str] = None
    #: Shell command printing a fresh bearer token on stdout (for expiring creds).
    token_cmd: Optional[str] = None
    #: False ⇒ APEXA refuses the endpoint: structured tool calling is mandatory.
    tool_calling: bool = True
    #: Operator-facing note surfaced at startup.
    notes: str = "Commercial OpenAI API — needs a personal API key and outbound egress."

    # ── credential resolution (lazy, so expiring tokens stay fresh) ─────────

    def resolve_key(self, username: str = "") -> str:
        """Credential for this request.

        Precedence: ``APEXA_LLM_API_KEY`` → preset ``api_key`` → ``token_cmd``
        output (cached) → ANL username (the argo-proxy convention).
        """
        explicit = (os.environ.get("APEXA_LLM_API_KEY") or "").strip()
        if explicit:
            return explicit
        if self.api_key:
            return self.api_key
        cmd = (os.environ.get("APEXA_LLM_TOKEN_CMD") or self.token_cmd or "").strip()
        if cmd:
            return self._token_from_cmd(cmd)
        return (username or "apexa").strip()

    @staticmethod
    def _token_from_cmd(cmd: str) -> str:
        now = time.monotonic()
        hit = _token_cache.get(cmd)
        if hit and (now - hit[0]) < _TOKEN_TTL_S:
            return hit[1]
        try:
            out = subprocess.run(shlex.split(cmd), capture_output=True, text=True,
                                 timeout=120, check=True).stdout.strip()
        except subprocess.CalledProcessError as e:
            raise EndpointRejected(
                f"token command failed ({cmd!r}): {(e.stderr or '').strip()[:200]}. "
                f"For ALCF, re-authenticate: `python inference_auth_token.py authenticate --force`."
            ) from e
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            raise EndpointRejected(f"token command unusable ({cmd!r}): {e}") from e
        if not out:
            raise EndpointRejected(f"token command {cmd!r} printed nothing.")
        # Tolerate helpers that print log lines before the token.
        token = out.splitlines()[-1].strip()
        _token_cache[cmd] = (now, token)
        return token

    def describe(self) -> str:
        return f"{self.name} → {self.base_url}"


@dataclass
class ArgoProxyEndpoint(OpenAIEndpoint):
    """Local argo-proxy sidecar fronting the Argo Gateway.

    Authenticates with the ANL **username** in the api_key slot — there is no
    secret, which is exactly why the sidecar must bind 127.0.0.1
    (see docs/ARGO_PROXY_DEPLOYMENT.md).
    """

    name: str = "argo-proxy"
    base_url: str = "http://127.0.0.1:44497/v1"
    tool_calling: bool = True
    notes: str = ("Argo via local sidecar. Shared ANL identity — inference is "
                  "attributed to whoever's username the proxy is configured with.")


@dataclass
class ALCFSophiaEndpoint(OpenAIEndpoint):
    """ALCF Inference Service, Sophia cluster (vLLM on A100).

    The multi-user answer: OpenAI-compatible, on-prem DOE hardware, and
    authenticated per user via Globus, so each beamline user runs under their own
    identity instead of a shared beamline account.

    Only models flagged **T** in the ALCF docs support tool calling — Llama 3.1/3.3/4,
    ``openai/gpt-oss-20b``/``-120b``, Gemma 3/4, Trinity, nemotron-3-super. APEXA is
    non-functional on a model without it.

    Cold starts run 10–15 min: only some nodes keep models hot and they unload after
    ~2 h idle. Prefer an always-hot (**H**) model for interactive beamline use.
    """

    name: str = "alcf-sophia"
    base_url: str = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    token_cmd: str = "python inference_auth_token.py get_access_token"
    tool_calling: bool = True
    notes: str = ("ALCF Inference Service (Sophia). Per-user Globus identity; tokens "
                  "expire after 48h and are refreshed by the token command. Use only "
                  "T-flagged (tool-calling) models. Cold start can be 10-15 min.")


@dataclass
class ALCFMetisEndpoint(OpenAIEndpoint):
    """ALCF Metis (SambaNova SN40L) — **unusable by APEXA**.

    ALCF documents that tool calling is not supported on Metis, with a known
    SambaNova sanitization bug producing ``Invalid function calling output.``
    Every APEXA capability is a tool call, so this is rejected up front rather
    than failing confusingly at the first query.
    """

    name: str = "alcf-metis"
    base_url: str = "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1"
    token_cmd: str = "python inference_auth_token.py get_access_token"
    tool_calling: bool = False
    notes: str = "Metis does not support tool calling — use alcf-sophia instead."


@dataclass
class AnthropicEndpoint(OpenAIEndpoint):
    name: str = "anthropic"
    base_url: str = "https://api.anthropic.com/v1"
    notes: str = ("Commercial Anthropic API — needs a personal API key (a Claude Pro "
                  "subscription is NOT API access) and outbound egress. Review data "
                  "governance before sending beamline paths to a third party.")


@dataclass
class CustomEndpoint(OpenAIEndpoint):
    """Any other OpenAI-compatible server (self-hosted vLLM, Ollama, …)."""

    name: str = "custom"
    base_url: str = "http://127.0.0.1:8000/v1"
    notes: str = "Custom OpenAI-compatible endpoint from APEXA_LLM_BASE_URL."


PRESETS: dict[str, type[OpenAIEndpoint]] = {
    "argo-proxy":  ArgoProxyEndpoint,
    "alcf-sophia": ALCFSophiaEndpoint,
    "alcf-metis":  ALCFMetisEndpoint,
    "openai":      OpenAIEndpoint,
    "anthropic":   AnthropicEndpoint,
    "custom":      CustomEndpoint,
}

DEFAULT_PRESET = "argo-proxy"


def active_endpoint() -> OpenAIEndpoint:
    """Resolve the configured endpoint.

    ``APEXA_LLM_PRESET`` picks the preset; ``APEXA_LLM_BASE_URL`` overrides its URL.
    An endpoint that cannot do tool calling is rejected here — APEXA has no
    non-tool mode, so failing at configuration time is far kinder than failing at
    the first query.
    """
    name = (os.environ.get("APEXA_LLM_PRESET") or DEFAULT_PRESET).strip().lower()
    cls = PRESETS.get(name)
    if cls is None:
        raise EndpointRejected(
            f"unknown APEXA_LLM_PRESET {name!r}. Choose one of: "
            f"{', '.join(sorted(PRESETS))}."
        )
    ep = cls()
    override = (os.environ.get("APEXA_LLM_BASE_URL") or "").strip()
    if override:
        ep.base_url = override
    if not ep.tool_calling:
        raise EndpointRejected(
            f"{ep.name} cannot be used by APEXA: {ep.notes} "
            f"Every APEXA capability is a tool call."
        )
    return ep


def describe_active() -> str:
    try:
        ep = active_endpoint()
    except EndpointRejected as e:
        return f"(unusable: {e})"
    return f"{ep.name} → {ep.base_url}"
