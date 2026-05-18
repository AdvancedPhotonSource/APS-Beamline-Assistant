"""
AutoGen single-agent baseline for APEXA-Bench evaluation.

Wraps AutoGen v0.7's AssistantAgent around an Argo-Gateway ChatCompletionClient
and reuses APEXA's text-based TOOL_CALL: protocol (Argo strips native
tool_calls, so any framework on top of Argo must use text parsing).

Used as `--config autogen` in benchmark/eval_harness.py.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)

from apexa_agents import ArgoProvider
from interaction_logger import InteractionEntry, ToolCallRecord


_TOOL_CALL_RE = re.compile(
    r"TOOL_CALL:\s*(\S+)\s*\n\s*ARGUMENTS:\s*(\{.*?\})\s*(?:\n|$)",
    re.DOTALL,
)


class _ArgoAutoGenClient(ChatCompletionClient):
    """Minimal AutoGen ChatCompletionClient that calls Argo Gateway.

    Reports function_calling=False because Argo strips native tool_calls;
    APEXA's text-based TOOL_CALL: protocol is handled by the outer loop, not
    by AutoGen's tool machinery.
    """

    def __init__(self, model: str, username: str):
        self._model = model
        self._username = username
        self._total_prompt = 0
        self._total_completion = 0
        self._last_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        family = ModelFamily.UNKNOWN
        if model.startswith("claude"):
            family = ModelFamily.CLAUDE_3_5_SONNET
        elif model.startswith("gpt"):
            family = ModelFamily.GPT_4O
        elif model.startswith("gemini"):
            family = ModelFamily.GEMINI_2_0_FLASH
        self._info: ModelInfo = {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": family,
            "structured_output": False,
            "multiple_system_messages": True,
        }

    @property
    def model_info(self) -> ModelInfo:
        return self._info

    @property
    def capabilities(self):
        return self._info

    @staticmethod
    def _to_argo(messages: Sequence[LLMMessage]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            if isinstance(m, SystemMessage):
                out.append({"role": "system", "content": content})
            elif isinstance(m, UserMessage):
                out.append({"role": "user", "content": content})
            elif isinstance(m, AssistantMessage):
                out.append({"role": "assistant", "content": content})
            else:
                # FunctionExecutionResultMessage etc — squash into user role
                out.append({"role": "user", "content": content})
        return out

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools=(),
        tool_choice="auto",
        json_output=None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        argo_msgs = self._to_argo(messages)
        provider = ArgoProvider(self._username, self._model)
        try:
            resp = await provider.chat(argo_msgs, temperature=0.5)
        finally:
            await provider.close()
        text = resp.content or ""
        # Approximate token usage so AutoGen's bookkeeping doesn't error.
        approx_prompt = sum(len(m.get("content", "")) for m in argo_msgs) // 4
        approx_completion = max(1, len(text) // 4)
        self._last_usage = RequestUsage(
            prompt_tokens=approx_prompt, completion_tokens=approx_completion,
        )
        self._total_prompt += approx_prompt
        self._total_completion += approx_completion
        return CreateResult(
            finish_reason="stop",
            content=text,
            usage=self._last_usage,
            cached=False,
        )

    async def create_stream(self, *args, **kwargs):
        result = await self.create(*args, **kwargs)
        yield result

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._last_usage

    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=self._total_prompt,
                             completion_tokens=self._total_completion)

    def count_tokens(self, messages, *, tools=()) -> int:
        return sum(len(getattr(m, "content", "")) for m in messages) // 4

    def remaining_tokens(self, messages, *, tools=()) -> int:
        return 100_000


def _build_system_message(tool_names: List[str]) -> str:
    tool_list = ", ".join(tool_names) if tool_names else "(none registered)"
    return (
        "You are a beamline assistant operating an autonomous agent. "
        "When the user's request requires running a tool, emit:\n"
        "TOOL_CALL: <tool_name>\n"
        "ARGUMENTS: {\"key\": value, ...}\n"
        "Use exactly that format on its own lines. Wait for the tool result "
        "(provided in the next user message) before continuing. Reply with "
        "TERMINATE on a line by itself when the task is complete and no more "
        "tool calls are needed.\n\n"
        f"Available tools: {tool_list}"
    )


async def run_autogen_task(
    query: str,
    model: str,
    available_tools: List[Dict],
    execute_tool_fn: Callable,
    log_entry: InteractionEntry,
    max_iterations: int = 6,
) -> str:
    """Run a single APEXA-Bench task through an AutoGen AssistantAgent.

    Returns the final text response. Records every TOOL_CALL execution in
    log_entry so the harness scorer sees the same shape it gets from the
    keyword/dspy paths.
    """
    username = os.getenv("ANL_USERNAME")
    if not username:
        raise RuntimeError("ANL_USERNAME must be set for AutoGen baseline")

    tool_names = [t.get("name") or t.get("function", {}).get("name", "")
                  for t in available_tools]
    tool_names = [n for n in tool_names if n]

    client = _ArgoAutoGenClient(model=model, username=username)
    agent = AssistantAgent(
        name="apexa_autogen_baseline",
        model_client=client,
        system_message=_build_system_message(tool_names),
        # No native tools — TOOL_CALL is handled by the outer loop below.
        reflect_on_tool_use=False,
    )

    transcript = query
    final_text = ""
    for _iter in range(max_iterations):
        response = await agent.on_messages(
            [TextMessage(content=transcript, source="user")],
            cancellation_token=CancellationToken(),
        )
        msg = response.chat_message
        text = getattr(msg, "content", "") if msg is not None else ""
        if not isinstance(text, str):
            text = str(text)
        final_text = text

        m = _TOOL_CALL_RE.search(text)
        if not m:
            break

        tool_name = m.group(1).strip()
        try:
            import json as _json
            args = _json.loads(m.group(2))
        except Exception as exc:
            args = {}
            log_entry.tool_calls.append(ToolCallRecord(
                name=tool_name, arguments={}, success=False,
                duration_ms=0, result_preview=f"argument parse error: {exc}",
            ))
            transcript = f"Tool {tool_name} arguments could not be parsed. Please retry with valid JSON."
            continue

        t0 = time.monotonic()
        try:
            tool_result = await execute_tool_fn(tool_name, args)
            success = True
        except Exception as exc:
            tool_result = f"Error: {exc}"
            success = False
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        result_str = (str(tool_result) if not isinstance(tool_result, str)
                      else tool_result)
        log_entry.tool_calls.append(ToolCallRecord(
            name=tool_name, arguments=args, success=success,
            duration_ms=elapsed_ms, result_preview=result_str[:300],
        ))

        # Truncate long tool results for the next prompt to stay in budget.
        truncated = result_str if len(result_str) <= 4000 else result_str[:4000] + " ...[truncated]"
        transcript = (
            f"TOOL_RESULT for {tool_name}:\n{truncated}\n\n"
            "Continue with the next TOOL_CALL or reply TERMINATE if done."
        )

        if "TERMINATE" in text:
            break

    await client.close()
    return final_text
