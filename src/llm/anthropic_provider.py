from __future__ import annotations

import logging
from typing import Any

import anthropic

from src.llm.base import BaseLLMProvider, LLMResponse, ToolCallRequest

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """LLM provider backed by the Anthropic API (claude-* models)."""

    def __init__(self, client: anthropic.AsyncAnthropic) -> None:
        self._client = client

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        max_tokens: int,
        model: str,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools  # already in Anthropic input_schema format

        response = await self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(ToolCallRequest(
                    id=block.id,
                    name=block.name,
                    input=dict(block.input),
                ))

        stop_reason = "tool_use" if tool_calls else "end_turn"

        return LLMResponse(
            text=" ".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            # Store the full content block list so subsequent calls work correctly.
            # The Anthropic SDK requires tool_use blocks to be present in history.
            assistant_turn={"role": "assistant", "content": response.content},
        )

    def build_tool_result_messages(
        self,
        tool_calls: list[ToolCallRequest],
        results: list[str],
    ) -> list[dict[str, Any]]:
        return [{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result,
                }
                for call, result in zip(tool_calls, results)
            ],
        }]
