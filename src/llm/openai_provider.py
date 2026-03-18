from __future__ import annotations

import json
import logging
from typing import Any

import openai

from src.llm.base import BaseLLMProvider, LLMResponse, ToolCallRequest

logger = logging.getLogger(__name__)

# Provider base URLs for OpenAI-compatible APIs
PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
}


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    LLM provider for any OpenAI-compatible API.
    Covers: OpenAI (GPT-4o etc.), OpenRouter (multi-model), DeepSeek.
    All share the same OpenAI SDK — only base_url and api_key differ.
    """

    def __init__(
        self,
        api_key: str,
        name: str = "openai",
        base_url: str | None = None,
    ) -> None:
        self._name = name
        resolved_url = base_url or PROVIDER_BASE_URLS.get(name)
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=resolved_url,
        )

    @property
    def provider_name(self) -> str:
        return self._name

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        max_tokens: int,
        model: str,
    ) -> LLMResponse:
        # OpenAI uses system as the first message in the messages list
        openai_messages: list[dict[str, Any]] = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(messages)

        # Convert Anthropic-format tools → OpenAI function-calling format
        openai_tools = [_anthropic_to_openai_tool(t) for t in tools] if tools else []

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": openai_messages,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = await self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        text = (message.content or "").strip()
        tool_calls: list[ToolCallRequest] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    parsed_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    parsed_input = {}
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    input=parsed_input,
                ))

        stop_reason = "tool_use" if tool_calls else "end_turn"

        # Build assistant turn in OpenAI format for subsequent loop iterations
        assistant_turn: dict[str, Any] = {
            "role": "assistant",
            "content": text or None,
        }
        if message.tool_calls:
            assistant_turn["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            assistant_turn=assistant_turn,
        )

    def build_tool_result_messages(
        self,
        tool_calls: list[ToolCallRequest],
        results: list[str],
    ) -> list[dict[str, Any]]:
        # OpenAI expects one tool message per tool call
        return [
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            }
            for call, result in zip(tool_calls, results)
        ]


def _anthropic_to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert Anthropic tool schema format → OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }
