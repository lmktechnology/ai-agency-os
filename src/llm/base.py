from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRequest:
    """Normalized tool call from the LLM (provider-agnostic)."""
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    text: str                                          # text content (empty if only tool calls)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    stop_reason: str = "end_turn"                      # "end_turn" | "tool_use"
    # Provider-specific assistant turn to append to messages when tool_use
    assistant_turn: dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """
    Abstract base for LLM providers.

    Contract:
    - chat() receives Anthropic-format tool schemas and simple {role, content} messages.
    - Implementations convert formats internally.
    - assistant_turn in LLMResponse is already formatted for this provider's message list.
    - build_tool_result_messages() returns messages in this provider's format.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name for logging."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],   # Anthropic-format: [{name, description, input_schema}]
        max_tokens: int,
        model: str,
    ) -> LLMResponse:
        """
        Call the LLM and return a normalized response.
        tools is always in Anthropic schema format — providers convert internally.
        """

    @abstractmethod
    def build_tool_result_messages(
        self,
        tool_calls: list[ToolCallRequest],
        results: list[str],
    ) -> list[dict[str, Any]]:
        """
        Build the message(s) containing tool results to feed back to the LLM.
        Returns a list because OpenAI uses one message per tool result,
        while Anthropic batches all results into a single user message.
        """
