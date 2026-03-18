from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.base import LLMResponse, ToolCallRequest
from src.llm.openai_provider import OpenAICompatibleProvider, _anthropic_to_openai_tool
from src.llm.factory import build_provider_registry, get_provider


# ---------------------------------------------------------------------------
# Schema conversion helpers
# ---------------------------------------------------------------------------

def test_anthropic_to_openai_tool_conversion():
    anthropic_tool = {
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }
    result = _anthropic_to_openai_tool(anthropic_tool)

    assert result["type"] == "function"
    assert result["function"]["name"] == "get_weather"
    assert result["function"]["description"] == "Get weather for a location"
    assert result["function"]["parameters"]["type"] == "object"
    assert "location" in result["function"]["parameters"]["properties"]


def test_anthropic_to_openai_tool_missing_input_schema():
    """Tool with no input_schema gets an empty object schema."""
    tool = {"name": "simple_tool", "description": "Does something"}
    result = _anthropic_to_openai_tool(tool)
    assert result["function"]["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# OpenAI provider — tool result message format
# ---------------------------------------------------------------------------

def test_openai_provider_tool_result_messages():
    provider = OpenAICompatibleProvider(api_key="test", name="openai")
    calls = [
        ToolCallRequest(id="call_1", name="get_weather", input={"location": "Paris"}),
        ToolCallRequest(id="call_2", name="get_time", input={}),
    ]
    results = ['{"temp": "20C"}', '{"time": "12:00"}']

    msgs = provider.build_tool_result_messages(calls, results)

    assert len(msgs) == 2
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "call_1"
    assert msgs[0]["content"] == '{"temp": "20C"}'
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["tool_call_id"] == "call_2"


# ---------------------------------------------------------------------------
# Anthropic provider — tool result message format
# ---------------------------------------------------------------------------

def test_anthropic_provider_tool_result_messages():
    from src.llm.anthropic_provider import AnthropicProvider
    provider = AnthropicProvider(client=MagicMock())
    calls = [
        ToolCallRequest(id="tu_1", name="web_search", input={"query": "AI news"}),
    ]
    results = ['{"results": []}']

    msgs = provider.build_tool_result_messages(calls, results)

    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "tool_result"
    assert content[0]["tool_use_id"] == "tu_1"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_build_provider_registry_anthropic_only():
    registry = build_provider_registry(anthropic_api_key="sk-ant-test")
    assert "anthropic" in registry
    assert len(registry) == 1


def test_build_provider_registry_multiple():
    registry = build_provider_registry(
        anthropic_api_key="sk-ant-test",
        openai_api_key="sk-oai-test",
        deepseek_api_key="sk-ds-test",
    )
    assert "anthropic" in registry
    assert "openai" in registry
    assert "deepseek" in registry


def test_build_provider_registry_empty_raises():
    with pytest.raises(ValueError, match="No LLM providers configured"):
        build_provider_registry()


def test_get_provider_returns_correct():
    registry = build_provider_registry(anthropic_api_key="sk-ant-test")
    provider = get_provider(registry, "anthropic")
    assert provider.provider_name == "anthropic"


def test_get_provider_falls_back_gracefully():
    registry = build_provider_registry(anthropic_api_key="sk-ant-test")
    # Request a provider that isn't configured — should fall back silently
    provider = get_provider(registry, "openai")
    assert provider.provider_name == "anthropic"  # falls back to first available
