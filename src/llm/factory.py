from __future__ import annotations

import logging

import anthropic

from src.llm.anthropic_provider import AnthropicProvider
from src.llm.base import BaseLLMProvider
from src.llm.openai_provider import OpenAICompatibleProvider

logger = logging.getLogger(__name__)


def build_provider_registry(
    anthropic_api_key: str = "",
    openai_api_key: str = "",
    openrouter_api_key: str = "",
    deepseek_api_key: str = "",
    ollama_base_url: str = "",
) -> dict[str, BaseLLMProvider]:
    """
    Build a registry of all configured LLM providers.
    Only providers with a set API key (or URL for Ollama) are registered.
    Returns a dict mapping provider name → BaseLLMProvider instance.
    """
    registry: dict[str, BaseLLMProvider] = {}

    if anthropic_api_key:
        client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        registry["anthropic"] = AnthropicProvider(client)
        logger.info("Registered LLM provider: anthropic")

    if openai_api_key:
        registry["openai"] = OpenAICompatibleProvider(
            api_key=openai_api_key,
            name="openai",
        )
        logger.info("Registered LLM provider: openai")

    if openrouter_api_key:
        registry["openrouter"] = OpenAICompatibleProvider(
            api_key=openrouter_api_key,
            name="openrouter",
        )
        logger.info("Registered LLM provider: openrouter")

    if deepseek_api_key:
        registry["deepseek"] = OpenAICompatibleProvider(
            api_key=deepseek_api_key,
            name="deepseek",
        )
        logger.info("Registered LLM provider: deepseek")

    if ollama_base_url:
        # Ollama uses OpenAI-compatible API with no API key required
        registry["ollama"] = OpenAICompatibleProvider(
            api_key="ollama",  # placeholder — Ollama ignores the key
            name="ollama",
            base_url=ollama_base_url,
        )
        logger.info("Registered LLM provider: ollama (%s)", ollama_base_url)

    if not registry:
        raise ValueError(
            "No LLM providers configured. Set at least ANTHROPIC_API_KEY in .env"
        )

    return registry


def get_provider(
    registry: dict[str, BaseLLMProvider],
    name: str,
) -> BaseLLMProvider:
    """
    Return the named provider, falling back to the first available one.
    Logs a warning if the requested provider is not configured.
    """
    if name in registry:
        return registry[name]

    fallback = next(iter(registry.values()))
    logger.warning(
        "Provider '%s' not configured — falling back to '%s'. "
        "Set the corresponding API key in .env to use '%s'.",
        name, fallback.provider_name, name,
    )
    return fallback
