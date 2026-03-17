from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

from src.config.models import ToolConfig
from src.tools.schema import tool_config_to_anthropic

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._configs: dict[str, ToolConfig] = {}
        self._callables: dict[str, Callable] = {}

    def register(self, config: ToolConfig) -> None:
        """Import module.function and register it under config.name."""
        try:
            module = importlib.import_module(config.module)
            fn = getattr(module, config.function)
        except (ImportError, AttributeError) as e:
            logger.error("Failed to load tool %s from %s.%s: %s", config.name, config.module, config.function, e)
            raise

        self._configs[config.name] = config
        self._callables[config.name] = fn
        logger.debug("Registered tool: %s", config.name)

    def get_callable(self, name: str) -> Callable:
        if name not in self._callables:
            raise KeyError(f"Tool not registered: {name}")
        return self._callables[name]

    def get_config(self, name: str) -> ToolConfig:
        if name not in self._configs:
            raise KeyError(f"Tool config not found: {name}")
        return self._configs[name]

    def get_anthropic_schemas(self, tool_names: list[str]) -> list[dict[str, Any]]:
        """Return Anthropic-format tool definitions for the given tool names."""
        schemas = []
        for name in tool_names:
            if name in self._configs:
                schemas.append(tool_config_to_anthropic(self._configs[name]))
            else:
                logger.warning("Tool %s not found in registry — skipping schema", name)
        return schemas

    def is_registered(self, name: str) -> bool:
        return name in self._callables
