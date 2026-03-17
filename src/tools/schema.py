from __future__ import annotations

from typing import Any

from src.config.models import ToolConfig


def tool_config_to_anthropic(config: ToolConfig) -> dict[str, Any]:
    """Convert a ToolConfig into the Anthropic API tool definition format."""
    return {
        "name": config.name,
        "description": config.description,
        "input_schema": config.parameters if config.parameters else {
            "type": "object",
            "properties": {},
        },
    }
