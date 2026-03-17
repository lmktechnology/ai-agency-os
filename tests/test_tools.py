from __future__ import annotations

import pytest

from src.config.models import ToolConfig
from src.tools.registry import ToolRegistry
from src.tools.schema import tool_config_to_anthropic


def _make_tool_config(**kwargs) -> ToolConfig:
    defaults = {
        "name": "get_datetime",
        "description": "Get current time",
        "module": "src.tools.builtins.datetime_tool",
        "function": "get_datetime",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "sandboxed": False,
    }
    defaults.update(kwargs)
    return ToolConfig(**defaults)


def test_registry_register_and_get():
    registry = ToolRegistry()
    config = _make_tool_config()
    registry.register(config)

    assert registry.is_registered("get_datetime")
    fn = registry.get_callable("get_datetime")
    assert callable(fn)


def test_registry_get_anthropic_schemas():
    registry = ToolRegistry()
    config = _make_tool_config()
    registry.register(config)

    schemas = registry.get_anthropic_schemas(["get_datetime"])
    assert len(schemas) == 1
    assert schemas[0]["name"] == "get_datetime"
    assert "input_schema" in schemas[0]


def test_registry_unknown_tool_returns_empty_schema():
    registry = ToolRegistry()
    schemas = registry.get_anthropic_schemas(["nonexistent_tool"])
    assert schemas == []


def test_registry_import_error_raises():
    registry = ToolRegistry()
    config = _make_tool_config(
        name="bad_tool",
        module="does.not.exist",
        function="foo",
    )
    with pytest.raises(Exception):
        registry.register(config)


def test_tool_config_to_anthropic_schema():
    config = _make_tool_config()
    schema = tool_config_to_anthropic(config)
    assert schema["name"] == "get_datetime"
    assert schema["description"] == "Get current time"
    assert "input_schema" in schema


def test_datetime_tool_returns_expected_keys():
    from src.tools.builtins.datetime_tool import get_datetime
    result = get_datetime(timezone="UTC")
    assert "datetime" in result
    assert "date" in result
    assert "time" in result
    assert result["timezone"] == "UTC"


def test_datetime_tool_invalid_timezone_falls_back_to_utc():
    from src.tools.builtins.datetime_tool import get_datetime
    result = get_datetime(timezone="Invalid/Zone")
    assert result["timezone"] == "UTC"
