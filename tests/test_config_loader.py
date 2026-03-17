from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.loader import load_agent_config, load_orchestrator_config
from src.config.models import AgentConfig, OrchestratorConfig


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)


def test_load_minimal_agent_config(tmp_path):
    data = {
        "name": "test_agent",
        "description": "A test agent",
        "system_prompt": "You are a test assistant.",
    }
    p = tmp_path / "test_agent.yaml"
    _write_yaml(p, data)

    config = load_agent_config(p)
    assert isinstance(config, AgentConfig)
    assert config.name == "test_agent"
    assert config.model == "claude-sonnet-4-6"
    assert config.tools == []
    assert config.skills == []


def test_load_agent_config_with_tools(tmp_path):
    data = {
        "name": "tool_agent",
        "system_prompt": "Test.",
        "tools": [
            {
                "name": "get_datetime",
                "description": "Get the time",
                "module": "src.tools.builtins.datetime_tool",
                "function": "get_datetime",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ],
    }
    p = tmp_path / "tool_agent.yaml"
    _write_yaml(p, data)

    config = load_agent_config(p)
    assert len(config.tools) == 1
    assert config.tools[0].name == "get_datetime"
    assert config.tools[0].sandboxed is True  # default


def test_load_orchestrator_config(tmp_path):
    data = {
        "default_agent": "general_assistant",
        "routing_rules": [
            {"pattern": "\\bsearch\\b", "agent": "research_agent", "priority": 10},
            {"pattern": ".*", "agent": "general_assistant", "priority": 0},
        ],
        "cron_jobs": [],
    }
    p = tmp_path / "orchestrator.yaml"
    _write_yaml(p, data)

    config = load_orchestrator_config(p)
    assert isinstance(config, OrchestratorConfig)
    assert config.default_agent == "general_assistant"
    assert len(config.routing_rules) == 2


def test_invalid_agent_config_raises(tmp_path):
    data = {"description": "Missing name field"}  # name is required
    p = tmp_path / "bad.yaml"
    _write_yaml(p, data)

    with pytest.raises(ValueError, match="Invalid agent config"):
        load_agent_config(p)
