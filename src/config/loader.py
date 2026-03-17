from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError

from src.config.models import AgentConfig, OrchestratorConfig


def load_agent_config(path: Union[str, Path]) -> AgentConfig:
    """Load and validate an agent YAML config file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    try:
        return AgentConfig.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid agent config at {path}: {e}") from e


def load_orchestrator_config(path: Union[str, Path]) -> OrchestratorConfig:
    """Load and validate the orchestrator YAML config file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    # Expand env vars in string fields (e.g. ${TELEGRAM_ADMIN_CHAT_ID})
    data = _expand_env_vars(data)
    try:
        return OrchestratorConfig.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid orchestrator config at {path}: {e}") from e


def _expand_env_vars(obj: object) -> object:
    """Recursively expand ${VAR} placeholders in string values."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj
