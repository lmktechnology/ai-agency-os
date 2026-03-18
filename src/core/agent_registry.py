from __future__ import annotations

import logging
from pathlib import Path

from src.config.loader import load_agent_config
from src.config.models import AgentConfig
from src.core.agent import BaseAgent
from src.llm.base import BaseLLMProvider
from src.llm.factory import get_provider
from src.memory.session import SessionManager
from src.skills.loader import SkillLoader
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_ORCHESTRATOR_FILENAME = "orchestrator.yaml"


class AgentRegistry:
    def __init__(self, agents_dir: str, tool_registry: ToolRegistry) -> None:
        self._agents_dir = Path(agents_dir)
        self._tool_registry = tool_registry
        self._configs: dict[str, AgentConfig] = {}

    def load_all(self) -> None:
        """Scan agents/*.yaml, validate, and cache all agent configs."""
        if not self._agents_dir.exists():
            logger.warning("Agents directory not found: %s", self._agents_dir)
            return

        for path in sorted(self._agents_dir.glob("*.yaml")):
            if path.name == _ORCHESTRATOR_FILENAME:
                continue
            try:
                config = load_agent_config(path)
                self._configs[config.name] = config
                self._register_agent_tools(config)
                logger.info(
                    "Loaded agent: %s (provider=%s model=%s)",
                    config.name, config.provider, config.model,
                )
            except Exception as e:
                logger.error("Failed to load agent config %s: %s", path, e)

    def _register_agent_tools(self, config: AgentConfig) -> None:
        for tool_cfg in config.tools:
            if not self._tool_registry.is_registered(tool_cfg.name):
                try:
                    self._tool_registry.register(tool_cfg)
                except Exception as e:
                    logger.error("Could not register tool %s: %s", tool_cfg.name, e)

    def get_config(self, name: str) -> AgentConfig:
        if name not in self._configs:
            raise KeyError(f"Agent not found: {name}. Available: {list(self._configs)}")
        return self._configs[name]

    def list_agents(self) -> list[str]:
        return list(self._configs.keys())

    def build_agent(
        self,
        name: str,
        skill_loader: SkillLoader,
        session_manager: SessionManager,
        provider_registry: dict[str, BaseLLMProvider],
    ) -> BaseAgent:
        config = self.get_config(name)
        provider = get_provider(provider_registry, config.provider)
        executor = ToolExecutor(self._tool_registry)

        return BaseAgent(
            config=config,
            tool_registry=self._tool_registry,
            tool_executor=executor,
            skill_loader=skill_loader,
            session_manager=session_manager,
            provider=provider,
        )
