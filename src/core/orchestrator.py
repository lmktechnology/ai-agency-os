from __future__ import annotations

import logging
import re
from typing import Any

from src.config.models import OrchestratorConfig
from src.core.agent_registry import AgentRegistry
from src.core.event import AgentResponse, Event
from src.gateway.base import BaseMessagingGateway
from src.llm.base import BaseLLMProvider
from src.memory.session import SessionManager
from src.skills.loader import SkillLoader

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        config: OrchestratorConfig,
        agent_registry: AgentRegistry,
        skill_loader: SkillLoader,
        session_manager: SessionManager,
        provider_registry: dict[str, BaseLLMProvider],
        gateway: BaseMessagingGateway | None = None,
    ) -> None:
        self._config = config
        self._agent_registry = agent_registry
        self._skill_loader = skill_loader
        self._session_manager = session_manager
        self._provider_registry = provider_registry
        self._gateway = gateway

        # Pre-compile routing rules sorted by descending priority
        self._rules: list[tuple[re.Pattern, str]] = [
            (re.compile(rule.pattern, re.IGNORECASE), rule.agent)
            for rule in sorted(config.routing_rules, key=lambda r: r.priority, reverse=True)
        ]

    def route(self, event: Event) -> str:
        """Return the agent name that should handle this event."""
        for pattern, agent_name in self._rules:
            if pattern.search(event.message):
                logger.debug("Routing '%s...' → agent '%s'", event.message[:40], agent_name)
                return agent_name
        logger.debug("No rule matched — default agent '%s'", self._config.default_agent)
        return self._config.default_agent

    async def handle_event(self, event: Event) -> AgentResponse:
        """Route event to the appropriate agent and send the response via gateway."""
        agent_name = self.route(event)

        try:
            agent = self._agent_registry.build_agent(
                name=agent_name,
                skill_loader=self._skill_loader,
                session_manager=self._session_manager,
                provider_registry=self._provider_registry,
            )
        except KeyError:
            logger.error("Agent '%s' not found; falling back to default", agent_name)
            agent = self._agent_registry.build_agent(
                name=self._config.default_agent,
                skill_loader=self._skill_loader,
                session_manager=self._session_manager,
                provider_registry=self._provider_registry,
            )

        response = await agent.run(event)

        if response.error:
            logger.error("Agent %s returned error: %s", agent_name, response.error)
            if self._gateway:
                await self._gateway.send_message(
                    event.session_id,
                    "Sorry, I encountered an error. Please try again.",
                )
        elif self._gateway:
            await self._gateway.send_message(event.session_id, response.content)

        return response
