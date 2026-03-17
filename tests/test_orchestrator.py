from __future__ import annotations

import pytest

from src.config.models import OrchestratorConfig, RoutingRule
from src.core.event import Event


def _make_orchestrator_config(rules: list[dict]) -> OrchestratorConfig:
    return OrchestratorConfig(
        default_agent="general_assistant",
        routing_rules=[RoutingRule(**r) for r in rules],
    )


def _make_event(message: str) -> Event:
    return Event(message=message, source="cli", session_id="test_session")


def test_routing_matches_high_priority_rule():
    from unittest.mock import MagicMock, AsyncMock
    from src.core.orchestrator import Orchestrator

    config = _make_orchestrator_config([
        {"pattern": "\\b(research|news)\\b", "agent": "research_agent", "priority": 10},
        {"pattern": ".*", "agent": "general_assistant", "priority": 0},
    ])

    orchestrator = Orchestrator(
        config=config,
        agent_registry=MagicMock(),
        skill_loader=MagicMock(),
        session_manager=MagicMock(),
        anthropic_client=MagicMock(),
        gateway=None,
    )

    assert orchestrator.route(_make_event("research the latest news")) == "research_agent"
    assert orchestrator.route(_make_event("tell me a joke")) == "general_assistant"


def test_routing_falls_back_to_default_when_no_match():
    from unittest.mock import MagicMock
    from src.core.orchestrator import Orchestrator

    config = _make_orchestrator_config([
        {"pattern": "\\bsearch\\b", "agent": "research_agent", "priority": 10},
    ])
    # No catch-all rule, so non-matching falls back to default_agent
    orchestrator = Orchestrator(
        config=config,
        agent_registry=MagicMock(),
        skill_loader=MagicMock(),
        session_manager=MagicMock(),
        anthropic_client=MagicMock(),
        gateway=None,
    )

    assert orchestrator.route(_make_event("what time is it?")) == "general_assistant"


def test_routing_is_case_insensitive():
    from unittest.mock import MagicMock
    from src.core.orchestrator import Orchestrator

    config = _make_orchestrator_config([
        {"pattern": "\\b(SEARCH|search)\\b", "agent": "research_agent", "priority": 10},
        {"pattern": ".*", "agent": "general_assistant", "priority": 0},
    ])

    orchestrator = Orchestrator(
        config=config,
        agent_registry=MagicMock(),
        skill_loader=MagicMock(),
        session_manager=MagicMock(),
        anthropic_client=MagicMock(),
        gateway=None,
    )

    assert orchestrator.route(_make_event("SEARCH for something")) == "research_agent"
    assert orchestrator.route(_make_event("Search for something")) == "research_agent"
