from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Event:
    message: str
    source: str = "cli"                      # "telegram" | "cron" | "cli"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class AgentResponse:
    event_id: str
    agent_name: str
    content: str
    session_id: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    react_steps: int = 0
    error: str | None = None
