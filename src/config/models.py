from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    name: str
    description: str
    module: str
    function: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    sandboxed: bool = True


class SkillRef(BaseModel):
    name: str


class MemoryConfig(BaseModel):
    enabled: bool = True
    vector_store: bool = False
    flush_interval_seconds: int = 30


class AgentConfig(BaseModel):
    name: str
    description: str = ""
    system_prompt: str = ""
    skills: list[SkillRef] = Field(default_factory=list)
    tools: list[ToolConfig] = Field(default_factory=list)
    # LLM provider selection — must match a key in the provider registry
    # Options: "anthropic" | "openai" | "openrouter" | "deepseek" | "ollama"
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 8096
    max_react_iterations: int = 10
    memory: MemoryConfig = Field(default_factory=MemoryConfig)


class RoutingRule(BaseModel):
    pattern: str
    agent: str
    priority: int = 0


class CronJobConfig(BaseModel):
    id: str
    cron: str
    agent: str
    message: str
    reply_chat_id: str | None = None


class HeartbeatConfig(BaseModel):
    """
    Interval-based autonomous heartbeat.
    Fires an agent every `interval_seconds` so it can proactively monitor
    state and act without any human prompting.

    Fields:
        id                  — unique identifier for this heartbeat
        interval_seconds    — how often to fire (min: 10s)
        agent               — which agent runs the heartbeat task
        message             — the prompt sent to the agent each tick
        reply_chat_id       — if set, non-empty agent responses are forwarded here
        silent_on_noop      — if True, responses starting with "NOOP" are suppressed
        startup_delay_seconds — delay before the first tick (default 5s)
        enabled             — set to false to disable without removing the config
    """
    id: str
    interval_seconds: int
    agent: str
    message: str
    reply_chat_id: str | None = None
    silent_on_noop: bool = True
    startup_delay_seconds: int = 5
    enabled: bool = True


class OrchestratorConfig(BaseModel):
    name: str = "orchestrator"
    default_agent: str
    routing_rules: list[RoutingRule] = Field(default_factory=list)
    cron_jobs: list[CronJobConfig] = Field(default_factory=list)
    heartbeats: list[HeartbeatConfig] = Field(default_factory=list)
    parallel_dispatch: bool = False
