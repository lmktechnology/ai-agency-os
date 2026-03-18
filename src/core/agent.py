from __future__ import annotations

import logging
from typing import Any

from src.config.models import AgentConfig
from src.core.event import AgentResponse, Event
from src.llm.base import BaseLLMProvider, ToolCallRequest
from src.memory.session import SessionManager
from src.skills.loader import SkillLoader
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class BaseAgent:
    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        skill_loader: SkillLoader,
        session_manager: SessionManager,
        provider: BaseLLMProvider,
    ) -> None:
        self._config = config
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._skill_loader = skill_loader
        self._session_manager = session_manager
        self._provider = provider

    @property
    def name(self) -> str:
        return self._config.name

    async def run(self, event: Event) -> AgentResponse:
        """Entry point: loads session, builds system prompt, runs ReAct loop."""
        session_id = event.session_id
        self._session_manager.get_or_create(session_id, source=event.source)

        system_prompt = self._build_system_prompt(session_id)
        history = self._session_manager.load_history(session_id, last_n=20)

        messages = [_strip_metadata(t) for t in history]
        messages.append({"role": "user", "content": event.message})

        try:
            final_text, tool_call_log, steps = await self._react_loop(
                messages=messages,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.exception("Agent %s failed for session %s", self.name, session_id)
            return AgentResponse(
                event_id=event.id,
                agent_name=self.name,
                content="",
                session_id=session_id,
                error=str(e),
            )

        # Persist the turn (plain text only — no provider-specific blocks)
        self._session_manager.append_turn(session_id, {
            "role": "user",
            "content": event.message,
        })
        self._session_manager.append_turn(session_id, {
            "role": "assistant",
            "content": final_text,
        })

        return AgentResponse(
            event_id=event.id,
            agent_name=self.name,
            content=final_text,
            session_id=session_id,
            tool_calls=tool_call_log,
            react_steps=steps,
        )

    def _build_system_prompt(self, session_id: str) -> str:
        parts = [self._config.system_prompt.strip()]

        skill_names = [s.name for s in self._config.skills]
        if skill_names:
            skills_text = self._skill_loader.load_many(skill_names)
            if skills_text:
                parts.append(f"\n\n---\n# Skills\n{skills_text}")

        memory_ctx = self._session_manager.get_memory_context(session_id)
        if memory_ctx:
            parts.append(f"\n\n---\n# Memory (previous session summary)\n{memory_ctx}")

        return "".join(parts)

    async def _react_loop(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
    ) -> tuple[str, list[dict[str, Any]], int]:
        """
        Provider-agnostic ReAct loop.
        Returns (final_text, tool_call_log, iterations).
        """
        tool_names = [t.name for t in self._config.tools]
        # Tools are always fetched in Anthropic schema format; providers convert internally
        tools = self._tool_registry.get_anthropic_schemas(tool_names) if tool_names else []

        tool_call_log: list[dict[str, Any]] = []
        iterations = 0
        final_text = ""
        current_messages = list(messages)

        while iterations < self._config.max_react_iterations:
            iterations += 1

            response = await self._provider.chat(
                messages=current_messages,
                system=system_prompt,
                tools=tools,
                max_tokens=self._config.max_tokens,
                model=self._config.model,
            )

            logger.debug(
                "Agent %s [%s] iteration %d: stop_reason=%s tools_requested=%d",
                self.name, self._provider.provider_name,
                iterations, response.stop_reason, len(response.tool_calls),
            )

            if response.text:
                final_text = response.text

            if response.stop_reason == "end_turn" or not response.tool_calls:
                break

            # Append assistant turn (provider-specific format) to history
            current_messages.append(response.assistant_turn)

            # Execute all requested tool calls
            results: list[str] = []
            for call in response.tool_calls:
                result = await self._execute_tool(call)
                tool_call_log.append({
                    "tool": call.name,
                    "input": call.input,
                    "result": result,
                    "iteration": iterations,
                })
                results.append(result)

            # Append tool results in provider-specific format
            tool_result_msgs = self._provider.build_tool_result_messages(
                response.tool_calls, results
            )
            current_messages.extend(tool_result_msgs)

        if iterations >= self._config.max_react_iterations and not final_text:
            logger.warning(
                "Agent %s hit max_react_iterations (%d)",
                self.name, self._config.max_react_iterations,
            )
            final_text = (
                "I reached the maximum number of reasoning steps. "
                "Please try a more specific question."
            )

        return final_text, tool_call_log, iterations

    async def _execute_tool(self, call: ToolCallRequest) -> str:
        config = self._tool_registry.get_config(call.name)
        logger.debug("Executing tool %s with input: %s", call.name, call.input)
        return await self._tool_executor.execute(
            name=call.name,
            input=call.input,
            sandboxed=config.sandboxed,
        )


def _strip_metadata(turn: dict[str, Any]) -> dict[str, Any]:
    """Remove internal _ts keys before passing to any LLM API."""
    return {k: v for k, v in turn.items() if not k.startswith("_")}
