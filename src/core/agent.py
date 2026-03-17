from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from src.config.models import AgentConfig
from src.core.event import AgentResponse, Event
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
        anthropic_client: anthropic.AsyncAnthropic,
    ) -> None:
        self._config = config
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._skill_loader = skill_loader
        self._session_manager = session_manager
        self._client = anthropic_client

    @property
    def name(self) -> str:
        return self._config.name

    async def run(self, event: Event) -> AgentResponse:
        """Entry point: loads session, builds system prompt, runs ReAct loop."""
        session_id = event.session_id
        self._session_manager.get_or_create(session_id, source=event.source)

        system_prompt = self._build_system_prompt(session_id)

        # Load prior conversation history (for multi-turn context)
        history = self._session_manager.load_history(session_id, last_n=20)

        # Strip internal metadata (_ts) from history before passing to API
        messages = [_strip_metadata(t) for t in history]
        messages.append({"role": "user", "content": event.message})

        try:
            final_text, tool_call_log, steps = await self._react_loop(
                messages=messages,
                system_prompt=system_prompt,
                session_id=session_id,
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

        # Persist the full turn
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

        # Inject skill content
        skill_names = [s.name for s in self._config.skills]
        if skill_names:
            skills_text = self._skill_loader.load_many(skill_names)
            if skills_text:
                parts.append(f"\n\n---\n# Skills\n{skills_text}")

        # Inject long-term memory summary
        memory_ctx = self._session_manager.get_memory_context(session_id)
        if memory_ctx:
            parts.append(f"\n\n---\n# Memory (previous session summary)\n{memory_ctx}")

        return "".join(parts)

    async def _react_loop(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        session_id: str,
    ) -> tuple[str, list[dict[str, Any]], int]:
        """
        Core ReAct loop using Anthropic tool_use blocks.
        Returns (final_text, tool_call_log, iterations).
        """
        tool_names = [t.name for t in self._config.tools]
        tools = self._tool_registry.get_anthropic_schemas(tool_names) if tool_names else []

        tool_call_log: list[dict[str, Any]] = []
        iterations = 0
        final_text = ""

        current_messages = list(messages)

        while iterations < self._config.max_react_iterations:
            iterations += 1

            kwargs: dict[str, Any] = {
                "model": self._config.model,
                "max_tokens": self._config.max_tokens,
                "system": system_prompt,
                "messages": current_messages,
            }
            if tools:
                kwargs["tools"] = tools

            response = await self._client.messages.create(**kwargs)

            stop_reason = response.stop_reason
            logger.debug(
                "Agent %s iteration %d: stop_reason=%s",
                self.name, iterations, stop_reason
            )

            # Extract text and tool_use blocks from response
            text_parts: list[str] = []
            tool_use_blocks: list[anthropic.types.ToolUseBlock] = []

            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif hasattr(block, "type") and block.type == "tool_use":
                    tool_use_blocks.append(block)

            if text_parts:
                final_text = " ".join(text_parts).strip()

            if stop_reason == "end_turn" or not tool_use_blocks:
                break

            # Append assistant message with all content blocks
            current_messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # Execute all tool_use blocks and collect results
            tool_results = []
            for block in tool_use_blocks:
                result_content = await self._execute_tool(block)
                tool_call_log.append({
                    "tool": block.name,
                    "input": block.input,
                    "result": result_content,
                    "iteration": iterations,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                })

            # Append tool results as user turn
            current_messages.append({
                "role": "user",
                "content": tool_results,
            })

        if iterations >= self._config.max_react_iterations and not final_text:
            logger.warning(
                "Agent %s hit max_react_iterations (%d) without end_turn",
                self.name, self._config.max_react_iterations,
            )
            final_text = "I reached the maximum number of reasoning steps. Please try a more specific question."

        return final_text, tool_call_log, iterations

    async def _execute_tool(self, block: anthropic.types.ToolUseBlock) -> str:
        config = self._tool_registry.get_config(block.name)
        logger.debug("Executing tool %s with input: %s", block.name, block.input)
        result = await self._tool_executor.execute(
            name=block.name,
            input=dict(block.input),
            sandboxed=config.sandboxed,
        )
        return result


def _strip_metadata(turn: dict[str, Any]) -> dict[str, Any]:
    """Remove internal metadata keys before passing to Anthropic API."""
    return {k: v for k, v in turn.items() if not k.startswith("_")}
