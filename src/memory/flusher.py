from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from src.memory.session import SessionManager

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """You are a memory summarizer. The conversation turns below are from an ongoing session.
Write a concise summary (max 300 words) of what has been discussed, decisions made, and any important facts to remember.
This summary will be injected into future sessions as long-term memory context.

Output only the summary text — no preamble, no headers."""


class MemoryFlusher:
    def __init__(
        self,
        anthropic_client: anthropic.AsyncAnthropic,
        flush_interval_seconds: int = 30,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._client = anthropic_client
        self._interval = flush_interval_seconds
        self._model = model
        self._running = False

    async def start(self, session_manager: SessionManager) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(self._interval)
            for session_id in session_manager.list_active_sessions():
                try:
                    await self.flush_session(session_id, session_manager)
                except Exception as e:
                    logger.warning("Memory flush failed for %s: %s", session_id, e)

    def stop(self) -> None:
        self._running = False

    async def flush_session(self, session_id: str, session_manager: SessionManager) -> None:
        turns = session_manager.load_history(session_id, last_n=30)
        if len(turns) < 5:
            return  # Not enough history to summarize yet

        # Build a plain text representation of the turns
        conversation_text = "\n".join(
            f"{t['role'].upper()}: {_extract_text(t.get('content', ''))}"
            for t in turns
        )

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": f"{_SUMMARIZE_PROMPT}\n\n---\n{conversation_text}",
                }
            ],
        )
        summary = response.content[0].text.strip()
        session_manager.write_memory_context(session_id, summary)
        logger.debug("Flushed memory for session %s", session_id)


def _extract_text(content: object) -> str:
    """Extract plain text from either a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif hasattr(block, "text"):
                parts.append(block.text)
        return " ".join(parts)
    return str(content)
