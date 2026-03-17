from __future__ import annotations

import asyncio
import logging
import sys
import uuid

from src.core.event import Event
from src.core.orchestrator import Orchestrator
from src.triggers.base import BaseTrigger

logger = logging.getLogger(__name__)

_BANNER = """
╔══════════════════════════════════════╗
║   AI Agency OS — CLI Mode            ║
║   Type 'exit' or Ctrl+C to quit      ║
╚══════════════════════════════════════╝
"""


class CLITrigger(BaseTrigger):
    def __init__(self, orchestrator: Orchestrator, session_id: str | None = None) -> None:
        self._orchestrator = orchestrator
        self._session_id = session_id or f"cli_{uuid.uuid4().hex[:8]}"
        self._running = False

    async def start(self) -> None:
        print(_BANNER)
        print(f"Session ID: {self._session_id}\n")
        self._running = True

        loop = asyncio.get_event_loop()

        while self._running:
            try:
                # Read input in a thread so we don't block the event loop
                user_input = await loop.run_in_executor(None, self._prompt)
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            event = Event(
                message=user_input,
                source="cli",
                session_id=self._session_id,
            )

            print()
            try:
                response = await self._orchestrator.handle_event(event)
                if response.error:
                    print(f"[Error] {response.error}")
                else:
                    print(f"[{response.agent_name}]: {response.content}")
            except Exception as e:
                print(f"[Error] {e}")
            print()

    async def stop(self) -> None:
        self._running = False

    def _prompt(self) -> str:
        try:
            return input("You: ").strip()
        except EOFError:
            raise
