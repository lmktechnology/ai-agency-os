from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMessagingGateway(ABC):
    @abstractmethod
    async def send_message(self, session_id: str, text: str) -> None:
        """Send a text message to the session."""

    async def send_typing(self, session_id: str) -> None:
        """Optionally signal that the bot is typing. No-op by default."""
