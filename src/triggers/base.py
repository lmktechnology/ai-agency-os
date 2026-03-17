from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTrigger(ABC):
    @abstractmethod
    async def start(self) -> None:
        """Start the trigger (blocks until stopped)."""

    async def stop(self) -> None:
        """Stop the trigger gracefully."""
