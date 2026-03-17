from __future__ import annotations

import logging

from telegram import Bot
from telegram.constants import ChatAction
from telegram.error import TelegramError

from src.gateway.base import BaseMessagingGateway

logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 4096


class TelegramGateway(BaseMessagingGateway):
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_message(
        self,
        session_id: str,
        text: str,
        parse_mode: str = "Markdown",
    ) -> None:
        """Send a message to a Telegram chat, splitting if over 4096 chars."""
        chat_id = int(session_id)
        chunks = _split_message(text, _MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            try:
                await self._bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=parse_mode,
                )
            except TelegramError as e:
                logger.error("Failed to send Telegram message to %s: %s", session_id, e)
                # Retry without parse_mode in case of formatting issue
                try:
                    await self._bot.send_message(chat_id=chat_id, text=chunk)
                except TelegramError:
                    logger.exception("Retry also failed for %s", session_id)

    async def send_typing(self, session_id: str) -> None:
        try:
            await self._bot.send_chat_action(
                chat_id=int(session_id),
                action=ChatAction.TYPING,
            )
        except TelegramError as e:
            logger.debug("send_typing failed: %s", e)


def _split_message(text: str, max_len: int) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:max_len])
        text = text[max_len:]
    return chunks
