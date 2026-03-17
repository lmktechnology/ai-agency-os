from __future__ import annotations

import logging

from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.core.event import Event
from src.core.orchestrator import Orchestrator
from src.gateway.telegram_gateway import TelegramGateway
from src.triggers.base import BaseTrigger

logger = logging.getLogger(__name__)


class TelegramTrigger(BaseTrigger):
    def __init__(
        self,
        token: str,
        orchestrator: Orchestrator,
        allowed_chat_ids: list[int] | None = None,
    ) -> None:
        self._token = token
        self._orchestrator = orchestrator
        self._allowed_chat_ids = set(allowed_chat_ids or [])

        self._app = Application.builder().token(token).build()
        # Inject the Telegram gateway so orchestrator can reply
        self._gateway = TelegramGateway(self._app.bot)
        orchestrator._gateway = self._gateway  # wire up late binding

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("help", self._on_help))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

    async def start(self) -> None:
        """Start polling for Telegram updates (blocks until stopped)."""
        logger.info("Starting Telegram polling...")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot is running.")
        # Keep running until stopped externally
        await self._app.updater.idle()

    async def stop(self) -> None:
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Hi! I'm your AI assistant. Send me a message to get started."
        )

    async def _on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Just send me any message and I'll respond. No special commands needed."
        )

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat:
            return

        chat_id = update.effective_chat.id

        # Security gate: only respond to allowed chats
        if self._allowed_chat_ids and chat_id not in self._allowed_chat_ids:
            logger.warning("Ignoring message from unauthorized chat_id: %s", chat_id)
            return

        user = update.effective_user
        user_id = str(user.id) if user else None

        # Send typing indicator before we start processing
        await self._gateway.send_typing(str(chat_id))

        event = Event(
            message=update.message.text or "",
            source="telegram",
            session_id=str(chat_id),
            user_id=user_id,
            metadata={
                "update_id": update.update_id,
                "chat_type": update.effective_chat.type,
                "username": user.username if user else None,
            },
        )

        try:
            await self._orchestrator.handle_event(event)
        except Exception as e:
            logger.exception("Error handling Telegram message from %s", chat_id)
            await self._gateway.send_message(
                str(chat_id),
                "Sorry, something went wrong. Please try again.",
            )
