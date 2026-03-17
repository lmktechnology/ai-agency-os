"""
AI Agency OS — Entry Point
Boots the agent OS and starts the selected trigger mode.

Usage:
  python main.py --trigger=telegram   (default; requires TELEGRAM_BOT_TOKEN)
  python main.py --trigger=cli        (local REPL for testing)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import anthropic
import structlog

from src.config.loader import load_orchestrator_config
from src.config.settings import GlobalSettings
from src.core.agent_registry import AgentRegistry
from src.core.orchestrator import Orchestrator
from src.memory.flusher import MemoryFlusher
from src.memory.session import SessionManager
from src.skills.loader import SkillLoader
from src.tools.registry import ToolRegistry
from src.triggers.cron import CronTrigger


def _configure_logging(level: str) -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


async def main(trigger_mode: str) -> None:
    settings = GlobalSettings()
    _configure_logging(settings.log_level)

    log = structlog.get_logger(__name__)
    log.info("Starting AI Agency OS", trigger=trigger_mode)

    # --- Core infrastructure ---
    anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    tool_registry = ToolRegistry()
    agent_registry = AgentRegistry(settings.agents_dir, tool_registry)
    agent_registry.load_all()

    skill_loader = SkillLoader(settings.skills_dir)
    session_manager = SessionManager(settings.data_dir)

    # --- Load orchestrator config ---
    import os
    orchestrator_config_path = os.path.join(settings.agents_dir, "orchestrator.yaml")
    try:
        orchestrator_config = load_orchestrator_config(orchestrator_config_path)
    except FileNotFoundError:
        log.error("orchestrator.yaml not found at %s", orchestrator_config_path)
        sys.exit(1)

    orchestrator = Orchestrator(
        config=orchestrator_config,
        agent_registry=agent_registry,
        skill_loader=skill_loader,
        session_manager=session_manager,
        anthropic_client=anthropic_client,
        gateway=None,  # gateway is wired up by the trigger
    )

    # --- Background memory flusher ---
    flusher = MemoryFlusher(anthropic_client)
    asyncio.create_task(flusher.start(session_manager))

    # --- Cron trigger ---
    cron_trigger = CronTrigger(orchestrator)
    cron_trigger.load_from_config(orchestrator_config.cron_jobs)
    await cron_trigger.start()

    # --- Main trigger ---
    if trigger_mode == "telegram":
        if not settings.telegram_bot_token:
            log.error("TELEGRAM_BOT_TOKEN is not set")
            sys.exit(1)

        from src.triggers.telegram import TelegramTrigger
        telegram = TelegramTrigger(
            token=settings.telegram_bot_token,
            orchestrator=orchestrator,
            allowed_chat_ids=settings.telegram_allowed_chat_ids or None,
        )
        log.info("Telegram trigger starting...")
        await telegram.start()

    elif trigger_mode == "cli":
        from src.triggers.cli import CLITrigger
        cli = CLITrigger(orchestrator)
        await cli.start()

    else:
        log.error("Unknown trigger mode: %s (choose: telegram, cli)", trigger_mode)
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Agency OS")
    parser.add_argument(
        "--trigger",
        default="telegram",
        choices=["telegram", "cli"],
        help="Trigger mode (default: telegram)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        asyncio.run(main(args.trigger))
    except KeyboardInterrupt:
        print("\nShutting down...")
