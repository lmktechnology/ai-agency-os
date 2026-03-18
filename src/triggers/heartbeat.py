from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.config.models import HeartbeatConfig
from src.core.event import Event
from src.core.orchestrator import Orchestrator
from src.triggers.base import BaseTrigger

logger = logging.getLogger(__name__)

# Responses starting with this prefix are treated as no-ops and not forwarded
_NOOP_PREFIX = "NOOP"


class HeartbeatTrigger(BaseTrigger):
    """
    Interval-based autonomous trigger that fires agents on a fixed cadence.

    Unlike CronTrigger (time-of-day), HeartbeatTrigger runs every N seconds,
    making the system proactively monitor state and act without human prompting.

    Each heartbeat tick:
      1. Fires the configured agent with the heartbeat message
      2. Forwards the response to reply_chat_id (if set and response is non-empty)
      3. Suppresses "NOOP: ..." responses when silent_on_noop is True

    The agent itself decides whether to act — it can return "NOOP: nothing to do"
    to indicate it checked and found nothing requiring action.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orchestrator = orchestrator
        self._scheduler = AsyncIOScheduler()
        self._heartbeats: list[HeartbeatConfig] = []

    def load_from_config(self, heartbeats: list[HeartbeatConfig]) -> None:
        """Register heartbeat configs. Call before start()."""
        self._heartbeats = [hb for hb in heartbeats if hb.enabled]
        skipped = len(heartbeats) - len(self._heartbeats)
        if skipped:
            logger.info("Skipped %d disabled heartbeat(s)", skipped)

    async def start(self) -> None:
        """Schedule all heartbeats and start the APScheduler."""
        for hb in self._heartbeats:
            if hb.interval_seconds < 10:
                logger.warning(
                    "Heartbeat '%s' interval %ds is very short — minimum is 10s, clamping.",
                    hb.id, hb.interval_seconds,
                )
                hb = hb.model_copy(update={"interval_seconds": 10})

            trigger = IntervalTrigger(
                seconds=hb.interval_seconds,
                # Delay the first tick so the system finishes booting first
                jitter=0,
            )
            self._scheduler.add_job(
                func=self._tick,
                trigger=trigger,
                id=hb.id,
                kwargs={"heartbeat": hb},
                replace_existing=True,
                # Fire the first tick after startup_delay_seconds, not immediately
                next_run_time=None,  # APScheduler will compute based on interval
            )
            logger.info(
                "Scheduled heartbeat '%s' every %ds → agent '%s'",
                hb.id, hb.interval_seconds, hb.agent,
            )

        self._scheduler.start()
        logger.info("HeartbeatTrigger started with %d heartbeat(s)", len(self._heartbeats))

        # Schedule the first ticks with startup delay using one-off jobs
        import asyncio
        for hb in self._heartbeats:
            asyncio.get_event_loop().call_later(
                hb.startup_delay_seconds,
                lambda h=hb: asyncio.ensure_future(self._tick(h)),
            )

    async def stop(self) -> None:
        self._scheduler.shutdown(wait=False)

    async def _tick(self, heartbeat: HeartbeatConfig) -> None:
        """Execute one heartbeat tick: run the agent, optionally forward response."""
        session_id = heartbeat.reply_chat_id or f"heartbeat_{heartbeat.id}"
        event = Event(
            message=heartbeat.message,
            source="heartbeat",
            session_id=session_id,
            metadata={"heartbeat_id": heartbeat.id},
        )
        logger.debug("Heartbeat tick '%s' → agent '%s'", heartbeat.id, heartbeat.agent)
        try:
            agent = self._orchestrator._agent_registry.build_agent(
                name=heartbeat.agent,
                skill_loader=self._orchestrator._skill_loader,
                session_manager=self._orchestrator._session_manager,
                provider_registry=self._orchestrator._provider_registry,
            )
            response = await agent.run(event)

            if response.error:
                logger.error("Heartbeat '%s' agent error: %s", heartbeat.id, response.error)
                return

            logger.debug(
                "Heartbeat '%s' completed (%d steps): %s",
                heartbeat.id, response.react_steps,
                response.content[:80] if response.content else "(empty)",
            )

            # Suppress no-op responses
            if heartbeat.silent_on_noop and response.content.startswith(_NOOP_PREFIX):
                logger.debug("Heartbeat '%s' — NOOP, suppressing forward", heartbeat.id)
                return

            # Forward to messaging gateway if configured
            if heartbeat.reply_chat_id and response.content and self._orchestrator._gateway:
                await self._orchestrator._gateway.send_message(
                    heartbeat.reply_chat_id, response.content
                )

        except Exception:
            logger.exception("Heartbeat '%s' tick failed", heartbeat.id)
