from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger as APSCronTrigger

from src.config.models import CronJobConfig
from src.core.event import Event
from src.core.orchestrator import Orchestrator
from src.triggers.base import BaseTrigger

logger = logging.getLogger(__name__)


class CronTrigger(BaseTrigger):
    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orchestrator = orchestrator
        self._scheduler = AsyncIOScheduler()
        self._jobs: list[dict[str, Any]] = []

    def add_job(
        self,
        job_id: str,
        cron_expression: str,
        agent_name: str,
        message: str,
        reply_chat_id: str | None = None,
    ) -> None:
        """Register a scheduled job. Call before start()."""
        self._jobs.append({
            "job_id": job_id,
            "cron": cron_expression,
            "agent_name": agent_name,
            "message": message,
            "reply_chat_id": reply_chat_id,
        })

    def load_from_config(self, jobs: list[CronJobConfig]) -> None:
        """Load cron jobs from orchestrator config."""
        for job in jobs:
            self.add_job(
                job_id=job.id,
                cron_expression=job.cron,
                agent_name=job.agent,
                message=job.message,
                reply_chat_id=job.reply_chat_id,
            )

    async def start(self) -> None:
        """Schedule all registered jobs and start the scheduler."""
        for job in self._jobs:
            cron_parts = job["cron"].split()
            if len(cron_parts) != 5:
                logger.error("Invalid cron expression for job %s: %s", job["job_id"], job["cron"])
                continue

            minute, hour, day, month, day_of_week = cron_parts
            trigger = APSCronTrigger(
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week,
            )

            self._scheduler.add_job(
                func=self._fire_job,
                trigger=trigger,
                id=job["job_id"],
                kwargs={
                    "job_id": job["job_id"],
                    "agent_name": job["agent_name"],
                    "message": job["message"],
                    "reply_chat_id": job["reply_chat_id"],
                },
                replace_existing=True,
            )
            logger.info("Scheduled cron job '%s' [%s]", job["job_id"], job["cron"])

        self._scheduler.start()
        logger.info("CronTrigger started with %d job(s)", len(self._jobs))

    async def stop(self) -> None:
        self._scheduler.shutdown(wait=False)

    async def _fire_job(
        self,
        job_id: str,
        agent_name: str,
        message: str,
        reply_chat_id: str | None,
    ) -> None:
        session_id = reply_chat_id or f"cron_{job_id}"
        event = Event(
            message=message,
            source="cron",
            session_id=session_id,
            metadata={"job_id": job_id, "agent_name": agent_name},
        )
        logger.info("Firing cron job '%s' → agent '%s'", job_id, agent_name)
        try:
            # Override routing: send directly to specified agent
            agent = self._orchestrator._agent_registry.build_agent(
                name=agent_name,
                skill_loader=self._orchestrator._skill_loader,
                session_manager=self._orchestrator._session_manager,
                anthropic_client=self._orchestrator._client,
            )
            response = await agent.run(event)
            if response.error:
                logger.error("Cron job '%s' agent error: %s", job_id, response.error)
            else:
                logger.info("Cron job '%s' completed (%d steps)", job_id, response.react_steps)
                if reply_chat_id and self._orchestrator._gateway:
                    await self._orchestrator._gateway.send_message(reply_chat_id, response.content)
        except Exception:
            logger.exception("Cron job '%s' failed", job_id)
