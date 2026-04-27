"""Migration scheduler using APScheduler."""

from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from adaptive_rag.core.config import get_settings
from adaptive_rag.core.logging import get_logger

logger = get_logger(__name__)


class MigrationScheduler:
    """Schedules periodic migration cycles and cluster maintenance."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.scheduler = AsyncIOScheduler()
        self._migration_job = None
        self._cluster_cleanup_job = None

    def start(
        self,
        migration_callback,
        cluster_cleanup_callback=None,
    ) -> None:
        """Start the migration scheduler.

        Args:
            migration_callback: Async function to call for migration.
            cluster_cleanup_callback: Optional async function to clean up stale clusters.
        """
        self._migration_job = self.scheduler.add_job(
            migration_callback,
            trigger=IntervalTrigger(
                minutes=self.settings.MIGRATION_INTERVAL_MINUTES,
            ),
            id="migration_cycle",
            name="Tier Migration Cycle",
            replace_existing=True,
        )
        if cluster_cleanup_callback is not None:
            self._cluster_cleanup_job = self.scheduler.add_job(
                cluster_cleanup_callback,
                trigger=CronTrigger(hour=3, minute=0),
                id="cluster_cleanup",
                name="Cluster Drift Cleanup",
                replace_existing=True,
            )
            logger.info("cluster_cleanup_scheduled", hour=3)

        self.scheduler.start()
        logger.info(
            "migration_scheduler_started",
            interval_minutes=self.settings.MIGRATION_INTERVAL_MINUTES,
        )

    def stop(self) -> None:
        """Stop the scheduler."""
        self.scheduler.shutdown()
        logger.info("migration_scheduler_stopped")

    async def trigger_now(self) -> None:
        """Trigger migration immediately."""
        if self._migration_job:
            self._migration_job.modify(next_run_time=datetime.utcnow())
