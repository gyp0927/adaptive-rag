from typing import Any

from hot_and_cold_memory.core.config import get_settings
from hot_and_cold_memory.core.llm_client import LLMClient
from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.monitoring.metrics import PROFILE_RECONCILER_RUNS_TOTAL

from .store import ProfileStore

logger = get_logger(__name__)


class ProfileReconciler:
    """Periodically reconcile profile facts and generate a summary."""

    def __init__(self, store: ProfileStore, llm_client: Any | None = None):
        self.settings = get_settings()
        self.store = store
        self.llm_client = llm_client or LLMClient()

    async def reconcile(self, user_id: str = "default") -> None:
        """Run reconciliation for a user."""
        if not self.settings.ENABLE_PROFILE_RECONCILER:
            return

        profile = await self.store.get_profile(user_id)
        if not profile or profile.is_empty():
            return

        try:
            summary = await self._generate_summary(profile)
            await self.store.metadata_store.upsert_profile(
                user_id=user_id,
                snapshot={"summary": summary},
            )
            logger.info("profile_reconciled", user_id=user_id, summary=summary[:100])
            PROFILE_RECONCILER_RUNS_TOTAL.labels(status="success").inc()
        except Exception as e:
            logger.warning("profile_reconcile_failed", user_id=user_id, error=str(e))
            PROFILE_RECONCILER_RUNS_TOTAL.labels(status="failure").inc()

    def _build_summary_prompt(self, profile: Any) -> str:
        parts = []
        if profile.identity:
            parts.append("Identity: " + ", ".join(f"{k}={v}" for k, v in profile.identity.items()))
        if profile.preferences:
            prefs = ", ".join(f"{p['key']}={p['value']}" for p in profile.preferences)
            parts.append("Preferences: " + prefs)
        if profile.goals:
            parts.append("Goals: " + ", ".join(profile.goals))
        if profile.constraints:
            parts.append("Constraints: " + ", ".join(profile.constraints))
        if profile.habits:
            parts.append("Habits: " + ", ".join(profile.habits))

        text = "\n".join(parts)
        return (
            "Summarize the following user profile in 1-2 sentences. "
            "Be concise and factual.\n\n" + text
        )

    async def _generate_summary(self, profile: Any) -> str:
        prompt = self._build_summary_prompt(profile)
        try:
            return (await self.llm_client.complete(
                prompt=prompt,
                model=self.settings.PROFILE_EXTRACTION_MODEL,
                max_tokens=256,
                temperature=0.0,
            )).strip()
        except Exception as e:
            logger.warning("profile_summary_generation_failed", error=str(e))
            return profile.summary or ""
