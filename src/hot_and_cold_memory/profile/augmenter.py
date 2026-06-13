import dataclasses
from typing import Any

from hot_and_cold_memory.core.config import get_settings
from hot_and_cold_memory.core.llm_client import LLMClient
from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.monitoring.metrics import (
    PROFILE_QUERY_REWRITES_TOTAL,
    PROFILE_RANKING_BOOSTS_TOTAL,
)
from hot_and_cold_memory.tiers.base import RetrievedMemory

from .schemas import Profile
from .store import ProfileStore

logger = get_logger(__name__)


class ProfileAugmenter:
    """Rewrite queries and re-rank results using the user profile."""

    def __init__(self, store: ProfileStore, llm_client: Any | None = None):
        self.settings = get_settings()
        self.store = store
        self.llm_client = llm_client or LLMClient()

    async def rewrite_query(self, query: str, user_id: str = "default") -> str:
        """Expand the query with profile context."""
        if not self.settings.ENABLE_PROFILE_QUERY_REWRITE:
            return query

        profile = await self.store.get_profile(user_id)
        if not profile or profile.is_empty():
            return query

        prompt = self._build_rewrite_prompt(query, profile)
        try:
            rewritten = await self.llm_client.complete(
                prompt=prompt,
                model=self.settings.PROFILE_EXTRACTION_MODEL,
                max_tokens=256,
                temperature=0.0,
            )
            result = rewritten.strip().strip('"')
            PROFILE_QUERY_REWRITES_TOTAL.labels(status="success").inc()
            return result
        except Exception as e:
            logger.warning("profile_query_rewrite_failed", error=str(e))
            PROFILE_QUERY_REWRITES_TOTAL.labels(status="failure").inc()
            return query

    async def rerank(
        self,
        query: str,
        memories: list[RetrievedMemory],
        user_id: str = "default",
    ) -> list[RetrievedMemory]:
        """Boost memories that match the profile."""
        if not self.settings.ENABLE_PROFILE_RANKING_BOOST:
            return memories

        profile = await self.store.get_profile(user_id)
        if not profile or profile.is_empty():
            return memories

        profile_text = self._profile_to_text(profile)
        boosted: list[RetrievedMemory] = []
        for memory in memories:
            boost = self._compute_match_score(profile_text.lower(), memory.content.lower())
            new_score = memory.score + boost * self.settings.PROFILE_BOOST_WEIGHT
            boosted.append(
                dataclasses.replace(memory, score=new_score)
            )

        PROFILE_RANKING_BOOSTS_TOTAL.inc()
        return sorted(boosted, key=lambda m: m.score, reverse=True)

    def _build_rewrite_prompt(self, query: str, profile: Profile) -> str:
        profile_text = self._profile_to_text(profile)
        return (
            "Rewrite the following user query to include relevant profile context. "
            "Keep the original intent. Return only the rewritten query, no explanation.\n\n"
            f"User profile:\n{profile_text}\n\n"
            f"Query: {query}\n\n"
            "Rewritten query:"
        )

    def _profile_to_text(self, profile: Profile) -> str:
        parts: list[str] = []
        if profile.summary:
            parts.append(f"Summary: {profile.summary}")
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
        if profile.attributes:
            parts.append("Attributes: " + ", ".join(f"{k}={v}" for k, v in profile.attributes.items()))
        return "\n".join(parts)

    def _compute_match_score(self, profile_text: str, content: str) -> float:
        profile_words = set(profile_text.split())
        content_words = set(content.split())
        if not profile_words:
            return 0.0
        overlap = len(profile_words & content_words)
        return min(1.0, overlap / len(profile_words))
