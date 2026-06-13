# src/hot_and_cold_memory/profile/builder.py
from typing import Any

from hot_and_cold_memory.core.logging import get_logger

from .schemas import ProfileFact
from .store import ProfileStore

logger = get_logger(__name__)


class ProfileBuilder:
    """Integrate extracted facts into the user's profile with conflict handling."""

    def __init__(self, store: ProfileStore):
        self.store = store

    async def integrate_facts(
        self,
        user_id: str,
        memory_id: Any | None,
        facts: list[ProfileFact],
    ) -> None:
        """Apply extracted facts and rebuild the snapshot."""
        for fact in facts:
            await self._apply_fact(user_id, memory_id, fact)
        await self.store.rebuild_snapshot(user_id)

    async def _apply_fact(
        self,
        user_id: str,
        memory_id: Any | None,
        fact: ProfileFact,
    ) -> None:
        current = await self.store.get_current_fact(user_id, fact.category, fact.key)
        if current is not None:
            if self._values_equal(current["value"], fact.value):
                merged_confidence = max(current["confidence"], fact.confidence)
                await self.store.update_fact_confidence(current["fact_id"], merged_confidence)
                return
            await self.store.expire_fact(current["fact_id"])
        await self.store.add_fact(user_id, memory_id, fact)

    def _values_equal(self, a: Any, b: Any) -> bool:
        """Compare two fact values for equality."""
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return list(a) == list(b)
        return a == b
