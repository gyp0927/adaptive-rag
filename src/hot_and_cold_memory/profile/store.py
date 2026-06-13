# src/hot_and_cold_memory/profile/store.py
from datetime import UTC, datetime
from typing import Any

from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.storage.metadata_store.base import BaseMetadataStore

from .schemas import Profile, ProfileFact

logger = get_logger(__name__)


class ProfileStore:
    """High-level storage for profile facts and snapshots."""

    def __init__(self, metadata_store: BaseMetadataStore):
        self.metadata_store = metadata_store

    async def get_current_fact(
        self,
        user_id: str,
        category: str,
        key: str,
    ) -> dict[str, Any] | None:
        return await self.metadata_store.get_current_profile_fact(user_id, category, key)

    async def add_fact(
        self,
        user_id: str,
        memory_id: Any | None,
        fact: ProfileFact,
    ) -> None:
        await self.metadata_store.create_profile_fact(
            user_id=user_id,
            memory_id=memory_id,
            category=fact.category,
            key=fact.key,
            value=fact.value,
            confidence=fact.confidence,
        )

    async def update_fact_confidence(self, fact_id: str, confidence: float) -> None:
        await self.metadata_store.update_profile_fact_confidence(fact_id, confidence)

    async def expire_fact(self, fact_id: str) -> None:
        await self.metadata_store.expire_profile_fact(fact_id)

    async def list_facts(
        self,
        user_id: str,
        category: str | None = None,
        key: str | None = None,
        is_current: bool | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return await self.metadata_store.list_profile_facts(
            user_id=user_id,
            category=category,
            key=key,
            is_current=is_current,
            limit=limit,
        )

    async def get_profile(self, user_id: str) -> Profile | None:
        data = await self.metadata_store.get_profile(user_id)
        if not data:
            return None
        user_id_val = data.get("user_id")
        if user_id_val is None:
            return None
        return Profile(
            user_id=user_id_val,
            identity=data.get("identity", {}),
            preferences=data.get("preferences", []),
            goals=data.get("goals", []),
            constraints=data.get("constraints", []),
            habits=data.get("habits", []),
            attributes=data.get("attributes", {}),
            summary=data.get("summary"),
            version=data.get("version", 1),
            updated_at=data.get("updated_at"),
        )

    async def rebuild_snapshot(self, user_id: str) -> None:
        """Rebuild the current profile snapshot from current facts.

        NOTE: This read-then-write pattern is not fully transactional and may
        race with concurrent writes. Consider wrapping in a transaction if the
        underlying metadata store supports it.
        """
        facts = await self.list_facts(user_id, is_current=True)

        identity: dict[str, Any] = {}
        preferences: list[dict[str, Any]] = []
        goals: list[str] = []
        constraints: list[str] = []
        habits: list[str] = []
        attributes: dict[str, Any] = {}

        def _safe_add_unique(items: list, value: Any) -> None:
            try:
                if value not in items:
                    items.append(value)
            except TypeError:
                items.append(value)

        for fact in facts:
            category = fact["category"]
            key = fact["key"]
            value = fact["value"]

            if category == "identity":
                identity[key] = value
            elif category == "preference":
                preferences.append({"key": key, "value": value, "confidence": fact["confidence"]})
            elif category == "goal":
                _safe_add_unique(goals, value)
            elif category == "constraint":
                _safe_add_unique(constraints, value)
            elif category == "habit":
                _safe_add_unique(habits, value)
            else:
                attributes[key] = value

        await self.metadata_store.upsert_profile(
            user_id=user_id,
            snapshot={
                "identity": identity,
                "preferences": preferences,
                "goals": goals,
                "constraints": constraints,
                "habits": habits,
                "attributes": attributes,
                "updated_at": datetime.now(UTC),
            },
        )
