"""Memory ingestion pipeline for agent memory system."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from hot_and_cold_memory.core.config import Tier, get_settings
from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.frequency.tracker import FrequencyTracker
from hot_and_cold_memory.migration.engine import MigrationEngine
from hot_and_cold_memory.monitoring.metrics import MEMORIES_TOTAL
from hot_and_cold_memory.storage.metadata_store.base import BaseMetadataStore
from hot_and_cold_memory.tiers.cold_tier import ColdTier
from hot_and_cold_memory.tiers.hot_tier import HotTier

from .embedder import Embedder
from .importance_scorer import ImportanceScorer

from hot_and_cold_memory.profile.builder import ProfileBuilder
from hot_and_cold_memory.profile.extractor import ProfileExtractor
from hot_and_cold_memory.profile.store import ProfileStore

logger = get_logger(__name__)


@dataclass
class MemoryWriteResult:
    """Result of writing a memory."""

    memory_id: uuid.UUID
    status: str = "pending"
    tier: str = ""
    error: str | None = None
    message: str | None = None
    processing_time_ms: float = 0.0


class MemoryPipeline:
    """Orchestrates memory ingestion into the system.

    Instead of blindly placing every new memory into the hot tier, the pipeline
    estimates the topic's historical popularity via the frequency tracker. Hot
    topics go to hot tier; new / cold topics skip compression and go directly to
    cold tier as raw text, saving LLM costs. After ingestion the hot tier
    capacity is checked and the coldest memories evicted if needed.
    """

    # Threshold above which a topic is considered "hot" at ingestion time.
    HOT_TOPIC_THRESHOLD: float = 0.5

    # Hard cap on hot tier memories. When exceeded, the coldest evict percent
    # of hot memories are pushed to cold tier.
    hot_tier_capacity: int = 10000
    evict_percent: float = 0.1

    def __init__(
        self,
        metadata_store: BaseMetadataStore,
        hot_tier: HotTier,
        cold_tier: ColdTier,
        embedder: Embedder,
        frequency_tracker: FrequencyTracker,
        migration_engine: MigrationEngine | None = None,
        profile_extractor: ProfileExtractor | None = None,
        profile_builder: ProfileBuilder | None = None,
    ) -> None:
        self.settings = get_settings()
        self.metadata_store = metadata_store
        self.hot_tier = hot_tier
        self.cold_tier = cold_tier
        self.embedder = embedder
        self.frequency_tracker = frequency_tracker
        self.hot_tier_capacity = self.settings.HOT_TIER_CAPACITY
        self.evict_percent = self.settings.HOT_TIER_EVICT_PERCENT
        self.migration_engine = migration_engine
        self.importance_scorer = ImportanceScorer()
        self.profile_extractor = profile_extractor or ProfileExtractor()
        self.profile_builder = profile_builder or ProfileBuilder(ProfileStore(metadata_store))

    async def write_memory(
        self,
        content: str,
        memory_type: str = "observation",
        source: str | None = None,
        importance: float = 0.5,
        tags: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        """Write a memory into the system.

        Args:
            content: Memory content text.
            memory_type: Type of memory (observation/fact/reflection/summary).
            source: Source identifier (e.g., conversation ID).
            importance: Initial importance score (0-1).
            tags: Optional tags.
            attributes: Optional additional attributes.

        Returns:
            Memory write result.
        """
        start_time = datetime.now(UTC)
        memory_id = uuid.uuid4()

        try:
            if not content.strip():
                return MemoryWriteResult(
                    memory_id=memory_id,
                    status="failed",
                    error="Memory content is empty",
                )

            # 1. Generate embedding
            embeddings = await self.embedder.embed_batch([content])
            embedding = embeddings[0]

            # 2. Check topic frequency to decide tier
            topic_info = await self.frequency_tracker.get_topic_frequency(embedding)
            is_hot = (
                topic_info.frequency >= self.HOT_TOPIC_THRESHOLD
                or topic_info.access_count >= self.settings.HOT_ACCESS_COUNT_THRESHOLD
            )

            from hot_and_cold_memory.tiers.base import MemoryEntry
            entry = MemoryEntry(
                memory_id=memory_id,
                content=content,
                tags=tags or [],
            )

            # Auto-score importance when user leaves the default
            if importance == 0.5:
                importance = await self.importance_scorer.score(content, memory_type)

            if is_hot:
                # Store in hot tier (short-term memory)
                await self.hot_tier.store_memories(
                    memories=[entry],
                    embeddings=[embedding],
                    memory_type=memory_type,
                    source=source,
                )
                tier = "hot"
            else:
                # Store in cold tier as raw (long-term memory, uncompressed)
                await self.cold_tier.store_raw_memories(
                    memories=[entry],
                    embeddings=[embedding],
                    memory_type=memory_type,
                    source=source,
                    initial_score=0.1,
                )
                tier = "cold"

            # Persist importance and attributes
            await self.metadata_store.update_memory(
                memory_id=memory_id,
                updates={"importance": importance, "attributes": attributes or {}},
            )

            # Hot tier capacity check
            await self._enforce_hot_tier_capacity()

            # Extract profile facts from the new memory
            try:
                facts = await self.profile_extractor.extract(content)
                if facts:
                    await self.profile_builder.integrate_facts(
                        user_id="default",
                        memory_id=memory_id,
                        facts=facts,
                    )
            except Exception as e:
                logger.warning("profile_extraction_after_write_failed", error=str(e))

            elapsed = (datetime.now(UTC) - start_time).total_seconds() * 1000

            # Update Prometheus gauges
            hot_total = await self.metadata_store.count_memories_by_tier(Tier.HOT)
            cold_total = await self.metadata_store.count_memories_by_tier(Tier.COLD)
            MEMORIES_TOTAL.labels(tier="hot").set(hot_total)
            MEMORIES_TOTAL.labels(tier="cold").set(cold_total)

            logger.info(
                "memory_written",
                memory_id=str(memory_id),
                tier=tier,
                memory_type=memory_type,
                elapsed_ms=elapsed,
            )

            return MemoryWriteResult(
                memory_id=memory_id,
                status="success",
                tier=tier,
                processing_time_ms=elapsed,
            )

        except Exception as e:
            logger.error("memory_write_failed", memory_id=str(memory_id), error=str(e))
            return MemoryWriteResult(
                memory_id=memory_id,
                status="failed",
                error=str(e),
            )

    async def write_memories_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[MemoryWriteResult]:
        """Write multiple memories in batch.

        Batches embedding generation and tier storage for efficiency.

        Args:
            items: List of memory dicts with keys: content, memory_type, source,
                   importance, tags, attributes.

        Returns:
            List of write results.
        """
        if not items:
            return []

        start_time = datetime.now(UTC)
        from hot_and_cold_memory.tiers.base import MemoryEntry

        # Pre-validate and assign IDs
        result_map: dict[int, MemoryWriteResult] = {}
        valid_items: list[tuple[int, dict[str, Any], uuid.UUID]] = []
        for i, item in enumerate(items):
            mid = uuid.uuid4()
            content = item.get("content", "")
            if not content or not content.strip():
                result_map[i] = MemoryWriteResult(
                    memory_id=mid,
                    status="failed",
                    error="Memory content is empty",
                )
            else:
                valid_items.append((i, item, mid))

        if not valid_items:
            return [result_map[i] for i, _ in enumerate(items)]

        # Batch generate embeddings
        contents = [item["content"] for _, item, _ in valid_items]
        embeddings = await self.embedder.embed_batch(contents)

        # Check topic frequencies in batch
        topic_infos = await self.frequency_tracker.get_topic_frequencies_batch(embeddings)

        hot_entries: list[tuple[int, MemoryEntry, list[float], dict[str, Any]]] = []
        cold_entries: list[tuple[int, MemoryEntry, list[float], dict[str, Any]]] = []

        for (orig_idx, item, mid), embedding, topic_info in zip(
            valid_items, embeddings, topic_infos, strict=True
        ):
            is_hot = (
                topic_info.frequency >= self.HOT_TOPIC_THRESHOLD
                or topic_info.access_count >= self.settings.HOT_ACCESS_COUNT_THRESHOLD
            )
            entry = MemoryEntry(
                memory_id=mid,
                content=item["content"],
                tags=item.get("tags") or [],
            )
            meta = {
                "memory_type": item.get("memory_type", "observation"),
                "source": item.get("source"),
                "importance": item.get("importance", 0.5),
                "attributes": item.get("attributes") or {},
            }
            if is_hot:
                hot_entries.append((orig_idx, entry, embedding, meta))
            else:
                cold_entries.append((orig_idx, entry, embedding, meta))

        # Batch store hot, grouped by (memory_type, source) to preserve per-item metadata
        if hot_entries:
            hot_groups: dict[tuple[str, str | None], list[tuple[int, Any, list[float], dict[str, Any]]]] = {}
            for orig_idx, entry, embedding, meta in hot_entries:
                key = (meta["memory_type"], meta["source"])
                hot_groups.setdefault(key, []).append((orig_idx, entry, embedding, meta))
            for (mtype, src), group in hot_groups.items():
                await self.hot_tier.store_memories(
                    memories=[e for _, e, _, _ in group],
                    embeddings=[emb for _, _, emb, _ in group],
                    memory_type=mtype,
                    source=src,
                )
        # Batch store cold, grouped by (memory_type, source)
        if cold_entries:
            cold_groups: dict[tuple[str, str | None], list[tuple[int, Any, list[float], dict[str, Any]]]] = {}
            for orig_idx, entry, embedding, meta in cold_entries:
                key = (meta["memory_type"], meta["source"])
                cold_groups.setdefault(key, []).append((orig_idx, entry, embedding, meta))
            for (mtype, src), group in cold_groups.items():
                await self.cold_tier.store_raw_memories(
                    memories=[e for _, e, _, _ in group],
                    embeddings=[emb for _, _, emb, _ in group],
                    memory_type=mtype,
                    source=src,
                    initial_score=0.1,
                )

        # Auto-score importance for defaults, then persist all importance values
        importance_updates: dict[uuid.UUID, dict[str, Any]] = {}
        auto_score_items: list[tuple[uuid.UUID, str, str]] = []
        for _, entry, _, meta in hot_entries + cold_entries:
            if meta["importance"] == 0.5:
                auto_score_items.append((entry.memory_id, entry.content, meta["memory_type"]))

        if auto_score_items:
            scored = await self.importance_scorer.score_batch(
                [(content, mt) for _, content, mt in auto_score_items]
            )
            for (mid, _content, _mt), score in zip(auto_score_items, scored, strict=True):
                importance_updates[mid] = {"importance": score}

        for _, entry, _, meta in hot_entries + cold_entries:
            if entry.memory_id not in importance_updates:
                importance_updates[entry.memory_id] = {
                    "importance": meta["importance"],
                }
            importance_updates[entry.memory_id]["attributes"] = meta["attributes"]

        if importance_updates:
            await self.metadata_store.update_memories_batch(importance_updates)

        # Build results
        for orig_idx, entry, _, _meta in hot_entries:
            result_map[orig_idx] = MemoryWriteResult(
                memory_id=entry.memory_id,
                status="success",
                tier="hot",
            )
        for orig_idx, entry, _, _meta in cold_entries:
            result_map[orig_idx] = MemoryWriteResult(
                memory_id=entry.memory_id,
                status="success",
                tier="cold",
            )

        # Insert results at correct positions
        for orig_idx, _, mid in valid_items:
            if orig_idx not in result_map:
                result_map[orig_idx] = MemoryWriteResult(
                    memory_id=mid,
                    status="failed",
                    error="Unknown batch error",
                )

        final_results = [result_map[i] for i, _ in enumerate(items)]

        # Hot tier capacity check
        await self._enforce_hot_tier_capacity()

        # Extract profile facts from the new memories
        for _, entry, _, _meta in hot_entries + cold_entries:
            try:
                facts = await self.profile_extractor.extract(entry.content)
                if facts:
                    await self.profile_builder.integrate_facts(
                        user_id="default",
                        memory_id=entry.memory_id,
                        facts=facts,
                    )
            except Exception as e:
                logger.warning("profile_extraction_after_write_failed", error=str(e))

        elapsed = (datetime.now(UTC) - start_time).total_seconds() * 1000
        for r in final_results:
            r.processing_time_ms = elapsed / max(len(items), 1)

        # Update metrics
        hot_total = await self.metadata_store.count_memories_by_tier(Tier.HOT)
        cold_total = await self.metadata_store.count_memories_by_tier(Tier.COLD)
        MEMORIES_TOTAL.labels(tier="hot").set(hot_total)
        MEMORIES_TOTAL.labels(tier="cold").set(cold_total)

        logger.info(
            "memory_batch_written",
            count=len(valid_items),
            hot=len(hot_entries),
            cold=len(cold_entries),
            elapsed_ms=elapsed,
        )
        return final_results

    async def delete_memory(self, memory_id: uuid.UUID) -> bool:
        """Delete a memory from all stores.

        Args:
            memory_id: Memory to delete.

        Returns:
            True if deleted.
        """
        meta = await self.metadata_store.get_memory(memory_id)
        if not meta:
            return False

        if meta.tier == Tier.HOT:
            await self.hot_tier.delete([memory_id])
        else:
            await self.cold_tier.delete([memory_id])

        logger.info("memory_deleted", memory_id=str(memory_id))
        return True

    async def _enforce_hot_tier_capacity(self) -> None:
        """If hot tier exceeds capacity, evict the coldest memories to cold tier."""
        try:
            hot_count = await self.metadata_store.count_memories_by_tier(tier=Tier.HOT)
            if hot_count > self.hot_tier_capacity:
                if self.migration_engine is not None:
                    evicted = await self.migration_engine.evict_coldest(
                        percent=self.evict_percent
                    )
                    logger.warning(
                        "hot_tier_capacity_exceeded",
                        hot_count=hot_count,
                        capacity=self.hot_tier_capacity,
                        evicted=len(evicted),
                    )
                else:
                    logger.warning(
                        "hot_tier_capacity_exceeded_no_migration_engine",
                        hot_count=hot_count,
                        capacity=self.hot_tier_capacity,
                    )
        except Exception as e:
            logger.error("hot_tier_capacity_check_failed", error=str(e))
