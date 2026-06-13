"""Frequency-driven query router."""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

from hot_and_cold_memory.core.config import RoutingStrategy, Tier, get_settings
from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.frequency.tracker import FrequencyTracker
from hot_and_cold_memory.ingestion.embedder import Embedder
from hot_and_cold_memory.monitoring.metrics import QUERY_DURATION, QUERY_TOTAL
from hot_and_cold_memory.storage.metadata_store.base import BaseMetadataStore
from hot_and_cold_memory.tiers.base import RetrievedMemory
from hot_and_cold_memory.tiers.cold_tier import ColdTier
from hot_and_cold_memory.tiers.hot_tier import HotTier

from .hybrid import HybridRanker
from .ranker import ResultRanker

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result of a routed query."""

    chunks: list[RetrievedMemory]
    routing_strategy: RoutingStrategy
    hot_results_count: int
    cold_results_count: int
    total_latency_ms: float
    topic_frequency: float


class FrequencyRouter:
    """Routes queries to appropriate tier(s) based on topic frequency.

    High-frequency topics -> Hot tier only (fast)
    Low-frequency topics -> Cold tier only (storage efficient)
    Medium frequency -> Both tiers (comprehensive)
    """

    def __init__(
        self,
        hot_tier: HotTier,
        cold_tier: ColdTier,
        frequency_tracker: FrequencyTracker,
        embedder: Embedder | None = None,
        metadata_store: BaseMetadataStore | None = None,
    ) -> None:
        self.settings = get_settings()
        self.hot_tier = hot_tier
        self.cold_tier = cold_tier
        self.frequency_tracker = frequency_tracker
        self.embedder = embedder or Embedder()
        self.metadata_store = metadata_store
        self.ranker = ResultRanker()
        self.hybrid_ranker = HybridRanker()
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def drain_background_tasks(self) -> None:
        """Wait for all pending background tasks to complete."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

    async def _record_access_safe(
        self,
        memory_ids: list[uuid.UUID],
        query_text: str,
        query_embedding: list[float],
    ) -> None:
        """Fire-and-forget access recording with exception swallowing."""
        try:
            await self.frequency_tracker.record_access(
                memory_ids=memory_ids,
                query_text=query_text,
                query_embedding=query_embedding,
            )
        except Exception as e:
            logger.warning("record_access_failed", error=str(e))

    async def route(
        self,
        query_text: str,
        query_embedding: list[float] | None = None,
        top_k: int = 10,
        tier_preference: Tier | None = None,
        force_decompress: bool = False,
        filters: dict[str, Any] | None = None,
        use_hybrid: bool = False,
    ) -> RetrievalResult:
        """Route query to appropriate tier(s) and return merged results.

        Args:
            query_text: Original query text.
            query_embedding: Pre-computed embedding (optional).
            top_k: Number of results to return.
            tier_preference: Force a specific tier.
            force_decompress: Decompress cold chunks.
            filters: Optional metadata filters.
            use_hybrid: If True, also search by keyword and fuse with RRF.

        Returns:
            Retrieval result with routing information.
        """
        start_time = time.time()

        # Generate embedding if not provided
        if query_embedding is None:
            query_embedding = await self.embedder.embed(query_text)

        # Determine routing strategy (fetch topic freq once and reuse)
        topic_info = await self.frequency_tracker.get_topic_frequency(
            query_embedding
        )
        strategy = self._determine_strategy_sync(
            topic_info.frequency,
            topic_info.access_count,
            tier_preference,
        )

        hot_results: list[RetrievedMemory] = []
        cold_results: list[RetrievedMemory] = []

        if strategy == RoutingStrategy.HOT_ONLY:
            hot_results = await self.hot_tier.retrieve(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )
        elif strategy == RoutingStrategy.COLD_ONLY:
            cold_results = await self.cold_tier.retrieve(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
                decompress=force_decompress,
            )
        elif strategy == RoutingStrategy.HOT_FIRST:
            # Try hot tier first, fall back to cold if insufficient
            hot_results = await self.hot_tier.retrieve(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )
            if len(hot_results) < top_k // 2:
                cold_results = await self.cold_tier.retrieve(
                    query_embedding=query_embedding,
                    top_k=top_k - len(hot_results),
                    filters=filters,
                    decompress=force_decompress,
                )
        elif strategy == RoutingStrategy.BOTH:
            # Query both tiers in parallel
            hot_task = self.hot_tier.retrieve(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )
            cold_task = self.cold_tier.retrieve(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
                decompress=force_decompress,
            )
            hot_results, cold_results = await asyncio.gather(hot_task, cold_task)

        # Merge and re-rank (vector results)
        vector_merged = self.ranker.merge_and_rank(
            hot_results, cold_results, top_k
        )

        # Hybrid search: fuse with keyword results
        final_chunks = vector_merged
        if use_hybrid and self.settings.ENABLE_HYBRID_SEARCH and self.metadata_store:
            keyword_items = await self.metadata_store.search_by_keyword(
                query_text=query_text,
                limit=top_k * 2,
            )
            keyword_results = [
                (item.memory_id, item.content) for item in keyword_items
            ]
            final_chunks = self.hybrid_ranker.fuse(
                vector_results=vector_merged,
                keyword_results=keyword_results,
                top_k=top_k,
            )

        # Record access async (fire-and-forget, exceptions swallowed)
        task = asyncio.create_task(
            self._record_access_safe(
                memory_ids=[c.memory_id for c in final_chunks],
                query_text=query_text,
                query_embedding=query_embedding,
            )
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        elapsed_ms = (time.time() - start_time) * 1000

        # Prometheus metrics
        QUERY_TOTAL.labels(tier=strategy.value, status="success").inc()
        QUERY_DURATION.labels(tier=strategy.value).observe(elapsed_ms / 1000.0)

        logger.info(
            "query_routed",
            strategy=strategy.value,
            hot_count=len(hot_results),
            cold_count=len(cold_results),
            merged_count=len(final_chunks),
            hybrid=use_hybrid,
            latency_ms=elapsed_ms,
        )

        return RetrievalResult(
            chunks=final_chunks,
            routing_strategy=strategy,
            hot_results_count=len(hot_results),
            cold_results_count=len(cold_results),
            total_latency_ms=elapsed_ms,
            topic_frequency=topic_info.frequency,
        )

    def _determine_strategy_sync(
        self,
        topic_freq: float,
        access_count: int,
        tier_preference: Tier | None,
    ) -> RoutingStrategy:
        """Determine routing strategy given pre-fetched topic frequency.

        Args:
            topic_freq: Already-fetched topic frequency score.
            access_count: Total historical access count for the topic.
            tier_preference: Explicit tier preference.

        Returns:
            Routing strategy.
        """
        # Respect explicit preference
        if tier_preference == Tier.HOT:
            return RoutingStrategy.HOT_ONLY
        if tier_preference == Tier.COLD:
            return RoutingStrategy.COLD_ONLY

        # High-frequency topic -> hot tier only (low latency)
        if topic_freq >= self.settings.COLD_TO_HOT_THRESHOLD:
            return RoutingStrategy.HOT_ONLY

        # Cumulative access count also qualifies for hot tier
        if access_count >= self.settings.HOT_ACCESS_COUNT_THRESHOLD:
            return RoutingStrategy.HOT_ONLY

        # Low-frequency topic -> cold tier only (avoid wasted hot lookups)
        if topic_freq <= self.settings.HOT_TO_COLD_THRESHOLD:
            return RoutingStrategy.COLD_ONLY

        # Medium frequency -> query both
        return RoutingStrategy.BOTH
