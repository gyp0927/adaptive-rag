"""Frequency-driven query router."""

import asyncio
from dataclasses import dataclass
from typing import Any

from adaptive_rag.core.config import Tier, RoutingStrategy, get_settings
from adaptive_rag.core.logging import get_logger
from adaptive_rag.frequency.tracker import FrequencyTracker
from adaptive_rag.ingestion.embedder import Embedder
from adaptive_rag.tiers.base import RetrievedChunk
from adaptive_rag.tiers.hot_tier import HotTier
from adaptive_rag.tiers.cold_tier import ColdTier

from .ranker import ResultRanker

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result of a routed query."""

    chunks: list[RetrievedChunk]
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
    ) -> None:
        self.settings = get_settings()
        self.hot_tier = hot_tier
        self.cold_tier = cold_tier
        self.frequency_tracker = frequency_tracker
        self.embedder = embedder or Embedder()
        self.ranker = ResultRanker()

    async def route(
        self,
        query_text: str,
        query_embedding: list[float] | None = None,
        top_k: int = 10,
        tier_preference: Tier | None = None,
        force_decompress: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Route query to appropriate tier(s) and return merged results.

        Args:
            query_text: Original query text.
            query_embedding: Pre-computed embedding (optional).
            top_k: Number of results to return.
            tier_preference: Force a specific tier.
            force_decompress: Decompress cold chunks.
            filters: Optional metadata filters.

        Returns:
            Retrieval result with routing information.
        """
        import time
        start_time = time.time()

        # Generate embedding if not provided
        if query_embedding is None:
            query_embedding = await self.embedder.embed(query_text)

        # Determine routing strategy
        strategy = await self._determine_strategy(
            query_embedding, tier_preference
        )

        hot_results: list[RetrievedChunk] = []
        cold_results: list[RetrievedChunk] = []

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

        # Merge and re-rank
        merged = self.ranker.merge_and_rank(
            hot_results, cold_results, top_k
        )

        # Record access async (fire-and-forget)
        asyncio.create_task(
            self.frequency_tracker.record_access(
                chunk_ids=[c.chunk_id for c in merged],
                query_text=query_text,
                query_embedding=query_embedding,
            )
        )

        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            "query_routed",
            strategy=strategy.value,
            hot_count=len(hot_results),
            cold_count=len(cold_results),
            merged_count=len(merged),
            latency_ms=elapsed_ms,
        )

        return RetrievalResult(
            chunks=merged,
            routing_strategy=strategy,
            hot_results_count=len(hot_results),
            cold_results_count=len(cold_results),
            total_latency_ms=elapsed_ms,
            topic_frequency=await self.frequency_tracker.get_topic_frequency(
                query_embedding
            ),
        )

    async def _determine_strategy(
        self,
        query_embedding: list[float],
        tier_preference: Tier | None,
    ) -> RoutingStrategy:
        """Determine routing strategy.

        Args:
            query_embedding: Query embedding.
            tier_preference: Explicit tier preference.

        Returns:
            Routing strategy.
        """
        # Respect explicit preference
        if tier_preference == Tier.HOT:
            return RoutingStrategy.HOT_ONLY
        if tier_preference == Tier.COLD:
            return RoutingStrategy.COLD_ONLY

        # Check topic frequency
        topic_freq = await self.frequency_tracker.get_topic_frequency(
            query_embedding
        )

        # High-frequency topic -> hot tier only (low latency)
        if topic_freq >= self.settings.COLD_TO_HOT_THRESHOLD:
            return RoutingStrategy.HOT_ONLY

        # Low-frequency topic -> cold tier only (avoid wasted hot lookups)
        if topic_freq <= self.settings.HOT_TO_COLD_THRESHOLD:
            return RoutingStrategy.COLD_ONLY

        # Medium frequency -> query both
        return RoutingStrategy.BOTH
