"""Unified retrieval interface."""

import hashlib
import time
from collections import OrderedDict
from typing import Any

from hot_and_cold_memory.core.config import Tier
from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.frequency.tracker import FrequencyTracker
from hot_and_cold_memory.ingestion.embedder import Embedder
from hot_and_cold_memory.storage.metadata_store.base import BaseMetadataStore
from hot_and_cold_memory.tiers.cold_tier import ColdTier
from hot_and_cold_memory.tiers.hot_tier import HotTier

from hot_and_cold_memory.profile.augmenter import ProfileAugmenter
from hot_and_cold_memory.profile.store import ProfileStore

from .router import FrequencyRouter, RetrievalResult

logger = get_logger(__name__)


class _TTLCache:
    """Simple TTL cache for query results."""

    def __init__(self, ttl_seconds: float = 5.0, maxsize: int = 200) -> None:
        self.ttl = ttl_seconds
        self.maxsize = maxsize
        self._store: OrderedDict[str, tuple[float, RetrievalResult]] = OrderedDict()

    def _key(
        self,
        query_text: str,
        top_k: int,
        tier: Tier | None,
        decompress: bool,
        filters: dict[str, Any] | None,
    ) -> str:
        parts = [query_text, str(top_k), str(tier), str(decompress)]
        if filters:
            parts.append(hashlib.sha256(str(sorted(filters.items())).encode()).hexdigest()[:16])
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def get(
        self,
        query_text: str,
        top_k: int,
        tier: Tier | None,
        decompress: bool,
        filters: dict[str, Any] | None,
    ) -> RetrievalResult | None:
        key = self._key(query_text, top_k, tier, decompress, filters)
        if key not in self._store:
            return None
        stored_at, result = self._store[key]
        if time.time() - stored_at > self.ttl:
            del self._store[key]
            return None
        # Move to end (LRU)
        self._store.move_to_end(key)
        return result

    def set(
        self,
        query_text: str,
        top_k: int,
        tier: Tier | None,
        decompress: bool,
        filters: dict[str, Any] | None,
        result: RetrievalResult,
    ) -> None:
        key = self._key(query_text, top_k, tier, decompress, filters)
        self._store[key] = (time.time(), result)
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.pop(next(iter(self._store)))


class UnifiedRetriever:
    """Unified retrieval interface that handles all retrieval operations.

    Caches the most recent query results for 5 seconds to avoid redundant
    retrievals when the same query is issued repeatedly (e.g. UI polling
    or rapid re-submits).
    """

    def __init__(
        self,
        hot_tier: HotTier,
        cold_tier: ColdTier,
        frequency_tracker: FrequencyTracker,
        embedder: Embedder | None = None,
        metadata_store: BaseMetadataStore | None = None,
        profile_augmenter: ProfileAugmenter | None = None,
    ) -> None:
        self.router = FrequencyRouter(
            hot_tier=hot_tier,
            cold_tier=cold_tier,
            frequency_tracker=frequency_tracker,
            embedder=embedder,
            metadata_store=metadata_store,
        )
        self._cache = _TTLCache(ttl_seconds=5.0, maxsize=200)
        self.profile_augmenter = profile_augmenter
        if self.profile_augmenter is None and metadata_store is not None:
            self.profile_augmenter = ProfileAugmenter(ProfileStore(metadata_store))

    async def drain_background_tasks(self) -> None:
        """Wait for any pending background access-recording tasks."""
        await self.router.drain_background_tasks()

    async def query(
        self,
        query_text: str,
        top_k: int = 10,
        tier: Tier | None = None,
        decompress: bool = False,
        filters: dict[str, Any] | None = None,
        use_hybrid: bool = False,
        use_profile: bool = True,
    ) -> RetrievalResult:
        """Execute a query and retrieve relevant chunks (with short-term cache).

        Args:
            query_text: User query.
            top_k: Number of results.
            tier: Force specific tier.
            decompress: Decompress cold chunks.
            filters: Metadata filters.
            use_hybrid: If True, fuse vector + keyword results with RRF.
            use_profile: If True, use user profile to rewrite query and boost results.

        Returns:
            Retrieval result.
        """
        cached = self._cache.get(query_text, top_k, tier, decompress, filters)
        if cached is not None:
            logger.debug("query_cache_hit", query=query_text[:50])
            return cached

        effective_query = query_text
        if use_profile and self.profile_augmenter:
            effective_query = await self.profile_augmenter.rewrite_query(query_text)

        result = await self.router.route(
            query_text=effective_query,
            top_k=top_k,
            tier_preference=tier,
            force_decompress=decompress,
            filters=filters,
            use_hybrid=use_hybrid,
        )

        if use_profile and self.profile_augmenter:
            result.chunks = await self.profile_augmenter.rerank(effective_query, result.chunks)

        self._cache.set(query_text, top_k, tier, decompress, filters, result)
        return result
