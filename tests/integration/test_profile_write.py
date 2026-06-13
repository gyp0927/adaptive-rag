from unittest.mock import AsyncMock, MagicMock

import pytest

from hot_and_cold_memory.ingestion.pipeline import MemoryPipeline


@pytest.mark.asyncio
async def test_pipeline_extracts_profile(metadata_store, vector_store, document_store):
    from hot_and_cold_memory.core.config import get_settings
    from hot_and_cold_memory.frequency.tracker import FrequencyTracker
    from hot_and_cold_memory.profile.builder import ProfileBuilder
    from hot_and_cold_memory.profile.extractor import ProfileExtractor
    from hot_and_cold_memory.profile.store import ProfileStore
    from hot_and_cold_memory.storage.cache.memory_cache import MemoryCache
    from hot_and_cold_memory.tiers.cold_tier import ColdTier
    from hot_and_cold_memory.tiers.compression import CompressionEngine
    from hot_and_cold_memory.tiers.hot_tier import HotTier

    cache = MemoryCache()
    embedder = MagicMock()
    embedder.embed_batch = AsyncMock(return_value=[[0.1] * 384])

    hot = HotTier(vector_store, metadata_store, document_store, cache)
    cold = ColdTier(vector_store, metadata_store, document_store, CompressionEngine(), cache, embedder)
    ft = FrequencyTracker(metadata_store, vector_store, embedder)

    # Ensure cold tier collection exists
    settings = get_settings()
    await vector_store.ensure_collection(f"{settings.VECTOR_DB_COLLECTION}_cold")

    extractor = ProfileExtractor(llm_client=AsyncMock())
    extractor.llm_client.complete = AsyncMock(
        return_value='{"facts": [{"category": "identity", "key": "role", "value": "agent engineer", "confidence": 0.95}]}'
    )

    store = ProfileStore(metadata_store)
    builder = ProfileBuilder(store)

    pipeline = MemoryPipeline(metadata_store, hot, cold, embedder, ft)
    pipeline.profile_extractor = extractor
    pipeline.profile_builder = builder

    result = await pipeline.write_memory("I am an agent engineer.", memory_type="fact")
    assert result.status == "success"

    profile = await store.get_profile("default")
    assert profile is not None
    assert profile.identity.get("role") == "agent engineer"
