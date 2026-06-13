import uuid

import pytest

from hot_and_cold_memory.core.config import Tier
from hot_and_cold_memory.profile.augmenter import ProfileAugmenter
from hot_and_cold_memory.profile.store import ProfileStore
from hot_and_cold_memory.tiers.base import RetrievedMemory


@pytest.mark.asyncio
async def test_rerank_boosts_matching_memory(metadata_store):
    store = ProfileStore(metadata_store)
    await store.metadata_store.upsert_profile(
        "default",
        {
            "identity": {"role": "backend engineer"},
            "preferences": [{"key": "language", "value": "Python", "confidence": 0.95}],
        },
    )

    augmenter = ProfileAugmenter(store, llm_client=None)
    memories = [
        RetrievedMemory(memory_id=uuid.uuid4(), content="User likes language=python", score=0.59, tier=Tier.HOT, is_decompressed=False),
        RetrievedMemory(memory_id=uuid.uuid4(), content="User went hiking", score=0.60, tier=Tier.HOT, is_decompressed=False),
    ]
    result = await augmenter.rerank("python", memories)
    assert result[0].content == "User likes language=python"


@pytest.mark.asyncio
async def test_rewrite_query_disabled(metadata_store, monkeypatch):
    store = ProfileStore(metadata_store)
    augmenter = ProfileAugmenter(store, llm_client=None)
    monkeypatch.setattr(augmenter.settings, "ENABLE_PROFILE_QUERY_REWRITE", False)
    assert await augmenter.rewrite_query("hello") == "hello"


@pytest.mark.asyncio
async def test_rewrite_query_empty_profile(metadata_store):
    store = ProfileStore(metadata_store)
    augmenter = ProfileAugmenter(store, llm_client=None)
    assert await augmenter.rewrite_query("hello") == "hello"


@pytest.mark.asyncio
async def test_rewrite_query_llm_exception(metadata_store, monkeypatch):
    store = ProfileStore(metadata_store)
    await store.metadata_store.upsert_profile(
        "default",
        {
            "identity": {"role": "backend engineer"},
        },
    )

    class FakeLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("boom")

    augmenter = ProfileAugmenter(store, llm_client=FakeLLM())
    assert await augmenter.rewrite_query("hello") == "hello"
