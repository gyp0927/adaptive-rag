import pytest
from unittest.mock import AsyncMock

from hot_and_cold_memory.profile.reconciler import ProfileReconciler
from hot_and_cold_memory.profile.store import ProfileStore


@pytest.mark.asyncio
async def test_reconciler_generates_summary(metadata_store):
    store = ProfileStore(metadata_store)
    await store.metadata_store.upsert_profile(
        "default",
        {
            "identity": {"role": "agent engineer"},
            "preferences": [{"key": "language", "value": "Python", "confidence": 0.95}],
        },
    )

    llm = AsyncMock()
    llm.complete = AsyncMock(return_value="Agent engineer who likes Python.")
    reconciler = ProfileReconciler(store, llm_client=llm)
    await reconciler.reconcile("default")

    profile = await store.get_profile("default")
    assert profile.summary == "Agent engineer who likes Python."
