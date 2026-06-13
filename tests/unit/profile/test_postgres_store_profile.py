
import pytest


@pytest.mark.asyncio
async def test_profile_crud(metadata_store):
    await metadata_store.create_profile_fact(
        user_id="default",
        memory_id=None,
        category="identity",
        key="role",
        value="agent engineer",
        confidence=0.95,
    )

    fact = await metadata_store.get_current_profile_fact("default", "identity", "role")
    assert fact is not None
    assert fact["value"] == "agent engineer"

    await metadata_store.expire_profile_fact(fact["fact_id"])
    expired = await metadata_store.get_current_profile_fact("default", "identity", "role")
    assert expired is None

    await metadata_store.upsert_profile("default", {"identity": {"role": "agent engineer"}})
    profile = await metadata_store.get_profile("default")
    assert profile["identity"]["role"] == "agent engineer"
