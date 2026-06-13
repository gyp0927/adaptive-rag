# tests/unit/profile/test_builder.py
import pytest

from hot_and_cold_memory.profile.builder import ProfileBuilder
from hot_and_cold_memory.profile.schemas import ProfileFact
from hot_and_cold_memory.profile.store import ProfileStore


@pytest.mark.asyncio
async def test_builder_conflict_resolution(metadata_store):
    store = ProfileStore(metadata_store)
    builder = ProfileBuilder(store)

    await builder.integrate_facts(
        "default",
        None,
        [ProfileFact(category="identity", key="role", value="hardware engineer", confidence=0.9)],
    )

    first = await store.get_current_fact("default", "identity", "role")
    assert first["value"] == "hardware engineer"

    await builder.integrate_facts(
        "default",
        None,
        [ProfileFact(category="identity", key="role", value="agent engineer", confidence=0.95)],
    )

    current = await store.get_current_fact("default", "identity", "role")
    assert current["value"] == "agent engineer"

    history = await store.list_facts("default", category="identity", key="role")
    assert len(history) == 2


@pytest.mark.asyncio
async def test_same_value_merges_confidence_max(metadata_store):
    store = ProfileStore(metadata_store)
    builder = ProfileBuilder(store)

    await builder.integrate_facts(
        "user_a",
        None,
        [ProfileFact(category="identity", key="role", value="engineer", confidence=0.9)],
    )

    await builder.integrate_facts(
        "user_a",
        None,
        [ProfileFact(category="identity", key="role", value="engineer", confidence=0.7)],
    )

    current = await store.get_current_fact("user_a", "identity", "role")
    assert current["value"] == "engineer"
    assert current["confidence"] == 0.9


@pytest.mark.asyncio
async def test_empty_fact_list_does_not_error(metadata_store):
    store = ProfileStore(metadata_store)
    builder = ProfileBuilder(store)

    await builder.integrate_facts("user_b", None, [])

    profile = await store.get_profile("user_b")
    assert profile is not None
    assert profile.identity == {}
    assert profile.goals == []


@pytest.mark.asyncio
async def test_rebuild_snapshot_produces_correct_fields(metadata_store):
    store = ProfileStore(metadata_store)
    builder = ProfileBuilder(store)

    facts = [
        ProfileFact(category="identity", key="name", value="Alice", confidence=0.95),
        ProfileFact(category="preference", key="theme", value="dark", confidence=0.8),
        ProfileFact(category="goal", key="", value="learn python", confidence=0.9),
        ProfileFact(category="constraint", key="", value="no dairy", confidence=0.85),
        ProfileFact(category="habit", key="", value="morning run", confidence=0.7),
        ProfileFact(category="attribute", key="skill", value="typing", confidence=0.6),
    ]

    await builder.integrate_facts("user_c", None, facts)

    profile = await store.get_profile("user_c")
    assert profile is not None
    assert profile.identity == {"name": "Alice"}
    assert profile.preferences == [{"key": "theme", "value": "dark", "confidence": 0.8}]
    assert profile.goals == ["learn python"]
    assert profile.constraints == ["no dairy"]
    assert profile.habits == ["morning run"]
    assert profile.attributes == {"skill": "typing"}
