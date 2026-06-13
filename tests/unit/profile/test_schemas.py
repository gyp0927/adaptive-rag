# tests/unit/profile/test_schemas.py
from hot_and_cold_memory.profile.schemas import Profile, ProfileFact


def test_profile_fact_creation():
    fact = ProfileFact(category="identity", key="role", value="agent engineer", confidence=0.95)
    assert fact.category == "identity"


def test_profile_empty():
    profile = Profile(user_id="default")
    assert profile.is_empty() is True


def test_profile_is_empty_with_identity():
    profile = Profile(user_id="default", identity={"role": "agent engineer"})
    assert profile.is_empty() is False


def test_profile_is_empty_with_preferences():
    profile = Profile(user_id="default", preferences=[{"theme": "dark"}])
    assert profile.is_empty() is False


def test_profile_is_empty_with_summary_only():
    profile = Profile(user_id="default", summary="A brief summary.")
    assert profile.is_empty() is True


def test_profile_fact_default_confidence():
    fact = ProfileFact(category="identity", key="role", value="agent engineer")
    assert fact.confidence == 0.5
