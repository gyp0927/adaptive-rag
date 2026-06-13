"""Unit tests for profile API Pydantic schemas."""

from datetime import datetime, UTC

from hot_and_cold_memory.api.schemas.profile import ProfileFacet, ProfileFactResponse, ProfileResponse


class TestProfileSchemas:
    """Tests for profile schema construction and validation."""

    def test_profile_facet_with_confidence(self):
        facet = ProfileFacet(key="language", value="Python", confidence=0.95)
        assert facet.key == "language"
        assert facet.value == "Python"
        assert facet.confidence == 0.95

    def test_profile_facet_without_confidence(self):
        facet = ProfileFacet(key="theme", value="dark")
        assert facet.confidence is None

    def test_profile_facet_confidence_null(self):
        facet = ProfileFacet(key="theme", value="dark", confidence=None)
        assert facet.confidence is None

    def test_profile_fact_response_full(self):
        fact = ProfileFactResponse(
            fact_id="f1",
            category="preference",
            key="language",
            value="Python",
            confidence=0.9,
            valid_from=datetime.now(UTC),
            valid_until=None,
            is_current=True,
        )
        assert fact.fact_id == "f1"
        assert fact.confidence == 0.9
        assert fact.valid_until is None

    def test_profile_response_schema(self):
        data = {
            "user_id": "default",
            "identity": {"role": "agent engineer"},
            "preferences": [{"key": "language", "value": "Python", "confidence": 0.95}],
            "goals": ["improve agent engineering"],
            "constraints": ["peanut allergy"],
            "habits": ["active late night"],
            "attributes": {},
            "summary": "Summary text",
            "updated_at": datetime.now(UTC),
        }
        response = ProfileResponse(**data)
        assert response.user_id == "default"
        assert response.identity["role"] == "agent engineer"
