# tests/unit/profile/test_extractor.py
import pytest

from hot_and_cold_memory.profile.extractor import ProfileExtractor
from hot_and_cold_memory.profile.schemas import ProfileFact


@pytest.mark.asyncio
async def test_extract_profile_facts(monkeypatch):
    class FakeLLM:
        async def complete(self, *args, **kwargs):
            return '{"facts": [{"category": "identity", "key": "role", "value": "agent engineer", "confidence": 0.95}]}'

    extractor = ProfileExtractor(llm_client=FakeLLM())
    facts = await extractor.extract("I am an agent engineer.")
    assert len(facts) == 1
    assert facts[0] == ProfileFact(category="identity", key="role", value="agent engineer", confidence=0.95)


@pytest.mark.asyncio
async def test_extract_empty_content_returns_empty_list():
    extractor = ProfileExtractor(llm_client=None)
    assert await extractor.extract("") == []
    assert await extractor.extract("   ") == []


@pytest.mark.asyncio
async def test_extract_disabled_feature_flag_returns_empty_list(monkeypatch):
    monkeypatch.setattr("hot_and_cold_memory.profile.extractor.get_settings", lambda: type("S", (), {
        "ENABLE_PROFILE_AUGMENTATION": False,
        "PROFILE_EXTRACTION_MODEL": "gpt-4o-mini",
        "PROFILE_MAX_FACTS_PER_MEMORY": 10,
    })())
    extractor = ProfileExtractor()
    assert await extractor.extract("I am an agent engineer.") == []


@pytest.mark.asyncio
async def test_extract_llm_exception_returns_empty_list():
    class FakeLLM:
        async def complete(self, *args, **kwargs):
            raise RuntimeError("LLM error")

    extractor = ProfileExtractor(llm_client=FakeLLM())
    assert await extractor.extract("I am an agent engineer.") == []


@pytest.mark.asyncio
async def test_extract_invalid_json_returns_empty_list():
    class FakeLLM:
        async def complete(self, *args, **kwargs):
            return "not json"

    extractor = ProfileExtractor(llm_client=FakeLLM())
    assert await extractor.extract("I am an agent engineer.") == []


@pytest.mark.asyncio
async def test_extract_missing_fields_skipped():
    class FakeLLM:
        async def complete(self, *args, **kwargs):
            return '{"facts": [{"category": "identity", "key": "role"}]}'

    extractor = ProfileExtractor(llm_client=FakeLLM())
    assert await extractor.extract("I am an agent engineer.") == []


@pytest.mark.asyncio
async def test_extract_max_facts_cap():
    class FakeLLM:
        async def complete(self, *args, **kwargs):
            facts = [{"category": "identity", "key": f"role_{i}", "value": f"v{i}", "confidence": 0.5} for i in range(20)]
            import json
            return json.dumps({"facts": facts})

    extractor = ProfileExtractor(llm_client=FakeLLM())
    facts = await extractor.extract("I am an agent engineer.")
    assert len(facts) == 10


@pytest.mark.asyncio
async def test_extract_default_confidence_fallback():
    class FakeLLM:
        async def complete(self, *args, **kwargs):
            return '{"facts": [{"category": "identity", "key": "role", "value": "agent engineer"}]}'

    extractor = ProfileExtractor(llm_client=FakeLLM())
    facts = await extractor.extract("I am an agent engineer.")
    assert len(facts) == 1
    assert facts[0].confidence == 0.5


@pytest.mark.asyncio
async def test_extract_confidence_clamping():
    class FakeLLM:
        async def complete(self, *args, **kwargs):
            return '{"facts": [{"category": "identity", "key": "a", "value": "1", "confidence": 1.5}, {"category": "identity", "key": "b", "value": "2", "confidence": -0.3}]}'

    extractor = ProfileExtractor(llm_client=FakeLLM())
    facts = await extractor.extract("I am an agent engineer.")
    assert len(facts) == 2
    assert facts[0].confidence == 1.0
    assert facts[1].confidence == 0.0


@pytest.mark.asyncio
async def test_extract_invalid_category_skipped():
    class FakeLLM:
        async def complete(self, *args, **kwargs):
            return '{"facts": [{"category": "invalid", "key": "a", "value": "1", "confidence": 0.5}, {"category": "identity", "key": "b", "value": "2", "confidence": 0.5}]}'

    extractor = ProfileExtractor(llm_client=FakeLLM())
    facts = await extractor.extract("I am an agent engineer.")
    assert len(facts) == 1
    assert facts[0].category == "identity"
