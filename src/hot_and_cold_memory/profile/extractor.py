# src/hot_and_cold_memory/profile/extractor.py
import json
from typing import Any

from hot_and_cold_memory.core.config import get_settings
from hot_and_cold_memory.core.llm_client import LLMClient
from hot_and_cold_memory.core.logging import get_logger

from .schemas import ProfileFact

logger = get_logger(__name__)


class ProfileExtractor:
    """Extract structured profile facts from memory content using an LLM."""

    def __init__(self, llm_client: Any | None = None):
        self.settings = get_settings()
        self.llm_client = llm_client or LLMClient()

    async def extract(self, content: str) -> list[ProfileFact]:
        """Extract profile facts from memory content."""
        if not self.settings.ENABLE_PROFILE_AUGMENTATION:
            return []

        if not content or not content.strip():
            return []

        prompt = self._build_prompt(content)
        try:
            response = await self.llm_client.complete(
                prompt=prompt,
                model=self.settings.PROFILE_EXTRACTION_MODEL,
                max_tokens=1024,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return self._parse_response(response)
        except Exception as e:
            logger.warning("profile_extraction_failed", error=str(e))
            return []

    def _build_prompt(self, content: str) -> str:
        return (
            "Extract structured profile facts from the following memory. "
            "Return a JSON object with a single key 'facts' containing a list of facts. "
            "Each fact must have: category (one of identity, preference, goal, constraint, habit), "
            "key (short field name in English), value (the fact value), and confidence (0.0-1.0). "
            "Only include facts that describe the user. If none, return an empty list.\n\n"
            f"Memory:\n{content}\n\n"
            "Example output:\n"
            '{"facts": [{"category": "identity", "key": "role", "value": "backend engineer", "confidence": 0.95}]}'
        )

    def _parse_response(self, response: str) -> list[ProfileFact]:
        try:
            data = json.loads(response)
            raw_facts = data.get("facts", []) if isinstance(data, dict) else []
        except json.JSONDecodeError as e:
            logger.warning("profile_extraction_parse_failed", response=response[:200], error=str(e))
            return []

        facts: list[ProfileFact] = []
        allowed_categories = {"identity", "preference", "goal", "constraint", "habit"}
        for item in raw_facts[: self.settings.PROFILE_MAX_FACTS_PER_MEMORY]:
            try:
                category = str(item["category"]).lower()
                if category not in allowed_categories:
                    logger.warning("profile_fact_invalid_category", item=item, category=category)
                    continue
                raw_confidence = float(item.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, raw_confidence))
                facts.append(
                    ProfileFact(
                        category=category,
                        key=str(item["key"]),
                        value=item["value"],
                        confidence=confidence,
                    )
                )
            except (KeyError, ValueError, TypeError) as e:
                logger.warning("profile_fact_parse_failed", item=item, error=str(e))
                continue
        return facts
