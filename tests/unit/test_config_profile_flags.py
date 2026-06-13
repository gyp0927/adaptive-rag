"""Tests for profile configuration flags."""

import pytest

from hot_and_cold_memory.core.config import get_settings


def test_profile_config_defaults(monkeypatch):
    """Profile feature flags and parameters should have correct defaults."""
    import hot_and_cold_memory.core.config as _config

    monkeypatch.setattr(_config, "_settings", None)
    settings = get_settings()
    assert settings.ENABLE_PROFILE_AUGMENTATION is True
    assert settings.ENABLE_PROFILE_QUERY_REWRITE is True
    assert settings.ENABLE_PROFILE_RANKING_BOOST is True
    assert settings.ENABLE_PROFILE_RECONCILER is True
    assert settings.PROFILE_RECONCILER_CRON == "0 3 * * *"
    assert settings.PROFILE_BOOST_WEIGHT == 0.15
    assert settings.PROFILE_EXTRACTION_MODEL == "gpt-4o-mini"
    assert settings.PROFILE_MAX_FACTS_PER_MEMORY == 10

    # Reset singleton to avoid cross-test pollution
    monkeypatch.setattr(_config, "_settings", None)
