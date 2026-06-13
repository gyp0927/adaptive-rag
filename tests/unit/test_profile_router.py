"""Tests for the profile API router."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from hot_and_cold_memory.api.main import create_app
from hot_and_cold_memory.api.routers import profile as profile_router


class TestProfileRouter:
    @pytest.fixture
    def client(self):
        app = create_app()
        mock_store = MagicMock()
        mock_store.metadata_store.get_profile = AsyncMock(return_value={
            "user_id": "default",
            "identity": {"role": "agent engineer"},
            "preferences": [],
            "goals": [],
            "constraints": [],
            "habits": [],
            "attributes": {},
            "summary": None,
            "version": 1,
            "updated_at": None,
        })
        mock_store.list_facts = AsyncMock(return_value=[])
        profile_router.set_profile_store(mock_store)
        return TestClient(app)

    def test_get_profile(self, client):
        response = client.get("/api/v1/profile")
        assert response.status_code == 200
        assert response.json()["identity"]["role"] == "agent engineer"
