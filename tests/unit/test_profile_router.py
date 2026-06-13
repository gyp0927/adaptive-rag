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

    def test_get_profile_history(self, client):
        response = client.get("/api/v1/profile/history")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_profile_history_with_is_current(self, client):
        response = client.get("/api/v1/profile/history?is_current=true")
        assert response.status_code == 200
        mock_store = profile_router._profile_store
        mock_store.list_facts.assert_awaited_once()
        call_kwargs = mock_store.list_facts.call_args.kwargs
        assert call_kwargs.get("is_current") is True

    def test_get_profile_503_when_store_not_initialized(self, client):
        original_store = profile_router._profile_store
        profile_router._profile_store = None
        try:
            response = client.get("/api/v1/profile")
            assert response.status_code == 503
            assert response.json()["detail"] == "Profile store not initialized"
        finally:
            profile_router._profile_store = original_store

    def test_get_profile_history_503_when_store_not_initialized(self, client):
        original_store = profile_router._profile_store
        profile_router._profile_store = None
        try:
            response = client.get("/api/v1/profile/history")
            assert response.status_code == 503
            assert response.json()["detail"] == "Profile store not initialized"
        finally:
            profile_router._profile_store = original_store

    def test_get_profile_empty_default_when_no_data(self, client):
        mock_store = profile_router._profile_store
        mock_store.metadata_store.get_profile = AsyncMock(return_value=None)
        response = client.get("/api/v1/profile")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "default"
        assert data["identity"] == {}
        assert data["preferences"] == []
        assert data["goals"] == []
        assert data["constraints"] == []
        assert data["habits"] == []
        assert data["attributes"] == {}
        assert data["summary"] is None
        assert data["updated_at"] is None
