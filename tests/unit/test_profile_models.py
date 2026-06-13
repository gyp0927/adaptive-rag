"""Unit tests for profile SQLAlchemy models."""

from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from hot_and_cold_memory.storage.metadata_store.models import ProfileFactModel, ProfileModel


@pytest.mark.asyncio
async def test_profile_fact_model_defaults(metadata_store):
    async with metadata_store.async_session() as session:
        fact = ProfileFactModel(
            user_id="defaults_user",
            category="identity",
            key="role",
            value="agent engineer",
        )
        session.add(fact)
        await session.commit()

        result = await session.execute(
            select(ProfileFactModel).where(ProfileFactModel.user_id == "defaults_user")
        )
        retrieved = result.scalar_one()
        assert retrieved.is_current is True
        assert retrieved.confidence == 0.5
        assert isinstance(retrieved.fact_id, str) and retrieved.fact_id
        assert isinstance(retrieved.created_at, datetime)
        assert isinstance(retrieved.updated_at, datetime)


@pytest.mark.asyncio
async def test_profile_model_defaults(metadata_store):
    async with metadata_store.async_session() as session:
        profile = ProfileModel(user_id="defaults_user")
        session.add(profile)
        await session.commit()

        result = await session.execute(
            select(ProfileModel).where(ProfileModel.user_id == "defaults_user")
        )
        retrieved = result.scalar_one()
        assert retrieved.version == 1
        assert retrieved.identity == {}
        assert retrieved.preferences == []
        assert retrieved.goals == []
        assert retrieved.constraints == []
        assert retrieved.habits == []
        assert retrieved.attributes == {}
        assert isinstance(retrieved.profile_id, str) and retrieved.profile_id
        assert isinstance(retrieved.updated_at, datetime)


@pytest.mark.asyncio
async def test_profile_model_user_id_unique(metadata_store):
    async with metadata_store.async_session() as session:
        profile1 = ProfileModel(user_id="unique_user", identity={"role": "agent engineer"})
        session.add(profile1)
        await session.commit()

        profile2 = ProfileModel(user_id="unique_user", identity={"role": "tester"})
        session.add(profile2)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


@pytest.mark.asyncio
async def test_profile_fact_model_create(metadata_store):
    async with metadata_store.async_session() as session:
        fact = ProfileFactModel(
            user_id="create_user",
            category="identity",
            key="role",
            value="agent engineer",
            confidence=0.95,
        )
        session.add(fact)
        await session.commit()

        result = await session.execute(
            select(ProfileFactModel).where(ProfileFactModel.user_id == "create_user")
        )
        assert result.scalar_one().confidence == 0.95


@pytest.mark.asyncio
async def test_profile_model_create(metadata_store):
    async with metadata_store.async_session() as session:
        profile = ProfileModel(user_id="create_user", identity={"role": "agent engineer"})
        session.add(profile)
        await session.commit()

        result = await session.execute(
            select(ProfileModel).where(ProfileModel.user_id == "create_user")
        )
        assert result.scalar_one().identity == {"role": "agent engineer"}
