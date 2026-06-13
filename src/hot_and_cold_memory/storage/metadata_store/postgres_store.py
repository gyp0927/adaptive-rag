"""Metadata store implementation using SQLAlchemy async (PostgreSQL/SQLite)."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, case, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hot_and_cold_memory.core.config import Tier, get_settings
from hot_and_cold_memory.core.logging import get_logger

from .base import (
    AccessLog,
    BaseMetadataStore,
    MemoryItem,
    MemoryLink,
    MigrationLog,
    TopicCluster,
)
from .models import (
    AccessLogModel,
    Base,
    MemoryLinkModel,
    MemoryModel,
    MigrationLogModel,
    ProfileFactModel,
    ProfileModel,
    TopicClusterModel,
)

logger = get_logger(__name__)


def _to_uuid_str(value: uuid.UUID | str) -> str:
    """Convert UUID to string."""
    return str(value) if isinstance(value, uuid.UUID) else value


def _memory_to_item(model: MemoryModel) -> MemoryItem:
    """Convert MemoryModel to MemoryItem dataclass."""
    return MemoryItem(
        memory_id=uuid.UUID(model.memory_id) if isinstance(model.memory_id, str) else model.memory_id,
        tier=Tier(model.tier),
        content=model.content or "",
        original_length=model.original_length,
        memory_type=model.memory_type,
        source=model.source,
        importance=model.importance,
        access_count=model.access_count,
        frequency_score=model.frequency_score,
        created_at=model.created_at,
        updated_at=model.updated_at,
        last_accessed_at=model.last_accessed_at,
        last_migrated_at=model.last_migrated_at,
        topic_cluster_id=uuid.UUID(model.topic_cluster_id) if model.topic_cluster_id else None,
        tags=list(model.tags) if model.tags else [],
        attributes=dict(model.attributes) if model.attributes else {},
        vector_id=model.vector_id,
        compressed=bool(model.compressed),
        expires_at=model.expires_at,
    )


def _cluster_to_dataclass(model: TopicClusterModel) -> TopicCluster:
    """Convert TopicClusterModel to TopicCluster dataclass."""
    return TopicCluster(
        cluster_id=uuid.UUID(model.cluster_id) if isinstance(model.cluster_id, str) else model.cluster_id,
        centroid=list(model.centroid) if model.centroid else [],
        representative_query=model.representative_query,
        access_count=model.access_count,
        frequency_score=model.frequency_score,
        member_count=model.member_count,
        created_at=model.created_at,
        last_accessed_at=model.last_accessed_at,
    )


class PostgresMetadataStore(BaseMetadataStore):
    """Metadata store implementation supporting PostgreSQL and SQLite."""

    def __init__(self) -> None:
        self.settings = get_settings()
        db_url = str(self.settings.METADATA_DB_URL)

        # SQLite doesn't support connection pooling
        if db_url.startswith("sqlite"):
            self.engine = create_async_engine(
                db_url,
                echo=self.settings.DEBUG,
            )
            from sqlalchemy import event
            @event.listens_for(self.engine.sync_engine, "connect")
            def _enable_sqlite_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            self.engine = create_async_engine(
                db_url,
                echo=self.settings.DEBUG,
                pool_size=10,
                max_overflow=20,
            )

        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def initialize(self) -> None:
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("metadata_store_initialized")

    # --- Memory operations ---

    async def create_memory(self, metadata: MemoryItem) -> None:
        """Create a new memory record."""
        async with self.async_session() as session:
            model = MemoryModel(
                memory_id=_to_uuid_str(metadata.memory_id),
                tier=metadata.tier.value,
                content=metadata.content,
                original_length=metadata.original_length,
                memory_type=metadata.memory_type,
                source=metadata.source,
                importance=metadata.importance,
                access_count=metadata.access_count,
                frequency_score=metadata.frequency_score,
                created_at=metadata.created_at,
                updated_at=metadata.updated_at,
                last_accessed_at=metadata.last_accessed_at,
                last_migrated_at=metadata.last_migrated_at,
                topic_cluster_id=_to_uuid_str(metadata.topic_cluster_id) if metadata.topic_cluster_id else None,
                tags=metadata.tags,
                attributes=metadata.attributes,
                vector_id=metadata.vector_id,
                compressed=metadata.compressed,
                expires_at=metadata.expires_at,
            )
            session.add(model)
            await session.commit()

    async def get_memory(self, memory_id: uuid.UUID) -> MemoryItem | None:
        """Get memory by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(MemoryModel).where(MemoryModel.memory_id == _to_uuid_str(memory_id))
            )
            model = result.scalar_one_or_none()
            return _memory_to_item(model) if model else None

    async def get_memories_batch(self, memory_ids: list[uuid.UUID]) -> list[MemoryItem]:
        """Get multiple memories by ID in a single query."""
        if not memory_ids:
            return []
        id_strs = [_to_uuid_str(mid) for mid in memory_ids]
        async with self.async_session() as session:
            result = await session.execute(
                select(MemoryModel).where(MemoryModel.memory_id.in_(id_strs))
            )
            models = result.scalars().all()
            return [_memory_to_item(m) for m in models]

    async def create_memories_batch(self, metadatas: list[MemoryItem]) -> None:
        """Create multiple memory records in a single transaction."""
        if not metadatas:
            return
        async with self.async_session() as session:
            models = [
                MemoryModel(
                    memory_id=_to_uuid_str(m.memory_id),
                    tier=m.tier.value,
                    content=m.content,
                    original_length=m.original_length,
                    memory_type=m.memory_type,
                    source=m.source,
                    importance=m.importance,
                    access_count=m.access_count,
                    frequency_score=m.frequency_score,
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                    last_accessed_at=m.last_accessed_at,
                    last_migrated_at=m.last_migrated_at,
                    topic_cluster_id=_to_uuid_str(m.topic_cluster_id) if m.topic_cluster_id else None,
                    tags=m.tags,
                    attributes=m.attributes,
                    vector_id=m.vector_id,
                    compressed=m.compressed,
                    expires_at=m.expires_at,
                )
                for m in metadatas
            ]
            session.add_all(models)
            await session.commit()

    async def update_memory(
        self,
        memory_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> MemoryItem | None:
        """Update memory fields."""
        async with self.async_session() as session:
            if "tier" in updates and isinstance(updates["tier"], Tier):
                updates["tier"] = updates["tier"].value

            result = await session.execute(
                update(MemoryModel)
                .where(MemoryModel.memory_id == _to_uuid_str(memory_id))
                .values(**updates, updated_at=datetime.now(UTC))
                .returning(MemoryModel)
            )
            await session.commit()
            model = result.scalar_one_or_none()
            return _memory_to_item(model) if model else None

    async def update_memories_batch(
        self,
        updates: dict[uuid.UUID, dict[str, Any]],
    ) -> None:
        """Update multiple memories in a single transaction using one UPDATE with CASE."""
        if not updates:
            return
        async with self.async_session() as session:
            id_strs = [_to_uuid_str(mid) for mid in updates]

            # Collect all columns being updated
            all_columns: set[str] = set()
            for upd in updates.values():
                all_columns.update(upd.keys())

            values_to_set: dict[str, Any] = {"updated_at": datetime.now(UTC)}

            for col in all_columns:
                col_attr = getattr(MemoryModel, col)
                whens: list[tuple[Any, Any]] = []
                for mid, upd in updates.items():
                    if col not in upd:
                        continue
                    val = upd[col]
                    if col == "tier" and isinstance(val, Tier):
                        val = val.value
                    whens.append((MemoryModel.memory_id == _to_uuid_str(mid), val))
                if whens:
                    values_to_set[col] = case(*whens, else_=col_attr)

            await session.execute(
                update(MemoryModel)
                .where(MemoryModel.memory_id.in_(id_strs))
                .values(**values_to_set)
            )
            await session.commit()

    async def delete_memories(self, memory_ids: list[uuid.UUID]) -> int:
        """Delete memories."""
        id_strs = [_to_uuid_str(mid) for mid in memory_ids]
        async with self.async_session() as session:
            result = await session.execute(
                delete(MemoryModel).where(MemoryModel.memory_id.in_(id_strs))
            )
            await session.commit()
            return result.rowcount or 0  # type: ignore[attr-defined]

    async def list_memories(
        self,
        memory_type: str | None = None,
        source: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryItem]:
        """List memories with optional filtering."""
        async with self.async_session() as session:
            stmt = select(MemoryModel)
            if memory_type:
                stmt = stmt.where(MemoryModel.memory_type == memory_type)
            if source:
                stmt = stmt.where(MemoryModel.source == source)
            result = await session.execute(
                stmt.order_by(MemoryModel.created_at.desc()).limit(limit).offset(offset)
            )
            models = result.scalars().all()
            return [_memory_to_item(m) for m in models]

    async def count_memories(
        self,
        memory_type: str | None = None,
        source: str | None = None,
    ) -> int:
        """Count memories with optional filtering."""
        async with self.async_session() as session:
            stmt = select(func.count(MemoryModel.memory_id))
            if memory_type:
                stmt = stmt.where(MemoryModel.memory_type == memory_type)
            if source:
                stmt = stmt.where(MemoryModel.source == source)
            result = await session.execute(stmt)
            return result.scalar() or 0

    async def count_memories_by_tier(self, tier: Tier) -> int:
        """Count memories in a given tier."""
        async with self.async_session() as session:
            result = await session.execute(
                select(func.count(MemoryModel.memory_id)).where(MemoryModel.tier == tier.value)
            )
            return result.scalar() or 0

    # Keyword search guardrails to prevent expensive full-table scans.
    _KEYWORD_MAX_TERMS: int = 5
    _KEYWORD_MIN_TERM_LENGTH: int = 2

    async def search_by_keyword(
        self,
        query_text: str,
        tier: Tier | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """Search memories by keyword match in content (cross-db compatible).

        Guardrails:
        - Up to 5 terms (excess dropped)
        - Terms shorter than 2 chars are skipped
        - Hard limit of 100 results
        """
        async with self.async_session() as session:
            conditions = []
            if tier is not None:
                conditions.append(MemoryModel.tier == tier.value)

            # Split query into terms and require each term to match somewhere
            raw_terms = [t.strip() for t in query_text.split() if t.strip()]
            terms = [
                t for t in raw_terms[: self._KEYWORD_MAX_TERMS]
                if len(t) >= self._KEYWORD_MIN_TERM_LENGTH
            ]
            if not terms:
                return []

            for term in terms:
                conditions.append(MemoryModel.content.ilike(f"%{term}%"))

            stmt = select(MemoryModel).where(and_(*conditions)).limit(min(limit, 100))
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [_memory_to_item(m) for m in models]

    async def close(self) -> None:
        """Dispose the database engine."""
        await self.engine.dispose()

    # --- Profile operations ---

    async def create_profile_fact(
        self,
        user_id: str,
        memory_id: uuid.UUID | None,
        category: str,
        key: str,
        value: Any,
        confidence: float,
    ) -> None:
        """Create a new profile fact."""
        async with self.async_session() as session:
            model = ProfileFactModel(
                user_id=user_id,
                memory_id=_to_uuid_str(memory_id) if memory_id else None,
                category=category,
                key=key,
                value=value,
                confidence=confidence,
            )
            session.add(model)
            await session.commit()

    async def get_current_profile_fact(
        self,
        user_id: str,
        category: str,
        key: str,
    ) -> dict[str, Any] | None:
        """Get current fact by user/category/key."""
        async with self.async_session() as session:
            result = await session.execute(
                select(ProfileFactModel)
                .where(
                    ProfileFactModel.user_id == user_id,
                    ProfileFactModel.category == category,
                    ProfileFactModel.key == key,
                    ProfileFactModel.is_current.is_(True),
                )
                .order_by(ProfileFactModel.valid_from.desc())
            )
            model = result.scalars().first()
            if not model:
                return None
            return self._profile_fact_to_dict(model)

    async def expire_profile_fact(self, fact_id: str) -> None:
        """Mark a fact as expired."""
        async with self.async_session() as session:
            await session.execute(
                update(ProfileFactModel)
                .where(ProfileFactModel.fact_id == fact_id)
                .values(valid_until=datetime.now(UTC), is_current=False)
            )
            await session.commit()

    async def update_profile_fact_confidence(
        self,
        fact_id: str,
        confidence: float,
    ) -> None:
        """Update fact confidence."""
        async with self.async_session() as session:
            await session.execute(
                update(ProfileFactModel)
                .where(ProfileFactModel.fact_id == fact_id)
                .values(confidence=confidence)
            )
            await session.commit()

    async def list_profile_facts(
        self,
        user_id: str,
        category: str | None = None,
        key: str | None = None,
        is_current: bool | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List profile facts."""
        async with self.async_session() as session:
            stmt = select(ProfileFactModel).where(ProfileFactModel.user_id == user_id)
            if category is not None:
                stmt = stmt.where(ProfileFactModel.category == category)
            if key is not None:
                stmt = stmt.where(ProfileFactModel.key == key)
            if is_current is not None:
                stmt = stmt.where(ProfileFactModel.is_current.is_(is_current))
            stmt = stmt.order_by(ProfileFactModel.valid_from.desc()).limit(limit)
            result = await session.execute(stmt)
            return [self._profile_fact_to_dict(m) for m in result.scalars().all()]

    async def get_profile(self, user_id: str) -> dict[str, Any] | None:
        """Get profile snapshot."""
        async with self.async_session() as session:
            result = await session.execute(
                select(ProfileModel).where(ProfileModel.user_id == user_id)
            )
            model = result.scalar_one_or_none()
            return self._profile_to_dict(model) if model else None

    async def upsert_profile(
        self,
        user_id: str,
        snapshot: dict[str, Any],
    ) -> None:
        """Upsert profile snapshot."""
        async with self.async_session() as session:
            existing = await session.execute(
                select(ProfileModel).where(ProfileModel.user_id == user_id)
            )
            model = existing.scalar_one_or_none()
            if model:
                for key, value in snapshot.items():
                    setattr(model, key, value)
                model.version += 1
                model.updated_at = datetime.now(UTC)
            else:
                model = ProfileModel(user_id=user_id, **snapshot)
                session.add(model)
            await session.commit()

    def _profile_fact_to_dict(self, model: ProfileFactModel) -> dict[str, Any]:
        return {
            "fact_id": model.fact_id,
            "user_id": model.user_id,
            "memory_id": model.memory_id,
            "category": model.category,
            "key": model.key,
            "value": model.value,
            "confidence": model.confidence,
            "valid_from": model.valid_from,
            "valid_until": model.valid_until,
            "is_current": model.is_current,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
        }

    def _profile_to_dict(self, model: ProfileModel) -> dict[str, Any]:
        return {
            "user_id": model.user_id,
            "identity": dict(model.identity) if model.identity else {},
            "preferences": list(model.preferences) if model.preferences else [],
            "goals": list(model.goals) if model.goals else [],
            "constraints": list(model.constraints) if model.constraints else [],
            "habits": list(model.habits) if model.habits else [],
            "attributes": dict(model.attributes) if model.attributes else {},
            "summary": model.summary,
            "version": model.version,
            "updated_at": model.updated_at,
        }

    async def query_memories_by_tier_and_score(
        self,
        tier: Tier,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int = 100,
        order_desc: bool = False,
    ) -> list[MemoryItem]:
        """Query memories by tier and frequency score range."""
        async with self.async_session() as session:
            conditions = [MemoryModel.tier == tier.value]

            if min_score is not None:
                conditions.append(MemoryModel.frequency_score >= min_score)
            if max_score is not None:
                conditions.append(MemoryModel.frequency_score <= max_score)

            stmt = (
                select(MemoryModel)
                .where(and_(*conditions))
                .limit(limit)
            )
            if order_desc:
                stmt = stmt.order_by(MemoryModel.frequency_score.desc())
            else:
                stmt = stmt.order_by(MemoryModel.frequency_score)

            result = await session.execute(stmt)
            models = result.scalars().all()
            return [_memory_to_item(m) for m in models]

    async def query_forgettable_memories(
        self,
        tier: Tier,
        max_importance: float,
        cutoff: datetime,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """Query memories eligible for deletion (forgetting)."""
        async with self.async_session() as session:
            conditions = [
                MemoryModel.tier == tier.value,
                MemoryModel.importance < max_importance,
                MemoryModel.compressed.is_(True),
            ]
            # Either never accessed (created_at < cutoff), last_accessed_at < cutoff,
            # or explicit expires_at has passed
            conditions.append(
                or_(
                    and_(
                        MemoryModel.last_accessed_at.is_(None),
                        MemoryModel.created_at < cutoff,
                    ),
                    MemoryModel.last_accessed_at < cutoff,
                    and_(
                        MemoryModel.expires_at.isnot(None),
                        MemoryModel.expires_at < datetime.now(UTC),
                    ),
                )
            )

            stmt = (
                select(MemoryModel)
                .where(and_(*conditions))
                .limit(limit)
            )
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [_memory_to_item(m) for m in models]

    async def increment_access(
        self,
        memory_ids: list[uuid.UUID],
        cluster_id: uuid.UUID | None,
        timestamp: datetime,
    ) -> None:
        """Increment access count for memories in a single UPDATE."""
        if not memory_ids:
            return
        id_strs = [_to_uuid_str(mid) for mid in memory_ids]
        cluster_str = _to_uuid_str(cluster_id) if cluster_id else None
        async with self.async_session() as session:
            values: dict[str, Any] = {
                "access_count": MemoryModel.access_count + 1,
                "last_accessed_at": timestamp,
            }
            if cluster_str is not None:
                values["topic_cluster_id"] = cluster_str
            await session.execute(
                update(MemoryModel)
                .where(MemoryModel.memory_id.in_(id_strs))
                .values(**values)
            )
            await session.commit()

    # --- Topic cluster operations ---

    async def create_cluster(self, cluster: TopicCluster) -> None:
        """Create a new topic cluster."""
        async with self.async_session() as session:
            model = TopicClusterModel(
                cluster_id=_to_uuid_str(cluster.cluster_id),
                representative_query=cluster.representative_query,
                access_count=cluster.access_count,
                frequency_score=cluster.frequency_score,
                member_count=cluster.member_count,
                created_at=cluster.created_at,
                last_accessed_at=cluster.last_accessed_at,
                centroid=cluster.centroid,
            )
            session.add(model)
            await session.commit()

    async def get_cluster(self, cluster_id: uuid.UUID) -> TopicCluster | None:
        """Get cluster by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(TopicClusterModel).where(TopicClusterModel.cluster_id == _to_uuid_str(cluster_id))
            )
            model = result.scalar_one_or_none()
            return _cluster_to_dataclass(model) if model else None

    async def update_cluster(
        self,
        cluster_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> TopicCluster | None:
        """Update cluster fields."""
        async with self.async_session() as session:
            await session.execute(
                update(TopicClusterModel)
                .where(TopicClusterModel.cluster_id == _to_uuid_str(cluster_id))
                .values(**updates)
            )
            await session.commit()

            result = await session.execute(
                select(TopicClusterModel).where(TopicClusterModel.cluster_id == _to_uuid_str(cluster_id))
            )
            model = result.scalar_one_or_none()
            return _cluster_to_dataclass(model) if model else None

    async def get_all_clusters(self) -> list[TopicCluster]:
        """Get all topic clusters."""
        async with self.async_session() as session:
            result = await session.execute(select(TopicClusterModel))
            models = result.scalars().all()
            return [_cluster_to_dataclass(m) for m in models]

    async def get_clusters_batch(self, cluster_ids: list[uuid.UUID]) -> list[TopicCluster]:
        """Get multiple clusters by ID in a single query."""
        if not cluster_ids:
            return []
        id_strs = [_to_uuid_str(cid) for cid in cluster_ids]
        async with self.async_session() as session:
            result = await session.execute(
                select(TopicClusterModel).where(TopicClusterModel.cluster_id.in_(id_strs))
            )
            models = result.scalars().all()
            return [_cluster_to_dataclass(m) for m in models]

    async def delete_clusters(self, cluster_ids: list[uuid.UUID]) -> int:
        """Delete topic clusters."""
        id_strs = [_to_uuid_str(cid) for cid in cluster_ids]
        async with self.async_session() as session:
            result = await session.execute(
                delete(TopicClusterModel).where(TopicClusterModel.cluster_id.in_(id_strs))
            )
            await session.commit()
            return result.rowcount or 0  # type: ignore[attr-defined]

    # --- Access / migration log operations ---

    async def create_access_log(self, log: AccessLog) -> None:
        """Create an access log entry."""
        async with self.async_session() as session:
            model = AccessLogModel(
                memory_id=_to_uuid_str(log.memory_id),
                query_cluster_id=_to_uuid_str(log.query_cluster_id) if log.query_cluster_id else None,
                query_text=log.query_text,
                retrieved_at=log.retrieved_at,
                response_time_ms=log.response_time_ms,
                tier_accessed=log.tier_accessed,
            )
            session.add(model)
            await session.commit()

    async def create_access_logs_batch(self, logs: list[AccessLog]) -> None:
        """Create multiple access log entries in a single transaction."""
        if not logs:
            return
        async with self.async_session() as session:
            models = [
                AccessLogModel(
                    memory_id=_to_uuid_str(log.memory_id),
                    query_cluster_id=_to_uuid_str(log.query_cluster_id) if log.query_cluster_id else None,
                    query_text=log.query_text,
                    retrieved_at=log.retrieved_at,
                    response_time_ms=log.response_time_ms,
                    tier_accessed=log.tier_accessed,
                )
                for log in logs
            ]
            session.add_all(models)
            await session.commit()

    async def create_migration_log(self, log: MigrationLog) -> None:
        """Create a migration log entry."""
        async with self.async_session() as session:
            model = MigrationLogModel(
                memory_id=_to_uuid_str(log.memory_id),
                direction=log.direction,
                original_size=log.original_size,
                new_size=log.new_size,
                compression_ratio=log.compression_ratio,
                started_at=log.started_at,
                completed_at=log.completed_at,
                status=log.status,
                error_message=log.error_message,
            )
            session.add(model)
            await session.commit()

    async def update_migration_log(
        self,
        log_id: int,
        updates: dict[str, Any],
    ) -> None:
        """Update migration log."""
        async with self.async_session() as session:
            await session.execute(
                update(MigrationLogModel)
                .where(MigrationLogModel.log_id == log_id)
                .values(**updates)
            )
            await session.commit()

    # --- Association graph operations ---

    async def create_link(self, link: MemoryLink) -> None:
        """Create a link between two memories (upsert on duplicate)."""
        async with self.async_session() as session:
            # Check if reverse link already exists; if so, update strength
            existing = await session.execute(
                select(MemoryLinkModel)
                .where(
                    or_(
                        and_(
                            MemoryLinkModel.source_memory_id == _to_uuid_str(link.source_memory_id),
                            MemoryLinkModel.target_memory_id == _to_uuid_str(link.target_memory_id),
                        ),
                        and_(
                            MemoryLinkModel.source_memory_id == _to_uuid_str(link.target_memory_id),
                            MemoryLinkModel.target_memory_id == _to_uuid_str(link.source_memory_id),
                        ),
                    )
                )
            )
            model = existing.scalar_one_or_none()
            if model:
                model.strength += link.strength * 0.1
                model.last_accessed_at = datetime.now(UTC)
            else:
                model = MemoryLinkModel(
                    source_memory_id=_to_uuid_str(link.source_memory_id),
                    target_memory_id=_to_uuid_str(link.target_memory_id),
                    link_type=link.link_type,
                    strength=link.strength,
                    created_at=link.created_at,
                    last_accessed_at=link.last_accessed_at,
                )
                session.add(model)
            await session.commit()

    async def get_related_memories(
        self,
        memory_id: uuid.UUID,
        link_type: str | None = None,
        min_strength: float | None = None,
        limit: int = 20,
    ) -> list[tuple[MemoryLink, MemoryItem]]:
        """Get memories related to a given memory."""
        mid_str = _to_uuid_str(memory_id)
        async with self.async_session() as session:
            conditions = [
                or_(
                    MemoryLinkModel.source_memory_id == mid_str,
                    MemoryLinkModel.target_memory_id == mid_str,
                )
            ]
            if link_type:
                conditions.append(MemoryLinkModel.link_type == link_type)
            if min_strength is not None:
                conditions.append(MemoryLinkModel.strength >= min_strength)

            stmt = (
                select(MemoryLinkModel)
                .where(and_(*conditions))
                .order_by(MemoryLinkModel.strength.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            link_models = result.scalars().all()

            # Fetch target memory details
            related_ids = []
            for lm in link_models:
                target = lm.target_memory_id if lm.source_memory_id == mid_str else lm.source_memory_id
                related_ids.append(target)

            if not related_ids:
                return []

            mem_result = await session.execute(
                select(MemoryModel).where(MemoryModel.memory_id.in_(related_ids))
            )
            mem_map = {m.memory_id: m for m in mem_result.scalars().all()}

            out: list[tuple[MemoryLink, MemoryItem]] = []
            for lm in link_models:
                target = lm.target_memory_id if lm.source_memory_id == mid_str else lm.source_memory_id
                mem = mem_map.get(target)
                if mem:
                    out.append((
                        MemoryLink(
                            source_memory_id=uuid.UUID(lm.source_memory_id),
                            target_memory_id=uuid.UUID(lm.target_memory_id),
                            link_type=lm.link_type,
                            strength=lm.strength,
                            created_at=lm.created_at,
                            last_accessed_at=lm.last_accessed_at,
                            link_id=lm.link_id,
                        ),
                        _memory_to_item(mem),
                    ))
            return out

    async def delete_links_for_memories(self, memory_ids: list[uuid.UUID]) -> int:
        """Delete all links involving any of the given memory IDs."""
        if not memory_ids:
            return 0
        id_strs = [_to_uuid_str(mid) for mid in memory_ids]
        async with self.async_session() as session:
            result = await session.execute(
                delete(MemoryLinkModel).where(
                    or_(
                        MemoryLinkModel.source_memory_id.in_(id_strs),
                        MemoryLinkModel.target_memory_id.in_(id_strs),
                    )
                )
            )
            await session.commit()
            return result.rowcount or 0  # type: ignore[attr-defined]
