"""Metadata store implementation using SQLAlchemy async (PostgreSQL/SQLite)."""

from datetime import datetime
from typing import Any
import uuid

from sqlalchemy import select, update, delete, and_
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from adaptive_rag.core.config import get_settings, Tier
from adaptive_rag.core.logging import get_logger

from .base import (
    BaseMetadataStore,
    ChunkMetadata,
    DocumentMetadata,
    QueryCluster,
    MigrationLog,
)
from .models import Base, ChunkModel, DocumentModel, QueryClusterModel, MigrationLogModel

logger = get_logger(__name__)


def _to_uuid_str(value: uuid.UUID | str) -> str:
    """Convert UUID to string."""
    return str(value) if isinstance(value, uuid.UUID) else value


def _chunk_to_metadata(model: ChunkModel) -> ChunkMetadata:
    """Convert ChunkModel to ChunkMetadata dataclass."""
    return ChunkMetadata(
        chunk_id=uuid.UUID(model.chunk_id) if isinstance(model.chunk_id, str) else model.chunk_id,
        document_id=uuid.UUID(model.document_id) if isinstance(model.document_id, str) else model.document_id,
        tier=Tier(model.tier),
        original_length=model.original_length,
        compressed_length=model.compressed_length,
        chunk_index=model.chunk_index,
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
    )


def _document_to_metadata(model: DocumentModel) -> DocumentMetadata:
    """Convert DocumentModel to DocumentMetadata dataclass."""
    return DocumentMetadata(
        document_id=uuid.UUID(model.document_id) if isinstance(model.document_id, str) else model.document_id,
        source_type=model.source_type,
        source_uri=model.source_uri,
        title=model.title,
        author=model.author,
        content_hash=model.content_hash,
        total_chunks=model.total_chunks,
        metadata=dict(model.doc_metadata) if model.doc_metadata else {},
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _cluster_to_dataclass(model: QueryClusterModel) -> QueryCluster:
    """Convert QueryClusterModel to QueryCluster dataclass."""
    return QueryCluster(
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

    async def create_chunk(self, metadata: ChunkMetadata) -> None:
        """Create a new chunk record."""
        async with self.async_session() as session:
            model = ChunkModel(
                chunk_id=_to_uuid_str(metadata.chunk_id),
                document_id=_to_uuid_str(metadata.document_id),
                tier=metadata.tier.value,
                original_length=metadata.original_length,
                compressed_length=metadata.compressed_length,
                chunk_index=metadata.chunk_index,
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
            )
            session.add(model)
            await session.commit()

    async def get_chunk(self, chunk_id: uuid.UUID) -> ChunkMetadata | None:
        """Get chunk by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(ChunkModel).where(ChunkModel.chunk_id == _to_uuid_str(chunk_id))
            )
            model = result.scalar_one_or_none()
            return _chunk_to_metadata(model) if model else None

    async def get_chunks_batch(self, chunk_ids: list[uuid.UUID]) -> list[ChunkMetadata]:
        """Get multiple chunks by ID in a single query."""
        if not chunk_ids:
            return []
        id_strs = [_to_uuid_str(cid) for cid in chunk_ids]
        async with self.async_session() as session:
            result = await session.execute(
                select(ChunkModel).where(ChunkModel.chunk_id.in_(id_strs))
            )
            models = result.scalars().all()
            return [_chunk_to_metadata(m) for m in models]

    async def create_chunks_batch(self, metadatas: list[ChunkMetadata]) -> None:
        """Create multiple chunk records in a single transaction."""
        if not metadatas:
            return
        async with self.async_session() as session:
            models = [
                ChunkModel(
                    chunk_id=_to_uuid_str(m.chunk_id),
                    document_id=_to_uuid_str(m.document_id),
                    tier=m.tier.value,
                    original_length=m.original_length,
                    compressed_length=m.compressed_length,
                    chunk_index=m.chunk_index,
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
                )
                for m in metadatas
            ]
            session.add_all(models)
            await session.commit()

    async def update_chunk(
        self,
        chunk_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> ChunkMetadata | None:
        """Update chunk fields."""
        async with self.async_session() as session:
            if "tier" in updates and isinstance(updates["tier"], Tier):
                updates["tier"] = updates["tier"].value

            await session.execute(
                update(ChunkModel)
                .where(ChunkModel.chunk_id == _to_uuid_str(chunk_id))
                .values(**updates, updated_at=datetime.utcnow())
            )
            await session.commit()

            result = await session.execute(
                select(ChunkModel).where(ChunkModel.chunk_id == _to_uuid_str(chunk_id))
            )
            model = result.scalar_one_or_none()
            return _chunk_to_metadata(model) if model else None

    async def update_chunks_batch(
        self,
        updates: dict[uuid.UUID, dict[str, Any]],
    ) -> None:
        """Update multiple chunks in a single transaction."""
        if not updates:
            return
        async with self.async_session() as session:
            for chunk_id, chunk_updates in updates.items():
                upd = dict(chunk_updates)
                if "tier" in upd and isinstance(upd["tier"], Tier):
                    upd["tier"] = upd["tier"].value
                upd["updated_at"] = datetime.utcnow()
                await session.execute(
                    update(ChunkModel)
                    .where(ChunkModel.chunk_id == _to_uuid_str(chunk_id))
                    .values(**upd)
                )
            await session.commit()

    async def delete_chunks(self, chunk_ids: list[uuid.UUID]) -> int:
        """Delete chunks."""
        id_strs = [_to_uuid_str(cid) for cid in chunk_ids]
        async with self.async_session() as session:
            result = await session.execute(
                delete(ChunkModel).where(ChunkModel.chunk_id.in_(id_strs))
            )
            await session.commit()
            return result.rowcount or 0

    async def count_chunks_by_tier(self, tier: Tier) -> int:
        """Count chunks in a given tier."""
        async with self.async_session() as session:
            from sqlalchemy import func
            result = await session.execute(
                select(func.count(ChunkModel.chunk_id)).where(ChunkModel.tier == tier.value)
            )
            return result.scalar() or 0

    async def query_chunks_by_tier_and_score(
        self,
        tier: Tier,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int = 100,
    ) -> list[ChunkMetadata]:
        """Query chunks by tier and frequency score range."""
        async with self.async_session() as session:
            conditions = [ChunkModel.tier == tier.value]

            if min_score is not None:
                conditions.append(ChunkModel.frequency_score >= min_score)
            if max_score is not None:
                conditions.append(ChunkModel.frequency_score <= max_score)

            result = await session.execute(
                select(ChunkModel)
                .where(and_(*conditions))
                .order_by(ChunkModel.frequency_score)
                .limit(limit)
            )
            models = result.scalars().all()
            return [_chunk_to_metadata(m) for m in models]

    async def increment_access(
        self,
        chunk_ids: list[uuid.UUID],
        cluster_id: uuid.UUID | None,
        timestamp: datetime,
    ) -> None:
        """Increment access count for chunks."""
        cluster_str = _to_uuid_str(cluster_id) if cluster_id else None
        async with self.async_session() as session:
            for chunk_id in chunk_ids:
                await session.execute(
                    update(ChunkModel)
                    .where(ChunkModel.chunk_id == _to_uuid_str(chunk_id))
                    .values(
                        access_count=ChunkModel.access_count + 1,
                        last_accessed_at=timestamp,
                        topic_cluster_id=cluster_str,
                    )
                )
            await session.commit()

    async def create_document(self, metadata: DocumentMetadata) -> None:
        """Create a new document record."""
        async with self.async_session() as session:
            model = DocumentModel(
                document_id=_to_uuid_str(metadata.document_id),
                source_type=metadata.source_type,
                source_uri=metadata.source_uri,
                title=metadata.title,
                author=metadata.author,
                content_hash=metadata.content_hash,
                total_chunks=metadata.total_chunks,
                doc_metadata=metadata.metadata,
                created_at=metadata.created_at,
                updated_at=metadata.updated_at,
            )
            session.add(model)
            await session.commit()

    async def get_document(self, document_id: uuid.UUID) -> DocumentMetadata | None:
        """Get document by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(DocumentModel).where(DocumentModel.document_id == _to_uuid_str(document_id))
            )
            model = result.scalar_one_or_none()
            return _document_to_metadata(model) if model else None

    async def get_document_by_hash(self, content_hash: str) -> DocumentMetadata | None:
        """Get document by content hash (for deduplication)."""
        async with self.async_session() as session:
            result = await session.execute(
                select(DocumentModel).where(DocumentModel.content_hash == content_hash)
            )
            model = result.scalar_one_or_none()
            return _document_to_metadata(model) if model else None

    async def update_document(
        self,
        document_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> DocumentMetadata | None:
        """Update document fields."""
        async with self.async_session() as session:
            await session.execute(
                update(DocumentModel)
                .where(DocumentModel.document_id == _to_uuid_str(document_id))
                .values(**updates, updated_at=datetime.utcnow())
            )
            await session.commit()

            result = await session.execute(
                select(DocumentModel).where(DocumentModel.document_id == _to_uuid_str(document_id))
            )
            model = result.scalar_one_or_none()
            return _document_to_metadata(model) if model else None

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentMetadata]:
        """List documents with pagination."""
        async with self.async_session() as session:
            result = await session.execute(
                select(DocumentModel)
                .order_by(DocumentModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            models = result.scalars().all()
            return [_document_to_metadata(m) for m in models]

    async def query_chunks_by_document(
        self,
        document_id: uuid.UUID,
        limit: int = 1000,
    ) -> list[ChunkMetadata]:
        """Get all chunks belonging to a document."""
        async with self.async_session() as session:
            result = await session.execute(
                select(ChunkModel)
                .where(ChunkModel.document_id == _to_uuid_str(document_id))
                .order_by(ChunkModel.chunk_index)
                .limit(limit)
            )
            models = result.scalars().all()
            return [_chunk_to_metadata(m) for m in models]

    async def delete_document(self, document_id: uuid.UUID) -> int:
        """Delete a document record."""
        async with self.async_session() as session:
            result = await session.execute(
                delete(DocumentModel).where(
                    DocumentModel.document_id == _to_uuid_str(document_id)
                )
            )
            await session.commit()
            return result.rowcount or 0

    async def create_cluster(self, cluster: QueryCluster) -> None:
        """Create a new query cluster."""
        async with self.async_session() as session:
            model = QueryClusterModel(
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

    async def get_cluster(self, cluster_id: uuid.UUID) -> QueryCluster | None:
        """Get cluster by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(QueryClusterModel).where(QueryClusterModel.cluster_id == _to_uuid_str(cluster_id))
            )
            model = result.scalar_one_or_none()
            return _cluster_to_dataclass(model) if model else None

    async def update_cluster(
        self,
        cluster_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> QueryCluster | None:
        """Update cluster fields."""
        async with self.async_session() as session:
            await session.execute(
                update(QueryClusterModel)
                .where(QueryClusterModel.cluster_id == _to_uuid_str(cluster_id))
                .values(**updates)
            )
            await session.commit()

            result = await session.execute(
                select(QueryClusterModel).where(QueryClusterModel.cluster_id == _to_uuid_str(cluster_id))
            )
            model = result.scalar_one_or_none()
            return _cluster_to_dataclass(model) if model else None

    async def get_all_clusters(self) -> list[QueryCluster]:
        """Get all query clusters."""
        async with self.async_session() as session:
            result = await session.execute(select(QueryClusterModel))
            models = result.scalars().all()
            return [_cluster_to_dataclass(m) for m in models]

    async def delete_clusters(self, cluster_ids: list[uuid.UUID]) -> int:
        """Delete query clusters."""
        id_strs = [_to_uuid_str(cid) for cid in cluster_ids]
        async with self.async_session() as session:
            result = await session.execute(
                delete(QueryClusterModel).where(QueryClusterModel.cluster_id.in_(id_strs))
            )
            await session.commit()
            return result.rowcount or 0

    async def create_migration_log(self, log: MigrationLog) -> None:
        """Create a migration log entry."""
        async with self.async_session() as session:
            model = MigrationLogModel(
                chunk_id=_to_uuid_str(log.chunk_id),
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
