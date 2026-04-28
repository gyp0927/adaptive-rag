"""Abstract base class for metadata stores."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid

from adaptive_rag.core.config import Tier


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    tier: Tier
    original_length: int
    compressed_length: int | None = None
    chunk_index: int = 0
    access_count: int = 0
    frequency_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: datetime | None = None
    last_migrated_at: datetime | None = None
    topic_cluster_id: uuid.UUID | None = None
    tags: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    vector_id: str | None = None


@dataclass
class DocumentMetadata:
    """Metadata for a source document."""

    document_id: uuid.UUID
    source_type: str
    source_uri: str
    title: str | None = None
    author: str | None = None
    content_hash: str = ""
    total_chunks: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QueryCluster:
    """A cluster of semantically similar queries."""

    cluster_id: uuid.UUID
    centroid: list[float]
    representative_query: str
    access_count: int = 0
    frequency_score: float = 0.0
    member_count: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: datetime | None = None


@dataclass
class AccessLog:
    """Record of a chunk access."""

    chunk_id: uuid.UUID
    log_id: int | None = None
    query_cluster_id: uuid.UUID | None = None
    query_text: str | None = None
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: int | None = None
    tier_accessed: str | None = None


@dataclass
class MigrationLog:
    """Record of a tier migration."""

    chunk_id: uuid.UUID
    direction: str
    original_size: int
    new_size: int
    log_id: int | None = None
    compression_ratio: float | None = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    status: str = "pending"
    error_message: str | None = None


class BaseMetadataStore(ABC):
    """Abstract interface for metadata database operations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the metadata store (create tables)."""
        pass

    # Chunk operations
    @abstractmethod
    async def create_chunk(self, metadata: ChunkMetadata) -> None:
        """Create a new chunk record."""
        pass

    @abstractmethod
    async def get_chunk(self, chunk_id: uuid.UUID) -> ChunkMetadata | None:
        """Get chunk by ID."""
        pass

    @abstractmethod
    async def get_chunks_batch(self, chunk_ids: list[uuid.UUID]) -> list[ChunkMetadata]:
        """Get multiple chunks by ID in a single query.

        Returns only found chunks; missing IDs are silently omitted.
        """
        pass

    @abstractmethod
    async def create_chunks_batch(self, metadatas: list[ChunkMetadata]) -> None:
        """Create multiple chunk records in a single transaction."""
        pass

    @abstractmethod
    async def update_chunk(
        self,
        chunk_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> ChunkMetadata | None:
        """Update chunk fields."""
        pass

    @abstractmethod
    async def update_chunks_batch(
        self,
        updates: dict[uuid.UUID, dict[str, Any]],
    ) -> None:
        """Update multiple chunks in a single transaction.

        Args:
            updates: Mapping from chunk_id to update dict.
        """
        pass

    @abstractmethod
    async def delete_chunks(self, chunk_ids: list[uuid.UUID]) -> int:
        """Delete chunks. Returns count deleted."""
        pass

    @abstractmethod
    async def query_chunks_by_tier_and_score(
        self,
        tier: Tier,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int = 100,
    ) -> list[ChunkMetadata]:
        """Query chunks by tier and frequency score range."""
        pass

    @abstractmethod
    async def count_chunks_by_tier(self, tier: Tier) -> int:
        """Count chunks in a given tier."""
        pass

    @abstractmethod
    async def increment_access(
        self,
        chunk_ids: list[uuid.UUID],
        cluster_id: uuid.UUID | None,
        timestamp: datetime,
    ) -> None:
        """Increment access count for chunks."""
        pass

    # Document operations
    @abstractmethod
    async def create_document(self, metadata: DocumentMetadata) -> None:
        """Create a new document record."""
        pass

    @abstractmethod
    async def get_document(self, document_id: uuid.UUID) -> DocumentMetadata | None:
        """Get document by ID."""
        pass

    @abstractmethod
    async def get_document_by_hash(self, content_hash: str) -> DocumentMetadata | None:
        """Get document by content hash (for deduplication)."""
        pass

    @abstractmethod
    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentMetadata]:
        """List documents with pagination."""
        pass

    @abstractmethod
    async def update_document(
        self,
        document_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> DocumentMetadata | None:
        """Update document fields."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: uuid.UUID) -> int:
        """Delete a document record. Returns count deleted."""
        pass

    @abstractmethod
    async def query_chunks_by_document(
        self,
        document_id: uuid.UUID,
        limit: int = 1000,
    ) -> list[ChunkMetadata]:
        """Get all chunks belonging to a document."""
        pass

    # Query cluster operations
    @abstractmethod
    async def create_cluster(self, cluster: QueryCluster) -> None:
        """Create a new query cluster."""
        pass

    @abstractmethod
    async def get_cluster(self, cluster_id: uuid.UUID) -> QueryCluster | None:
        """Get cluster by ID."""
        pass

    @abstractmethod
    async def update_cluster(
        self,
        cluster_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> QueryCluster | None:
        """Update cluster fields."""
        pass

    @abstractmethod
    async def get_all_clusters(self) -> list[QueryCluster]:
        """Get all query clusters."""
        pass

    @abstractmethod
    async def delete_clusters(self, cluster_ids: list[uuid.UUID]) -> int:
        """Delete query clusters. Returns count deleted."""
        pass

    # Migration log operations
    @abstractmethod
    async def create_migration_log(self, log: MigrationLog) -> None:
        """Create a migration log entry."""
        pass

    @abstractmethod
    async def update_migration_log(
        self,
        log_id: int,
        updates: dict[str, Any],
    ) -> None:
        """Update migration log."""
        pass
