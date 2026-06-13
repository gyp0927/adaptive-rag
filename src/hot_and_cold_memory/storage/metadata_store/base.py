"""Abstract base class for metadata stores."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from hot_and_cold_memory.core.config import Tier


@dataclass
class MemoryItem:
    """A single memory item."""

    memory_id: uuid.UUID
    tier: Tier
    content: str = ""
    original_length: int = 0
    memory_type: str = "observation"
    source: str | None = None
    importance: float = 0.5
    access_count: int = 0
    frequency_score: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime | None = None
    last_migrated_at: datetime | None = None
    topic_cluster_id: uuid.UUID | None = None
    tags: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    vector_id: str | None = None
    compressed: bool = False
    expires_at: datetime | None = None


@dataclass
class TopicCluster:
    """A cluster of semantically similar memories/topics."""

    cluster_id: uuid.UUID
    centroid: list[float]
    representative_query: str
    access_count: int = 0
    frequency_score: float = 0.0
    member_count: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime | None = None


@dataclass
class AccessLog:
    """Record of a memory access."""

    memory_id: uuid.UUID
    log_id: int | None = None
    query_cluster_id: uuid.UUID | None = None
    query_text: str | None = None
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    response_time_ms: int | None = None
    tier_accessed: str | None = None


@dataclass
class MigrationLog:
    """Record of a tier migration."""

    memory_id: uuid.UUID
    direction: str
    original_size: int
    new_size: int
    log_id: int | None = None
    compression_ratio: float | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    status: str = "pending"
    error_message: str | None = None


@dataclass
class MemoryLink:
    """A link/association between two memories."""

    source_memory_id: uuid.UUID
    target_memory_id: uuid.UUID
    link_type: str = "coaccess"
    strength: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime | None = None
    link_id: int | None = None


class BaseMetadataStore(ABC):
    """Abstract interface for metadata database operations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the metadata store (create tables)."""
        pass

    # Memory operations
    @abstractmethod
    async def create_memory(self, metadata: MemoryItem) -> None:
        """Create a new memory record."""
        pass

    @abstractmethod
    async def get_memory(self, memory_id: uuid.UUID) -> MemoryItem | None:
        """Get memory by ID."""
        pass

    @abstractmethod
    async def get_memories_batch(self, memory_ids: list[uuid.UUID]) -> list[MemoryItem]:
        """Get multiple memories by ID in a single query.

        Returns only found memories; missing IDs are silently omitted.
        """
        pass

    @abstractmethod
    async def create_memories_batch(self, metadatas: list[MemoryItem]) -> None:
        """Create multiple memory records in a single transaction."""
        pass

    @abstractmethod
    async def update_memory(
        self,
        memory_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> MemoryItem | None:
        """Update memory fields."""
        pass

    @abstractmethod
    async def update_memories_batch(
        self,
        updates: dict[uuid.UUID, dict[str, Any]],
    ) -> None:
        """Update multiple memories in a single transaction.

        Args:
            updates: Mapping from memory_id to update dict.
        """
        pass

    @abstractmethod
    async def delete_memories(self, memory_ids: list[uuid.UUID]) -> int:
        """Delete memories. Returns count deleted."""
        pass

    @abstractmethod
    async def list_memories(
        self,
        memory_type: str | None = None,
        source: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryItem]:
        """List memories with optional filtering."""
        pass

    @abstractmethod
    async def query_memories_by_tier_and_score(
        self,
        tier: Tier,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int = 100,
        order_desc: bool = False,
    ) -> list[MemoryItem]:
        """Query memories by tier and frequency score range."""
        pass

    @abstractmethod
    async def search_by_keyword(
        self,
        query_text: str,
        tier: Tier | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """Search memories by keyword match in content.

        Args:
            query_text: Space-separated keywords.
            tier: Filter by tier (optional).
            limit: Max results.

        Returns:
            Matching memory items ordered by relevance.
        """
        pass

    @abstractmethod
    async def query_forgettable_memories(
        self,
        tier: Tier,
        max_importance: float,
        cutoff: datetime,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """Query memories eligible for deletion (forgetting).

        Args:
            tier: Tier to query.
            max_importance: Importance must be below this threshold.
            cutoff: last_accessed_at must be older than this (or None).
            limit: Max results.

        Returns:
            List of memory items eligible for forgetting.
        """
        pass

    @abstractmethod
    async def count_memories(
        self,
        memory_type: str | None = None,
        source: str | None = None,
    ) -> int:
        """Count memories with optional filtering."""
        pass

    @abstractmethod
    async def count_memories_by_tier(self, tier: Tier) -> int:
        """Count memories in a given tier."""
        pass

    @abstractmethod
    async def increment_access(
        self,
        memory_ids: list[uuid.UUID],
        cluster_id: uuid.UUID | None,
        timestamp: datetime,
    ) -> None:
        """Increment access count for memories."""
        pass

    # Topic cluster operations
    @abstractmethod
    async def create_cluster(self, cluster: TopicCluster) -> None:
        """Create a new topic cluster."""
        pass

    @abstractmethod
    async def get_cluster(self, cluster_id: uuid.UUID) -> TopicCluster | None:
        """Get cluster by ID."""
        pass

    @abstractmethod
    async def update_cluster(
        self,
        cluster_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> TopicCluster | None:
        """Update cluster fields."""
        pass

    @abstractmethod
    async def get_all_clusters(self) -> list[TopicCluster]:
        """Get all topic clusters."""
        pass

    @abstractmethod
    async def get_clusters_batch(self, cluster_ids: list[uuid.UUID]) -> list[TopicCluster]:
        """Get multiple clusters by ID in a single query."""
        pass

    @abstractmethod
    async def delete_clusters(self, cluster_ids: list[uuid.UUID]) -> int:
        """Delete topic clusters. Returns count deleted."""
        pass

    # Access / migration log operations
    @abstractmethod
    async def create_access_log(self, log: AccessLog) -> None:
        """Create an access log entry."""
        pass

    @abstractmethod
    async def create_access_logs_batch(self, logs: list[AccessLog]) -> None:
        """Create multiple access log entries in a single transaction."""
        pass

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

    # Association graph operations
    @abstractmethod
    async def create_link(self, link: MemoryLink) -> None:
        """Create a link between two memories."""
        pass

    @abstractmethod
    async def get_related_memories(
        self,
        memory_id: uuid.UUID,
        link_type: str | None = None,
        min_strength: float | None = None,
        limit: int = 20,
    ) -> list[tuple[MemoryLink, MemoryItem]]:
        """Get memories related to a given memory.

        Returns:
            List of (link, memory_item) tuples.
        """
        pass

    @abstractmethod
    async def delete_links_for_memories(self, memory_ids: list[uuid.UUID]) -> int:
        """Delete all links involving any of the given memory IDs."""
        pass

    # --- Profile operations ---

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_current_profile_fact(
        self,
        user_id: str,
        category: str,
        key: str,
    ) -> dict[str, Any] | None:
        """Get the current (is_current=true) fact for a given key."""
        pass

    @abstractmethod
    async def expire_profile_fact(self, fact_id: str) -> None:
        """Mark a fact as no longer current."""
        pass

    @abstractmethod
    async def update_profile_fact_confidence(
        self,
        fact_id: str,
        confidence: float,
    ) -> None:
        """Update the confidence of a fact."""
        pass

    @abstractmethod
    async def list_profile_facts(
        self,
        user_id: str,
        category: str | None = None,
        key: str | None = None,
        is_current: bool | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List profile facts with optional filters."""
        pass

    @abstractmethod
    async def get_profile(self, user_id: str) -> dict[str, Any] | None:
        """Get the current profile snapshot."""
        pass

    @abstractmethod
    async def upsert_profile(
        self,
        user_id: str,
        snapshot: dict[str, Any],
    ) -> None:
        """Insert or update the profile snapshot."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close store connections."""
        pass
