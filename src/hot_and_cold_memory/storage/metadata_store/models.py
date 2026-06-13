"""SQLAlchemy models for metadata store (PostgreSQL/SQLite compatible)."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy base class."""

    pass


class MemoryModel(Base):
    """Memory item metadata table."""

    __tablename__ = "memories"

    __table_args__ = (
        CheckConstraint("tier IN ('hot', 'cold')", name="ck_memory_tier"),
        Index("ix_memories_created_at", "created_at"),
        Index("ix_memories_updated_at", "updated_at"),
        Index("ix_memories_source_memory_type", "source", "memory_type"),
    )

    memory_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    tier: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    original_length: Mapped[int] = mapped_column(Integer, nullable=False)

    memory_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="observation", index=True
    )
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frequency_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now()
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_migrated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    topic_cluster_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("topic_clusters.cluster_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    compression_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    compressed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class TopicClusterModel(Base):
    """Topic cluster table for memory organization."""

    __tablename__ = "topic_clusters"

    cluster_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    representative_query: Mapped[str] = mapped_column(Text, nullable=False)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frequency_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    centroid: Mapped[list[float]] = mapped_column(JSON, nullable=False)


class AccessLogModel(Base):
    """Access log table."""

    __tablename__ = "access_logs"

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("memories.memory_id", ondelete="CASCADE"), nullable=False
    )
    query_cluster_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("topic_clusters.cluster_id", ondelete="SET NULL"), nullable=True
    )
    query_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tier_accessed: Mapped[str | None] = mapped_column(String(10), nullable=True)


class MigrationLogModel(Base):
    """Migration log table."""

    __tablename__ = "migration_logs"

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("memories.memory_id", ondelete="CASCADE"), nullable=False
    )
    direction: Mapped[str] = mapped_column(String(20), nullable=False)
    original_size: Mapped[int] = mapped_column(Integer, nullable=False)
    new_size: Mapped[int] = mapped_column(Integer, nullable=False)
    compression_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class MemoryLinkModel(Base):
    """Association/links between memories."""

    __tablename__ = "memory_links"

    link_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("memories.memory_id", ondelete="CASCADE"), nullable=False
    )
    target_memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("memories.memory_id", ondelete="CASCADE"), nullable=False
    )
    link_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="coaccess", index=True
    )
    strength: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_memory_links_source_target", "source_memory_id", "target_memory_id"),
        Index("ix_memory_links_target", "target_memory_id"),
    )


class ProfileFactModel(Base):
    """A single profile assertion with a validity timeline."""

    __tablename__ = "profile_facts"

    __table_args__ = (
        Index("ix_profile_facts_user_current", "user_id", "is_current"),
        Index("ix_profile_facts_user_cat_key", "user_id", "category", "key"),
    )

    fact_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(64), nullable=False, default="default", index=True
    )
    memory_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("memories.memory_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    category: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now()
    )


class ProfileModel(Base):
    """Cached snapshot of the current effective profile."""

    __tablename__ = "profiles"

    profile_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, default="default"
    )

    identity: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    preferences: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    goals: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    constraints: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    habits: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now()
    )
