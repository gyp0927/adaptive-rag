"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import uuid


@dataclass
class VectorSearchResult:
    """Result from a vector search."""

    chunk_id: uuid.UUID
    score: float
    vector: list[float] | None = None
    payload: dict[str, Any] | None = None


class BaseVectorStore(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store (create collections, etc.)."""
        pass

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        ids: list[uuid.UUID],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """Store or update vectors."""
        pass

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def search_batch(
        self,
        collection: str,
        query_vectors: list[list[float]],
        limit: int = 1,
    ) -> list[list[VectorSearchResult]]:
        """Batch search for multiple query vectors.

        Returns one result list per query vector, in the same order.
        """
        pass

    @abstractmethod
    async def delete(
        self,
        collection: str,
        ids: list[uuid.UUID],
    ) -> int:
        """Delete vectors by ID. Returns count deleted."""
        pass

    @abstractmethod
    async def get_by_id(
        self,
        collection: str,
        chunk_id: uuid.UUID,
    ) -> VectorSearchResult | None:
        """Get a vector by ID."""
        pass

    @abstractmethod
    async def count(self, collection: str) -> int:
        """Count vectors in collection."""
        pass
