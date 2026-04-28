"""Local Qdrant vector store (no server needed).

Uses qdrant_client's local persistence mode.
All operations run in thread pool to avoid blocking asyncio.
"""

import asyncio
from typing import Any
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from adaptive_rag.core.config import get_settings
from adaptive_rag.core.exceptions import VectorStoreError
from adaptive_rag.core.logging import get_logger

from .base import BaseVectorStore, VectorSearchResult

logger = get_logger(__name__)


def _parse_uuid(value: Any) -> uuid.UUID:
    """Safely parse a Qdrant point ID into a UUID.

    Qdrant may return IDs as strings (UUIDs) or integers.
    This helper handles both cases gracefully.
    """
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str):
        return uuid.UUID(value)
    if isinstance(value, int):
        return uuid.UUID(int=value)
    # Fallback: try string conversion
    return uuid.UUID(str(value))


class LocalQdrantStore(BaseVectorStore):
    """Local file-based Qdrant store (no server required)."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client: QdrantClient | None = None
        self._path = "./data/qdrant_storage"

    async def initialize(self) -> None:
        """Initialize local Qdrant store."""
        import os
        os.makedirs(self._path, exist_ok=True)

        self.client = await asyncio.to_thread(
            QdrantClient,
            path=self._path,
        )
        await self.ensure_collection(self.settings.VECTOR_DB_COLLECTION)
        await self.ensure_collection("query_clusters")
        logger.info("local_qdrant_initialized", path=self._path)

    async def ensure_collection(self, name: str) -> None:
        """Create collection if it doesn't exist."""
        if not self.client:
            raise VectorStoreError("Client not initialized")

        collections = await asyncio.to_thread(self.client.get_collections)
        existing = {c.name for c in collections.collections}

        if name not in existing:
            await asyncio.to_thread(
                self.client.create_collection,
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("collection_created", collection=name)

    async def upsert(
        self,
        collection: str,
        ids: list[uuid.UUID],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """Store or update vectors."""
        if not self.client:
            raise VectorStoreError("Client not initialized")

        if payloads is None:
            payloads = [{} for _ in ids]

        points = [
            PointStruct(
                id=str(id_),
                vector=vec,
                payload=payload,
            )
            for id_, vec, payload in zip(ids, vectors, payloads)
        ]

        await asyncio.to_thread(
            self.client.upsert,
            collection_name=collection,
            points=points,
        )

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        if not self.client:
            raise VectorStoreError("Client not initialized")

        # Build filter for local mode
        filter_kwargs = {}
        if filters:
            # Simple key-value matching for local mode
            must_conditions = []
            for key, value in filters.items():
                from qdrant_client.models import FieldCondition, MatchValue
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if must_conditions:
                from qdrant_client.models import Filter
                filter_kwargs["query_filter"] = Filter(must=must_conditions)

        results = await asyncio.to_thread(
            self.client.query_points,
            collection_name=collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            **filter_kwargs,
        )

        return [
            VectorSearchResult(
                chunk_id=_parse_uuid(r.id),
                score=r.score,
                vector=None,
                payload=r.payload or {},
            )
            for r in results.points
        ]

    async def delete(
        self,
        collection: str,
        ids: list[uuid.UUID],
    ) -> int:
        """Delete vectors by ID."""
        if not self.client:
            raise VectorStoreError("Client not initialized")

        await asyncio.to_thread(
            self.client.delete,
            collection_name=collection,
            points_selector=PointIdsList(
                points=[str(id_) for id_ in ids],
            ),
        )
        return len(ids)

    async def get_by_id(
        self,
        collection: str,
        chunk_id: uuid.UUID,
    ) -> VectorSearchResult | None:
        """Get a vector by ID."""
        if not self.client:
            raise VectorStoreError("Client not initialized")

        results = await asyncio.to_thread(
            self.client.retrieve,
            collection_name=collection,
            ids=[str(chunk_id)],
            with_payload=True,
            with_vectors=True,
        )

        if not results:
            return None

        r = results[0]
        return VectorSearchResult(
            chunk_id=_parse_uuid(r.id),
            score=1.0,
            vector=r.vector,
            payload=r.payload or {},
        )

    async def count(self, collection: str) -> int:
        """Count vectors in collection."""
        if not self.client:
            raise VectorStoreError("Client not initialized")

        result = await asyncio.to_thread(self.client.count, collection_name=collection)
        return result.count

    async def search_batch(
        self,
        collection: str,
        query_vectors: list[list[float]],
        limit: int = 1,
    ) -> list[list[VectorSearchResult]]:
        """Batch search for multiple query vectors.

        Local-mode QdrantClient does not support search_batch, so we
        run individual searches concurrently via asyncio.gather.
        Returns one result list per query vector, in the same order.
        """
        if not self.client:
            raise VectorStoreError("Client not initialized")

        if not query_vectors:
            return []

        results = await asyncio.gather(
            *[self.search(collection, qv, limit=limit) for qv in query_vectors]
        )
        return list(results)
