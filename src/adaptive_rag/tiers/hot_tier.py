"""Hot tier: stores full original text with embeddings."""

from datetime import datetime
from typing import Any
import uuid

from adaptive_rag.core.config import Tier, get_settings
from adaptive_rag.core.exceptions import TierError
from adaptive_rag.core.logging import get_logger
from adaptive_rag.ingestion.chunker import Chunk
from adaptive_rag.storage.cache.base import BaseCache
from adaptive_rag.storage.document_store.base import BaseDocumentStore
from adaptive_rag.storage.metadata_store.base import (
    BaseMetadataStore,
    ChunkMetadata,
)
from adaptive_rag.storage.vector_store.base import BaseVectorStore

from .base import BaseTier, RetrievedChunk

logger = get_logger(__name__)


class HotTier(BaseTier):
    """Hot tier stores full original text with dense embeddings.

    Optimized for low-latency retrieval of frequently accessed content.
    Chunks with high topic frequency are routed here by the ingestion pipeline.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        metadata_store: BaseMetadataStore,
        document_store: BaseDocumentStore,
        cache: BaseCache | None = None,
    ) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.document_store = document_store
        self.cache = cache
        self.collection = self.settings.VECTOR_DB_COLLECTION

    @property
    def tier_type(self) -> Tier:
        return Tier.HOT

    async def store_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> list[ChunkMetadata]:
        """Store chunks in the hot tier.

        Args:
            chunks: Document chunks.
            embeddings: Embedding vectors for each chunk.

        Returns:
            List of chunk metadata.
        """
        if len(chunks) != len(embeddings):
            raise TierError("Chunks and embeddings count mismatch")

        chunk_ids = [c.chunk_id for c in chunks]

        # 1. Store original text in document store
        await self.document_store.store_batch([
            (c.chunk_id, c.text) for c in chunks
        ])

        # 2. Store vectors
        payloads = [{
            "chunk_id": str(c.chunk_id),
            "document_id": str(c.document_id),
            "tier": Tier.HOT.value,
            "chunk_index": c.index,
            "tags": c.tags or [],
        } for c in chunks]

        await self.vector_store.upsert(
            collection=self.collection,
            ids=chunk_ids,
            vectors=embeddings,
            payloads=payloads,
        )

        # 3. Store metadata (new chunks start with max frequency)
        now = datetime.utcnow()
        metadata_list = [
            ChunkMetadata(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                tier=Tier.HOT,
                original_length=len(chunk.text),
                access_count=0,
                frequency_score=1.0,  # New chunks start hot
                created_at=now,
                updated_at=now,
                chunk_index=chunk.index,
                tags=chunk.tags or [],
            )
            for chunk in chunks
        ]
        await self.metadata_store.create_chunks_batch(metadata_list)

        logger.info(
            "hot_tier_stored",
            chunk_count=len(chunks),
        )
        return metadata_list

    async def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks by vector similarity from hot tier."""
        # Search vectors
        results = await self.vector_store.search(
            collection=self.collection,
            query_vector=query_embedding,
            limit=top_k,
            filters=filters,
        )

        # Batch fetch metadata to avoid N+1 queries
        chunk_ids = [r.chunk_id for r in results]
        meta_map = {
            m.chunk_id: m
            for m in await self.metadata_store.get_chunks_batch(chunk_ids)
        }

        chunks = []
        for result in results:
            chunk_id = result.chunk_id

            # Try cache first
            text = None
            if self.cache:
                text = await self.cache.get(f"chunk:{chunk_id}")

            # Fetch from document store
            if text is None:
                text = await self.document_store.get(chunk_id)
                if text and self.cache:
                    await self.cache.set(f"chunk:{chunk_id}", text)

            if text is None:
                logger.warning("chunk_not_found", chunk_id=str(chunk_id))
                continue

            meta = meta_map.get(chunk_id)
            chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                document_id=uuid.UUID(result.payload.get("document_id")) if result.payload else uuid.UUID(int=0),
                content=text,
                score=result.score,
                tier=Tier.HOT,
                is_decompressed=False,
                access_count=meta.access_count if meta else 0,
                frequency_score=meta.frequency_score if meta else 0.0,
                metadata=result.payload or {},
            ))

        return chunks

    async def get_by_id(self, chunk_id: uuid.UUID) -> RetrievedChunk | None:
        """Get a specific chunk by ID."""
        # Get from vector store for payload
        vec_result = await self.vector_store.get_by_id(self.collection, chunk_id)
        if not vec_result:
            return None

        # Get text
        text = await self.document_store.get(chunk_id)
        if text is None:
            return None

        # Get metadata
        meta = await self.metadata_store.get_chunk(chunk_id)

        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=uuid.UUID(vec_result.payload.get("document_id")) if vec_result.payload else uuid.UUID(int=0),
            content=text,
            score=1.0,
            tier=Tier.HOT,
            is_decompressed=False,
            access_count=meta.access_count if meta else 0,
            frequency_score=meta.frequency_score if meta else 0.0,
            metadata=vec_result.payload or {},
        )

    async def delete(self, chunk_ids: list[uuid.UUID]) -> int:
        """Delete chunks from hot tier."""
        # Delete from all stores
        await self.vector_store.delete(self.collection, chunk_ids)
        await self.document_store.delete(chunk_ids)
        count = await self.metadata_store.delete_chunks(chunk_ids)

        # Clear cache
        if self.cache:
            for chunk_id in chunk_ids:
                await self.cache.delete(f"chunk:{chunk_id}")

        return count

    async def exists(self, chunk_id: uuid.UUID) -> bool:
        """Check if chunk exists in hot tier."""
        result = await self.vector_store.get_by_id(self.collection, chunk_id)
        return result is not None
