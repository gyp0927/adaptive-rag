"""Cold tier: stores compressed summaries with summary embeddings."""

from datetime import datetime
from typing import Any
import uuid

from adaptive_rag.core.config import Tier, get_settings
from adaptive_rag.core.exceptions import TierError
from adaptive_rag.core.logging import get_logger
from adaptive_rag.ingestion.chunker import Chunk
from adaptive_rag.ingestion.embedder import Embedder
from adaptive_rag.storage.cache.base import BaseCache
from adaptive_rag.storage.document_store.base import BaseDocumentStore
from adaptive_rag.storage.metadata_store.base import (
    BaseMetadataStore,
    ChunkMetadata,
)
from adaptive_rag.storage.vector_store.base import BaseVectorStore

from .base import BaseTier, RetrievedChunk
from .compression import CompressionEngine, CompressedChunk
from .decompression import DecompressionEngine

logger = get_logger(__name__)


class ColdTier(BaseTier):
    """Cold tier stores LLM-compressed summaries with summary embeddings.

    Storage-efficient but requires decompression for full detail.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        metadata_store: BaseMetadataStore,
        document_store: BaseDocumentStore,
        compression_engine: CompressionEngine,
        decompression_engine: DecompressionEngine,
        cache: BaseCache | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.document_store = document_store
        self.compression_engine = compression_engine
        self.decompression_engine = decompression_engine
        self.cache = cache
        self.embedder = embedder or Embedder()
        self.collection = f"{self.settings.VECTOR_DB_COLLECTION}_cold"

    @property
    def tier_type(self) -> Tier:
        return Tier.COLD

    async def store_chunks(
        self,
        chunks: list[Chunk],
        original_embeddings: list[list[float]] | None = None,
    ) -> list[ChunkMetadata]:
        """Compress and store chunks in the cold tier.

        Args:
            chunks: Document chunks with full text.
            original_embeddings: Optional original embeddings (unused for cold tier).

        Returns:
            List of chunk metadata.
        """
        # 1. Compress chunks via LLM
        compressed = await self.compression_engine.compress_batch(chunks)

        # 2. Generate embeddings for summaries
        summary_texts = [c.summary_text for c in compressed]
        summary_embeddings = await self.embedder.embed_batch(summary_texts)

        # 3. Store compressed summaries in document store
        await self.document_store.store_batch([
            (c.chunk_id, comp.summary_text)
            for c, comp in zip(chunks, compressed)
        ])

        # 4. Store summary vectors
        chunk_ids = [c.chunk_id for c in chunks]
        payloads = [{
            "chunk_id": str(c.chunk_id),
            "document_id": str(c.document_id),
            "tier": Tier.COLD.value,
            "chunk_index": c.index,
            "tags": c.tags or [],
            "compressed": True,
        } for c in chunks]

        await self.vector_store.upsert(
            collection=self.collection,
            ids=chunk_ids,
            vectors=summary_embeddings,
            payloads=payloads,
        )

        # 5. Store metadata
        now = datetime.utcnow()
        metadata_list = []
        for chunk, comp in zip(chunks, compressed):
            meta = ChunkMetadata(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                tier=Tier.COLD,
                original_length=len(chunk.text),
                compressed_length=len(comp.summary_text),
                access_count=0,
                frequency_score=0.0,
                created_at=now,
                updated_at=now,
                chunk_index=chunk.index,
                tags=chunk.tags or [],
            )
            metadata_list.append(meta)
            await self.metadata_store.create_chunk(meta)

        logger.info(
            "cold_tier_stored",
            chunk_count=len(chunks),
            avg_compression=sum(len(c.summary_text) for c in compressed) / max(sum(len(c.text) for c in chunks), 1),
        )
        return metadata_list

    async def store_raw_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        initial_score: float = 0.1,
    ) -> list[ChunkMetadata]:
        """Store raw (uncompressed) chunks directly into the cold tier.

        This skips LLM compression and stores the original text, saving LLM
        costs for newly ingested cold-topic content. The chunks can be later
        compressed during scheduled migration cycles when the off-peak window
        arrives.

        Args:
            chunks: Document chunks with full text.
            embeddings: Pre-computed embedding vectors for each chunk.
            initial_score: Starting frequency score (default 0.1 for cold topics).

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

        # 2. Store vectors (original text embeddings for similarity search)
        payloads = [{
            "chunk_id": str(c.chunk_id),
            "document_id": str(c.document_id),
            "tier": Tier.COLD.value,
            "chunk_index": c.index,
            "tags": c.tags or [],
            "compressed": False,
        } for c in chunks]

        await self.vector_store.upsert(
            collection=self.collection,
            ids=chunk_ids,
            vectors=embeddings,
            payloads=payloads,
        )

        # 3. Store metadata
        now = datetime.utcnow()
        metadata_list = []
        for chunk in chunks:
            meta = ChunkMetadata(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                tier=Tier.COLD,
                original_length=len(chunk.text),
                compressed_length=None,
                access_count=0,
                frequency_score=initial_score,
                created_at=now,
                updated_at=now,
                chunk_index=chunk.index,
                tags=chunk.tags or [],
            )
            metadata_list.append(meta)
            await self.metadata_store.create_chunk(meta)

        logger.info(
            "cold_tier_raw_stored",
            chunk_count=len(chunks),
            initial_score=initial_score,
        )
        return metadata_list

    async def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        decompress: bool = False,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks from cold tier.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results.
            filters: Optional metadata filters.
            decompress: Whether to decompress summaries.
        """
        results = await self.vector_store.search(
            collection=self.collection,
            query_vector=query_embedding,
            limit=top_k,
            filters=filters,
        )

        chunks = []
        for result in results:
            chunk_id = result.chunk_id

            # Try cache
            text = None
            if self.cache:
                text = await self.cache.get(f"chunk:{chunk_id}")

            # Fetch summary from document store
            if text is None:
                text = await self.document_store.get(chunk_id)
                if text and self.cache:
                    await self.cache.set(f"chunk:{chunk_id}", text)

            if text is None:
                continue

            is_decompressed = False

            # On-demand decompression
            if decompress:
                text = await self.decompression_engine.decompress(text)
                is_decompressed = True

            meta = await self.metadata_store.get_chunk(chunk_id)

            chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                document_id=uuid.UUID(result.payload.get("document_id")) if result.payload else uuid.UUID(int=0),
                content=text,
                score=result.score,
                tier=Tier.COLD,
                is_decompressed=is_decompressed,
                access_count=meta.access_count if meta else 0,
                frequency_score=meta.frequency_score if meta else 0.0,
                metadata=result.payload or {},
            ))

        return chunks

    async def get_by_id(self, chunk_id: uuid.UUID) -> RetrievedChunk | None:
        """Get a specific chunk by ID."""
        vec_result = await self.vector_store.get_by_id(self.collection, chunk_id)
        if not vec_result:
            return None

        text = await self.document_store.get(chunk_id)
        if text is None:
            return None

        meta = await self.metadata_store.get_chunk(chunk_id)

        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=uuid.UUID(vec_result.payload.get("document_id")) if vec_result.payload else uuid.UUID(int=0),
            content=text,
            score=1.0,
            tier=Tier.COLD,
            is_decompressed=False,
            access_count=meta.access_count if meta else 0,
            frequency_score=meta.frequency_score if meta else 0.0,
            metadata=vec_result.payload or {},
        )

    async def delete(self, chunk_ids: list[uuid.UUID]) -> int:
        """Delete chunks from cold tier."""
        await self.vector_store.delete(self.collection, chunk_ids)
        await self.document_store.delete(chunk_ids)
        count = await self.metadata_store.delete_chunks(chunk_ids)

        if self.cache:
            for chunk_id in chunk_ids:
                await self.cache.delete(f"chunk:{chunk_id}")

        return count

    async def exists(self, chunk_id: uuid.UUID) -> bool:
        """Check if chunk exists in cold tier."""
        result = await self.vector_store.get_by_id(self.collection, chunk_id)
        return result is not None
