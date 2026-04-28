"""Document ingestion pipeline."""

import asyncio
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from adaptive_rag.core.config import Tier, get_settings
from adaptive_rag.core.exceptions import IngestionError
from adaptive_rag.core.logging import get_logger
from adaptive_rag.frequency.tracker import FrequencyTracker
from adaptive_rag.ingestion.chunker import Chunk
from adaptive_rag.storage.document_store.base import BaseDocumentStore
from adaptive_rag.storage.metadata_store.base import (
    BaseMetadataStore,
    DocumentMetadata,
    ChunkMetadata,
)
from adaptive_rag.storage.vector_store.base import BaseVectorStore
from adaptive_rag.tiers.cold_tier import ColdTier
from adaptive_rag.tiers.hot_tier import HotTier

from .chunker import LLMChunker, RecursiveChunker
from .embedder import Embedder
from .extractors.text import extract_text

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Result of document ingestion."""

    document_id: uuid.UUID
    status: str = "pending"
    chunks_created: int = 0
    total_chunks: int = 0
    hot_chunks: int = 0
    cold_chunks: int = 0
    error: str | None = None
    processing_time_ms: float = 0.0


class IngestionPipeline:
    """Orchestrates document ingestion into the system.

    Instead of blindly placing every new document into the hot tier, the pipeline
    estimates the topic's historical popularity via the frequency tracker. Hot
    topics go to hot tier; new / cold topics skip compression and go directly to
    cold tier as raw text (``cold_raw``), saving LLM costs. After ingestion the
    hot tier capacity is checked and the coldest chunks evicted if needed.
    """

    # Threshold above which a topic is considered "hot" at ingestion time.
    HOT_TOPIC_THRESHOLD: float = 0.5

    # Hard cap on hot tier chunks. When exceeded, the coldest ``EVICT_PERCENT``
    # of hot chunks are pushed to cold tier.
    HOT_TIER_CAPACITY: int = 10000
    EVICT_PERCENT: float = 0.1

    def __init__(
        self,
        metadata_store: BaseMetadataStore,
        hot_tier: HotTier,
        cold_tier: ColdTier,
        embedder: Embedder,
        frequency_tracker: FrequencyTracker,
        chunker: RecursiveChunker | LLMChunker | None = None,
        migration_engine=None,
    ) -> None:
        self.settings = get_settings()
        self.metadata_store = metadata_store
        self.hot_tier = hot_tier
        self.cold_tier = cold_tier
        self.embedder = embedder
        self.frequency_tracker = frequency_tracker

        if chunker is not None:
            self.chunker = chunker
        elif self.settings.CHUNK_STRATEGY.value == "llm":
            self.chunker = LLMChunker(
                chunk_size=self.settings.CHUNK_SIZE,
                chunk_overlap=self.settings.CHUNK_OVERLAP,
            )
        else:
            self.chunker = RecursiveChunker(
                chunk_size=self.settings.CHUNK_SIZE,
                chunk_overlap=self.settings.CHUNK_OVERLAP,
            )
        self.migration_engine = migration_engine

    async def ingest_text(
        self,
        text: str,
        source_uri: str = "inline",
        title: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest a text document.

        Args:
            text: Document text content.
            source_uri: Source identifier.
            title: Optional document title.
            tags: Optional tags.
            metadata: Optional metadata dict.

        Returns:
            Ingestion result.
        """
        start_time = datetime.utcnow()
        document_id = uuid.uuid4()

        try:
            # 1. Compute content hash and check for duplicates
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            existing = await self.metadata_store.get_document_by_hash(content_hash)
            if existing:
                logger.info(
                    "ingestion_duplicate_detected",
                    document_id=str(existing.document_id),
                    source_uri=source_uri,
                )
                return IngestionResult(
                    document_id=existing.document_id,
                    status="duplicate",
                    chunks_created=existing.total_chunks,
                    total_chunks=existing.total_chunks,
                    message="Document already exists (same content hash)",
                )

            # 2. Store document metadata
            doc_meta = DocumentMetadata(
                document_id=document_id,
                source_type="text",
                source_uri=source_uri,
                title=title,
                content_hash=content_hash,
                total_chunks=0,
                metadata=metadata or {},
            )
            await self.metadata_store.create_document(doc_meta)

            # 3. Chunk the document
            chunks = await self.chunker.chunk(
                text=text,
                document_id=document_id,
                tags=tags or [],
            )
            logger.info(
                "ingestion_chunked",
                document_id=str(document_id),
                chunk_count=len(chunks),
                text_len=len(text),
                previews=[c.text[:80] for c in chunks[:3]],
            )

            if len(chunks) == 0:
                logger.error(
                    "ingestion_zero_chunks",
                    document_id=str(document_id),
                    text_len=len(text),
                    source_uri=source_uri,
                )
                return IngestionResult(
                    document_id=document_id,
                    status="failed",
                    error="Document produced zero chunks (text may be empty or unchunkable)",
                )

            # 4. Generate embeddings
            texts = [c.text for c in chunks]
            embeddings = await self.embedder.embed_batch(texts)
            logger.info(
                "ingestion_embedded",
                document_id=str(document_id),
                embeddings=len(embeddings),
            )

            # 5. Classify each chunk by historical topic frequency and route
            topic_freqs = await self.frequency_tracker.get_topic_frequencies_batch(
                embeddings
            )
            logger.info(
                "ingestion_frequencies",
                document_id=str(document_id),
                freqs=[round(f, 3) for f in topic_freqs],
            )

            hot_chunks: list[Chunk] = []
            hot_embeddings: list[list[float]] = []
            cold_chunks: list[Chunk] = []
            cold_embeddings: list[list[float]] = []

            for chunk, embedding, topic_freq in zip(chunks, embeddings, topic_freqs):
                if topic_freq >= self.HOT_TOPIC_THRESHOLD:
                    hot_chunks.append(chunk)
                    hot_embeddings.append(embedding)
                else:
                    cold_chunks.append(chunk)
                    cold_embeddings.append(embedding)

            logger.info(
                "ingestion_routing",
                document_id=str(document_id),
                hot=len(hot_chunks),
                cold=len(cold_chunks),
            )

            # 6. Store hot chunks
            if hot_chunks:
                hot_meta = await self.hot_tier.store_chunks(
                    chunks=hot_chunks,
                    embeddings=hot_embeddings,
                )
                logger.info(
                    "ingestion_hot_stored",
                    document_id=str(document_id),
                    stored=len(hot_meta),
                )
                # Batch update frequency scores
                await self.metadata_store.update_chunks_batch({
                    chunk.chunk_id: {"frequency_score": 0.6}
                    for chunk in hot_chunks
                })

            # 7. Store cold chunks as raw (skip LLM compression)
            if cold_chunks:
                cold_meta = await self.cold_tier.store_raw_chunks(
                    chunks=cold_chunks,
                    embeddings=cold_embeddings,
                    initial_score=0.1,
                )
                logger.info(
                    "ingestion_cold_stored",
                    document_id=str(document_id),
                    stored=len(cold_meta),
                )

            # 8. Update document with chunk count
            await self.metadata_store.update_document(
                document_id=document_id,
                updates={"total_chunks": len(chunks)},
            )

            # 9. Hot tier capacity check
            await self._enforce_hot_tier_capacity()

            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.info(
                "ingestion_complete",
                document_id=str(document_id),
                chunks=len(chunks),
                hot_chunks=len(hot_chunks),
                cold_chunks=len(cold_chunks),
                elapsed_ms=elapsed,
            )

            return IngestionResult(
                document_id=document_id,
                status="success",
                chunks_created=len(chunks),
                total_chunks=len(chunks),
                hot_chunks=len(hot_chunks),
                cold_chunks=len(cold_chunks),
                processing_time_ms=elapsed,
            )

        except Exception as e:
            logger.error("ingestion_failed", document_id=str(document_id), error=str(e))
            return IngestionResult(
                document_id=document_id,
                status="failed",
                error=str(e),
            )

    async def _enforce_hot_tier_capacity(self) -> None:
        """If hot tier exceeds capacity, evict the coldest chunks to cold tier."""
        try:
            hot_count = await self.metadata_store.count_chunks_by_tier(tier=Tier.HOT)
            if hot_count > self.HOT_TIER_CAPACITY:
                if self.migration_engine is not None:
                    evicted = await self.migration_engine.evict_coldest(
                        percent=self.EVICT_PERCENT
                    )
                    logger.warning(
                        "hot_tier_capacity_exceeded",
                        hot_count=hot_count,
                        capacity=self.HOT_TIER_CAPACITY,
                        evicted=len(evicted),
                    )
                else:
                    logger.warning(
                        "hot_tier_capacity_exceeded_no_migration_engine",
                        hot_count=hot_count,
                        capacity=self.HOT_TIER_CAPACITY,
                    )
        except Exception as e:
            logger.error("hot_tier_capacity_check_failed", error=str(e))

    async def delete_document(self, document_id: uuid.UUID) -> int:
        """Delete a document and all its chunks from every store.

        Args:
            document_id: Document to delete.

        Returns:
            Number of chunks deleted.
        """
        # 1. Find all chunks belonging to this document
        chunks = await self.metadata_store.query_chunks_by_document(document_id)
        if not chunks:
            # No chunks found — still try to delete the document metadata
            pass

        hot_ids = [c.chunk_id for c in chunks if c.tier == Tier.HOT]
        cold_ids = [c.chunk_id for c in chunks if c.tier == Tier.COLD]

        # 2. Delete from tiers (vector store + document store + metadata)
        deleted = 0
        if hot_ids:
            deleted += await self.hot_tier.delete(hot_ids)
        if cold_ids:
            deleted += await self.cold_tier.delete(cold_ids)

        # 3. Delete document metadata (critical: without this the document
        # record lingers and appears as an empty ghost on refresh)
        await self.metadata_store.delete_document(document_id)

        logger.info(
            "document_deleted",
            document_id=str(document_id),
            chunks_deleted=deleted,
            hot_chunks=len(hot_ids),
            cold_chunks=len(cold_ids),
        )
        return deleted

    async def ingest_file(
        self,
        file_path: str,
        title: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest a file from disk.

        Args:
            file_path: Path to file.
            title: Optional title.
            tags: Optional tags.
            metadata: Optional metadata.

        Returns:
            Ingestion result.
        """
        text = extract_text(file_path)
        path = Path(file_path)

        return await self.ingest_text(
            text=text,
            source_uri=str(path.absolute()),
            title=title or path.name,
            tags=tags,
            metadata=metadata,
        )
