"""Frequency tracking with time-decay and semantic clustering."""

import asyncio
from datetime import datetime
from typing import Any
import uuid

from adaptive_rag.core.config import get_settings
from adaptive_rag.core.logging import get_logger
from adaptive_rag.ingestion.embedder import Embedder
from adaptive_rag.storage.metadata_store.base import BaseMetadataStore, QueryCluster
from adaptive_rag.storage.vector_store.base import BaseVectorStore

from .clustering import QueryClusterStore
from .decay import DecayEngine

logger = get_logger(__name__)


class FrequencyTracker:
    """Tracks access frequency with time-decay and semantic clustering.

    Uses query-topic clustering where similar queries contribute
    to the same topic's frequency score.
    """

    def __init__(
        self,
        metadata_store: BaseMetadataStore,
        vector_store: BaseVectorStore,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = get_settings()
        self.metadata_store = metadata_store
        self.cluster_store = QueryClusterStore(vector_store, metadata_store)
        self.decay_engine = DecayEngine()
        self.embedder = embedder or Embedder()

    async def record_access(
        self,
        chunk_ids: list[uuid.UUID],
        query_text: str,
        query_embedding: list[float] | None = None,
    ) -> None:
        """Record access to chunks and update frequency scores.

        Args:
            chunk_ids: IDs of accessed chunks.
            query_text: Original query text.
            query_embedding: Pre-computed query embedding (optional).
        """
        timestamp = datetime.utcnow()

        # 1. Get or create query cluster
        cluster_id = await self._get_or_create_cluster(
            query_text, query_embedding
        )

        # 2. Update per-chunk access counts
        await self.metadata_store.increment_access(
            chunk_ids=chunk_ids,
            cluster_id=cluster_id,
            timestamp=timestamp,
        )

        # 3. Update cluster access count
        await self.cluster_store.increment_access(cluster_id, timestamp)

        # 4. Recalculate frequency scores
        await self._recalculate_scores(chunk_ids, timestamp)

        logger.debug(
            "access_recorded",
            chunks=len(chunk_ids),
            cluster_id=str(cluster_id),
        )

    async def get_frequency_score(self, chunk_id: uuid.UUID) -> float:
        """Get the current (decayed) frequency score for a chunk.

        Args:
            chunk_id: Chunk ID.

        Returns:
            Current frequency score.
        """
        metadata = await self.metadata_store.get_chunk(chunk_id)
        if not metadata:
            return 0.0

        return self.decay_engine.apply_decay(
            base_score=metadata.frequency_score,
            last_accessed=metadata.last_accessed_at,
            access_count=metadata.access_count,
        )

    async def get_topic_frequency(self, query_embedding: list[float]) -> float:
        """Get aggregate frequency for a topic cluster.

        Used by the router to decide which tier to query.

        Args:
            query_embedding: Query embedding vector.

        Returns:
            Topic frequency score (0 if no matching cluster).
        """
        cluster = await self.cluster_store.find_nearest_cluster(
            query_embedding,
        )
        if not cluster:
            return 0.0  # Unknown topic -> routed to cold tier (freq <= HOT_TO_COLD_THRESHOLD)

        return self.decay_engine.apply_decay(
            base_score=cluster.frequency_score,
            last_accessed=cluster.last_accessed_at,
            access_count=cluster.access_count,
        )

    async def get_topic_frequencies_batch(
        self,
        query_embeddings: list[list[float]],
    ) -> list[float]:
        """Get topic frequencies for multiple embeddings in one batch.

        Uses search_batch under the hood to cut vector-store round-trips
        from N to 1 when checking many chunks (e.g. during ingestion).

        Args:
            query_embeddings: List of query embedding vectors.

        Returns:
            List of frequency scores, same order as input.
        """
        if not query_embeddings:
            return []

        clusters = await self.cluster_store.find_nearest_clusters_batch(
            query_embeddings,
        )

        return [
            self.decay_engine.apply_decay(
                base_score=c.frequency_score,
                last_accessed=c.last_accessed_at,
                access_count=c.access_count,
            )
            if c
            else 0.0
            for c in clusters
        ]

    async def _get_or_create_cluster(
        self,
        query_text: str,
        query_embedding: list[float] | None,
    ) -> uuid.UUID:
        """Find existing cluster or create new one.

        Args:
            query_text: Query text.
            query_embedding: Pre-computed embedding.

        Returns:
            Cluster ID.
        """
        if query_embedding is None:
            query_embedding = (await self.embedder.embed_batch([query_text]))[0]

        existing = await self.cluster_store.find_nearest_cluster(
            query_embedding,
        )

        if existing:
            return existing.cluster_id

        # Create new cluster
        new_cluster = QueryCluster(
            cluster_id=uuid.uuid4(),
            centroid=query_embedding,
            representative_query=query_text,
            access_count=0,
            frequency_score=0.0,
            member_count=1,
            created_at=datetime.utcnow(),
            last_accessed_at=None,
        )
        await self.cluster_store.create_cluster(new_cluster)
        return new_cluster.cluster_id

    async def _recalculate_scores(
        self,
        chunk_ids: list[uuid.UUID],
        timestamp: datetime,
    ) -> None:
        """Recalculate frequency scores for affected chunks.

        Batch-optimized: fetches all chunk metadata and needed clusters
        in two queries, then applies updates in a single transaction.

        Args:
            chunk_ids: Chunks to update.
            timestamp: Current timestamp.
        """
        if not chunk_ids:
            return

        # 1. Batch fetch all chunk metadata
        metadatas = await self.metadata_store.get_chunks_batch(chunk_ids)
        meta_by_id = {m.chunk_id: m for m in metadatas}

        # 2. Collect unique cluster IDs
        cluster_ids = {
            m.topic_cluster_id for m in metadatas if m.topic_cluster_id
        }

        # 3. Batch fetch all needed clusters
        cluster_scores: dict[uuid.UUID, float] = {}
        if cluster_ids:
            # get_all_clusters is the only batch cluster API available;
            # filter locally to avoid N+1.
            all_clusters = await self.metadata_store.get_all_clusters()
            cluster_scores = {
                c.cluster_id: c.frequency_score
                for c in all_clusters
                if c.cluster_id in cluster_ids
            }

        # 4. Compute new scores and build batch update
        batch_updates: dict[uuid.UUID, dict[str, Any]] = {}
        for chunk_id in chunk_ids:
            metadata = meta_by_id.get(chunk_id)
            if not metadata:
                continue

            cluster_score = cluster_scores.get(metadata.topic_cluster_id, 0.0)
            new_score = self.decay_engine.compute_score(
                access_count=metadata.access_count,
                last_accessed=metadata.last_accessed_at,
                created_at=metadata.created_at,
                cluster_score=cluster_score,
            )
            batch_updates[chunk_id] = {
                "frequency_score": new_score,
                "updated_at": timestamp,
            }

        # 5. Single transaction update
        if batch_updates:
            await self.metadata_store.update_chunks_batch(batch_updates)
