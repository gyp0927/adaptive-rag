"""Semantic query clustering for topic-based frequency tracking."""

import asyncio
from datetime import datetime, timedelta
from typing import Any
import uuid

from adaptive_rag.core.config import get_settings
from adaptive_rag.core.exceptions import ClusterNotFoundError
from adaptive_rag.core.logging import get_logger
from adaptive_rag.storage.metadata_store.base import BaseMetadataStore, QueryCluster
from adaptive_rag.storage.vector_store.base import BaseVectorStore

logger = get_logger(__name__)


class QueryClusterStore:
    """Manages query clusters using vector store for similarity search."""

    # Clusters inactive longer than this are considered stale and removed.
    STALE_DAYS: int = 7
    # Clusters with more members than this are split to keep boundaries sharp.
    MAX_CLUSTER_SIZE: int = 500
    # Number of sub-clusters when splitting an oversized cluster.
    SPLIT_SUBCLUSTERS: int = 3

    def __init__(
        self,
        vector_store: BaseVectorStore,
        metadata_store: BaseMetadataStore,
    ) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.collection = "query_clusters"

    async def find_nearest_cluster(
        self,
        query_embedding: list[float],
        threshold: float | None = None,
    ) -> QueryCluster | None:
        """Find the nearest cluster within similarity threshold.

        Args:
            query_embedding: Query embedding vector.
            threshold: Minimum cosine similarity (default from config).

        Returns:
            Nearest cluster or None if none within threshold.
        """
        threshold = threshold or self.settings.QUERY_CLUSTERING_THRESHOLD

        results = await self.vector_store.search(
            collection=self.collection,
            query_vector=query_embedding,
            limit=1,
        )

        if not results:
            return None

        # Qdrant returns cosine similarity directly (higher = more similar)
        if results[0].score < threshold:
            return None

        cluster_id = uuid.UUID(results[0].payload.get("cluster_id")) if results[0].payload else None
        if not cluster_id:
            return None

        return await self.metadata_store.get_cluster(cluster_id)

    async def find_nearest_clusters_batch(
        self,
        query_embeddings: list[list[float]],
        threshold: float | None = None,
    ) -> list[QueryCluster | None]:
        """Find the nearest cluster for multiple embeddings in one batch request.

        Uses vector_store.search_batch to reduce round-trips when checking
        topic frequency for many chunks at once (e.g. during ingestion).

        Args:
            query_embeddings: List of embedding vectors.
            threshold: Minimum cosine similarity (default from config).

        Returns:
            List of nearest clusters (or None), in the same order as input.
        """
        if not query_embeddings:
            return []

        threshold = threshold or self.settings.QUERY_CLUSTERING_THRESHOLD

        batch_results = await self.vector_store.search_batch(
            collection=self.collection,
            query_vectors=query_embeddings,
            limit=1,
        )

        # Collect all cluster IDs that passed the threshold
        cluster_ids: list[uuid.UUID | None] = []
        for results in batch_results:
            if not results or results[0].score < threshold:
                cluster_ids.append(None)
                continue
            cid = uuid.UUID(results[0].payload.get("cluster_id")) if results[0].payload else None
            cluster_ids.append(cid)

        # Fetch all clusters in parallel (typically very few unique IDs)
        unique_ids = {cid for cid in cluster_ids if cid is not None}
        cluster_map: dict[uuid.UUID, QueryCluster | None] = {}
        if unique_ids:
            clusters = await asyncio.gather(*[
                self.metadata_store.get_cluster(cid) for cid in unique_ids
            ])
            cluster_map = {cid: c for cid, c in zip(unique_ids, clusters)}

        return [cluster_map.get(cid) for cid in cluster_ids]

    async def create_cluster(self, cluster: QueryCluster) -> None:
        """Create a new query cluster.

        Args:
            cluster: Cluster to create.
        """
        # Store centroid in vector store
        await self.vector_store.upsert(
            collection=self.collection,
            ids=[cluster.cluster_id],
            vectors=[cluster.centroid],
            payloads=[{
                "cluster_id": str(cluster.cluster_id),
                "representative_query": cluster.representative_query,
                "access_count": cluster.access_count,
                "frequency_score": cluster.frequency_score,
                "member_count": cluster.member_count,
            }],
        )

        # Store metadata
        await self.metadata_store.create_cluster(cluster)

        logger.info(
            "cluster_created",
            cluster_id=str(cluster.cluster_id),
            query=cluster.representative_query[:100],
        )

    async def delete_cluster(self, cluster_id: uuid.UUID) -> None:
        """Delete a cluster from both vector and metadata stores.

        Args:
            cluster_id: Cluster to delete.
        """
        await self.vector_store.delete(self.collection, [cluster_id])
        # Metadata deletion is best-effort; not all stores implement it
        try:
            await self.metadata_store.delete_clusters([cluster_id])
        except AttributeError:
            pass

        logger.info("cluster_deleted", cluster_id=str(cluster_id))

    async def update_cluster(
        self,
        cluster_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> None:
        """Update cluster fields.

        Args:
            cluster_id: Cluster ID.
            updates: Fields to update.
        """
        await self.metadata_store.update_cluster(cluster_id, updates)

    async def increment_access(
        self,
        cluster_id: uuid.UUID,
        timestamp: datetime,
    ) -> None:
        """Increment cluster access count.

        Args:
            cluster_id: Cluster ID.
            timestamp: Access timestamp.
        """
        cluster = await self.metadata_store.get_cluster(cluster_id)
        if not cluster:
            return

        await self.metadata_store.update_cluster(
            cluster_id=cluster_id,
            updates={
                "access_count": cluster.access_count + 1,
                "last_accessed_at": timestamp,
                "member_count": cluster.member_count + 1,
            },
        )

    async def merge_clusters(
        self,
        cluster_id_1: uuid.UUID,
        cluster_id_2: uuid.UUID,
    ) -> QueryCluster:
        """Merge two clusters into one.

        Args:
            cluster_id_1: First cluster ID.
            cluster_id_2: Second cluster ID.

        Returns:
            Merged cluster.
        """
        c1 = await self.metadata_store.get_cluster(cluster_id_1)
        c2 = await self.metadata_store.get_cluster(cluster_id_2)

        if not c1 or not c2:
            raise ValueError("One or both clusters not found")

        # Weighted average of centroids
        total = c1.member_count + c2.member_count
        new_centroid = [
            (c1.centroid[i] * c1.member_count + c2.centroid[i] * c2.member_count) / total
            for i in range(len(c1.centroid))
        ]

        merged = QueryCluster(
            cluster_id=uuid.uuid4(),
            centroid=new_centroid,
            representative_query=c1.representative_query,
            access_count=c1.access_count + c2.access_count,
            frequency_score=max(c1.frequency_score, c2.frequency_score),
            member_count=total,
            created_at=min(c1.created_at, c2.created_at),
            last_accessed_at=max(
                c1.last_accessed_at or c1.created_at,
                c2.last_accessed_at or c2.created_at,
            ),
        )

        # Delete old clusters from vector store
        await self.vector_store.delete(
            collection=self.collection,
            ids=[cluster_id_1, cluster_id_2],
        )

        # Create merged cluster
        await self.create_cluster(merged)

        logger.info(
            "clusters_merged",
            cluster_1=str(cluster_id_1),
            cluster_2=str(cluster_id_2),
            merged=str(merged.cluster_id),
        )

        return merged

    async def cleanup_stale_clusters(self) -> tuple[int, int]:
        """Remove stale clusters and split oversized ones.

        A cluster is stale if it has been inactive for more than
        ``STALE_DAYS``. A cluster is oversized if its ``member_count`` exceeds
        ``MAX_CLUSTER_SIZE`` and gets split into ``SPLIT_SUBCLUSTERS``
        sub-clusters.

        Returns:
            Tuple of (deleted_count, split_count).
        """
        all_clusters = await self.metadata_store.get_all_clusters()
        deleted = 0
        split = 0
        stale_cutoff = datetime.utcnow() - timedelta(days=self.STALE_DAYS)

        for cluster in all_clusters:
            last_active = cluster.last_accessed_at or cluster.created_at
            if last_active < stale_cutoff:
                await self.delete_cluster(cluster.cluster_id)
                deleted += 1
                continue

            if cluster.member_count > self.MAX_CLUSTER_SIZE:
                await self.split_cluster(cluster.cluster_id)
                split += 1

        logger.info(
            "cluster_cleanup_complete",
            deleted=deleted,
            split=split,
            total_clusters=len(all_clusters),
        )
        return deleted, split

    async def split_cluster(self, cluster_id: uuid.UUID) -> list[QueryCluster]:
        """Split an oversized cluster into sub-clusters using K-Means.

        Because the metadata store doesn't keep per-query embeddings, we
        synthesise fake members by adding small random perturbations to the
        centroid.  This is a pragmatic approximation; a production system
        should instead store actual member embeddings.

        Args:
            cluster_id: Cluster to split.

        Returns:
            List of newly created sub-clusters.
        """
        cluster = await self.metadata_store.get_cluster(cluster_id)
        if not cluster:
            raise ClusterNotFoundError(f"Cluster {cluster_id} not found")

        try:
            from sklearn.cluster import KMeans
            import numpy as np
        except ImportError as e:
            logger.error("sklearn_not_installed_for_cluster_split", error=str(e))
            raise ImportError(
                "scikit-learn is required for cluster splitting. "
                "Install it: pip install scikit-learn"
            ) from e

        centroid = cluster.centroid
        member_count = cluster.member_count
        n_sub = min(self.SPLIT_SUBCLUSTERS, max(2, member_count // 2))

        # Synthesise fake member embeddings around the centroid
        rng = np.random.default_rng(seed=42)
        std_dev = 0.05  # small perturbation for cosine-similarity-normalised vectors
        members = np.array([
            np.array(centroid) + rng.normal(0, std_dev, len(centroid))
            for _ in range(member_count)
        ])

        kmeans = KMeans(n_clusters=n_sub, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(members)

        sub_clusters: list[QueryCluster] = []
        for i in range(n_sub):
            sub_members = members[labels == i]
            sub_centroid_raw = kmeans.cluster_centers_[i]
            # Normalise to unit length so Qdrant cosine search is consistent
            norm = np.linalg.norm(sub_centroid_raw)
            sub_centroid = (
                (sub_centroid_raw / norm).tolist()
                if norm > 0
                else sub_centroid_raw.tolist()
            )
            sub_count = len(sub_members)

            sub_cluster = QueryCluster(
                cluster_id=uuid.uuid4(),
                centroid=sub_centroid,
                representative_query=cluster.representative_query,
                access_count=max(1, cluster.access_count // n_sub),
                frequency_score=cluster.frequency_score,
                member_count=sub_count,
                created_at=datetime.utcnow(),
                last_accessed_at=cluster.last_accessed_at,
            )
            await self.create_cluster(sub_cluster)
            sub_clusters.append(sub_cluster)

        # Remove the original oversized cluster
        await self.delete_cluster(cluster_id)

        logger.info(
            "cluster_split",
            original_cluster_id=str(cluster_id),
            sub_clusters=len(sub_clusters),
            original_members=member_count,
        )
        return sub_clusters
