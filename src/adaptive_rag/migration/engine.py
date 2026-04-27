"""Tier migration engine."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid

from adaptive_rag.core.config import Tier, get_settings
from adaptive_rag.core.exceptions import MigrationError, ChunkNotFoundError
from adaptive_rag.core.logging import get_logger
from adaptive_rag.ingestion.chunker import Chunk
from adaptive_rag.ingestion.embedder import Embedder
from adaptive_rag.storage.metadata_store.base import (
    BaseMetadataStore,
    MigrationLog,
)
from adaptive_rag.tiers.base import RetrievedChunk
from adaptive_rag.tiers.hot_tier import HotTier
from adaptive_rag.tiers.cold_tier import ColdTier
from adaptive_rag.tiers.compression import CompressionEngine
from adaptive_rag.tiers.decompression import DecompressionEngine

from .policies import MigrationPolicy

logger = get_logger(__name__)


@dataclass
class MigrationResult:
    """Result of a single migration."""

    chunk_id: uuid.UUID
    direction: str
    original_size: int
    new_size: int
    compression_ratio: float
    success: bool = True
    error: str | None = None


@dataclass
class MigrationReport:
    """Report of a migration cycle."""

    started_at: datetime
    completed_at: datetime | None = None
    hot_to_cold: list[MigrationResult] = field(default_factory=list)
    cold_to_hot: list[MigrationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total_processed: int = 0
    skipped_off_peak: bool = False


class MigrationEngine:
    """Orchestrates chunk migration between hot and cold tiers."""

    # Default off-peak window for hot->cold migrations (server local time).
    # Hot->cold runs LLM compression and is expensive, so we restrict it to
    # the quiet hours by default. Cold->hot is cheap and runs any time.
    DEFAULT_OFF_PEAK_START_HOUR = 2
    DEFAULT_OFF_PEAK_END_HOUR = 5

    def __init__(
        self,
        hot_tier: HotTier,
        cold_tier: ColdTier,
        metadata_store: BaseMetadataStore,
        policy: MigrationPolicy | None = None,
        embedder: Embedder | None = None,
        off_peak_start_hour: int | None = None,
        off_peak_end_hour: int | None = None,
    ) -> None:
        self.settings = get_settings()
        self.hot_tier = hot_tier
        self.cold_tier = cold_tier
        self.metadata_store = metadata_store
        self.policy = policy or MigrationPolicy()
        self.embedder = embedder or Embedder()
        self._lock = asyncio.Lock()
        self.off_peak_start_hour = (
            off_peak_start_hour
            if off_peak_start_hour is not None
            else self.DEFAULT_OFF_PEAK_START_HOUR
        )
        self.off_peak_end_hour = (
            off_peak_end_hour
            if off_peak_end_hour is not None
            else self.DEFAULT_OFF_PEAK_END_HOUR
        )

    def _is_off_peak(self) -> bool:
        """Whether the current local hour is inside the configured off-peak window."""
        hour = datetime.now().hour
        start, end = self.off_peak_start_hour, self.off_peak_end_hour
        if start <= end:
            return start <= hour <= end
        # Wrap-around window (e.g. 22..3)
        return hour >= start or hour <= end

    async def run_migration_cycle(self, force: bool = False) -> MigrationReport:
        """Execute one migration cycle.

        Hot->cold migrations are expensive (LLM compression) and only run inside
        the off-peak window unless ``force=True``. Cold->hot is cheap and always
        runs.

        Args:
            force: If True, bypass the off-peak gate for hot->cold migrations.

        Returns:
            Migration report.
        """
        report = MigrationReport(
            started_at=datetime.utcnow(),
        )

        async with self._lock:
            # Phase 1: Identify candidates
            cold_candidates = await self._identify_cold_to_hot_candidates()

            run_hot_to_cold = force or self._is_off_peak()
            hot_candidates: list[uuid.UUID] = []
            if run_hot_to_cold:
                hot_candidates = await self._identify_hot_to_cold_candidates()
            else:
                report.skipped_off_peak = True
                logger.info(
                    "migration_hot_to_cold_skipped_peak_hours",
                    current_hour=datetime.now().hour,
                    off_peak_window=f"{self.off_peak_start_hour}-{self.off_peak_end_hour}",
                )

            logger.info(
                "migration_candidates",
                hot_to_cold=len(hot_candidates),
                cold_to_hot=len(cold_candidates),
            )

            # Phase 2: Execute hot -> cold migrations as a batch (one LLM call per group)
            if hot_candidates:
                batch_results = await self._migrate_hot_to_cold_batch(hot_candidates)
                for result in batch_results:
                    if isinstance(result, Exception):
                        report.errors.append(str(result))
                    else:
                        report.hot_to_cold.append(result)

            # Phase 3: Execute cold -> hot migrations
            semaphore = asyncio.Semaphore(self.policy.thresholds.max_concurrent)

            async def migrate_cold_to_hot(chunk_id: uuid.UUID) -> MigrationResult:
                async with semaphore:
                    return await self._migrate_cold_to_hot(chunk_id)

            cold_results = await asyncio.gather(*[
                migrate_cold_to_hot(cid) for cid in cold_candidates
            ], return_exceptions=True)

            for result in cold_results:
                if isinstance(result, Exception):
                    report.errors.append(str(result))
                else:
                    report.cold_to_hot.append(result)

        report.completed_at = datetime.utcnow()
        report.total_processed = len(report.hot_to_cold) + len(report.cold_to_hot)

        logger.info(
            "migration_cycle_complete",
            hot_to_cold=len(report.hot_to_cold),
            cold_to_hot=len(report.cold_to_hot),
            errors=len(report.errors),
            skipped_off_peak=report.skipped_off_peak,
            duration_seconds=(report.completed_at - report.started_at).total_seconds(),
        )

        return report

    async def _migrate_hot_to_cold_batch(
        self,
        chunk_ids: list[uuid.UUID],
    ) -> list[MigrationResult | Exception]:
        """Migrate a batch of chunks hot->cold using grouped LLM compression.

        Compresses up to ``COMPRESSION_BATCH_SIZE`` chunks per LLM call to drive
        down cost (~10x reduction vs per-chunk).
        """
        # Pre-fetch hot chunks
        chunk_records = await asyncio.gather(*[
            self.hot_tier.get_by_id(cid) for cid in chunk_ids
        ], return_exceptions=True)

        valid_pairs: list[tuple[uuid.UUID, RetrievedChunk]] = []
        results: list[MigrationResult | Exception] = []
        for cid, record in zip(chunk_ids, chunk_records):
            if isinstance(record, Exception):
                results.append(MigrationError(f"Hot fetch failed for {cid}: {record}"))
                continue
            if record is None:
                results.append(ChunkNotFoundError(f"Chunk {cid} not found in hot tier"))
                continue
            valid_pairs.append((cid, record))

        # Group into batches and compress each group with a single LLM call
        group_size = max(1, self.settings.COMPRESSION_BATCH_SIZE)
        compression_engine: CompressionEngine = self.cold_tier.compression_engine

        for i in range(0, len(valid_pairs), group_size):
            group = valid_pairs[i:i + group_size]
            chunks_for_llm = [
                Chunk(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    text=record.content,
                    tags=record.metadata.get("tags", []) if record.metadata else [],
                )
                for _, record in group
            ]

            try:
                compressed_list = await compression_engine.compress_group(chunks_for_llm)
            except Exception as e:
                for cid, _ in group:
                    results.append(MigrationError(f"Hot->cold batch compress failed for {cid}: {e}"))
                continue

            # Persist each compressed result
            for (cid, record), compressed in zip(group, compressed_list):
                try:
                    result = await self._persist_hot_to_cold(record, compressed)
                    results.append(result)
                except Exception as e:
                    results.append(MigrationError(f"Hot->cold persist failed for {cid}: {e}"))

        return results

    async def _persist_hot_to_cold(
        self,
        record: RetrievedChunk,
        compressed,
    ) -> MigrationResult:
        """Persist an already-compressed chunk into the cold tier and clean up hot."""
        log = MigrationLog(
            chunk_id=record.chunk_id,
            direction="hot_to_cold",
            original_size=len(record.content),
            new_size=len(compressed.summary_text),
            started_at=datetime.utcnow(),
        )

        try:
            # 1. Embed the summary so cold tier search uses the summary's vector
            summary_embedding = await self.embedder.embed(compressed.summary_text)

            # 2. Drop existing metadata to avoid PK conflicts when re-inserting
            await self.metadata_store.delete_chunks([record.chunk_id])

            # 3. Write to cold tier stores directly using the precomputed summary
            await self.cold_tier.document_store.store_batch([
                (record.chunk_id, compressed.summary_text)
            ])
            await self.cold_tier.vector_store.upsert(
                collection=self.cold_tier.collection,
                ids=[record.chunk_id],
                vectors=[summary_embedding],
                payloads=[{
                    "chunk_id": str(record.chunk_id),
                    "document_id": str(record.document_id),
                    "tier": Tier.COLD.value,
                    "tags": record.metadata.get("tags", []) if record.metadata else [],
                    "compressed": True,
                }],
            )

            # 4. Recreate metadata in cold tier
            from adaptive_rag.storage.metadata_store.base import ChunkMetadata
            now = datetime.utcnow()
            await self.metadata_store.create_chunk(ChunkMetadata(
                chunk_id=record.chunk_id,
                document_id=record.document_id,
                tier=Tier.COLD,
                original_length=len(record.content),
                compressed_length=len(compressed.summary_text),
                access_count=record.access_count,
                frequency_score=record.frequency_score,
                created_at=now,
                updated_at=now,
                last_migrated_at=now,
                tags=record.metadata.get("tags", []) if record.metadata else [],
            ))

            # 5. Remove from hot tier
            await self.hot_tier.delete([record.chunk_id])

            ratio = len(compressed.summary_text) / max(len(record.content), 1)
            log.compression_ratio = ratio
            log.completed_at = datetime.utcnow()
            log.status = "success"
            await self.metadata_store.create_migration_log(log)

            logger.info(
                "migrated_hot_to_cold",
                chunk_id=str(record.chunk_id),
                original=len(record.content),
                compressed=len(compressed.summary_text),
                ratio=ratio,
            )

            return MigrationResult(
                chunk_id=record.chunk_id,
                direction="hot_to_cold",
                original_size=len(record.content),
                new_size=len(compressed.summary_text),
                compression_ratio=ratio,
            )
        except Exception as e:
            log.status = "failed"
            log.error_message = str(e)
            log.completed_at = datetime.utcnow()
            await self.metadata_store.create_migration_log(log)
            raise MigrationError(f"Hot to cold persist failed for {record.chunk_id}: {e}") from e

    async def _migrate_hot_to_cold(self, chunk_id: uuid.UUID) -> MigrationResult:
        """Migrate a chunk from hot to cold tier (legacy single-chunk path).

        Kept for callers that need per-chunk migration; the cycle runner uses
        the batched path for cost efficiency.

        Args:
            chunk_id: Chunk to migrate.

        Returns:
            Migration result.
        """
        log = MigrationLog(
            chunk_id=chunk_id,
            direction="hot_to_cold",
            original_size=0,
            new_size=0,
            started_at=datetime.utcnow(),
        )

        try:
            # 1. Retrieve from hot tier
            chunk = await self.hot_tier.get_by_id(chunk_id)
            if not chunk:
                raise ChunkNotFoundError(f"Chunk {chunk_id} not found in hot tier")

            # 2. Delete old metadata first (avoid unique constraint)
            await self.metadata_store.delete_chunks([chunk_id])

            # 3. Store in cold tier (compresses automatically)
            await self.cold_tier.store_chunks(
                chunks=[Chunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.content,
                    tags=chunk.metadata.get("tags", []) if chunk.metadata else [],
                )],
            )

            # 4. Delete from hot tier
            await self.hot_tier.delete([chunk_id])

            # 5. Get compressed info
            meta = await self.metadata_store.get_chunk(chunk_id)
            original_size = len(chunk.content)
            new_size = meta.compressed_length if meta else original_size
            ratio = new_size / original_size if original_size > 0 else 1.0

            # 6. Log migration
            log.original_size = original_size
            log.new_size = new_size
            log.compression_ratio = ratio
            log.completed_at = datetime.utcnow()
            log.status = "success"
            await self.metadata_store.create_migration_log(log)

            logger.info(
                "migrated_hot_to_cold",
                chunk_id=str(chunk_id),
                original=original_size,
                compressed=new_size,
                ratio=ratio,
            )

            return MigrationResult(
                chunk_id=chunk_id,
                direction="hot_to_cold",
                original_size=original_size,
                new_size=new_size,
                compression_ratio=ratio,
            )

        except Exception as e:
            log.status = "failed"
            log.error_message = str(e)
            log.completed_at = datetime.utcnow()
            await self.metadata_store.create_migration_log(log)
            raise MigrationError(f"Hot to cold migration failed for {chunk_id}: {e}") from e

    async def evict_coldest(self, percent: float = 0.1) -> list[MigrationResult]:
        """Evict the coldest ``percent`` of hot chunks down to cold tier.

        Used as an emergency-relief mechanism when the hot tier is over capacity.
        This bypasses the off-peak window because it's reacting to a capacity
        constraint, not a scheduled cleanup.

        Args:
            percent: Fraction of hot chunks to evict (0..1).

        Returns:
            Migration results (one entry per chunk attempted).
        """
        if percent <= 0:
            return []

        # Pull a generous superset; we'll trim below
        candidates = await self.metadata_store.query_chunks_by_tier_and_score(
            tier=Tier.HOT,
            limit=self.policy.thresholds.batch_size * 4,
        )
        if not candidates:
            return []

        # Lowest frequency_score first - already ordered by the query
        evict_count = max(1, int(len(candidates) * percent))
        ids_to_evict = [c.chunk_id for c in candidates[:evict_count]]

        logger.warning(
            "hot_tier_eviction_triggered",
            count=len(ids_to_evict),
            percent=percent,
        )
        raw_results = await self._migrate_hot_to_cold_batch(ids_to_evict)
        return [r for r in raw_results if isinstance(r, MigrationResult)]

    async def _migrate_cold_to_hot(self, chunk_id: uuid.UUID) -> MigrationResult:
        """Migrate a chunk from cold to hot tier.

        Args:
            chunk_id: Chunk to migrate.

        Returns:
            Migration result.
        """
        log = MigrationLog(
            chunk_id=chunk_id,
            direction="cold_to_hot",
            original_size=0,
            new_size=0,
            started_at=datetime.utcnow(),
        )

        try:
            # 1. Retrieve from cold tier
            chunk = await self.cold_tier.get_by_id(chunk_id)
            if not chunk:
                raise ChunkNotFoundError(f"Chunk {chunk_id} not found in cold tier")

            # Get original content (summary)
            summary = chunk.content

            # 2. Decompress
            decompressed = await self.cold_tier.decompression_engine.decompress(summary)

            # 3. Generate embedding
            embedding = await self.embedder.embed(decompressed)

            # 4. Delete old metadata first (avoid unique constraint)
            await self.metadata_store.delete_chunks([chunk_id])

            # 5. Store in hot tier
            await self.hot_tier.store_chunks(
                chunks=[Chunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=decompressed,
                    tags=chunk.metadata.get("tags", []) if chunk.metadata else [],
                )],
                embeddings=[embedding],
            )

            # 6. Delete from cold tier
            await self.cold_tier.delete([chunk_id])

            # 6. Log migration
            log.original_size = len(summary)
            log.new_size = len(decompressed)
            log.compression_ratio = len(summary) / len(decompressed) if len(decompressed) > 0 else 1.0
            log.completed_at = datetime.utcnow()
            log.status = "success"
            await self.metadata_store.create_migration_log(log)

            logger.info(
                "migrated_cold_to_hot",
                chunk_id=str(chunk_id),
                summary_len=len(summary),
                expanded_len=len(decompressed),
            )

            return MigrationResult(
                chunk_id=chunk_id,
                direction="cold_to_hot",
                original_size=len(summary),
                new_size=len(decompressed),
                compression_ratio=log.compression_ratio or 0,
            )

        except Exception as e:
            log.status = "failed"
            log.error_message = str(e)
            log.completed_at = datetime.utcnow()
            await self.metadata_store.create_migration_log(log)
            raise MigrationError(f"Cold to hot migration failed for {chunk_id}: {e}") from e

    async def _identify_hot_to_cold_candidates(self) -> list[uuid.UUID]:
        """Identify hot chunks with low frequency for demotion.

        Returns:
            List of chunk IDs to demote.
        """
        chunks = await self.metadata_store.query_chunks_by_tier_and_score(
            tier=Tier.HOT,
            max_score=self.policy.thresholds.hot_to_cold,
            limit=self.policy.thresholds.batch_size,
        )
        return [c.chunk_id for c in chunks]

    async def _identify_cold_to_hot_candidates(self) -> list[uuid.UUID]:
        """Identify cold chunks with high frequency for promotion.

        Returns:
            List of chunk IDs to promote.
        """
        chunks = await self.metadata_store.query_chunks_by_tier_and_score(
            tier=Tier.COLD,
            min_score=self.policy.thresholds.cold_to_hot,
            limit=self.policy.thresholds.batch_size,
        )
        return [c.chunk_id for c in chunks]
