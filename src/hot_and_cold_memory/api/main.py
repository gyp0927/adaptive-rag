"""FastAPI application factory for Adaptive Memory."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from hot_and_cold_memory.core.config import get_settings
from hot_and_cold_memory.core.exceptions import AdaptiveMemoryError
from hot_and_cold_memory.core.logging import get_logger, setup_logging
from hot_and_cold_memory.frequency.tracker import FrequencyTracker
from hot_and_cold_memory.ingestion.embedder import Embedder
from hot_and_cold_memory.ingestion.pipeline import MemoryPipeline
from hot_and_cold_memory.migration.engine import MigrationEngine
from hot_and_cold_memory.migration.scheduler import MigrationScheduler
from hot_and_cold_memory.profile.augmenter import ProfileAugmenter
from hot_and_cold_memory.profile.builder import ProfileBuilder
from hot_and_cold_memory.profile.extractor import ProfileExtractor
from hot_and_cold_memory.profile.store import ProfileStore
from hot_and_cold_memory.retrieval.retriever import UnifiedRetriever
from hot_and_cold_memory.storage.cache.memory_cache import MemoryCache
from hot_and_cold_memory.storage.cache.redis_cache import RedisCache
from hot_and_cold_memory.storage.document_store.local_store import LocalDocumentStore
from hot_and_cold_memory.storage.metadata_store.postgres_store import PostgresMetadataStore
from hot_and_cold_memory.storage.vector_store.local_qdrant_store import LocalQdrantStore
from hot_and_cold_memory.tiers.cold_tier import ColdTier
from hot_and_cold_memory.tiers.compression import CompressionEngine
from hot_and_cold_memory.tiers.hot_tier import HotTier

from .routers import admin, health, memories, profile, retrieve

logger = get_logger(__name__)

# Global service instances
_services: dict[str, Any] = {}


async def initialize_services() -> dict[str, Any]:
    """Initialize all storage and service components."""
    settings = get_settings()

    # Storage layer
    vector_store = LocalQdrantStore()
    await vector_store.initialize()
    # Ensure cold tier collection exists
    await vector_store.ensure_collection(f"{settings.VECTOR_DB_COLLECTION}_cold")

    metadata_store = PostgresMetadataStore()
    await metadata_store.initialize()

    document_store = LocalDocumentStore()

    # Cache layer
    cache: RedisCache | MemoryCache
    if settings.CACHE_URL:
        cache = RedisCache()
        await cache.initialize()
        logger.info("cache_initialized", type="redis", url=settings.CACHE_URL)
    else:
        cache = MemoryCache()
        logger.info("cache_initialized", type="memory")

    # Embedding
    embedder = Embedder()

    # Tiers
    hot_tier = HotTier(
        vector_store=vector_store,
        metadata_store=metadata_store,
        document_store=document_store,
        cache=cache,
    )

    compression_engine = CompressionEngine()

    cold_tier = ColdTier(
        vector_store=vector_store,
        metadata_store=metadata_store,
        document_store=document_store,
        compression_engine=compression_engine,
        cache=cache,
        embedder=embedder,
    )

    # Frequency tracking
    frequency_tracker = FrequencyTracker(
        metadata_store=metadata_store,
        vector_store=vector_store,
        embedder=embedder,
    )

    # Migration
    migration_engine = MigrationEngine(
        hot_tier=hot_tier,
        cold_tier=cold_tier,
        metadata_store=metadata_store,
        embedder=embedder,
    )

    # Profile services
    profile_store = ProfileStore(metadata_store)
    profile_extractor = ProfileExtractor()
    profile_builder = ProfileBuilder(profile_store)
    profile_augmenter = ProfileAugmenter(profile_store)

    # Memory pipeline
    pipeline = MemoryPipeline(
        metadata_store=metadata_store,
        hot_tier=hot_tier,
        cold_tier=cold_tier,
        embedder=embedder,
        frequency_tracker=frequency_tracker,
        migration_engine=migration_engine,
        profile_extractor=profile_extractor,
        profile_builder=profile_builder,
    )

    # Retrieval (with profile augmentation)
    retriever = UnifiedRetriever(
        hot_tier=hot_tier,
        cold_tier=cold_tier,
        frequency_tracker=frequency_tracker,
        embedder=embedder,
        metadata_store=metadata_store,
        profile_augmenter=profile_augmenter,
    )

    # Scheduler
    migration_scheduler = MigrationScheduler()

    return {
        "vector_store": vector_store,
        "metadata_store": metadata_store,
        "document_store": document_store,
        "cache": cache,
        "embedder": embedder,
        "hot_tier": hot_tier,
        "cold_tier": cold_tier,
        "frequency_tracker": frequency_tracker,
        "retriever": retriever,
        "pipeline": pipeline,
        "migration_engine": migration_engine,
        "migration_scheduler": migration_scheduler,
        "profile_store": profile_store,
        "profile_extractor": profile_extractor,
        "profile_builder": profile_builder,
        "profile_augmenter": profile_augmenter,
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Application lifespan handler."""
    setup_logging(get_settings().LOG_LEVEL)
    logger.info("starting_up")

    global _services
    _services = await initialize_services()

    # Wire up routers
    retrieve.set_retriever(_services["retriever"])
    memories.set_pipeline(_services["pipeline"])
    memories.set_metadata_store(_services["metadata_store"])
    admin.set_migration_engine(_services["migration_engine"])
    admin.set_metadata_store(_services["metadata_store"])
    health.set_stores(_services["metadata_store"], _services["vector_store"])
    profile.set_profile_store(_services["profile_store"])

    # Profile reconciler
    from hot_and_cold_memory.profile.reconciler import ProfileReconciler
    profile_reconciler = ProfileReconciler(_services["profile_store"])

    # Start background migration scheduler
    scheduler = _services["migration_scheduler"]
    scheduler.start(
        migration_callback=_services["migration_engine"].run_migration_cycle,
        cluster_cleanup_callback=_services["frequency_tracker"].cluster_store.cleanup_stale_clusters,
        profile_reconcile_callback=lambda: profile_reconciler.reconcile("default"),
    )

    logger.info("services_initialized")
    yield

    # Graceful shutdown
    scheduler.stop()

    # Drain background tasks
    retriever = _services.get("retriever")
    if retriever and hasattr(retriever, "drain_background_tasks"):
        try:
            await retriever.drain_background_tasks()
            logger.info("background_tasks_drained")
        except Exception as e:
            logger.warning("drain_background_tasks_failed", error=str(e))

    # Close storage connections
    for name in ("vector_store", "metadata_store", "document_store", "cache"):
        store = _services.get(name)
        if store and hasattr(store, "close"):
            try:
                await store.close()
                logger.info("store_closed", name=name)
            except Exception as e:
                logger.warning("store_close_failed", name=name, error=str(e))

    logger.info("shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description="Adaptive Agent Memory with Frequency-Driven Tiered Storage",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS (restrict in production via CORS_ALLOW_ORIGINS env var)
    origins = (
        [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",") if o.strip()]
        if settings.CORS_ALLOW_ORIGINS != "*"
        else ["*"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # Register routers
    app.include_router(memories.router, prefix="/api/v1")
    app.include_router(retrieve.router, prefix="/api/v1")
    app.include_router(profile.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")
    app.include_router(health.router)

    # Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {"message": "Adaptive Memory API", "version": "0.1.0"}

    # Global exception handler for custom exception hierarchy
    @app.exception_handler(AdaptiveMemoryError)
    async def handle_adaptive_memory_error(request: Request, exc: AdaptiveMemoryError) -> JSONResponse:
        logger.error("adaptive_memory_error", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()


def main() -> None:
    """Entry point for running the API server."""
    settings = get_settings()
    import uvicorn
    uvicorn.run(
        "hot_and_cold_memory.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
