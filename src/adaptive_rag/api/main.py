"""FastAPI application factory."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from adaptive_rag.core.config import get_settings
from adaptive_rag.core.logging import setup_logging, get_logger
from adaptive_rag.ingestion.embedder import Embedder
from adaptive_rag.ingestion.pipeline import IngestionPipeline
from adaptive_rag.storage.cache.memory_cache import MemoryCache
from adaptive_rag.storage.document_store.local_store import LocalDocumentStore
from adaptive_rag.storage.metadata_store.postgres_store import PostgresMetadataStore
from adaptive_rag.storage.vector_store.local_qdrant_store import LocalQdrantStore
from adaptive_rag.tiers.hot_tier import HotTier
from adaptive_rag.tiers.cold_tier import ColdTier
from adaptive_rag.tiers.compression import CompressionEngine
from adaptive_rag.tiers.decompression import DecompressionEngine
from adaptive_rag.frequency.tracker import FrequencyTracker
from adaptive_rag.retrieval.retriever import UnifiedRetriever
from adaptive_rag.migration.engine import MigrationEngine

from .routers import query, documents, admin, health

logger = get_logger(__name__)

# Global service instances
_services: dict = {}


async def initialize_services() -> dict:
    """Initialize all storage and service components."""
    settings = get_settings()

    # Storage layer (local mode - no Docker needed)
    vector_store = LocalQdrantStore()
    await vector_store.initialize()
    # Ensure cold tier collection exists
    await vector_store.ensure_collection(f"{settings.VECTOR_DB_COLLECTION}_cold")

    metadata_store = PostgresMetadataStore()
    await metadata_store.initialize()

    document_store = LocalDocumentStore()
    cache = MemoryCache()

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
    decompression_engine = DecompressionEngine()

    cold_tier = ColdTier(
        vector_store=vector_store,
        metadata_store=metadata_store,
        document_store=document_store,
        compression_engine=compression_engine,
        decompression_engine=decompression_engine,
        cache=cache,
        embedder=embedder,
    )

    # Frequency tracking
    frequency_tracker = FrequencyTracker(
        metadata_store=metadata_store,
        vector_store=vector_store,
        embedder=embedder,
    )

    # Retrieval
    retriever = UnifiedRetriever(
        hot_tier=hot_tier,
        cold_tier=cold_tier,
        frequency_tracker=frequency_tracker,
        embedder=embedder,
    )

    # Migration (must be created before ingestion pipeline)
    migration_engine = MigrationEngine(
        hot_tier=hot_tier,
        cold_tier=cold_tier,
        metadata_store=metadata_store,
        embedder=embedder,
    )

    # Ingestion
    pipeline = IngestionPipeline(
        metadata_store=metadata_store,
        hot_tier=hot_tier,
        cold_tier=cold_tier,
        embedder=embedder,
        frequency_tracker=frequency_tracker,
        migration_engine=migration_engine,
    )

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
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    setup_logging(get_settings().LOG_LEVEL)
    logger.info("starting_up")

    global _services
    _services = await initialize_services()

    # Wire up routers
    query.set_retriever(_services["retriever"])
    documents.set_pipeline(_services["pipeline"])
    documents.set_stores(_services["metadata_store"], _services["document_store"])
    admin.set_migration_engine(_services["migration_engine"])

    logger.info("services_initialized")
    yield
    logger.info("shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description="Frequency-Driven Adaptive RAG Tiered Architecture",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS - allow_credentials=False when using wildcard origins
    # (browsers reject credentials with wildcard; set specific origins to enable credentials)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")
    app.include_router(health.router)

    # Static files
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def root():
        return FileResponse(os.path.join(static_dir, "index.html"))

    return app


app = create_app()


def main() -> None:
    """Entry point for running the API server."""
    settings = get_settings()
    import uvicorn
    uvicorn.run(
        "adaptive_rag.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
