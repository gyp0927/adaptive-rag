"""Health check endpoints."""

from fastapi import APIRouter, HTTPException

from adaptive_rag.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])

# Global stores (set from main.py)
_metadata_store = None
_vector_store = None


def set_stores(metadata_store, vector_store) -> None:
    """Set store references for health checks."""
    global _metadata_store, _vector_store
    _metadata_store = metadata_store
    _vector_store = vector_store


@router.get("/health")
async def health_check() -> dict:
    """Liveness probe."""
    return {"status": "ok", "service": "adaptive-rag"}


@router.get("/ready")
async def readiness_check() -> dict:
    """Readiness probe — verifies DB and vector store connectivity."""
    checks = {}
    healthy = True

    if _metadata_store:
        try:
            from adaptive_rag.core.config import Tier
            hot = await _metadata_store.count_chunks_by_tier(Tier.HOT)
            checks["metadata_store"] = {"status": "ok", "hot_chunks": hot}
        except Exception as e:
            checks["metadata_store"] = {"status": "error", "detail": str(e)}
            healthy = False
            logger.warning("health_check_metadata_failed", error=str(e))
    else:
        checks["metadata_store"] = {"status": "not_configured"}
        healthy = False

    if _vector_store:
        try:
            from adaptive_rag.core.config import get_settings
            collection = get_settings().VECTOR_DB_COLLECTION
            count = await _vector_store.count(collection)
            checks["vector_store"] = {"status": "ok", "collection_count": count}
        except Exception as e:
            checks["vector_store"] = {"status": "error", "detail": str(e)}
            healthy = False
            logger.warning("health_check_vector_failed", error=str(e))
    else:
        checks["vector_store"] = {"status": "not_configured"}
        healthy = False

    if not healthy:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks": checks},
        )

    return {"status": "ready", "checks": checks}
