"""Memory retrieval endpoints."""

from fastapi import APIRouter, HTTPException

from hot_and_cold_memory.api.schemas.retrieve import (
    RetrievedMemorySchema,
    RetrieveRequest,
    RetrieveResponse,
)
from hot_and_cold_memory.core.config import Tier
from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.retrieval.retriever import UnifiedRetriever

logger = get_logger(__name__)
router = APIRouter(prefix="/retrieve", tags=["Retrieve"])

# Global retriever instance
_retriever: UnifiedRetriever | None = None


def set_retriever(retriever: UnifiedRetriever) -> None:
    """Set the global retriever instance."""
    global _retriever
    _retriever = retriever


@router.post("", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Retrieve relevant memories for a query."""
    if not _retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        tier = None
        if request.tier and request.tier != "both":
            tier = Tier(request.tier)

        result = await _retriever.query(
            query_text=request.query,
            top_k=request.top_k,
            tier=tier,
            filters=request.filters,
            use_hybrid=request.use_hybrid,
            use_profile=request.use_profile,
        )

        return RetrieveResponse(
            memories=[
                RetrievedMemorySchema(
                    memory_id=c.memory_id,
                    content=c.content,
                    score=c.score,
                    tier=c.tier.value,
                    access_count=c.access_count,
                    frequency_score=c.frequency_score,
                    memory_type=c.memory_type,
                )
                for c in result.chunks
            ],
            routing_strategy=result.routing_strategy.value,
            hot_results_count=result.hot_results_count,
            cold_results_count=result.cold_results_count,
            total_latency_ms=result.total_latency_ms,
            topic_frequency=result.topic_frequency,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("retrieve_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal retrieval error") from e
