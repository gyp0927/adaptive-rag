"""Query endpoints."""

from fastapi import APIRouter, HTTPException, Depends

from adaptive_rag.api.schemas.query import QueryRequest, QueryResponse, RetrievedChunkSchema
from adaptive_rag.core.config import Tier
from adaptive_rag.core.logging import get_logger
from adaptive_rag.retrieval.retriever import UnifiedRetriever

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


# Global retriever instance (initialized in main.py)
_retriever: UnifiedRetriever | None = None


def set_retriever(retriever: UnifiedRetriever) -> None:
    """Set the global retriever instance."""
    global _retriever
    _retriever = retriever


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Submit a query and retrieve relevant chunks."""
    if not _retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        # Map tier preference. "both" is not a valid Tier enum value;
        # passing None lets the router use BOTH automatically.
        tier = None
        if request.tier and request.tier != "both":
            tier = Tier(request.tier)

        result = await _retriever.query(
            query_text=request.query,
            top_k=request.top_k,
            tier=tier,
            decompress=request.decompress,
            filters=request.filters,
        )

        return QueryResponse(
            chunks=[
                RetrievedChunkSchema(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    content=c.content,
                    score=c.score,
                    tier=c.tier.value,
                    is_decompressed=c.is_decompressed,
                    access_count=c.access_count,
                    frequency_score=c.frequency_score,
                )
                for c in result.chunks
            ],
            routing_strategy=result.routing_strategy.value,
            hot_results_count=result.hot_results_count,
            cold_results_count=result.cold_results_count,
            total_latency_ms=result.total_latency_ms,
            topic_frequency=result.topic_frequency,
        )

    except Exception as e:
        logger.error("query_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
