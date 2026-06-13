"""API request/response schemas for memory retrieval."""

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    """Memory retrieval request."""

    query: str = Field(..., min_length=1, max_length=10000)
    top_k: int = Field(default=10, ge=1, le=100)
    tier: Literal["hot", "cold", "both"] | None = Field(
        default=None,
        description="Tier preference. None = auto-route based on frequency.",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Metadata filters.",
    )
    use_hybrid: bool = Field(
        default=False,
        description="If True, fuse vector similarity with keyword search using RRF.",
    )
    use_profile: bool = Field(
        default=True,
        description="If True, use user profile to rewrite query and boost results.",
    )


class RetrievedMemorySchema(BaseModel):
    """A retrieved memory in the response."""

    memory_id: uuid.UUID
    content: str
    score: float
    tier: Literal["hot", "cold"]
    access_count: int
    frequency_score: float
    memory_type: str


class RetrieveResponse(BaseModel):
    """Memory retrieval response."""

    memories: list[RetrievedMemorySchema]
    routing_strategy: Literal["hot_only", "cold_only", "hot_first", "both"]
    hot_results_count: int
    cold_results_count: int
    total_latency_ms: float
    topic_frequency: float
