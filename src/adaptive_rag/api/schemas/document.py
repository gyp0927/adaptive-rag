"""API request/response schemas for documents."""

from pydantic import BaseModel, Field
from typing import Literal
import uuid


class DocumentUploadRequest(BaseModel):
    """Document upload request."""

    text: str = Field(..., min_length=1, max_length=1000000)
    chunking_strategy: Literal["fixed", "recursive"] = "recursive"
    chunk_size: int = Field(default=512, ge=100, le=4096)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    title: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """Document upload response."""

    document_id: uuid.UUID
    status: str
    chunks_created: int
    message: str


class DocumentListItem(BaseModel):
    """Document list item."""

    document_id: uuid.UUID
    source_type: str
    source_uri: str
    title: str | None = None
    total_chunks: int
    created_at: str


class DocumentListResponse(BaseModel):
    """Document list response."""

    documents: list[DocumentListItem]
    total: int


class ChunkContent(BaseModel):
    """Chunk with its content."""

    chunk_id: uuid.UUID
    chunk_index: int
    tier: str
    frequency_score: float
    access_count: int
    content: str


class DocumentDetailResponse(BaseModel):
    """Document detail with chunks."""

    document_id: uuid.UUID
    source_type: str
    source_uri: str
    title: str | None = None
    content_hash: str
    total_chunks: int
    created_at: str
    updated_at: str
    chunks: list[ChunkContent]
