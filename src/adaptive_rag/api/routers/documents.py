"""Document ingestion endpoints."""

import os
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from adaptive_rag.api.schemas.document import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentListItem,
    DocumentDetailResponse,
    ChunkContent,
)
from adaptive_rag.core.logging import get_logger
from adaptive_rag.ingestion.extractors.pdf import extract_pdf
from adaptive_rag.ingestion.extractors.docx import extract_docx
from adaptive_rag.ingestion.extractors.image import extract_image
from adaptive_rag.ingestion.extractors.text import extract_text_from_bytes
from adaptive_rag.ingestion.pipeline import IngestionPipeline
from adaptive_rag.storage.metadata_store.postgres_store import PostgresMetadataStore
from adaptive_rag.storage.document_store.local_store import LocalDocumentStore

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

# Global pipeline instance
_pipeline: IngestionPipeline | None = None
_metadata_store: PostgresMetadataStore | None = None
_document_store: LocalDocumentStore | None = None


def set_pipeline(pipeline: IngestionPipeline) -> None:
    """Set the global ingestion pipeline."""
    global _pipeline
    _pipeline = pipeline


def set_stores(
    metadata_store: PostgresMetadataStore,
    document_store: LocalDocumentStore,
) -> None:
    """Set storage references for document retrieval."""
    global _metadata_store, _document_store
    _metadata_store = metadata_store
    _document_store = document_store


def _extract_text(filename: str, content: bytes) -> str:
    """Extract text from file based on extension.

    Args:
        filename: Original filename.
        content: File content as bytes.

    Returns:
        Extracted text.

    Raises:
        HTTPException: If file type is unsupported or extraction fails.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".txt", ".md", ".markdown", ".rst", ".json", ".csv"):
        return extract_text_from_bytes(content)

    elif ext == ".pdf":
        return extract_pdf(content)

    elif ext in (".docx", ".doc"):
        return extract_docx(content)

    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"):
        return extract_image(content)

    else:
        # Fallback: try to decode as plain text with binary-garbage guard
        try:
            return extract_text_from_bytes(content)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type or binary content: {ext}. "
                       f"Supported: .txt, .md, .pdf, .docx, .png, .jpg, .jpeg. ({e})"
            )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    tags: str = Form(""),
) -> DocumentUploadResponse:
    """Upload a document for ingestion.

    Supports: .txt, .md, .pdf, .docx, .png, .jpg, .jpeg
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        content = await file.read()
        text = _extract_text(file.filename or "upload", content)

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content extracted from file")

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        result = await _pipeline.ingest_text(
            text=text,
            source_uri=file.filename or "upload",
            title=title or file.filename,
            tags=tag_list,
        )

        return DocumentUploadResponse(
            document_id=result.document_id,
            status=result.status,
            chunks_created=result.chunks_created,
            message="Document ingested successfully" if result.status == "success" else result.error or "Failed",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("upload_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/text", response_model=DocumentUploadResponse)
async def upload_text(request: DocumentUploadRequest) -> DocumentUploadResponse:
    """Upload text content directly."""
    raise HTTPException(status_code=501, detail="Direct text upload not yet implemented")


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = 100,
    offset: int = 0,
) -> DocumentListResponse:
    """List all uploaded documents."""
    if not _metadata_store:
        raise HTTPException(status_code=503, detail="Metadata store not initialized")

    docs = await _metadata_store.list_documents(limit=limit, offset=offset)
    return DocumentListResponse(
        documents=[
            DocumentListItem(
                document_id=d.document_id,
                source_type=d.source_type,
                source_uri=d.source_uri,
                title=d.title,
                total_chunks=d.total_chunks,
                created_at=d.created_at.isoformat() if d.created_at else "",
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(document_id: str) -> DocumentDetailResponse:
    """Get document detail with all chunk contents."""
    if not _metadata_store or not _document_store:
        raise HTTPException(status_code=503, detail="Stores not initialized")

    import uuid as uuid_mod
    try:
        doc_uuid = uuid_mod.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    doc = await _metadata_store.get_document(doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks_meta = await _metadata_store.query_chunks_by_document(doc_uuid)

    chunk_contents: list[ChunkContent] = []
    for meta in chunks_meta:
        content = await _document_store.get(meta.chunk_id)
        chunk_contents.append(ChunkContent(
            chunk_id=meta.chunk_id,
            chunk_index=meta.chunk_index,
            tier=meta.tier.value,
            frequency_score=meta.frequency_score,
            access_count=meta.access_count,
            content=content or "",
        ))

    return DocumentDetailResponse(
        document_id=doc.document_id,
        source_type=doc.source_type,
        source_uri=doc.source_uri,
        title=doc.title,
        content_hash=doc.content_hash,
        total_chunks=doc.total_chunks,
        created_at=doc.created_at.isoformat() if doc.created_at else "",
        updated_at=doc.updated_at.isoformat() if doc.updated_at else "",
        chunks=chunk_contents,
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str) -> dict:
    """Delete a document and all its chunks."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    import uuid as uuid_mod
    try:
        doc_uuid = uuid_mod.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    try:
        deleted = await _pipeline.delete_document(doc_uuid)
        return {"success": True, "deleted_chunks": deleted}
    except Exception as e:
        logger.error("delete_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
