"""Maintenance API routes — operational tooling for pipeline health.

Thin HTTP layer only.  All business logic lives in
:mod:`event_driven_rag_service.services.requeue_service`.
"""
from __future__ import annotations

import logging

import aio_pika
from fastapi import APIRouter, Request, status
from pydantic import BaseModel

from event_driven_rag_service.repository.maintenance_repository import MaintenanceRepository
from event_driven_rag_service.services.requeue_service import (
    RequeueService,
    RmqEmbedTaskPublisher,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/maintenance", tags=["maintenance"])


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class RequeueMissingEmbeddingsResponse(BaseModel):
    requeued_chunks: int
    tasks_published: int
    tables_scanned: int
    tables_skipped: int


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/requeue-missing-embeddings",
    response_model=RequeueMissingEmbeddingsResponse,
    status_code=status.HTTP_200_OK,
)
async def requeue_missing_embeddings(request: Request) -> RequeueMissingEmbeddingsResponse:
    """Scan all chunk tables and re-queue any chunks that have no embedding.

    Idempotent: only chunks with ``embedding IS NULL`` at query time are
    requeued.  Chunks that are already embedded are never touched.

    Safe to call while the GPU worker is active - the worker's text-hash
    deduplication prevents any observable side-effect from double-processing.
    """
    repo: MaintenanceRepository = request.app.state.maintenance_repo
    rmq: aio_pika.abc.AbstractRobustConnection = request.app.state.rmq

    service = RequeueService(
        reader=repo,
        publisher=RmqEmbedTaskPublisher(rmq),
    )
    result = await service.requeue_missing_embeddings()

    return RequeueMissingEmbeddingsResponse(
        requeued_chunks=result.requeued_chunks,
        tasks_published=result.tasks_published,
        tables_scanned=result.tables_scanned,
        tables_skipped=result.tables_skipped,
    )
