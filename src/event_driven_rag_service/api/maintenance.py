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
from event_driven_rag_service.services.chunk_backfill_service import (
    ChunkBackfillService,
    RmqChunkTaskPublisher,
)
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


class RequeueMissingChunksResponse(BaseModel):
    posts_scanned: int
    chunk_tasks_published: int
    post_tables_scanned: int
    requeued_chunks: int
    embed_tasks_published: int
    chunk_tables_scanned: int
    chunk_tables_skipped: int


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


@router.post(
    "/requeue-missing-chunks",
    response_model=RequeueMissingChunksResponse,
    status_code=status.HTTP_200_OK,
)
async def requeue_missing_chunks(request: Request) -> RequeueMissingChunksResponse:
    """Backfill both missing chunks and missing embeddings for post summaries.

    Two steps, run together because bringing a new ``summary_title`` embedding
    model online needs both:

    1. Publish a summary_title ChunkTask for every post that has a summary,
       across every library's post table. ``ChunkPostHandler`` dedupes by
       ``text_hash``, so a post whose chunk already exists under the current
       model is skipped, not duplicated. Chunking happens asynchronously on
       the CPU worker; each newly-created chunk auto-dispatches its own
       EmbedTask via the normal chunks.created -> ChunkDispatcher path, so
       most new chunks get embedded without any further action here.
    2. Scan chunk tables for rows that are still missing an embedding and
       re-publish those directly. This catches chunks that already existed
       under this model but never got embedded (e.g. the GPU worker was down
       when the original EmbedTask was published) - a gap step 1 alone can't
       close, since it only creates chunks that don't yet exist.

    Safe to call repeatedly or while workers are active.
    """
    repo: MaintenanceRepository = request.app.state.maintenance_repo
    rmq: aio_pika.abc.AbstractRobustConnection = request.app.state.rmq

    chunk_service = ChunkBackfillService(
        reader=repo,
        publisher=RmqChunkTaskPublisher(rmq),
    )
    chunk_result = await chunk_service.requeue_missing_chunks()

    embed_service = RequeueService(
        reader=repo,
        publisher=RmqEmbedTaskPublisher(rmq),
    )
    embed_result = await embed_service.requeue_missing_embeddings()

    return RequeueMissingChunksResponse(
        posts_scanned=chunk_result.posts_scanned,
        chunk_tasks_published=chunk_result.tasks_published,
        post_tables_scanned=chunk_result.tables_scanned,
        requeued_chunks=embed_result.requeued_chunks,
        embed_tasks_published=embed_result.tasks_published,
        chunk_tables_scanned=embed_result.tables_scanned,
        chunk_tables_skipped=embed_result.tables_skipped,
    )
