"""Chunk backfill service — recover posts missing summary_title chunks.

This module contains:
  - ``PostSummaryReader`` protocol — DB-side contract (list post tables, fetch
    posts with a summary)
  - ``ChunkTaskPublisher`` protocol — queue-side contract (publish one
    ChunkTask)
  - ``ChunkBackfillResult`` — typed result returned to the caller
  - ``ChunkBackfillService`` — orchestrates scanning + publishing

Companion to :mod:`event_driven_rag_service.services.requeue_service`, which
re-embeds chunks that already exist but lack an embedding. This service
instead recreates chunks that were never created in the first place — the
scenario that arises when a new embedding model is added to EMBED_CONFIGS for
``summary_title`` and every post's summary needs (re)chunking under it.

Republishing is safe to call repeatedly: ``ChunkPostHandler`` dedupes by
``text_hash`` before inserting, so a post whose summary_title chunk already
exists under the current model is a no-op here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import aio_pika

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.tasks.chunk_task import ChunkTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols (Dependency-Inversion interfaces)
# ---------------------------------------------------------------------------

@runtime_checkable
class PostSummaryReader(Protocol):
    """Read-only view of post table data needed for backfill decisions."""

    async def list_post_tables(self) -> list[str]:
        """Return names of all post tables (one per library) in the database."""
        ...

    async def fetch_posts_with_summary(self, table_name: str) -> list[int]:
        """Return ``post_id`` values for posts with a non-empty summary."""
        ...


@runtime_checkable
class ChunkTaskPublisher(Protocol):
    """Publish a single ChunkTask to the task queue."""

    async def publish(self, task: ChunkTask) -> None:
        """Publish *task* to the ingestion exchange's chunk routing key."""
        ...


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ChunkBackfillResult:
    """Summary of a requeue_missing_chunks run."""

    posts_scanned: int = 0
    tasks_published: int = 0
    tables_scanned: int = 0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ChunkBackfillService:
    """Scan all libraries for posts with a summary and (re)queue their
    summary_title chunk task.

    Args:
        reader:    Provides DB reads (list post tables, fetch posts with summary).
        publisher: Publishes ChunkTask messages to the broker.
    """

    def __init__(
        self,
        reader: PostSummaryReader,
        publisher: ChunkTaskPublisher,
    ) -> None:
        self._reader = reader
        self._publisher = publisher

    async def requeue_missing_chunks(self) -> ChunkBackfillResult:
        """Scan all post tables; requeue a summary_title ChunkTask per post
        that has a summary.

        Idempotent: ``ChunkPostHandler`` dedupes by ``text_hash``, so a post
        whose summary_title chunk already exists under the current model is
        re-chunked into an identical row and skipped, not duplicated.

        Returns:
            A :class:`ChunkBackfillResult` summarising what was done.
        """
        embed_model = EMBED_CONFIGS["summary_title"].primary
        post_tables = await self._reader.list_post_tables()

        result = ChunkBackfillResult(tables_scanned=len(post_tables))

        logger.info(
            "chunk_backfill_service: scanning %d post table(s)", len(post_tables)
        )

        for post_table in post_tables:
            post_ids = await self._reader.fetch_posts_with_summary(post_table)
            result.posts_scanned += len(post_ids)

            for post_id in post_ids:
                task = ChunkTask(
                    task_type="summary_title",
                    post_id=post_id,
                    post_table=post_table,
                    embed_model=embed_model,
                )
                await self._publisher.publish(task)
                result.tasks_published += 1

            logger.info(
                "chunk_backfill_service: table '%s' — requeued %d post(s) with a summary",
                post_table,
                len(post_ids),
            )

        logger.info(
            "chunk_backfill_service: done — %d task(s) published across %d post(s) "
            "in %d table(s)",
            result.tasks_published,
            result.posts_scanned,
            result.tables_scanned,
        )
        return result


# ---------------------------------------------------------------------------
# Concrete publisher (production wiring)
# ---------------------------------------------------------------------------

class RmqChunkTaskPublisher:
    """Concrete ChunkTaskPublisher backed by an aio_pika RabbitMQ connection.

    Each call opens a channel, resolves the ingestion exchange, publishes, and
    closes the channel. This keeps the publisher stateless and safe to
    construct per-request.
    """

    def __init__(self, connection: aio_pika.abc.AbstractRobustConnection) -> None:
        self._connection = connection

    async def publish(self, task: ChunkTask) -> None:
        from event_driven_rag_service.tasks.registry import TASK_ROUTES

        route = TASK_ROUTES["chunk"]
        channel = await self._connection.channel()
        try:
            exchange = await channel.declare_exchange(
                route.exchange, aio_pika.ExchangeType.TOPIC, durable=True
            )
            await exchange.publish(
                aio_pika.Message(task.model_dump_json().encode()),
                routing_key=route.routing_key,
            )
        finally:
            await channel.close()
