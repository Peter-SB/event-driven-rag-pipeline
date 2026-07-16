"""Requeue service — business logic for recovering un-embedded chunks.

This module contains:
  - ``ChunkTableReader`` protocol — DB-side contract (list tables, fetch nulls)
  - ``EmbedTaskPublisher`` protocol — queue-side contract (publish one task)
  - ``RequeueResult`` — typed result returned to the caller
  - ``RequeueService`` — orchestrates scanning + publishing; depends only on
    the two protocols above so it can be unit-tested without a DB or broker

Design (SOLID)
--------------
S — Single responsibility: this module only decides *which* chunks to requeue
    and *how* to group them.  It delegates DB reads to ChunkTableReader and
    queue writes to EmbedTaskPublisher.

O — Open/closed: new chunk types or models are handled automatically because
    the suffix map is built from EMBED_CONFIGS at call time — no code changes
    needed when a new model is added to the config.

L — Both protocols can be substituted by any compatible implementation; tests
    use in-memory fakes, production uses asyncpg + aio_pika.

I — Protocols are minimal (two methods each) — callers are not forced to
    depend on anything they don't use.

D — RequeueService depends on the Protocol abstractions, never on concrete
    classes (asyncpg.Pool, aio_pika.Connection, …).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.utils.build_table_names import (
    build_chunk_table_suffix_map,
    parse_chunk_table_name,
)

import aio_pika

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols (Dependency-Inversion interfaces)
# ---------------------------------------------------------------------------

@runtime_checkable
class ChunkTableReader(Protocol):
    """Read-only view of chunk table data needed for requeue decisions."""

    async def list_chunk_tables(self) -> list[str]:
        """Return names of all chunk tables in the database."""
        ...

    async def fetch_unembedded_chunks(
        self, table_name: str
    ) -> list[tuple[str, int]]:
        """Return ``(chunk_id, post_id)`` pairs for rows with ``embedding IS NULL``."""
        ...


@runtime_checkable
class EmbedTaskPublisher(Protocol):
    """Publish a single EmbedTask to the task queue."""

    async def publish(self, task: EmbedTask, routing_key: str) -> None:
        """Publish *task* to the given *routing_key* on the embedding exchange."""
        ...


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RequeueResult:
    """Summary of a requeue_missing_embeddings run."""

    requeued_chunks: int = 0
    tasks_published: int = 0
    tables_scanned: int = 0
    tables_skipped: int = 0
    skipped_table_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class RequeueService:
    """Scan all chunk tables and re-queue any chunks that have no embedding.

    Args:
        reader:    Provides DB reads (list tables, fetch unembedded chunks).
        publisher: Publishes EmbedTask messages to the broker.
    """

    def __init__(
        self,
        reader: ChunkTableReader,
        publisher: EmbedTaskPublisher,
    ) -> None:
        self._reader = reader
        self._publisher = publisher

    async def requeue_missing_embeddings(self) -> RequeueResult:
        """Scan all chunk tables; requeue un-embedded chunks as EmbedTasks.

        Idempotent: only chunks with ``embedding IS NULL`` at query time are
        requeued.  Calling this endpoint while the GPU worker is active is safe
        — the worker's text-hash deduplication means double-processing a chunk
        has no visible side-effect beyond a small amount of redundant work.

        Returns:
            A :class:`RequeueResult` summarising what was done.
        """
        suffix_map = build_chunk_table_suffix_map(EMBED_CONFIGS)
        chunk_tables = await self._reader.list_chunk_tables()

        result = RequeueResult(tables_scanned=len(chunk_tables))

        logger.info(
            "requeue_service: scanning %d chunk table(s)", len(chunk_tables)
        )

        for table_name in chunk_tables:
            parsed = parse_chunk_table_name(table_name, suffix_map)
            if parsed is None:
                logger.warning(
                    "requeue_service: skipping unrecognised chunk table '%s' "
                    "(not in EMBED_CONFIGS)",
                    table_name,
                )
                result.tables_skipped += 1
                result.skipped_table_names.append(table_name)
                continue

            post_table, _task_type, embed_cfg = parsed
            unembedded = await self._reader.fetch_unembedded_chunks(table_name)

            if not unembedded:
                logger.debug(
                    "requeue_service: table '%s' — no missing embeddings", table_name
                )
                continue

            tasks_published = await self._publish_tasks(
                post_table=post_table,
                chunk_table=table_name,
                embed_cfg=embed_cfg,
                unembedded=unembedded,
            )

            logger.info(
                "requeue_service: table '%s' — requeued %d chunk(s) in %d task(s) "
                "(model=%s, queue=%s)",
                table_name,
                len(unembedded),
                tasks_published,
                embed_cfg.model,
                embed_cfg.queue,
            )

            result.requeued_chunks += len(unembedded)
            result.tasks_published += tasks_published

        logger.info(
            "requeue_service: done — %d chunk(s) requeued, %d task(s) published, "
            "%d/%d table(s) skipped",
            result.requeued_chunks,
            result.tasks_published,
            result.tables_skipped,
            result.tables_scanned,
        )
        return result

    async def _publish_tasks(
        self,
        post_table: str,
        chunk_table: str,
        embed_cfg: object,
        unembedded: list[tuple[str, int]],
    ) -> int:
        """Group chunks by post_id and publish one EmbedTask per group.

        Grouping by post_id mirrors what the normal pipeline does
        (ChunkDispatcher emits one EmbedTask per chunks.created event, which is
        one per post per chunk type).

        Returns the number of tasks published.
        """
        by_post: dict[int, list[str]] = defaultdict(list)
        for chunk_id, post_id in unembedded:
            by_post[post_id].append(chunk_id)

        for post_id, chunk_ids in by_post.items():
            task = EmbedTask(
                task_type="chunk",
                model_name=embed_cfg.model,  # type: ignore[attr-defined]
                post_id=post_id,
                post_table=post_table,
                chunk_ids=chunk_ids,
                chunk_table=chunk_table,
            )
            await self._publisher.publish(task, routing_key=embed_cfg.queue)  # type: ignore[attr-defined]

        return len(by_post)


# ---------------------------------------------------------------------------
# Concrete publisher (production wiring)
# ---------------------------------------------------------------------------

class RmqEmbedTaskPublisher:
    """Concrete EmbedTaskPublisher backed by an aio_pika RabbitMQ connection.

    Each call opens a channel, resolves the embedding exchange, publishes, and
    closes the channel.  This keeps the publisher stateless and safe to
    construct per-request.
    """

    def __init__(self, connection: aio_pika.abc.AbstractRobustConnection) -> None:
        self._connection = connection

    async def publish(self, task: EmbedTask, routing_key: str) -> None:
        from event_driven_rag_service.tasks.registry import TASK_ROUTES

        route = TASK_ROUTES["embed"]
        channel = await self._connection.channel()
        try:
            exchange = await channel.declare_exchange(
                route.exchange, aio_pika.ExchangeType.TOPIC, durable=True
            )
            await exchange.publish(
                aio_pika.Message(task.model_dump_json().encode()),
                routing_key=routing_key,
            )
        finally:
            await channel.close()
