"""
CPU chunk worker — processes ChunkTask messages from ``cpu.chunk.post``.

Synchronous single-message consumer.  One message is fully processed before
the next is fetched.  Spin up additional worker processes for concurrency.

All business logic (text resolution, chunking, idempotency, event publish)
lives in ChunkPostHandler.  This worker is a thin RabbitMQ shim:
  1. Receive message → deserialise to ChunkTask
  2. Delegate to ChunkPostHandler.handle()
  3. Ack on success, nack to DLQ on any exception (handled by BaseWorker)

Async bridge
------------
ChunkPostHandler uses async repository protocols (asyncpg).  Pass the same
asyncio event loop that owns the DB connection pool via ``loop``; the worker
bridges the sync/async boundary with ``loop.run_until_complete()``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from event_driven_rag_service.tasks.chunk_task import ChunkTask
from event_driven_rag_service.worker.base_worker import BaseWorker
from event_driven_rag_service.handlers.chunk_handler import ChunkPostHandler

logger = logging.getLogger(__name__)


class CpuChunkWorker(BaseWorker):
    """
    Parameters
    ----------
    rabbitmq_url : pika-compatible RabbitMQ URL
    handler      : ChunkPostHandler with injected DB / event-bus dependencies
    loop         : asyncio event loop that owns the DB connection pool;
                   a new loop is created if not provided (useful in tests)
    prefetch     : messages pulled at a time (default 4 — CPU work is I/O-bound)
    """

    QUEUE = "cpu.chunk.post"

    def __init__(
        self,
        rabbitmq_url: str,
        handler: ChunkPostHandler,
        loop: asyncio.AbstractEventLoop | None = None,
        prefetch: int = 4,
    ) -> None:
        super().__init__(rabbitmq_url, self.QUEUE, prefetch=prefetch)
        self._handler = handler
        self._loop = loop or asyncio.new_event_loop()

    def process(self, payload: dict[str, Any]) -> None:
        task = ChunkTask.model_validate(payload)
        self._loop.run_until_complete(self._handler.handle(task))

