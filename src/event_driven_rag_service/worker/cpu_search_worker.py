"""CPU search worker — processes SearchRunTask messages from ``cpu.search.run``.

All business logic lives in SearchHandler.  This worker is a thin RabbitMQ shim:
  1. Receive message → deserialise to SearchRunTask
  2. Delegate to SearchHandler.handle()
  3. Ack on success, nack to DLQ on any exception (handled by BaseWorker)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from event_driven_rag_service.tasks.search_tasks import SearchRunTask
from event_driven_rag_service.worker.base_worker import BaseWorker
from event_driven_rag_service.handlers.search_handler import SearchHandler

logger = logging.getLogger(__name__)


class CpuSearchWorker(BaseWorker):
    """
    Parameters
    ----------
    rabbitmq_url : pika-compatible RabbitMQ URL
    handler      : SearchHandler with injected DB / event-bus dependencies
    loop         : asyncio event loop that owns the DB connection pool;
                   a new loop is created if not provided (useful in tests)
    prefetch     : messages pulled at a time (default 4 — I/O-bound work)
    """

    QUEUE = "cpu.search.run"

    def __init__(
        self,
        rabbitmq_url: str,
        handler: SearchHandler,
        loop: asyncio.AbstractEventLoop | None = None,
        prefetch: int = 4,
    ) -> None:
        super().__init__(rabbitmq_url, self.QUEUE, prefetch=prefetch)
        self._handler = handler
        self._loop = loop or asyncio.new_event_loop()

    def process(self, payload: dict[str, Any]) -> None:
        task = SearchRunTask.model_validate(payload)
        self._loop.run_until_complete(self._handler.handle(task))
