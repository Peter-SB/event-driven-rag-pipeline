"""
GPU embedding worker — processes EmbedTask messages from model-specific queues.

Synchronous worker using pika.  Keeps one embedding model warm at a time and
processes messages in explicit batches to maximise GPU throughput.

Design
------
1. ``model_queues`` is a priority-ordered list of (model_name, queue_name) pairs.
   The worker polls queues in order; after processing any batch it restarts from
   the first (highest-priority) queue.  This ensures latency-sensitive tasks are
   never starved by large batch workloads.

2. Model lifecycle:
   - A model is loaded when the first message for it arrives.
   - It stays warm as long as the same queue has work.
   - It is unloaded when a different model is needed, or after ``idle_timeout_s``
     seconds with no work in any queue.

3. Async bridge: chunk-text fetches, embedding saves, and event publishes all use
   asyncpg/EventBus which have async APIs.  A single asyncio event loop (passed in
   at construction, owning the DB pool) bridges this via ``loop.run_until_complete()``.

Run directly::

    python -m event_driven_rag_service.worker.entrypoints.gpu
"""
from __future__ import annotations

import asyncio
import gc
import logging
import time
from dataclasses import dataclass

from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.worker.base_worker import BaseWorker
from event_driven_rag_service.handlers.embed_handler import EmbedHandler, EmbeddingModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal message type
# ---------------------------------------------------------------------------

@dataclass
class WorkerMessage:
    """A parsed RabbitMQ message carrying a typed EmbedTask."""
    task: EmbedTask
    delivery_tag: int


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class GpuEmbedWorker(BaseWorker):
    """
    Long-lived GPU embedding worker.

    Parameters
    ----------
    rabbitmq_url    : pika-compatible URL
    model_queues    : priority-ordered list of (model_name, queue_name) pairs;
                      index 0 has the highest priority
    model_loader    : callable(model_name: str) -> EmbeddingModel  (blocking)
    handler         : EmbedHandler with injected DB / event-bus dependencies
    loop            : asyncio event loop that owns the DB pool;
                      a new loop is created if not provided
    max_batch       : max messages collected per queue poll
    idle_timeout_s  : seconds idle before the model is unloaded
    idle_sleep_s    : sleep duration between empty-queue polls
    """

    def __init__(
        self,
        rabbitmq_url: str,
        model_queues: list[tuple[str, str]],
        model_loader,
        handler: EmbedHandler,
        loop: asyncio.AbstractEventLoop | None = None,
        max_batch: int = 32,
        idle_timeout_s: float = 300.0,
        idle_sleep_s: float = 1.0,
    ) -> None:
        # Queue name is blank — this worker polls multiple queues manually
        # via _poll_queue; basic_consume is not used.
        super().__init__(rabbitmq_url, queue_name="", prefetch=max_batch)
        self._model_queues = model_queues
        self._load_model = model_loader
        self._handler = handler
        self._loop = loop or asyncio.new_event_loop()
        self._max_batch = max_batch
        self._idle_timeout_s = idle_timeout_s
        self._idle_sleep_s = idle_sleep_s

        self._model: EmbeddingModel | None = None
        self._current_model_name: str | None = None

    # ------------------------------------------------------------------
    # Channel setup — declare model-specific queues
    # ------------------------------------------------------------------

    def _open_channel(self):
        channel = super()._open_channel()
        for _, queue_name in self._model_queues:
            # Use passive=True: assert the queue exists without redeclaring it.
            # The full topology (including DLX args) is declared by setup_topology
            # in the API at startup. Redeclaring here with different args causes
            # PRECONDITION_FAILED from RabbitMQ.
            channel.queue_declare(queue=queue_name, passive=True)
        return channel

    # ------------------------------------------------------------------
    # Custom batch loop (replaces default basic_consume pattern)
    # ------------------------------------------------------------------

    def _run_inner(self, channel) -> None:
        logger.info(
            "GpuEmbedWorker consuming — queue priority: %s",
            [q for _, q in self._model_queues],
        )
        idle_since: float | None = None

        while self._running:
            found_work = False

            for model_name, queue_name in self._model_queues:
                if not self._running:
                    break

                raw = self._poll_queue(channel, queue_name, self._max_batch)
                if not raw:
                    continue

                batch = self._parse_batch(channel, raw)
                if not batch:
                    continue

                found_work = True
                idle_since = None

                self._ensure_model(model_name)
                ok, failed = self._process_batch(batch, model_name)

                for msg in ok:
                    self._ack(channel, msg.delivery_tag)
                for msg in failed:
                    self._nack(channel, msg.delivery_tag, requeue=False)

                # Restart priority scan from the top after each batch.
                break

            if not found_work:
                if idle_since is None:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since > self._idle_timeout_s:
                    self._unload_model()
                    idle_since = None
                time.sleep(self._idle_sleep_s)

    def _parse_batch(
        self,
        channel,
        raw: list[tuple[int, bytes]],
    ) -> list[WorkerMessage]:
        """Deserialize raw (delivery_tag, body) pairs; nack malformed messages."""
        batch: list[WorkerMessage] = []
        for delivery_tag, body in raw:
            try:
                task = EmbedTask.model_validate_json(body)
                batch.append(WorkerMessage(task=task, delivery_tag=delivery_tag))
            except Exception:
                logger.warning(
                    "GpuEmbedWorker: malformed embed message (tag=%s) — discarding",
                    delivery_tag,
                )
                self._nack(channel, delivery_tag, requeue=False)
        return batch

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _ensure_model(self, model_name: str) -> None:
        if self._current_model_name == model_name:
            return
        self._unload_model()
        logger.info("Loading embedding model: %s", model_name)
        self._model = self._load_model(model_name)
        self._current_model_name = model_name
        logger.info("Model ready: %s", model_name)

    def _unload_model(self) -> None:
        if self._model is None:
            return
        logger.info("Unloading model: %s", self._current_model_name)
        self._model = None
        self._current_model_name = None
        gc.collect()

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _process_batch(
        self,
        batch: list[WorkerMessage],
        model_name: str,
    ) -> tuple[list[WorkerMessage], list[WorkerMessage]]:
        assert self._model is not None, "_process_batch called without a loaded model"

        chunk_msgs = [m for m in batch if m.task.task_type == "chunk"]
        query_msgs = [m for m in batch if m.task.task_type == "query"]

        ok: list[WorkerMessage] = []
        failed: list[WorkerMessage] = []

        if chunk_msgs:
            chunk_tasks = [m.task for m in chunk_msgs]
            task_to_msg = {id(t): m for t, m in zip(chunk_tasks, chunk_msgs)}
            ok_tasks, fail_tasks = self._loop.run_until_complete(
                self._handler.embed_chunks(chunk_tasks, model_name, self._model)
            )
            ok.extend(task_to_msg[id(t)] for t in ok_tasks if id(t) in task_to_msg)
            failed.extend(task_to_msg[id(t)] for t in fail_tasks if id(t) in task_to_msg)

        for msg in query_msgs:
            success = self._loop.run_until_complete(
                self._handler.embed_query(msg.task, model_name, self._model)
            )
            (ok if success else failed).append(msg)

        return ok, failed


