"""
Business logic for EmbedTask processing.

Separated from GpuEmbedWorker so it can be unit-tested without RabbitMQ or a
real GPU — inject fakes for ChunkFetcher, EmbeddingStore, and EmbeddingModel.

Two task types
--------------
chunk — fetch chunk texts from DB, encode in batch, persist vectors,
         emit ``embedding.completed`` grouped by (post_id, chunk_table)
query — encode the query string carried inline, persist the vector,
         emit ``search_query.embedded``
"""
from __future__ import annotations

import logging
import time
from typing import Protocol, TypedDict

from opentelemetry import trace

from event_driven_rag_service.events.embedding_events import EmbeddingCompletedEvent
from event_driven_rag_service.events.search_events import SearchQueryEmbeddedEvent
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.infrastructure.metrics import (
    record_embeddings_generated,
    record_embedding_latency,
    record_encode_latency,
    record_failure,
)
from event_driven_rag_service.exceptions import ChunkTableNotFoundError
from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.utils.tracing_utils import extract_trace_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed row models for embedding persistence
# ---------------------------------------------------------------------------

class ChunkEmbeddingRow(TypedDict):
    """Row passed to EmbeddingStore.save_batch for a chunk embedding."""
    chunk_id: str
    model_name: str
    embedding: list[float]
    chunk_table: str


class QueryEmbeddingRow(TypedDict):
    """Row passed to EmbeddingStore.save_batch for a query embedding."""
    query_job_id: str
    model_name: str
    embedding: list[float]


EmbeddingRow = ChunkEmbeddingRow | QueryEmbeddingRow


# ---------------------------------------------------------------------------
# Protocols — inject real implementations at startup, fakes in tests
# ---------------------------------------------------------------------------

class EmbeddingModel(Protocol):
    @property
    def name(self) -> str: ...

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text (blocking compute)."""
        ...


class ChunkFetcher(Protocol):
    async def fetch_texts(
        self, chunk_ids: list[str], table_name: str
    ) -> list[tuple[str, str]]:
        """Return (chunk_id, text) pairs in the same order as chunk_ids."""
        ...


class EmbeddingStore(Protocol):
    async def save_batch(self, rows: list[EmbeddingRow]) -> None:
        """Persist embedding rows: ChunkEmbeddingRow or QueryEmbeddingRow."""
        ...


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class EmbedHandler:
    """
    Handles EmbedTask processing: fetch texts, encode, persist, emit events.

    Parameters
    ----------
    chunk_fetcher   : fetches chunk texts from DB (async)
    embedding_store : persists computed embedding vectors (async)
    event_log       : emits embedding completion events (async)
    """

    def __init__(
        self,
        chunk_fetcher: ChunkFetcher,
        embedding_store: EmbeddingStore,
        event_log: EventBusBase,
    ) -> None:
        self._chunks = chunk_fetcher
        self._embeddings = embedding_store
        self._event_log = event_log

    async def embed_chunks(
        self,
        tasks: list[EmbedTask],
        model_name: str,
        encoder: EmbeddingModel,
    ) -> tuple[list[EmbedTask], list[EmbedTask]]:
        """Embed a batch of chunk tasks.

        Fetches texts from DB, computes vectors, persists, and emits
        ``embedding.completed`` grouped by (post_id, chunk_table).

        Returns (ok_tasks, failed_tasks).
        """
        total_chunks = sum(len(t.chunk_ids or []) for t in tasks)
        logger.info(
            "EmbedHandler.embed_chunks: starting batch (tasks=%d total_chunks=%d model=%s)",
            len(tasks),
            total_chunks,
            model_name,
        )
        start_time = time.time()

        # Use the first task's trace context as the parent span.
        # All tasks in a batch share the same trace_id (they came from the same event).
        first = tasks[0] if tasks else None
        parent_ctx = extract_trace_context(
            first.trace_id if first else None,
            first.parent_span_id if first else None,
        )
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("embed_chunks", context=parent_ctx) as span:
            span.set_attribute("model", model_name)
            span.set_attribute("task_count", len(tasks))

            ok_tasks, failed_tasks = await self._embed_chunks_inner(tasks, model_name, encoder, span)

            # Record metrics
            embeddings_count = sum(len(t.chunk_ids or []) for t in ok_tasks)
            if embeddings_count > 0:
                record_embeddings_generated(embeddings_count, model_name)

            if failed_tasks:
                record_failure("encode_failed", "gpu-worker")

            latency_seconds = time.time() - start_time
            record_embedding_latency(latency_seconds, model_name)

            return ok_tasks, failed_tasks

    async def _embed_chunks_inner(
        self,
        tasks: list[EmbedTask],
        model_name: str,
        encoder: EmbeddingModel,
        span,
    ) -> tuple[list[EmbedTask], list[EmbedTask]]:
        # Flatten to (task, chunk_id, text) triples across all tasks.
        triples: list[tuple[EmbedTask, str, str]] = []
        fetch_failed: list[EmbedTask] = []
        for task in tasks:
            if not task.chunk_ids or not task.chunk_table:
                continue
            try:
                pairs = await self._chunks.fetch_texts(task.chunk_ids, task.chunk_table)
            except ChunkTableNotFoundError:
                logger.warning(
                    "EmbedHandler: chunk table %r does not exist — skipping task (task_id=%s). "
                    "The CPU worker likely failed before creating the table.",
                    task.chunk_table,
                    task.task_id,
                )
                continue
            except Exception:
                logger.exception(
                    "EmbedHandler: fetch_texts failed (chunk_table=%s, task_id=%s) — failing task",
                    task.chunk_table,
                    task.task_id,
                )
                fetch_failed.append(task)
                continue
            for chunk_id, text in pairs:
                triples.append((task, chunk_id, text))

        if not triples:
            logger.info(
                "EmbedHandler: no chunks to encode after fetch (fetch_failed=%d)",
                len(fetch_failed),
            )
            return [t for t in tasks if t not in fetch_failed], fetch_failed

        texts = [t[2] for t in triples]
        logger.info("EmbedHandler: encoding %d texts with %s", len(texts), encoder.name)
        encode_start = time.time()
        try:
            vectors = encoder.encode(texts)
        except Exception:
            logger.exception("EmbedHandler: encode() failed — failing entire batch")
            return [], tasks
        encode_time = time.time() - encode_start
        record_encode_latency(encode_time, model_name, len(texts))
        logger.info(
            "EmbedHandler: encoding complete (texts=%d vectors=%d latency=%.2fs throughput=%.0f texts/s per_text=%.3fs)",
            len(texts),
            len(vectors),
            encode_time,
            len(texts) / encode_time if encode_time > 0 else 0,
            encode_time / len(texts) if len(texts) > 0 else 0,
        )

        rows: list[EmbeddingRow] = []
        for i in range(len(triples)):
            task = triples[i][0]
            if task.chunk_table:
                rows.append(
                    ChunkEmbeddingRow(
                        chunk_id=triples[i][1],
                        model_name=model_name,
                        embedding=vectors[i],
                        chunk_table=task.chunk_table,
                    )
                )
            else:
                logger.warning(
                    "EmbedHandler: skipping chunk embedding without chunk_table (chunk_id=%s, post_id=%s)",
                    triples[i][1],
                    task.post_id,
                )

        logger.info(
            "EmbedHandler: persisting %d embedding rows to store",
            len(rows),
        )
        await self._embeddings.save_batch(rows)
        logger.info(
            "EmbedHandler: persistence complete (rows=%d)",
            len(rows),
        )

        # Emit one embedding.completed per (post_id, chunk_table) group.
        # Falls back to the first task's trace_id when OTEL is disabled.
        from event_driven_rag_service.utils.tracing_utils import propagate_trace
        first_task_trace = triples[0][0].trace_id if triples else None
        trace_id, parent_span_id = propagate_trace(first_task_trace)

        GroupKey = tuple[int | None, str | None]
        group_chunks: dict[GroupKey, list[str]] = {}
        group_meta: dict[GroupKey, tuple[str | None, str | None]] = {}
        for task, chunk_id, _ in triples:
            key: GroupKey = (task.post_id, task.chunk_table)
            if key not in group_chunks:
                group_chunks[key] = []
                group_meta[key] = (task.post_table, task.chunk_table)
            group_chunks[key].append(chunk_id)

        logger.info(
            "EmbedHandler: grouped embeddings into %d event(s) by (post_id, chunk_table)",
            len(group_chunks),
        )

        for key, chunk_ids in group_chunks.items():
            post_table, chunk_table = group_meta[key]
            event = EmbeddingCompletedEvent(
                post_id=key[0],
                post_table=post_table,
                chunk_ids=chunk_ids,
                chunk_table=chunk_table,
                model_name=model_name,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
            )
            await self._event_log.publish(event.event_type, event.to_dict())
            logger.info(
                "embedding.completed chunks=%d model=%s post_id=%s table=%s",
                len(chunk_ids),
                model_name,
                key[0],
                chunk_table,
            )

        # Deduplicate: one task may have contributed multiple chunk_ids.
        seen: dict[int, EmbedTask] = {id(t[0]): t[0] for t in triples}
        return list(seen.values()), fetch_failed

    async def embed_query(
        self,
        task: EmbedTask,
        model_name: str,
        encoder: EmbeddingModel,
    ) -> bool:
        """Embed a single search query and emit ``search_query.embedded``.

        Returns True on success, False on failure.
        Missing query/job_id is treated as ok (logged and skipped) to avoid
        poisoning the DLQ with bad data.
        """
        start_time = time.time()

        from event_driven_rag_service.utils.tracing_utils import current_trace_ids
        parent_ctx = extract_trace_context(task.trace_id, task.parent_span_id)
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("embed_query", context=parent_ctx) as span:
            span.set_attribute("model", model_name)
            if task.query_job_id:
                span.set_attribute("query_job_id", task.query_job_id)

            if not task.query or not task.query_job_id:
                logger.warning(
                    "EmbedHandler: query task missing query or job_id — skipping (task_id=%s)",
                    task.task_id,
                )
                return True

            try:
                encode_start = time.time()
                vector = encoder.encode([task.query])[0]
                encode_time = time.time() - encode_start
                record_encode_latency(encode_time, model_name, 1)

                await self._embeddings.save_batch(
                    [
                        QueryEmbeddingRow(
                            query_job_id=task.query_job_id,
                            model_name=model_name,
                            embedding=vector,
                        )
                    ]
                )

                from event_driven_rag_service.utils.tracing_utils import propagate_trace
                trace_id, parent_span_id = propagate_trace(task.trace_id)
                event = SearchQueryEmbeddedEvent(
                    query_job_id=task.query_job_id,
                    model_name=model_name,
                    trace_id=trace_id,
                    parent_span_id=parent_span_id,
                )
                await self._event_log.publish(event.event_type, event.to_dict())

                # Record success
                record_embeddings_generated(1, model_name)
                latency_seconds = time.time() - start_time
                record_embedding_latency(latency_seconds, model_name)

                return True
            except Exception:
                logger.exception(
                    "EmbedHandler: query embed failed (job_id=%s)",
                    task.query_job_id,
                )
                record_failure("encode_failed", "gpu-worker")
                return False
