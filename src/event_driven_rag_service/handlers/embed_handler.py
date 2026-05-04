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
from typing import Any, Protocol

from event_driven_rag_service.events.embedding_events import EmbeddingCompletedEvent
from event_driven_rag_service.events.search_events import SearchQueryEmbeddedEvent
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.tasks.embed_task import EmbedTask

logger = logging.getLogger(__name__)


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
        self, chunk_ids: list[str], table: str
    ) -> list[tuple[str, str]]:
        """Return (chunk_id, text) pairs in the same order as chunk_ids."""
        ...


class EmbeddingStore(Protocol):
    async def save_batch(self, rows: list[dict[str, Any]]) -> None:
        """Persist embedding rows: chunk_id / query_job_id / model_name / embedding."""
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
        # Flatten to (task, chunk_id, text) triples across all tasks.
        triples: list[tuple[EmbedTask, str, str]] = []
        for task in tasks:
            if not task.chunk_ids or not task.chunk_table:
                continue
            pairs = await self._chunks.fetch_texts(task.chunk_ids, task.chunk_table)
            for chunk_id, text in pairs:
                triples.append((task, chunk_id, text))

        if not triples:
            return tasks, []

        texts = [t[2] for t in triples]
        try:
            vectors = encoder.encode(texts)
        except Exception:
            logger.exception("EmbedHandler: encode() failed — failing entire batch")
            return [], tasks

        rows = [
            {
                "chunk_id":    triples[i][1],
                "model_name":  model_name,
                "embedding":   vectors[i],
                "chunk_table": triples[i][0].chunk_table,
            }
            for i in range(len(triples))
        ]
        await self._embeddings.save_batch(rows)

        # Emit one embedding.completed per (post_id, chunk_table) group.
        groups: dict[tuple[int | None, str | None], dict] = {}
        for task, chunk_id, _ in triples:
            key = (task.post_id, task.chunk_table)
            if key not in groups:
                groups[key] = {
                    "post_id":     task.post_id,
                    "post_table":  task.post_table,
                    "chunk_table": task.chunk_table,
                    "model_name":  model_name,
                    "chunk_ids":   [],
                    "trace_id":    task.trace_id,
                }
            groups[key]["chunk_ids"].append(chunk_id)

        for data in groups.values():
            event = EmbeddingCompletedEvent(
                post_id=data["post_id"],
                post_table=data["post_table"],
                chunk_ids=data["chunk_ids"],
                chunk_table=data["chunk_table"],
                model_name=data["model_name"],
                trace_id=data["trace_id"],
            )
            await self._event_log.publish(event.event_type, event.to_dict())

        # Deduplicate: one task may have contributed multiple chunk_ids.
        seen: dict[int, EmbedTask] = {id(t[0]): t[0] for t in triples}
        return list(seen.values()), []

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
        if not task.query or not task.query_job_id:
            logger.warning(
                "EmbedHandler: query task missing query or job_id — skipping (task_id=%s)",
                task.task_id,
            )
            return True

        try:
            vector = encoder.encode([task.query])[0]
            await self._embeddings.save_batch(
                [
                    {
                        "query_job_id": task.query_job_id,
                        "model_name":   model_name,
                        "embedding":    vector,
                    }
                ]
            )
            event = SearchQueryEmbeddedEvent(
                query_job_id=task.query_job_id,
                model_name=model_name,
                trace_id=task.trace_id,
            )
            await self._event_log.publish(event.event_type, event.to_dict())
            return True
        except Exception:
            logger.exception(
                "EmbedHandler: query embed failed (job_id=%s)",
                task.query_job_id,
            )
            return False
