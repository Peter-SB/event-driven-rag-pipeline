"""Business logic for SearchRunTask processing.

SearchHandler is the single unit of work here.  It reads the job's stored
query embedding, executes a pgvector ANN search against the chunk table,
persists results, and fires ``search_job.completed``.

Separated from CpuSearchWorker so it can be unit-tested without RabbitMQ.

Repository protocols
--------------------
SearchJobStore and ChunkSearcher are Protocol interfaces so real Postgres
implementations and in-memory fakes are both accepted.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Protocol

from event_driven_rag_service.events.search_events import SearchJobCompletedEvent
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.tasks.search_tasks import SearchRunTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repository protocols
# ---------------------------------------------------------------------------

class SearchJobStore(Protocol):
    async def get_job(self, job_id: str) -> Dict[str, Any] | None: ...
    async def mark_searching(self, job_id: str) -> None: ...
    async def complete_job(self, job_id: str, results: List[Dict]) -> None: ...
    async def fail_job(self, job_id: str, error: str) -> None: ...


class ChunkSearcher(Protocol):
    async def search_nearest(
        self, table_name: str, query_vector: list[float], k: int
    ) -> list[Dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class SearchHandler:
    """Processes a SearchRunTask: load embedding, run ANN search, persist results.

    Parameters
    ----------
    job_store      : fetch job metadata and persist results
    chunk_searcher : execute pgvector ANN search against a chunk table
    event_log      : emit ``search_job.completed`` on success
    """

    def __init__(
        self,
        job_store: SearchJobStore,
        chunk_searcher: ChunkSearcher,
        event_log: EventBusBase,
    ) -> None:
        self._jobs = job_store
        self._chunks = chunk_searcher
        self._event_log = event_log

    async def handle(self, task: SearchRunTask) -> None:
        """Execute a search job and persist results.

        Raises on unexpected errors so CpuSearchWorker can nack to DLQ.
        """
        job = await self._jobs.get_job(task.job_id)
        if not job:
            logger.warning("SearchHandler: job %s not found — skipping", task.job_id)
            return

        embedding = job.get("embedding")
        if not embedding:
            msg = f"Job {task.job_id} has no embedding stored — cannot search"
            logger.error("SearchHandler: %s", msg)
            await self._jobs.fail_job(task.job_id, msg)
            return

        await self._jobs.mark_searching(task.job_id)

        try:
            raw = await self._chunks.search_nearest(
                job["chunks_table"], embedding, job["k"]
            )
            results = [
                {
                    "chunk_id": row["id"],
                    "post_id": row["post_id"],
                    "text": row["text"],
                    "metadata": row.get("metadata"),
                    "score": row["score"],
                }
                for row in raw
            ]
            await self._jobs.complete_job(task.job_id, results)

            event = SearchJobCompletedEvent(
                query_job_id=task.job_id,
                trace_id=task.trace_id,
            )
            await self._event_log.publish(event.event_type, event.to_dict())

            logger.info(
                "SearchHandler: job %s complete — %d results from %s",
                task.job_id,
                len(results),
                job["chunks_table"],
            )

        except Exception as exc:
            logger.exception("SearchHandler: job %s failed", task.job_id)
            await self._jobs.fail_job(task.job_id, str(exc))
            raise
