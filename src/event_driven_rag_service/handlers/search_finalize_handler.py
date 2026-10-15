"""Business logic for SearchFinalizeTask processing.

Dispatched when a search job did *not* ask for reranking. No model involved —
just truncate the ANN candidate pool to k, hydrate full chunk rows, and
complete the job. Exists as a real task/handler (not an inline branch inside
SearchHandler) so both the rerank and no-rerank paths look the same in the
event log and both go through a dispatched task.

Separated from CpuSearchFinalizeWorker so it can be unit-tested without
RabbitMQ.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Protocol

from opentelemetry import trace

from event_driven_rag_service.events.search_events import SearchJobCompletedEvent
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.infrastructure.metrics import (
    record_search_job_completed,
    record_failure,
)
from event_driven_rag_service.tasks.search_tasks import SearchFinalizeTask
from event_driven_rag_service.utils.tracing_utils import extract_trace_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repository protocols
# ---------------------------------------------------------------------------

class SearchJobStore(Protocol):
    async def get_job(self, job_id: str) -> Dict[str, Any] | None: ...
    async def complete_job(self, job_id: str, results: List[Dict]) -> None: ...
    async def fail_job(self, job_id: str, error: str) -> None: ...


class ChunkHydrator(Protocol):
    async def get_chunks_by_ids(
        self, chunk_ids: list[str], table_name: str
    ) -> list[Dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class SearchFinalizeHandler:
    """Truncates the ANN candidate pool to k and completes the search job.

    Parameters
    ----------
    job_store : fetch job metadata and persist results
    chunks    : hydrate candidate chunk_ids into full rows
    event_log : emit ``search_job.completed`` on success
    """

    def __init__(
        self,
        job_store: SearchJobStore,
        chunks: ChunkHydrator,
        event_log: EventBusBase,
    ) -> None:
        self._jobs = job_store
        self._chunks = chunks
        self._event_log = event_log

    async def handle(self, task: SearchFinalizeTask) -> None:
        """Complete a non-reranked search job.

        Raises on unexpected errors so the CPU worker can nack to DLQ.
        """
        parent_ctx = extract_trace_context(task.trace_id, task.parent_span_id)
        tracer = trace.get_tracer(__name__)
        start_time = time.time()

        with tracer.start_as_current_span("search_finalize", context=parent_ctx) as span:
            span.set_attribute("job_id", task.job_id)

            job = await self._jobs.get_job(task.job_id)
            if not job:
                logger.warning("SearchFinalizeHandler: job %s not found — skipping", task.job_id)
                span.set_attribute("skipped_reason", "job_not_found")
                return

            try:
                top_candidates = task.candidates[: job["k"]]
                score_by_id = {c.chunk_id: c.score for c in top_candidates}
                rows = await self._chunks.get_chunks_by_ids(
                    list(score_by_id.keys()), job["chunks_table"]
                )
                rows_by_id = {r["id"]: r for r in rows}

                results = [
                    {
                        "chunk_id": c.chunk_id,
                        "post_id": rows_by_id[c.chunk_id]["post_id"],
                        "text": rows_by_id[c.chunk_id]["text"],
                        "metadata": rows_by_id[c.chunk_id].get("metadata"),
                        "score": c.score,
                    }
                    for c in top_candidates
                    if c.chunk_id in rows_by_id
                ]
                await self._jobs.complete_job(task.job_id, results)

                event = SearchJobCompletedEvent(
                    query_job_id=task.job_id,
                    trace_id=task.trace_id,
                )
                await self._event_log.publish(event.event_type, event.to_dict())

                span.set_attribute("result_count", len(results))
                record_search_job_completed("completed")

                logger.info(
                    "SearchFinalizeHandler: job %s complete — %d results (no rerank, elapsed=%.3fs)",
                    task.job_id,
                    len(results),
                    time.time() - start_time,
                )

            except Exception as exc:
                logger.exception("SearchFinalizeHandler: job %s failed", task.job_id)
                await self._jobs.fail_job(task.job_id, str(exc))
                record_search_job_completed("failed")
                record_failure("storage_failed", "cpu-worker")
                raise
