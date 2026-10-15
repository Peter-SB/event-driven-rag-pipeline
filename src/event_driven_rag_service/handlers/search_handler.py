"""Business logic for SearchRunTask processing.

SearchHandler is the single unit of work here.  It reads the job's stored
query embedding, executes a pgvector ANN search against the chunk table for a
candidate pool, and always announces ``search_candidates.retrieved`` —
whether or not the job asked for reranking. That keeps the event log uniform
across both branches: SearchDispatcher decides from ``event.rerank`` whether
to route to SearchRankTask (GPU rerank) or SearchFinalizeTask (CPU truncate),
but SearchHandler itself doesn't special-case either path.

Separated from CpuSearchWorker so it can be unit-tested without RabbitMQ.

Repository protocols
--------------------
SearchJobStore and ChunkSearcher are Protocol interfaces so real Postgres
implementations and in-memory fakes are both accepted.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Protocol

from opentelemetry import trace

from event_driven_rag_service.events.search_events import (
    CandidateChunkRef,
    SearchCandidatesRetrievedEvent,
)
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.infrastructure.metrics import (
    record_search_job_completed,
    record_search_latency,
    record_failure,
)
from event_driven_rag_service.tasks.search_tasks import SearchRunTask
from event_driven_rag_service.utils.tracing_utils import extract_trace_context

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
    event_log      : emit ``search_candidates.retrieved`` on success
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
        parent_ctx = extract_trace_context(task.trace_id, task.parent_span_id)
        tracer = trace.get_tracer(__name__)
        start_time = time.time()

        with tracer.start_as_current_span("search_run", context=parent_ctx) as span:
            span.set_attribute("job_id", task.job_id)

            job = await self._jobs.get_job(task.job_id)
            if not job:
                logger.warning("SearchHandler: job %s not found — skipping", task.job_id)
                span.set_attribute("skipped_reason", "job_not_found")
                return

            embedding = job.get("embedding")
            if not embedding:
                msg = f"Job {task.job_id} has no embedding stored — cannot search"
                logger.error("SearchHandler: %s", msg)
                await self._jobs.fail_job(task.job_id, msg)
                record_search_job_completed("failed")
                record_failure("storage_failed", "cpu-worker")
                span.set_attribute("skipped_reason", "no_embedding")
                return

            await self._jobs.mark_searching(task.job_id)

            k = job["k"]
            rerank = bool(job.get("rerank", False))
            rerank_profile = job.get("rerank_profile")
            # Overfetch a candidate pool for rerank to have something to
            # re-order — a fixed 3x multiplier, no floor/ceiling. When
            # rerank is off this still returns exactly k (harmless no-op
            # multiply), so the ANN query itself doesn't need to branch.
            candidate_pool = k * 3 if rerank else k

            try:
                raw = await self._chunks.search_nearest(
                    job["chunks_table"], embedding, candidate_pool
                )
                candidates = [
                    CandidateChunkRef(chunk_id=row["id"], score=row["score"])
                    for row in raw
                ]

                event = SearchCandidatesRetrievedEvent(
                    query_job_id=task.job_id,
                    candidates=candidates,
                    k=k,
                    rerank=rerank,
                    rerank_profile=rerank_profile,
                    trace_id=task.trace_id,
                )
                await self._event_log.publish(event.event_type, event.to_dict())

                span.set_attribute("candidate_count", len(candidates))
                span.set_attribute("rerank", rerank)
                record_search_latency(time.time() - start_time)

                logger.info(
                    "SearchHandler: job %s — %d candidates from %s (rerank=%s)",
                    task.job_id,
                    len(candidates),
                    job["chunks_table"],
                    rerank,
                )

            except Exception as exc:
                logger.exception("SearchHandler: job %s failed", task.job_id)
                await self._jobs.fail_job(task.job_id, str(exc))
                record_search_job_completed("failed")
                record_failure("encode_failed", "cpu-worker")
                raise
