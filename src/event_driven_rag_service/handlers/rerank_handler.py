"""Business logic for SearchRankTask processing.

Dispatched when a search job asked for reranking. Hydrates the ANN candidate
pool into full text, scores it with a reranker model (GPU, warm-loaded by
GpuEmbedWorker — see worker/gpu_worker.py and worker/entrypoints/gpu.py),
re-sorts by the reranker's score, truncates to k, and completes the job.

Mirrors EmbedHandler.embed_query's shape deliberately: the model is loaded
and kept warm by the worker, not by this handler, so `handle()` takes the
model as a parameter rather than owning it. Returns bool (True=ack,
False=nack) instead of raising, for the same reason EmbedHandler does —
GpuEmbedWorker acks/nacks a whole polled batch based on per-message success,
and a single malformed job shouldn't take the batch's other messages down
with it.
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
from event_driven_rag_service.tasks.search_tasks import SearchRankTask
from event_driven_rag_service.utils.tracing_utils import extract_trace_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols — inject real implementations at startup, fakes in tests
# ---------------------------------------------------------------------------

class RerankerModel(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document, same order as `documents`.
        Higher = more relevant. Not necessarily bounded to [0, 1] — callers
        only rely on relative order, never on absolute magnitude."""
        ...


class SearchJobStore(Protocol):
    async def get_job(self, job_id: str) -> Dict[str, Any] | None: ...
    async def mark_reranking(self, job_id: str) -> None: ...
    async def complete_job(self, job_id: str, results: List[Dict]) -> None: ...
    async def fail_job(self, job_id: str, error: str) -> None: ...


class ChunkHydrator(Protocol):
    async def get_chunks_by_ids(
        self, chunk_ids: list[str], table_name: str
    ) -> list[Dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class RerankHandler:
    """
    Handles SearchRankTask processing: hydrate candidates, score, persist.

    Parameters
    ----------
    job_store : fetch job metadata and persist results
    chunks    : hydrate candidate chunk_ids into full rows
    event_log : emits ``search_job.completed`` on success
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

    async def handle(
        self,
        task: SearchRankTask,
        model_name: str,
        reranker: RerankerModel,
    ) -> bool:
        """Rerank a candidate pool and complete the job.

        Returns True on success (including the "job vanished" / "no
        candidates survived hydration" no-op cases — nothing to retry),
        False on failure so the worker nacks the message.
        """
        start_time = time.time()
        parent_ctx = extract_trace_context(task.trace_id, task.parent_span_id)
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("search_rerank", context=parent_ctx) as span:
            span.set_attribute("job_id", task.job_id)
            span.set_attribute("model", model_name)
            span.set_attribute("candidate_count", len(task.candidates))

            job = await self._jobs.get_job(task.job_id)
            if not job:
                logger.warning("RerankHandler: job %s not found — skipping", task.job_id)
                span.set_attribute("skipped_reason", "job_not_found")
                return True

            await self._jobs.mark_reranking(task.job_id)

            try:
                score_by_id = {c.chunk_id: c.score for c in task.candidates}
                rows = await self._chunks.get_chunks_by_ids(
                    list(score_by_id.keys()), job["chunks_table"]
                )
                rows_by_id = {r["id"]: r for r in rows}
                # Preserve ANN order for anything hydration couldn't find
                # (deleted between search and rerank) — just drop it.
                ordered = [c for c in task.candidates if c.chunk_id in rows_by_id]

                if not ordered:
                    await self._jobs.complete_job(task.job_id, [])
                    await self._publish_completed(task)
                    span.set_attribute("result_count", 0)
                    record_search_job_completed("completed")
                    logger.info(
                        "RerankHandler: job %s complete — 0 results (no candidates survived hydration)",
                        task.job_id,
                    )
                    return True

                texts = [rows_by_id[c.chunk_id]["text"] for c in ordered]

                score_start = time.time()
                scores = reranker.score(job["query"], texts)
                score_time = time.time() - score_start
                logger.info(
                    "RerankHandler: scored %d candidates with %s in %.2fs",
                    len(texts), model_name, score_time,
                )

                ranked = sorted(zip(ordered, scores), key=lambda pair: pair[1], reverse=True)
                top = ranked[: job["k"]]

                results = [
                    {
                        "chunk_id": candidate.chunk_id,
                        "post_id": rows_by_id[candidate.chunk_id]["post_id"],
                        "text": rows_by_id[candidate.chunk_id]["text"],
                        "metadata": rows_by_id[candidate.chunk_id].get("metadata"),
                        "score": float(score),
                    }
                    for candidate, score in top
                ]
                await self._jobs.complete_job(task.job_id, results)
                await self._publish_completed(task)

                span.set_attribute("result_count", len(results))
                record_search_job_completed("completed")

                logger.info(
                    "RerankHandler: job %s complete — %d results (elapsed=%.3fs)",
                    task.job_id, len(results), time.time() - start_time,
                )
                return True

            except Exception as exc:
                logger.exception("RerankHandler: job %s failed", task.job_id)
                await self._jobs.fail_job(task.job_id, str(exc))
                record_search_job_completed("failed")
                record_failure("encode_failed", "gpu-worker")
                return False

    async def _publish_completed(self, task: SearchRankTask) -> None:
        event = SearchJobCompletedEvent(query_job_id=task.job_id, trace_id=task.trace_id)
        await self._event_log.publish(event.event_type, event.to_dict())
