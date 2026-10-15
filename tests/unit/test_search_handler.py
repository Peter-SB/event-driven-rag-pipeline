"""Unit tests for SearchHandler.

Uses in-memory fakes for all dependencies — no Postgres, no RabbitMQ.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from event_driven_rag_service.handlers.search_handler import SearchHandler
from event_driven_rag_service.tasks.search_tasks import SearchRunTask
from tests.utils.factories import FakeEventBus


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSearchJobStore:
    def __init__(self, job: Optional[Dict[str, Any]] = None) -> None:
        self._job = job
        self.marked_searching: list[str] = []
        self.completed: list[tuple[str, list]] = []
        self.failed: list[tuple[str, str]] = []

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._job

    async def mark_searching(self, job_id: str) -> None:
        self.marked_searching.append(job_id)

    async def complete_job(self, job_id: str, results: List[Dict]) -> None:
        self.completed.append((job_id, results))

    async def fail_job(self, job_id: str, error: str) -> None:
        self.failed.append((job_id, error))


class FakeChunkSearcher:
    def __init__(self, results: list[Dict[str, Any]] | None = None) -> None:
        self._results = results or []
        self.calls: list[tuple[str, list, int]] = []

    async def search_nearest(
        self, table_name: str, query_vector: list[float], k: int
    ) -> list[Dict[str, Any]]:
        self.calls.append((table_name, query_vector, k))
        return self._results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(job_id: str = "job-1") -> SearchRunTask:
    return SearchRunTask(job_id=job_id)


def _make_job(
    job_id: str = "job-1",
    embedding: list[float] | None = None,
    k: int = 5,
    chunks_table: str = "posts_main_chunks_body_baai_bge_base_en_v1_5",
) -> Dict[str, Any]:
    return {
        "id": job_id,
        "status": "embedding",
        "query": "test query",
        "k": k,
        "embedding_profile": "BAAI/bge-base-en-v1.5",
        "chunks_table": chunks_table,
        "embedding": embedding or [0.1] * 768,
        "results": None,
        "error": None,
    }


def _make_search_result(chunk_id: str = "c1", score: float = 0.9) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "post_id": 1,
        "text": "sample chunk text",
        "metadata": {"title": "Post"},
        "score": score,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_success_emits_candidates_retrieved_event():
    """SearchHandler should mark job as searching and announce candidates —
    never completes the job itself (that's SearchFinalizeHandler/RerankHandler's job)."""
    job = _make_job()
    fake_results = [_make_search_result("c1", 0.9), _make_search_result("c2", 0.7)]

    job_store = FakeSearchJobStore(job=job)
    searcher = FakeChunkSearcher(results=fake_results)
    bus = FakeEventBus()

    handler = SearchHandler(job_store=job_store, chunk_searcher=searcher, event_log=bus)
    await handler.handle(_make_task("job-1"))

    assert job_store.marked_searching == ["job-1"]
    assert job_store.completed == []

    events = bus.drain_topic("search_candidates.retrieved")
    assert len(events) == 1
    event = events[0]
    assert event["query_job_id"] == "job-1"
    assert event["event_type"] == "search_candidates.retrieved"
    assert event["rerank"] is False
    assert event["k"] == job["k"]
    candidates = event["candidates"]
    assert len(candidates) == 2
    assert candidates[0]["chunk_id"] == "c1"
    assert candidates[0]["score"] == 0.9
    assert candidates[1]["chunk_id"] == "c2"


@pytest.mark.asyncio
async def test_handle_rerank_true_overfetches_candidate_pool():
    """When the job asked for rerank, SearchHandler should overfetch k*3
    candidates (no floor/ceiling) instead of exactly k, and carry rerank=True
    through the emitted event."""
    job = _make_job(k=5)
    job["rerank"] = True
    job["rerank_profile"] = "qwen3-reranker-0.6b"

    job_store = FakeSearchJobStore(job=job)
    searcher = FakeChunkSearcher(results=[_make_search_result("c1", 0.9)])
    bus = FakeEventBus()

    handler = SearchHandler(job_store=job_store, chunk_searcher=searcher, event_log=bus)
    await handler.handle(_make_task("job-1"))

    assert len(searcher.calls) == 1
    _, _, result_k = searcher.calls[0]
    assert result_k == 15  # k * 3, no floor/ceiling

    events = bus.drain_topic("search_candidates.retrieved")
    assert events[0]["rerank"] is True
    assert events[0]["rerank_profile"] == "qwen3-reranker-0.6b"
    assert events[0]["k"] == 5


@pytest.mark.asyncio
async def test_handle_job_not_found_returns_without_error():
    """SearchHandler should skip silently when job_id does not exist."""
    job_store = FakeSearchJobStore(job=None)
    searcher = FakeChunkSearcher()
    bus = FakeEventBus()

    handler = SearchHandler(job_store=job_store, chunk_searcher=searcher, event_log=bus)
    await handler.handle(_make_task("missing-job"))

    assert job_store.marked_searching == []
    assert job_store.completed == []
    assert job_store.failed == []
    assert searcher.calls == []


@pytest.mark.asyncio
async def test_handle_no_embedding_marks_job_failed():
    """SearchHandler should fail the job when embedding is missing."""
    job = _make_job(embedding=None)
    job["embedding"] = None

    job_store = FakeSearchJobStore(job=job)
    searcher = FakeChunkSearcher()
    bus = FakeEventBus()

    handler = SearchHandler(job_store=job_store, chunk_searcher=searcher, event_log=bus)
    await handler.handle(_make_task("job-1"))

    assert len(job_store.failed) == 1
    failed_id, error = job_store.failed[0]
    assert failed_id == "job-1"
    assert "embedding" in error.lower()
    assert job_store.completed == []
    assert searcher.calls == []


@pytest.mark.asyncio
async def test_handle_search_passes_correct_params_to_searcher():
    """SearchHandler should pass chunk_table, embedding, and k to the searcher."""
    k = 7
    embedding = [0.5] * 768
    chunks_table = "posts_work_chunks_body_baai_bge_base_en_v1_5"
    job = _make_job(embedding=embedding, k=k, chunks_table=chunks_table)

    job_store = FakeSearchJobStore(job=job)
    searcher = FakeChunkSearcher(results=[_make_search_result()])
    bus = FakeEventBus()

    handler = SearchHandler(job_store=job_store, chunk_searcher=searcher, event_log=bus)
    await handler.handle(_make_task("job-1"))

    assert len(searcher.calls) == 1
    table, vec, result_k = searcher.calls[0]
    assert table == chunks_table
    assert vec == embedding
    assert result_k == k


@pytest.mark.asyncio
async def test_handle_search_error_marks_job_failed_and_reraises():
    """SearchHandler should fail the job and re-raise when the searcher throws."""
    job = _make_job()
    job_store = FakeSearchJobStore(job=job)
    bus = FakeEventBus()

    class _BrokenSearcher:
        async def search_nearest(self, table, vec, k):
            raise RuntimeError("pgvector is down")

    handler = SearchHandler(job_store=job_store, chunk_searcher=_BrokenSearcher(), event_log=bus)

    with pytest.raises(RuntimeError, match="pgvector is down"):
        await handler.handle(_make_task("job-1"))

    assert len(job_store.failed) == 1
    assert "pgvector is down" in job_store.failed[0][1]
    assert job_store.completed == []
    assert bus.drain_topic("search_candidates.retrieved") == []


@pytest.mark.asyncio
async def test_handle_empty_results_still_emits_candidates_event():
    """SearchHandler should still announce candidates (empty list) when no chunks match —
    downstream (SearchFinalizeHandler) is responsible for completing the job as empty."""
    job = _make_job()
    job_store = FakeSearchJobStore(job=job)
    searcher = FakeChunkSearcher(results=[])
    bus = FakeEventBus()

    handler = SearchHandler(job_store=job_store, chunk_searcher=searcher, event_log=bus)
    await handler.handle(_make_task("job-1"))

    assert job_store.completed == []
    events = bus.drain_topic("search_candidates.retrieved")
    assert len(events) == 1
    assert events[0]["candidates"] == []
