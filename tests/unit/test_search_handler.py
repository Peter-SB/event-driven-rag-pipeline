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
async def test_handle_success_stores_results_and_emits_event():
    """SearchHandler should mark job as searching, store results, and emit search_job.completed."""
    job = _make_job()
    fake_results = [_make_search_result("c1", 0.9), _make_search_result("c2", 0.7)]

    job_store = FakeSearchJobStore(job=job)
    searcher = FakeChunkSearcher(results=fake_results)
    bus = FakeEventBus()

    handler = SearchHandler(job_store=job_store, chunk_searcher=searcher, event_log=bus)
    await handler.handle(_make_task("job-1"))

    assert job_store.marked_searching == ["job-1"]
    assert len(job_store.completed) == 1
    completed_id, results = job_store.completed[0]
    assert completed_id == "job-1"
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["score"] == 0.9
    assert results[1]["chunk_id"] == "c2"

    events = bus.drain_topic("search_job.completed")
    assert len(events) == 1
    assert events[0]["query_job_id"] == "job-1"
    assert events[0]["event_type"] == "search_job.completed"


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
    assert bus.drain_topic("search_job.completed") == []


@pytest.mark.asyncio
async def test_handle_empty_results_stored_as_empty_list():
    """SearchHandler should complete job even when no chunks match."""
    job = _make_job()
    job_store = FakeSearchJobStore(job=job)
    searcher = FakeChunkSearcher(results=[])
    bus = FakeEventBus()

    handler = SearchHandler(job_store=job_store, chunk_searcher=searcher, event_log=bus)
    await handler.handle(_make_task("job-1"))

    assert len(job_store.completed) == 1
    _, results = job_store.completed[0]
    assert results == []
    assert len(bus.drain_topic("search_job.completed")) == 1
