"""Unit tests for SearchFinalizeHandler.

Uses in-memory fakes for all dependencies — no Postgres, no RabbitMQ, no model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from event_driven_rag_service.handlers.search_finalize_handler import SearchFinalizeHandler
from event_driven_rag_service.tasks.search_tasks import CandidateChunkRef, SearchFinalizeTask
from tests.utils.factories import FakeEventBus


class FakeSearchJobStore:
    def __init__(self, job: Optional[Dict[str, Any]] = None) -> None:
        self._job = job
        self.completed: list[tuple[str, list]] = []
        self.failed: list[tuple[str, str]] = []

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._job

    async def complete_job(self, job_id: str, results: List[Dict]) -> None:
        self.completed.append((job_id, results))

    async def fail_job(self, job_id: str, error: str) -> None:
        self.failed.append((job_id, error))


class FakeChunkHydrator:
    def __init__(self, rows: dict[str, Dict[str, Any]] | None = None) -> None:
        self._rows = rows or {}

    async def get_chunks_by_ids(self, chunk_ids: list[str], table_name: str) -> list[Dict[str, Any]]:
        return [self._rows[cid] for cid in chunk_ids if cid in self._rows]


def _row(chunk_id: str, text: str, post_id: int = 1) -> Dict[str, Any]:
    return {"id": chunk_id, "post_id": post_id, "text": text, "metadata": {"k": "v"}}


def _job(job_id: str = "job-1", k: int = 2) -> Dict[str, Any]:
    return {"id": job_id, "k": k, "chunks_table": "posts_main_chunks_body_baai_bge_base_en_v1_5"}


@pytest.mark.asyncio
async def test_handle_truncates_to_k_preserving_ann_order():
    job = _job(k=2)
    rows = {
        "c1": _row("c1", "text one"),
        "c2": _row("c2", "text two"),
        "c3": _row("c3", "text three"),
    }
    job_store = FakeSearchJobStore(job=job)
    hydrator = FakeChunkHydrator(rows=rows)
    bus = FakeEventBus()

    handler = SearchFinalizeHandler(job_store=job_store, chunks=hydrator, event_log=bus)
    task = SearchFinalizeTask(
        job_id="job-1",
        candidates=[
            CandidateChunkRef(chunk_id="c1", score=0.9),
            CandidateChunkRef(chunk_id="c2", score=0.7),
            CandidateChunkRef(chunk_id="c3", score=0.5),
        ],
    )
    await handler.handle(task)

    assert len(job_store.completed) == 1
    job_id, results = job_store.completed[0]
    assert job_id == "job-1"
    assert len(results) == 2
    assert results[0] == {"chunk_id": "c1", "post_id": 1, "text": "text one", "metadata": {"k": "v"}, "score": 0.9}
    assert results[1]["chunk_id"] == "c2"

    events = bus.drain_topic("search_job.completed")
    assert len(events) == 1
    assert events[0]["query_job_id"] == "job-1"


@pytest.mark.asyncio
async def test_handle_preserves_ann_score_not_a_new_score():
    """No model is involved — the ANN cosine score from the candidate ref
    must pass straight through unchanged."""
    job = _job(k=1)
    rows = {"c1": _row("c1", "text one")}
    job_store = FakeSearchJobStore(job=job)
    hydrator = FakeChunkHydrator(rows=rows)
    bus = FakeEventBus()

    handler = SearchFinalizeHandler(job_store=job_store, chunks=hydrator, event_log=bus)
    task = SearchFinalizeTask(job_id="job-1", candidates=[CandidateChunkRef(chunk_id="c1", score=0.4242)])
    await handler.handle(task)

    _, results = job_store.completed[0]
    assert results[0]["score"] == 0.4242


@pytest.mark.asyncio
async def test_handle_job_not_found_skips_silently():
    job_store = FakeSearchJobStore(job=None)
    hydrator = FakeChunkHydrator()
    bus = FakeEventBus()

    handler = SearchFinalizeHandler(job_store=job_store, chunks=hydrator, event_log=bus)
    task = SearchFinalizeTask(job_id="missing-job", candidates=[])
    await handler.handle(task)

    assert job_store.completed == []
    assert job_store.failed == []


@pytest.mark.asyncio
async def test_handle_empty_candidates_completes_with_empty_results():
    job = _job()
    job_store = FakeSearchJobStore(job=job)
    hydrator = FakeChunkHydrator()
    bus = FakeEventBus()

    handler = SearchFinalizeHandler(job_store=job_store, chunks=hydrator, event_log=bus)
    task = SearchFinalizeTask(job_id="job-1", candidates=[])
    await handler.handle(task)

    assert job_store.completed == [("job-1", [])]
    assert len(bus.drain_topic("search_job.completed")) == 1


@pytest.mark.asyncio
async def test_handle_hydration_failure_fails_job_and_reraises():
    job = _job()
    job_store = FakeSearchJobStore(job=job)
    bus = FakeEventBus()

    class _BrokenHydrator:
        async def get_chunks_by_ids(self, chunk_ids, table_name):
            raise RuntimeError("table does not exist")

    handler = SearchFinalizeHandler(job_store=job_store, chunks=_BrokenHydrator(), event_log=bus)
    task = SearchFinalizeTask(job_id="job-1", candidates=[CandidateChunkRef(chunk_id="c1", score=0.5)])

    with pytest.raises(RuntimeError, match="table does not exist"):
        await handler.handle(task)

    assert len(job_store.failed) == 1
    assert job_store.completed == []
