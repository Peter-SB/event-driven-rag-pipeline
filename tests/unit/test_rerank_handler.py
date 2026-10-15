"""Unit tests for RerankHandler.

Uses in-memory fakes for all dependencies — no Postgres, no RabbitMQ, no
real model. The reranker is a fake that scores by a caller-supplied mapping.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from event_driven_rag_service.handlers.rerank_handler import RerankHandler
from event_driven_rag_service.tasks.search_tasks import CandidateChunkRef, SearchRankTask
from tests.utils.factories import FakeEventBus


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSearchJobStore:
    def __init__(self, job: Optional[Dict[str, Any]] = None) -> None:
        self._job = job
        self.marked_reranking: list[str] = []
        self.completed: list[tuple[str, list]] = []
        self.failed: list[tuple[str, str]] = []

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._job

    async def mark_reranking(self, job_id: str) -> None:
        self.marked_reranking.append(job_id)

    async def complete_job(self, job_id: str, results: List[Dict]) -> None:
        self.completed.append((job_id, results))

    async def fail_job(self, job_id: str, error: str) -> None:
        self.failed.append((job_id, error))


class FakeChunkHydrator:
    def __init__(self, rows: dict[str, Dict[str, Any]] | None = None) -> None:
        self._rows = rows or {}

    async def get_chunks_by_ids(self, chunk_ids: list[str], table_name: str) -> list[Dict[str, Any]]:
        return [self._rows[cid] for cid in chunk_ids if cid in self._rows]


class FakeReranker:
    """Scores by a fixed {chunk_id -> score} mapping (keyed by matching text)."""

    def __init__(self, score_by_text: dict[str, float]) -> None:
        self._score_by_text = score_by_text
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def name(self) -> str:
        return "fake-reranker"

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, documents))
        return [self._score_by_text.get(doc, 0.0) for doc in documents]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(chunk_id: str, text: str, post_id: int = 1) -> Dict[str, Any]:
    return {"id": chunk_id, "post_id": post_id, "text": text, "metadata": {"k": "v"}}


def _job(job_id: str = "job-1", k: int = 2, query: str = "test query") -> Dict[str, Any]:
    return {
        "id": job_id,
        "query": query,
        "k": k,
        "chunks_table": "posts_main_chunks_body_baai_bge_base_en_v1_5",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_reorders_by_reranker_score_and_truncates_to_k():
    """Candidates should be re-sorted by reranker score (not ANN score) and
    truncated to job['k'], even though the ANN order was different."""
    job = _job(k=2)
    rows = {
        "c1": _row("c1", "low relevance text"),
        "c2": _row("c2", "high relevance text"),
        "c3": _row("c3", "medium relevance text"),
    }
    job_store = FakeSearchJobStore(job=job)
    hydrator = FakeChunkHydrator(rows=rows)
    reranker = FakeReranker({
        "low relevance text": 0.1,
        "high relevance text": 0.9,
        "medium relevance text": 0.5,
    })
    bus = FakeEventBus()

    handler = RerankHandler(job_store=job_store, chunks=hydrator, event_log=bus)
    # ANN order: c1 (highest ANN score) first, c2 last — reranker should flip this.
    task = SearchRankTask(
        job_id="job-1",
        candidates=[
            CandidateChunkRef(chunk_id="c1", score=0.9),
            CandidateChunkRef(chunk_id="c2", score=0.3),
            CandidateChunkRef(chunk_id="c3", score=0.6),
        ],
    )

    success = await handler.handle(task, "fake-reranker-model", reranker)

    assert success is True
    assert job_store.marked_reranking == ["job-1"]
    assert len(job_store.completed) == 1
    job_id, results = job_store.completed[0]
    assert job_id == "job-1"
    assert len(results) == 2  # truncated to k=2
    assert results[0]["chunk_id"] == "c2"  # highest reranker score
    assert results[0]["score"] == 0.9
    assert results[1]["chunk_id"] == "c3"
    assert results[1]["score"] == 0.5

    events = bus.drain_topic("search_job.completed")
    assert len(events) == 1
    assert events[0]["query_job_id"] == "job-1"


@pytest.mark.asyncio
async def test_handle_passes_job_query_to_reranker():
    job = _job(query="what is retrieval augmented generation?")
    rows = {"c1": _row("c1", "RAG combines retrieval and generation")}
    job_store = FakeSearchJobStore(job=job)
    hydrator = FakeChunkHydrator(rows=rows)
    reranker = FakeReranker({"RAG combines retrieval and generation": 0.8})
    bus = FakeEventBus()

    handler = RerankHandler(job_store=job_store, chunks=hydrator, event_log=bus)
    task = SearchRankTask(job_id="job-1", candidates=[CandidateChunkRef(chunk_id="c1", score=0.5)])
    await handler.handle(task, "fake-reranker-model", reranker)

    assert len(reranker.calls) == 1
    called_query, called_docs = reranker.calls[0]
    assert called_query == "what is retrieval augmented generation?"
    assert called_docs == ["RAG combines retrieval and generation"]


@pytest.mark.asyncio
async def test_handle_job_not_found_returns_true_without_error():
    job_store = FakeSearchJobStore(job=None)
    hydrator = FakeChunkHydrator()
    reranker = FakeReranker({})
    bus = FakeEventBus()

    handler = RerankHandler(job_store=job_store, chunks=hydrator, event_log=bus)
    task = SearchRankTask(job_id="missing-job", candidates=[])
    success = await handler.handle(task, "fake-reranker-model", reranker)

    assert success is True
    assert job_store.marked_reranking == []
    assert job_store.completed == []


@pytest.mark.asyncio
async def test_handle_no_candidates_survive_hydration_completes_empty():
    """Candidates that no longer exist in the chunk table (deleted between
    search and rerank) should be dropped, not crash the batch."""
    job = _job()
    job_store = FakeSearchJobStore(job=job)
    hydrator = FakeChunkHydrator(rows={})  # nothing hydrates
    reranker = FakeReranker({})
    bus = FakeEventBus()

    handler = RerankHandler(job_store=job_store, chunks=hydrator, event_log=bus)
    task = SearchRankTask(job_id="job-1", candidates=[CandidateChunkRef(chunk_id="ghost", score=0.5)])
    success = await handler.handle(task, "fake-reranker-model", reranker)

    assert success is True
    assert job_store.completed == [("job-1", [])]
    assert reranker.calls == []  # never called with an empty document list


@pytest.mark.asyncio
async def test_handle_scoring_error_fails_job_and_returns_false():
    job = _job()
    rows = {"c1": _row("c1", "some text")}
    job_store = FakeSearchJobStore(job=job)
    hydrator = FakeChunkHydrator(rows=rows)
    bus = FakeEventBus()

    class _BrokenReranker:
        name = "broken"

        def score(self, query, documents):
            raise RuntimeError("CUDA out of memory")

    handler = RerankHandler(job_store=job_store, chunks=hydrator, event_log=bus)
    task = SearchRankTask(job_id="job-1", candidates=[CandidateChunkRef(chunk_id="c1", score=0.5)])
    success = await handler.handle(task, "broken-model", _BrokenReranker())

    assert success is False
    assert len(job_store.failed) == 1
    assert "CUDA out of memory" in job_store.failed[0][1]
    assert job_store.completed == []
    assert bus.drain_topic("search_job.completed") == []
