"""Tests for EmbedHandler business logic.

EmbedHandler is the unit of work extracted from GpuEmbedWorker.
All tests use in-memory fakes for I/O — no DB or RabbitMQ required.

Tested behaviours
-----------------
- embed_chunks() fetches texts, encodes, persists, and emits embedding.completed
- embed_query() encodes a single search query and emits search_query.embedded
- embedding.completed is grouped by (post_id, chunk_table)
- Failed encoding returns all tasks to DLQ
- Missing query or job_id in embed_query is logged and treated as ok
- save_batch is called with properly formatted rows (ChunkEmbeddingRow, QueryEmbeddingRow)
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.handlers.embed_handler import EmbedHandler
from tests.utils.factories import FakeEventBus


# ---------------------------------------------------------------------------
# Protocol fakes
# ---------------------------------------------------------------------------

class FakeChunkFetcher:
    """In-memory chunk fetcher that returns predefined texts."""

    def __init__(self, chunks_by_id: dict[str, str]) -> None:
        """
        Args:
            chunks_by_id: Map of chunk_id → text
        """
        self._chunks = chunks_by_id

    async def fetch_texts(
        self, chunk_ids: list[str], table: str
    ) -> list[tuple[str, str]]:
        """Return (chunk_id, text) pairs in the same order as chunk_ids."""
        return [(cid, self._chunks.get(cid, "")) for cid in chunk_ids]


class FakeEmbeddingStore:
    """In-memory embedding store that records all saves."""

    def __init__(self) -> None:
        self.saved_rows: list[dict[str, Any]] = []

    async def save_batch(self, rows: list[dict[str, Any]]) -> None:
        self.saved_rows.extend(rows)


class MockEmbeddingModel:
    """Deterministic mock embedding model for testing."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self._name = "mock-model"

    @property
    def name(self) -> str:
        return self._name

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings (same input = same output)."""
        import hashlib

        embeddings = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            base = [b / 255.0 for b in digest[:32]]  # Use first 32 bytes
            # Tile/trim to required dimension
            vec = (base * ((self.dim // len(base)) + 1))[: self.dim]
            embeddings.append(vec)
        return embeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler(
    chunks_by_id: dict[str, str] | None = None,
) -> tuple[EmbedHandler, FakeChunkFetcher, FakeEmbeddingStore, FakeEventBus]:
    """Create an EmbedHandler with fakes."""
    bus = FakeEventBus()
    store = FakeEmbeddingStore()
    fetcher = FakeChunkFetcher(chunks_by_id or {})
    handler = EmbedHandler(
        chunk_fetcher=fetcher,
        embedding_store=store,
        event_log=bus,
    )
    return handler, fetcher, store, bus


# ---------------------------------------------------------------------------
# Happy path: embed_chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_chunks_fetches_and_encodes_texts():
    """Verify embed_chunks fetches chunk texts and calls encode()."""
    chunks_by_id = {
        "chunk-1": "This is the first chunk text.",
        "chunk-2": "This is the second chunk text.",
    }
    handler, fetcher, store, _ = _make_handler(chunks_by_id)

    task = EmbedTask(
        task_id="task-1",
        task_type="chunk",
        post_id=1,
        post_table="posts",
        chunk_ids=["chunk-1", "chunk-2"],
        chunk_table="chunks_body_bge_base_v1_5",
        model_name="bge-base-v1.5",
    )
    model = MockEmbeddingModel(dim=768)

    ok_tasks, failed_tasks = await handler.embed_chunks([task], model.name, model)

    assert len(store.saved_rows) == 2
    assert failed_tasks == []
    assert task in ok_tasks


@pytest.mark.asyncio
async def test_embed_chunks_saves_vectors_to_store():
    """Verify embed_chunks saves embedding vectors with correct schema."""
    chunks_by_id = {
        "chunk-1": "First text",
        "chunk-2": "Second text",
    }
    handler, _, store, _ = _make_handler(chunks_by_id)

    task = EmbedTask(
        task_id="task-1",
        task_type="chunk",
        post_id=5,
        post_table="posts",
        chunk_ids=["chunk-1", "chunk-2"],
        chunk_table="chunks_body_bge_base_v1_5",
        model_name="bge-base-v1.5",
    )
    model = MockEmbeddingModel(dim=768)

    await handler.embed_chunks([task], model.name, model)

    # Verify saved rows have correct structure
    assert len(store.saved_rows) == 2
    for row in store.saved_rows:
        assert "chunk_id" in row
        assert "embedding" in row
        assert "model_name" in row
        assert "chunk_table" in row
        assert len(row["embedding"]) == 768


@pytest.mark.asyncio
async def test_embed_chunks_emits_embedding_completed_event():
    """Verify embed_chunks emits embedding.completed event."""
    chunks_by_id = {
        "chunk-1": "Text 1",
        "chunk-2": "Text 2",
    }
    handler, _, _, bus = _make_handler(chunks_by_id)

    task = EmbedTask(
        task_id="task-1",
        task_type="chunk",
        post_id=7,
        post_table="posts",
        chunk_ids=["chunk-1", "chunk-2"],
        chunk_table="chunks_body_bge_base_v1_5",
        model_name="bge-base-v1.5",
        trace_id="trace-123",
    )
    model = MockEmbeddingModel(dim=768)

    await handler.embed_chunks([task], model.name, model)

    events = bus.peek_topic("embedding.completed")
    assert len(events) == 1
    event = events[0]
    assert event["post_id"] == 7
    assert event["chunk_table"] == "chunks_body_bge_base_v1_5"
    assert event["model_name"] == "mock-model"
    assert set(event["chunk_ids"]) == {"chunk-1", "chunk-2"}


@pytest.mark.asyncio
async def test_embed_chunks_groups_by_post_id_and_chunk_table():
    """Multiple tasks → one event per (post_id, chunk_table) group."""
    chunks_by_id = {
        "chunk-1": "Text 1",
        "chunk-2": "Text 2",
        "chunk-3": "Text 3",
    }
    handler, _, _, bus = _make_handler(chunks_by_id)

    # Two tasks, same post_id and chunk_table → should emit 1 event
    task1 = EmbedTask(
        task_id="task-1",
        task_type="chunk",
        post_id=10,
        post_table="posts",
        chunk_ids=["chunk-1"],
        chunk_table="chunks_body_bge_base_v1_5",
        model_name="bge-base-v1.5",
    )
    task2 = EmbedTask(
        task_id="task-2",
        task_type="chunk",
        post_id=10,
        post_table="posts",
        chunk_ids=["chunk-2"],
        chunk_table="chunks_body_bge_base_v1_5",
        model_name="bge-base-v1.5",
    )
    model = MockEmbeddingModel(dim=768)

    await handler.embed_chunks([task1, task2], model.name, model)

    events = bus.peek_topic("embedding.completed")
    assert len(events) == 1
    event = events[0]
    assert set(event["chunk_ids"]) == {"chunk-1", "chunk-2"}


# ---------------------------------------------------------------------------
# Edge cases: empty / missing data
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_chunks_returns_all_tasks_when_no_chunk_ids():
    """If a task has no chunk_ids, it's skipped but returned as ok."""
    handler, _, store, bus = _make_handler({})

    task_empty = EmbedTask(
        task_id="task-empty",
        task_type="chunk",
        post_id=1,
        post_table="posts",
        chunk_ids=[],  # Empty
        chunk_table="chunks_body_bge_base_v1_5",
        model_name="bge-base-v1.5",
    )
    model = MockEmbeddingModel(dim=768)

    ok_tasks, failed_tasks = await handler.embed_chunks([task_empty], model.name, model)

    assert ok_tasks == [task_empty]
    assert failed_tasks == []
    assert store.saved_rows == []
    assert bus.peek_topic("embedding.completed") == []


@pytest.mark.asyncio
async def test_embed_chunks_fails_entire_batch_on_encode_error():
    """If encode() raises, all tasks are failed (returned as failed_tasks)."""
    chunks_by_id = {"chunk-1": "Text 1"}
    handler, _, _, _ = _make_handler(chunks_by_id)

    task = EmbedTask(
        task_id="task-1",
        task_type="chunk",
        post_id=1,
        post_table="posts",
        chunk_ids=["chunk-1"],
        chunk_table="chunks_body_bge_base_v1_5",
        model_name="bge-base-v1.5",
    )

    # Create a broken model that raises
    broken_model = MagicMock()
    broken_model.name = "broken-model"
    broken_model.encode.side_effect = RuntimeError("GPU error")

    ok_tasks, failed_tasks = await handler.embed_chunks([task], broken_model.name, broken_model)

    assert ok_tasks == []
    assert failed_tasks == [task]


# ---------------------------------------------------------------------------
# Happy path: embed_query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_query_encodes_and_persists_vector():
    """Verify embed_query encodes a search query and saves the vector."""
    handler, _, store, _ = _make_handler({})

    task = EmbedTask(
        task_id="task-query",
        task_type="query",
        post_id=None,
        post_table=None,
        chunk_ids=[],
        chunk_table=None,
        model_name="bge-base-v1.5",
        query="How to learn Python?",
        query_job_id="job-123",
    )
    model = MockEmbeddingModel(dim=768)

    result = await handler.embed_query(task, model.name, model)

    assert result is True
    assert len(store.saved_rows) == 1
    row = store.saved_rows[0]
    assert row["query_job_id"] == "job-123"
    assert row["model_name"] == "mock-model"
    assert len(row["embedding"]) == 768


@pytest.mark.asyncio
async def test_embed_query_emits_search_query_embedded_event():
    """Verify embed_query emits search_query.embedded event."""
    handler, _, _, bus = _make_handler({})

    task = EmbedTask(
        task_id="task-query",
        task_type="query",
        post_id=None,
        post_table=None,
        chunk_ids=[],
        chunk_table=None,
        model_name="bge-base-v1.5",
        query="Search query text",
        query_job_id="job-456",
        trace_id="trace-query",
    )
    model = MockEmbeddingModel(dim=768)

    await handler.embed_query(task, model.name, model)

    events = bus.peek_topic("search_query.embedded")
    assert len(events) == 1
    event = events[0]
    assert event["query_job_id"] == "job-456"
    assert event["model_name"] == "mock-model"
    assert event["trace_id"] == "trace-query"


# ---------------------------------------------------------------------------
# Edge cases: embed_query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_query_skips_when_query_missing():
    """Missing query is logged but treated as success (no DLQ poison)."""
    handler, _, store, _ = _make_handler({})

    task = EmbedTask(
        task_id="task-bad-query",
        task_type="query",
        post_id=None,
        post_table=None,
        chunk_ids=[],
        chunk_table=None,
        model_name="bge-base-v1.5",
        query=None,  # Missing
        query_job_id="job-789",
    )
    model = MockEmbeddingModel(dim=768)

    result = await handler.embed_query(task, model.name, model)

    assert result is True  # Treated as ok to avoid DLQ poisoning
    assert store.saved_rows == []  # Nothing persisted


@pytest.mark.asyncio
async def test_embed_query_skips_when_job_id_missing():
    """Missing job_id is logged but treated as success."""
    handler, _, store, _ = _make_handler({})

    task = EmbedTask(
        task_id="task-bad-job",
        task_type="query",
        post_id=None,
        post_table=None,
        chunk_ids=[],
        chunk_table=None,
        model_name="bge-base-v1.5",
        query="Search query",
        query_job_id=None,  # Missing
    )
    model = MockEmbeddingModel(dim=768)

    result = await handler.embed_query(task, model.name, model)

    assert result is True
    assert store.saved_rows == []


@pytest.mark.asyncio
async def test_embed_query_fails_on_encode_error():
    """Encode failure is logged and returns False."""
    handler, _, _, _ = _make_handler({})

    task = EmbedTask(
        task_id="task-query",
        task_type="query",
        post_id=None,
        post_table=None,
        chunk_ids=[],
        chunk_table=None,
        model_name="bge-base-v1.5",
        query="Query text",
        query_job_id="job-fail",
    )

    # Broken model
    broken_model = MagicMock()
    broken_model.name = "broken-model"
    broken_model.encode.side_effect = RuntimeError("GPU error")

    result = await handler.embed_query(task, broken_model.name, broken_model)

    assert result is False
