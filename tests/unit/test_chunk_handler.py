"""Tests for ChunkPostHandler business logic.

ChunkPostHandler is the unit of work extracted from CpuChunkWorker.
All tests use in-memory fakes for I/O — no DB or RabbitMQ required.

Tested behaviours
-----------------
- handle() inserts chunks and publishes chunks.created for a new post
- chunks.created event carries correct post_id, chunk_table, and chunk_ids
- handle() skips ALL chunks when every text_hash already exists (up-to-date)
- handle() inserts only NEW chunks when some text_hashes are new
- Empty / missing text results in no insert and no event published
- chunk_table name is derived from the task (not hardcoded)
- _resolve_text routes correctly for body / summary_title / analysis task types
"""
from __future__ import annotations

from typing import Any

import pytest

from event_driven_rag_service.data_models.chunk import Chunk
from event_driven_rag_service.tasks.chunk_task import ChunkTask
from event_driven_rag_service.handlers.chunk_handler import (
    ChunkPostHandler,
    _build_chunks,
)
from tests.utils.factories import FakeEventBus


# ---------------------------------------------------------------------------
# Protocol fakes
# ---------------------------------------------------------------------------

class FakePostFetcher:
    def __init__(self, post: dict[str, Any]) -> None:
        self._post = post

    async def fetch(self, post_id: int, table_name: str) -> dict[str, Any]:
        return self._post


class FakeChunkStore:
    def __init__(self) -> None:
        self.inserted: list[Chunk] = []

    async def ensure_table(self, table_name: str, vector_dim: int) -> None:
        pass

    async def bulk_insert(self, chunks: list[Chunk], table_name: str) -> None:
        self.inserted.extend(chunks)


class FakeVersionChecker:
    def __init__(self, existing_hashes: dict[str, str] | None = None) -> None:
        self._hashes = existing_hashes or {}

    async def get_text_hashes(self, post_id: int, table_name: str) -> dict[str, str]:
        return self._hashes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler(
    post: dict,
    existing_hashes: dict | None = None,
) -> tuple[ChunkPostHandler, FakeChunkStore, FakeEventBus]:
    bus = FakeEventBus()
    store = FakeChunkStore()
    checker = FakeVersionChecker(existing_hashes)
    handler = ChunkPostHandler(
        post_fetcher=FakePostFetcher(post),
        chunk_store=store,
        version_checker=checker,
        event_log=bus,
    )
    return handler, store, bus


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_inserts_chunks_for_new_post():
    post = {"body_text": "interesting article " * 100, "title": "Test", "updated_at": "2024-01-01T00:00:00+00:00"}
    task = ChunkTask(task_type="body", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    handler, store, _ = _make_handler(post)

    chunk_ids = await handler.handle(task)

    assert len(store.inserted) > 0
    assert set(chunk_ids) == {c.id for c in store.inserted}


@pytest.mark.asyncio
async def test_handle_publishes_chunks_created_event():
    post = {"body_text": "interesting article " * 100, "title": "Test", "updated_at": "2024-01-01T00:00:00+00:00"}
    task = ChunkTask(task_type="body", post_id=5, post_table="posts_main", embed_model="bge-base-v1.5")
    handler, _, bus = _make_handler(post)

    await handler.handle(task)

    events = bus.peek_topic("chunks.created")
    assert len(events) == 1
    assert events[0]["post_id"] == 5


@pytest.mark.asyncio
async def test_chunks_created_event_carries_chunk_ids():
    post = {"body_text": "word " * 200, "title": "T", "updated_at": "2024-01-01T00:00:00+00:00"}
    task = ChunkTask(task_type="body", post_id=6, post_table="posts_main", embed_model="bge-base-v1.5")
    handler, store, bus = _make_handler(post)

    await handler.handle(task)

    event = bus.peek_topic("chunks.created")[0]
    stored_ids = {c.id for c in store.inserted}
    assert set(event["chunk_ids"]) == stored_ids


@pytest.mark.asyncio
async def test_chunks_created_event_has_correct_chunk_table():
    post = {"body_text": "word " * 200, "title": "T", "updated_at": "2024-01-01T00:00:00+00:00"}
    task = ChunkTask(task_type="body", post_id=7, post_table="posts_main", embed_model="bge-base-v1.5")
    handler, _, bus = _make_handler(post)

    await handler.handle(task)

    event = bus.peek_topic("chunks.created")[0]
    assert event["chunk_table"] == "posts_main_chunks_body_bge_base_v1_5"


# ---------------------------------------------------------------------------
# Idempotency — skip already-current chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_chunks_skipped_when_hashes_already_exist():
    """If all text_hashes are present, no insert and no event should occur."""
    body = "word " * 200
    chunks = _build_chunks(8, "2024-01-01T00:00:00+00:00", body, None)
    existing = {c.text_hash: c.id for c in chunks}

    post = {"body_text": body, "title": "T", "updated_at": "2024-01-01T00:00:00+00:00"}
    task = ChunkTask(task_type="body", post_id=8, post_table="posts_main", embed_model="bge-base-v1.5")
    handler, store, bus = _make_handler(post, existing_hashes=existing)

    result = await handler.handle(task)

    assert result == []
    assert store.inserted == []
    assert bus.peek_topic("chunks.created") == []


@pytest.mark.asyncio
async def test_only_new_chunks_inserted_on_partial_update():
    """Some hashes already stored, some are new — only new chunks inserted."""
    body = "word " * 400
    chunks = _build_chunks(9, "2024-01-01T00:00:00+00:00", body, None)
    if len(chunks) < 2:
        pytest.skip("Not enough chunks to split for this test")

    existing = {chunks[0].text_hash: chunks[0].id}

    post = {"body_text": body, "title": "T", "updated_at": "2024-01-01T00:00:00+00:00"}
    task = ChunkTask(task_type="body", post_id=9, post_table="posts_main", embed_model="bge-base-v1.5")
    handler, store, bus = _make_handler(post, existing_hashes=existing)

    await handler.handle(task)

    inserted_hashes = {c.text_hash for c in store.inserted}
    assert chunks[0].text_hash not in inserted_hashes
    assert len(store.inserted) >= 1


# ---------------------------------------------------------------------------
# Empty / missing text — no work done
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_skips_when_body_text_is_empty():
    post = {"body_text": "", "title": "T", "updated_at": "2024-01-01T00:00:00+00:00"}
    task = ChunkTask(task_type="body", post_id=10, post_table="posts_main", embed_model="bge-base-v1.5")
    handler, store, bus = _make_handler(post)

    result = await handler.handle(task)

    assert result == []
    assert store.inserted == []
    assert bus.peek_topic("chunks.created") == []


@pytest.mark.asyncio
async def test_handle_skips_when_body_text_is_none():
    post = {"body_text": None, "title": "T", "updated_at": "2024-01-01T00:00:00+00:00"}
    task = ChunkTask(task_type="body", post_id=11, post_table="posts_main", embed_model="bge-base-v1.5")
    handler, store, _ = _make_handler(post)

    result = await handler.handle(task)

    assert result == []
    assert store.inserted == []


# ---------------------------------------------------------------------------
# _resolve_text — text routing by task_type
# ---------------------------------------------------------------------------

def test_resolve_text_prefers_custom_body_over_body_text():
    task = ChunkTask(task_type="body", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    post = {"body_text": "original", "custom_body": "override"}
    assert ChunkPostHandler._resolve_text(task, post) == "override"


def test_resolve_text_falls_back_to_body_text():
    task = ChunkTask(task_type="body", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    post = {"body_text": "original", "custom_body": None}
    assert ChunkPostHandler._resolve_text(task, post) == "original"


def test_resolve_text_summary_title_combines_title_and_summary():
    task = ChunkTask(task_type="summary_title", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    post = {"title": "My Title", "summary": "A brief summary."}
    result = ChunkPostHandler._resolve_text(task, post)
    assert "My Title" in result
    assert "A brief summary." in result


def test_resolve_text_summary_title_handles_missing_summary():
    task = ChunkTask(task_type="summary_title", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    post = {"title": "Only Title", "summary": None}
    assert ChunkPostHandler._resolve_text(task, post) == "Only Title"


def test_resolve_text_returns_none_when_both_title_and_summary_absent():
    task = ChunkTask(task_type="summary_title", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    post = {"title": None, "summary": None}
    assert ChunkPostHandler._resolve_text(task, post) is None


def test_resolve_text_analysis_uses_inline_text():
    task = ChunkTask(
        task_type="analysis",
        post_id=1,
        post_table="posts_main",
        embed_model="qwen3-0.6b",
        analysis_text="Inline analysis content.",
    )
    post = {"analysis_text": "Stale post column value"}
    assert ChunkPostHandler._resolve_text(task, post) == "Inline analysis content."


def test_resolve_text_analysis_falls_back_to_post_column():
    task = ChunkTask(
        task_type="analysis",
        post_id=1,
        post_table="posts_main",
        embed_model="qwen3-0.6b",
        analysis_text=None,
    )
    post = {"analysis_text": "Stored analysis text."}
    assert ChunkPostHandler._resolve_text(task, post) == "Stored analysis text."
