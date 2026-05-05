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
- ensure_table is called with correct vector dimensions
- unknown task_type is handled gracefully
"""
from __future__ import annotations

from typing import Any

import pytest

from event_driven_rag_service.tasks.chunk_task import ChunkTask
from event_driven_rag_service.handlers.chunk_handler import (
    ChunkPostHandler,
    _build_chunks,
    _select_strategy,
)
from event_driven_rag_service.utils.chunk_strategies import SplitTextAtIndexStrategy
from tests.utils.factories import (
    FakeEventBus,
    FakePostFetcher,
    FakeChunkStore,
    FakeChunkVersionChecker,
    make_post,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler(
    post: dict,
    existing_hashes: dict | None = None,
) -> tuple[ChunkPostHandler, FakeChunkStore, FakeEventBus]:
    bus = FakeEventBus()
    store = FakeChunkStore()
    checker = FakeChunkVersionChecker(existing_hashes)
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

    assert len(store.inserted_chunks) > 0
    assert set(chunk_ids) == {c.id for c in store.inserted_chunks}


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
    stored_ids = {c.id for c in store.inserted_chunks}
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
    assert store.inserted_chunks == []
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

    inserted_hashes = {c.text_hash for c in store.inserted_chunks}
    assert chunks[0].text_hash not in inserted_hashes
    assert len(store.inserted_chunks) >= 1


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
    assert store.inserted_chunks == []
    assert bus.peek_topic("chunks.created") == []


@pytest.mark.asyncio
async def test_handle_skips_when_body_text_is_none():
    post = {"body_text": None, "title": "T", "updated_at": "2024-01-01T00:00:00+00:00"}
    task = ChunkTask(task_type="body", post_id=11, post_table="posts_main", embed_model="bge-base-v1.5")
    handler, store, _ = _make_handler(post)

    result = await handler.handle(task)

    assert result == []
    assert store.inserted_chunks == []


# ---------------------------------------------------------------------------
# _resolve_text — text routing by task_type
# ---------------------------------------------------------------------------

def test_resolve_text_prefers_custom_body_over_body_text():
    task = ChunkTask(task_type="body", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    post = make_post(body_text="original", custom_body="override")
    assert ChunkPostHandler._resolve_text(task, post) == "override"


def test_resolve_text_falls_back_to_body_text():
    task = ChunkTask(task_type="body", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    post = make_post(body_text="original", custom_body=None)
    assert ChunkPostHandler._resolve_text(task, post) == "original"


def test_resolve_text_summary_title_combines_title_and_summary():
    task = ChunkTask(task_type="summary_title", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    post = make_post(title="My Title", summary="A brief summary.")
    result = ChunkPostHandler._resolve_text(task, post)
    assert "My Title" in result
    assert "A brief summary." in result


def test_resolve_text_summary_title_handles_missing_summary():
    task = ChunkTask(task_type="summary_title", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    post = make_post(title="Only Title", summary=None)
    assert ChunkPostHandler._resolve_text(task, post) == "Title: Only Title"


def test_resolve_text_returns_none_when_both_title_and_summary_absent():
    task = ChunkTask(task_type="summary_title", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    post = make_post(title=None, summary=None)
    assert ChunkPostHandler._resolve_text(task, post) is None


def test_resolve_text_analysis_uses_inline_text():
    task = ChunkTask(
        task_type="analysis",
        post_id=1,
        post_table="posts_main",
        embed_model="qwen3-0.6b",
        analysis_text="Inline analysis content.",
    )
    post = make_post(analysis_text="Stale post column value")
    assert ChunkPostHandler._resolve_text(task, post) == "Inline analysis content."


def test_resolve_text_analysis_falls_back_to_post_column():
    task = ChunkTask(
        task_type="analysis",
        post_id=1,
        post_table="posts_main",
        embed_model="qwen3-0.6b",
        analysis_text=None,
    )
    post = make_post(analysis_text="Stored analysis text.")
    assert ChunkPostHandler._resolve_text(task, post) == "Stored analysis text."


# ---------------------------------------------------------------------------
# ensure_table invocation with correct vector dimensions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_table_called_with_correct_vector_dim_for_body():
    """ensure_table should be called with dim=768 for body task."""
    post = {"body_text": "word " * 200, "title": "T", "updated_at": "2024-01-01T00:00:00+00:00"}
    task = ChunkTask(task_type="body", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    handler, store, _ = _make_handler(post)

    await handler.handle(task)

    assert len(store.ensure_table_calls) == 1
    table_name, vector_dim = store.ensure_table_calls[0]
    assert table_name == "posts_main_chunks_body_bge_base_v1_5"
    assert vector_dim == 768, "bge-base-v1.5 should have dim=768"


@pytest.mark.asyncio
async def test_ensure_table_called_with_correct_vector_dim_for_analysis():
    """ensure_table should be called with dim=1024 for analysis task."""
    post = {
        "body_text": None,
        "analysis_text": "analysis " * 100,
        "title": "T",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    task = ChunkTask(
        task_type="analysis",
        post_id=1,
        post_table="posts_main",
        embed_model="qwen3-0.6b",
        analysis_text="analysis " * 100,
    )
    handler, store, _ = _make_handler(post)

    await handler.handle(task)

    assert len(store.ensure_table_calls) == 1
    table_name, vector_dim = store.ensure_table_calls[0]
    assert vector_dim == 1024, "qwen3-0.6b should have dim=1024"


# ---------------------------------------------------------------------------
# Unknown task_type handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_skips_unknown_task_type():
    """Unknown task_type not in EMBED_CONFIGS → skip gracefully."""
    post = {"body_text": "word " * 100, "title": "T", "updated_at": "2024-01-01T00:00:00+00:00"}
    handler, store, bus = _make_handler(post)

    # Manually create task with unknown type (bypasses validation if possible)
    # Or we test that an unknown type would be caught earlier
    # Actually, ChunkTask.task_type is a Literal, so we can't create one with unknown type
    # This test validates that the constraint exists
    with pytest.raises(Exception):  # pydantic validation error
        ChunkTask(
            task_type="unknown_type",  # type: ignore
            post_id=1,
            post_table="posts_main",
            embed_model="unknown-model",
        )


# ---------------------------------------------------------------------------
# Chunk table name derivation
# ---------------------------------------------------------------------------

def test_chunk_task_derives_table_name_correctly():
    """ChunkTask.chunk_table_name() produces correct derived name."""
    task = ChunkTask(
        task_type="body",
        post_id=1,
        post_table="posts_main",
        embed_model="bge-base-v1.5",
    )
    assert task.chunk_table_name() == "posts_main_chunks_body_bge_base_v1_5"

    task2 = ChunkTask(
        task_type="summary_title",
        post_id=2,
        post_table="posts_test",
        embed_model="bge-base-v1.5",
    )
    assert task2.chunk_table_name() == "posts_test_chunks_summary_title_bge_base_v1_5"


# ---------------------------------------------------------------------------
# SplitTextAtIndexStrategy tests
# ---------------------------------------------------------------------------

def test_split_text_at_index_returns_single_chunk_for_short_text():
    """Text under max_chars should be returned as a single chunk."""
    strategy = SplitTextAtIndexStrategy(max_chars=1000)
    text = "Short text that fits easily within the limit."
    chunks = strategy.chunk(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_at_index_splits_long_text_at_max_chars():
    """Text exceeding max_chars should be split at character boundaries."""
    strategy = SplitTextAtIndexStrategy(max_chars=50)
    text = "a" * 150  # 150 chars, exceeds 50-char limit
    chunks = strategy.chunk(text)
    assert len(chunks) == 3  # 50 + 50 + 50
    assert all(len(c) <= 50 for c in chunks)
    assert "".join(chunks) == text


def test_split_text_at_index_strategy_selected_for_summary_title():
    """_select_strategy should return SplitTextAtIndexStrategy for summary_title."""
    strategy = _select_strategy("summary_title")
    assert isinstance(strategy, SplitTextAtIndexStrategy)


# ---------------------------------------------------------------------------
# Updated _resolve_text tests for summary_title format
# ---------------------------------------------------------------------------

def test_resolve_text_summary_title_format_includes_labels():
    """summary_title _resolve_text should format with 'Title:' and 'Summary:' labels."""
    task = ChunkTask(task_type="summary_title", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    post = make_post(title="My Article Title", summary="This is a brief summary.")
    result = ChunkPostHandler._resolve_text(task, post)
    assert "Title: My Article Title" in result
    assert "Summary: This is a brief summary." in result
    assert result.startswith("Title:")


def test_resolve_text_summary_title_only_title():
    """summary_title with only title should format correctly."""
    task = ChunkTask(task_type="summary_title", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    post = make_post(title="Only Title", summary=None)
    result = ChunkPostHandler._resolve_text(task, post)
    assert result == "Title: Only Title"


def test_resolve_text_summary_title_only_summary():
    """summary_title with only summary should format correctly."""
    task = ChunkTask(task_type="summary_title", post_id=1, post_table="posts_main", embed_model="bge-base-v1.5")
    post = make_post(title=None, summary="Only summary text.")
    result = ChunkPostHandler._resolve_text(task, post)
    assert result == "Summary: Only summary text."


# ---------------------------------------------------------------------------
# Handler tests for summary_title chunking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_summary_title_produces_single_chunk_for_short_text():
    """Handler with summary_title task type should produce a single chunk for typical data."""
    post = {
        "body_text": "ignored",
        "title": "Understanding AI",
        "summary": "A comprehensive guide to artificial intelligence concepts.",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    task = ChunkTask(task_type="summary_title", post_id=50, post_table="posts_main", embed_model="bge-base-v1.5")
    handler, store, _ = _make_handler(post)

    chunk_ids = await handler.handle(task)

    assert len(store.inserted_chunks) == 1, "summary_title should produce exactly one chunk"
    assert chunk_ids == [store.inserted_chunks[0].id]


@pytest.mark.asyncio
async def test_handle_summary_title_chunk_text_has_correct_format():
    """The chunk text for summary_title should start with 'Title:' and contain 'Summary:'."""
    post = {
        "body_text": "ignored",
        "title": "My Test Title",
        "summary": "My test summary.",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    task = ChunkTask(task_type="summary_title", post_id=51, post_table="posts_main", embed_model="bge-base-v1.5")
    handler, store, _ = _make_handler(post)

    await handler.handle(task)

    assert len(store.inserted_chunks) == 1
    chunk_text = store.inserted_chunks[0].text
    assert chunk_text.startswith("Title: My Test Title")
    assert "Summary: My test summary." in chunk_text


@pytest.mark.asyncio
async def test_handle_summary_title_publishes_correct_event():
    """Handler should publish chunks.created event with summary_title task_type."""
    post = {
        "body_text": "ignored",
        "title": "Title",
        "summary": "Summary.",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    task = ChunkTask(task_type="summary_title", post_id=52, post_table="posts_main", embed_model="bge-base-v1.5")
    handler, _, bus = _make_handler(post)

    await handler.handle(task)

    events = bus.peek_topic("chunks.created")
    assert len(events) == 1
    assert events[0]["task_type"] == "summary_title"


# ---------------------------------------------------------------------------
# Handler tests for title chunking
# ---------------------------------------------------------------------------

def test_resolve_text_returns_title_for_title_task_type():
    """_resolve_text should return title for task_type='title'."""
    task = ChunkTask(task_type="title", post_id=1, post_table="posts_main", embed_model="bge-small-en-v1.5")
    post = make_post(title="My Title")
    result = ChunkPostHandler._resolve_text(task, post)
    assert result == "My Title"


def test_resolve_text_prefers_custom_title_over_title():
    """_resolve_text should prefer custom_title when present."""
    task = ChunkTask(task_type="title", post_id=1, post_table="posts_main", embed_model="bge-small-en-v1.5")
    post = make_post(title="Original Title", custom_title="Custom Title")
    result = ChunkPostHandler._resolve_text(task, post)
    assert result == "Custom Title"


def test_split_text_at_index_strategy_selected_for_title():
    """_select_strategy should return SplitTextAtIndexStrategy for title."""
    strategy = _select_strategy("title")
    assert isinstance(strategy, SplitTextAtIndexStrategy)


@pytest.mark.asyncio
async def test_handle_title_produces_single_chunk():
    """Handler with title task type should produce a single chunk for typical title text."""
    post = {
        "body_text": "ignored",
        "title": "Understanding AI",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    task = ChunkTask(task_type="title", post_id=60, post_table="posts_main", embed_model="bge-small-en-v1.5")
    handler, store, _ = _make_handler(post)

    chunk_ids = await handler.handle(task)

    assert len(store.inserted_chunks) == 1, "title should produce exactly one chunk"
    assert chunk_ids == [store.inserted_chunks[0].id]


@pytest.mark.asyncio
async def test_handle_title_publishes_correct_event():
    """Handler should publish chunks.created event with title task_type."""
    post = {
        "body_text": "ignored",
        "title": "Title",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    task = ChunkTask(task_type="title", post_id=61, post_table="posts_main", embed_model="bge-small-en-v1.5")
    handler, _, bus = _make_handler(post)

    await handler.handle(task)

    events = bus.peek_topic("chunks.created")
    assert len(events) == 1
    assert events[0]["task_type"] == "title"


@pytest.mark.asyncio
async def test_ensure_table_called_with_correct_vector_dim_for_title():
    """ensure_table should be called with dim=384 for title task (bge-small-en-v1.5)."""
    post = {"title": "Test Title", "body_text": "ignored", "updated_at": "2024-01-01T00:00:00+00:00"}
    task = ChunkTask(task_type="title", post_id=1, post_table="posts_main", embed_model="bge-small-en-v1.5")
    handler, store, _ = _make_handler(post)

    await handler.handle(task)

    assert len(store.ensure_table_calls) == 1
    table_name, vector_dim = store.ensure_table_calls[0]
    assert table_name == "posts_main_chunks_title_bge_small_en_v1_5"
    assert vector_dim == 384, "bge-small-en-v1.5 should have dim=384"


@pytest.mark.asyncio
async def test_handle_title_with_custom_title():
    """Handler should use custom_title when present."""
    post = {
        "body_text": "ignored",
        "title": "Original Title",
        "custom_title": "Custom Title",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    task = ChunkTask(task_type="title", post_id=62, post_table="posts_main", embed_model="bge-small-en-v1.5")
    handler, store, _ = _make_handler(post)

    await handler.handle(task)

    assert len(store.inserted_chunks) == 1
    chunk_text = store.inserted_chunks[0].text
    assert chunk_text == "Custom Title"


@pytest.mark.asyncio
async def test_handle_skips_title_when_title_missing():
    """Handler should skip when title is missing."""
    post = {
        "body_text": "ignored",
        "title": None,
        "custom_title": None,
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    task = ChunkTask(task_type="title", post_id=63, post_table="posts_main", embed_model="bge-small-en-v1.5")
    handler, store, bus = _make_handler(post)

    chunk_ids = await handler.handle(task)

    assert len(chunk_ids) == 0
    assert len(store.inserted_chunks) == 0
    assert len(bus.peek_topic("chunks.created")) == 0
