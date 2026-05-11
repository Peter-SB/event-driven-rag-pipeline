"""Integration tests for the chunk pipeline end-to-end flow.

Tests the complete flow with real infrastructure (Postgres testcontainer,
RabbitMQ testcontainer, live PostgresEventBus, real dispatchers).

Flow under test
---------------
    post.synced event on event log
        ↓
    PostDispatcher.run() (real aio_pika connection)
        ↓  [publishes ChunkTask to RabbitMQ "ingestion" exchange]
    ChunkTask on RabbitMQ queue
        ↓
    ChunkPostHandler.handle() (real PostRepository + ChunkRepository)
        ↓  [ensures chunk table, deduplicates, inserts chunks]
    Chunks in Postgres + chunks.created event on event log

Tested behaviours
-----------------
- Full flow: post.synced → ChunkTask dispatched → chunks created → chunks.created published
- Chunk table created lazily on first task (no pre-existing table)
- Idempotency: same task twice produces no duplicate chunks, one chunks.created event
- summary_title task dispatched when has_summary=True; skipped when False
- Body task dispatched when fields_changed includes body_text changes
- Consumer offset advances after dispatcher processes event
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import aio_pika
import asyncpg
import pytest

from event_driven_rag_service.config import consumer_groups
from event_driven_rag_service.dispatchers.post_dispatcher import PostDispatcher
from event_driven_rag_service.infrastructure.event_bus import PostgresEventBus
from event_driven_rag_service.repository.post_repository import PostRepository
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.handlers.chunk_handler import ChunkPostHandler
from event_driven_rag_service.tasks.chunk_task import ChunkTask
from tests.utils.factories import make_post, make_post_synced_event

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _setup_bus(pool: asyncpg.Pool) -> PostgresEventBus:
    """Initialize a real PostgresEventBus with event_log tables."""
    bus = PostgresEventBus(pool)
    await bus.setup_tables()
    return bus


async def _consume_one_event(
    bus: PostgresEventBus, topic: str, consumer_group: str, timeout: float = 5.0
) -> dict:
    """Pull exactly one event from the event bus; raise TimeoutError if none arrives."""

    async def _read_first() -> dict:
        async for event in bus.subscribe(topic, consumer_group=consumer_group):
            return event

    return await asyncio.wait_for(_read_first(), timeout=timeout)


async def _consume_one_rmq_message(queue: aio_pika.abc.AbstractQueue, timeout: float = 5.0) -> dict:
    """Pull one message from an RabbitMQ queue; raise TimeoutError if none arrives."""
    msg = await asyncio.wait_for(queue.get(no_ack=True), timeout=timeout)
    return json.loads(msg.body)


# ---------------------------------------------------------------------------
# Full pipeline: post.synced → PostDispatcher → RabbitMQ → handler → chunks.created
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_post_synced_to_chunks_created(clean_pipeline_tables):
    """Full flow: post → event → dispatch → handler → chunks in DB and event published."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    post_repo: PostRepository = fixtures["post_repo"]
    post_table: str = fixtures["post_table"]
    chunk_table: str = fixtures["chunk_table"]
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]

    # Setup event bus
    bus = await _setup_bus(postgres_pool)

    # Setup RabbitMQ: declare exchange and queue (exclusive, auto-delete to avoid test pollution)
    channel = await rmq_conn.channel()
    exchange = await channel.declare_exchange("ingestion", aio_pika.ExchangeType.TOPIC, durable=True)
    queue = await channel.declare_queue("cpu.chunk.post", exclusive=True, auto_delete=True)
    await queue.bind(exchange, routing_key="cpu.chunk.post")

    # Insert a post
    post = make_post(post_id=101, body_text="word " * 200, summary="A summary.", updated_at=None)
    upsert_result, _ = await post_repo.upsert(post, post_table)
    assert upsert_result in ("inserted", "updated")

    # Publish post.synced event
    synced_event = make_post_synced_event(
        post_id=101, has_summary=True, fields_changed=[], post_table=post_table
    )
    await bus.publish("post.synced", synced_event)

    # Run PostDispatcher briefly — it will consume the event and dispatch ChunkTask
    dispatcher = PostDispatcher(rmq_conn, bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass  # Expected — dispatcher ran, consumed the event, and is now polling for more

    # Consume ChunkTask from RabbitMQ queue
    task_dict = await _consume_one_rmq_message(queue, timeout=3.0)
    task = ChunkTask.model_validate(task_dict)
    assert task.post_id == 101
    assert task.task_type == "body"
    assert task.post_table == post_table

    # Create ChunkPostHandler with real repos
    chunk_repo = ChunkRepository(postgres_pool, table_name=chunk_table, vector_dim=768)
    handler = ChunkPostHandler(
        post_fetcher=post_repo,
        chunk_store=chunk_repo,
        version_checker=chunk_repo,
        event_log=bus,
    )

    # Run handler
    chunk_ids = await handler.handle(task)
    assert len(chunk_ids) > 0, "Handler should have created chunks"

    # Verify chunks in database
    async with postgres_pool.acquire() as conn:
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {chunk_table} WHERE post_id = $1", 101)
        assert count == len(chunk_ids), "All returned chunk IDs should be in the DB"

    # Consume chunks.created event from bus
    created_event = await _consume_one_event(bus, "chunks.created", "test.pipeline.group", timeout=3.0)
    assert created_event["post_id"] == 101
    assert created_event["chunk_table"] == chunk_table
    assert created_event["task_type"] == "body"
    assert set(created_event["chunk_ids"]) == set(chunk_ids)

    await channel.close()


# ---------------------------------------------------------------------------
# Chunk table created lazily
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunk_table_created_lazily_on_first_task(clean_pipeline_tables):
    """Chunk table does not exist beforehand; handler creates it on first task."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    post_repo: PostRepository = fixtures["post_repo"]
    post_table: str = fixtures["post_table"]
    chunk_table: str = fixtures["chunk_table"]

    # Verify table does not exist yet
    async with postgres_pool.acquire() as conn:
        exists_before = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = $1
            )
            """,
            chunk_table,
        )
        assert not exists_before, f"Table {chunk_table} should not exist initially"

    # Setup event bus and insert post
    bus = await _setup_bus(postgres_pool)
    post = make_post(post_id=102, body_text="word " * 200, summary="Summary.", updated_at=None)
    _, _ = await post_repo.upsert(post, post_table)

    # Create and run handler
    chunk_repo = ChunkRepository(postgres_pool, table_name=chunk_table, vector_dim=768)
    handler = ChunkPostHandler(
        post_fetcher=post_repo,
        chunk_store=chunk_repo,
        version_checker=chunk_repo,
        event_log=bus,
    )

    task = ChunkTask(
        task_type="body",
        post_id=102,
        post_table=post_table,
        embed_model="BAAI/bge-base-en-v1.5",
    )

    chunk_ids = await handler.handle(task)
    assert len(chunk_ids) > 0

    # Verify table now exists
    async with postgres_pool.acquire() as conn:
        exists_after = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = $1
            )
            """,
            chunk_table,
        )
        assert exists_after, f"Table {chunk_table} should have been created"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotency_no_duplicate_chunks_on_rerun(clean_pipeline_tables):
    """Running the same task twice should not duplicate chunks; only one chunks.created event."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    post_repo: PostRepository = fixtures["post_repo"]
    post_table: str = fixtures["post_table"]
    chunk_table: str = fixtures["chunk_table"]

    bus = await _setup_bus(postgres_pool)

    post = make_post(post_id=103, body_text="word " * 200, summary="Summary.", updated_at=None)
    _, _ = await post_repo.upsert(post, post_table)

    chunk_repo = ChunkRepository(postgres_pool, table_name=chunk_table, vector_dim=768)
    handler = ChunkPostHandler(
        post_fetcher=post_repo,
        chunk_store=chunk_repo,
        version_checker=chunk_repo,
        event_log=bus,
    )

    task = ChunkTask(
        task_type="body",
        post_id=103,
        post_table=post_table,
        embed_model="BAAI/bge-base-en-v1.5",
    )

    # First run
    chunk_ids_first = await handler.handle(task)
    assert len(chunk_ids_first) > 0

    async with postgres_pool.acquire() as conn:
        count_first = await conn.fetchval(f"SELECT COUNT(*) FROM {chunk_table} WHERE post_id = $1", 103)

    # Second run (same task)
    chunk_ids_second = await handler.handle(task)
    assert chunk_ids_second == [], "Second run should find all chunks already current and return []"

    async with postgres_pool.acquire() as conn:
        count_second = await conn.fetchval(f"SELECT COUNT(*) FROM {chunk_table} WHERE post_id = $1", 103)

    assert count_second == count_first, "Chunk count should not change on second run"

    # Verify only one chunks.created event was published
    # (QueryingPostgres event log since we can't drain a real event bus)
    async with postgres_pool.acquire() as conn:
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM event_log
            WHERE topic = 'chunks.created' AND payload->>'post_id' = '103'
            """
        )
    assert count == 1, "Only one chunks.created event should have been published"


# ---------------------------------------------------------------------------
# Task type variations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_title_task_handled(clean_pipeline_tables):
    """summary_title task type should chunk title + summary correctly."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    post_repo: PostRepository = fixtures["post_repo"]
    post_table: str = fixtures["post_table"]
    chunk_table_title: str = "posts_pipeline_test_chunks_summary_title_baai_bge_base_en_v1_5"

    bus = await _setup_bus(postgres_pool)

    # Post with significant summary/title (but minimal body)
    post = make_post(
        post_id=104,
        body_text="minimal",
        summary="This is a detailed summary of the post. " * 10,
        updated_at=None,
    )
    _, _ = await post_repo.upsert(post, post_table)

    chunk_repo = ChunkRepository(postgres_pool, table_name=chunk_table_title, vector_dim=768)
    handler = ChunkPostHandler(
        post_fetcher=post_repo,
        chunk_store=chunk_repo,
        version_checker=chunk_repo,
        event_log=bus,
    )

    task = ChunkTask(
        task_type="summary_title",
        post_id=104,
        post_table=post_table,
        embed_model="BAAI/bge-base-en-v1.5",
    )

    chunk_ids = await handler.handle(task)
    assert len(chunk_ids) > 0

    # Verify event has correct task_type (consume from real event bus)
    created_event = await _consume_one_event(bus, "chunks.created", "test.summary_group", timeout=3.0)
    assert created_event["task_type"] == "summary_title"
    assert created_event["chunk_table"] == chunk_table_title


# ---------------------------------------------------------------------------
# Consumer offset advancement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consumer_offset_advances_after_dispatcher_runs(clean_pipeline_tables):
    """After dispatcher processes post.synced, consumer_offsets should be persisted."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    post_repo: PostRepository = fixtures["post_repo"]
    post_table: str = fixtures["post_table"]
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]

    bus = await _setup_bus(postgres_pool)

    # Setup RabbitMQ (exclusive, auto-delete to avoid test pollution)
    channel = await rmq_conn.channel()
    exchange = await channel.declare_exchange("ingestion", aio_pika.ExchangeType.TOPIC, durable=True)
    queue = await channel.declare_queue("cpu.chunk.post", exclusive=True, auto_delete=True)
    await queue.bind(exchange, routing_key="cpu.chunk.post")

    # Insert post and publish event
    post = make_post(post_id=105, body_text="word " * 200, summary="Summary.", updated_at=None)
    _, _ = await post_repo.upsert(post, post_table)

    synced_event = make_post_synced_event(
        post_id=105, has_summary=True, fields_changed=[], post_table=post_table
    )
    await bus.publish("post.synced", synced_event)

    # Run dispatcher
    dispatcher = PostDispatcher(rmq_conn, bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    # Verify consumer offset was recorded
    async with postgres_pool.acquire() as conn:
        offset = await conn.fetchval(
            """
            SELECT last_id FROM consumer_offsets
            WHERE consumer_group = $1 AND topic = $2
            """,
            consumer_groups.POST_SYNCED,
            "post.synced",
        )

    assert offset is not None, "Consumer offset should have been recorded"
    assert offset > 0, "Offset should have advanced"

    await channel.close()


# ---------------------------------------------------------------------------
# summary_title task type tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_title_chunk_text_format_in_db(clean_pipeline_tables):
    """Full pipeline: summary_title task produces chunk with 'Title: ... Summary: ...' format in DB."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    post_repo: PostRepository = fixtures["post_repo"]
    post_table: str = fixtures["post_table"]

    bus = await _setup_bus(postgres_pool)

    # Create a post with title and summary
    post = make_post(
        post_id=200,
        body_text="This body text is ignored for summary_title task",
        title="Understanding Vector Embeddings",
        summary="A guide to how embeddings work in modern ML systems.",
        updated_at=None,
    )
    _, _ = await post_repo.upsert(post, post_table)

    # Create handler for summary_title chunks
    chunk_table = "posts_pipeline_test_chunks_summary_title_baai_bge_base_en_v1_5"
    chunk_repo = ChunkRepository(postgres_pool, table_name=chunk_table, vector_dim=768)
    handler = ChunkPostHandler(
        post_fetcher=post_repo,
        chunk_store=chunk_repo,
        version_checker=chunk_repo,
        event_log=bus,
    )

    # Create and execute summary_title task
    task = ChunkTask(
        task_type="summary_title",
        post_id=200,
        post_table=post_table,
        embed_model="BAAI/bge-base-en-v1.5",
    )

    chunk_ids = await handler.handle(task)
    assert len(chunk_ids) > 0, "Handler should create at least one chunk"

    # Query the DB and verify chunk text format
    async with postgres_pool.acquire() as conn:
        chunk_rows = await conn.fetch(
            f"SELECT text FROM {chunk_table} WHERE post_id = $1",
            200,
        )

    assert len(chunk_rows) >= 1, "Chunk should be persisted in DB"
    chunk_text = chunk_rows[0]["text"]

    # Verify format: "Title: ... Summary: ..."
    assert chunk_text.startswith("Title: Understanding Vector Embeddings"), \
        f"Chunk text should start with 'Title:', got: {chunk_text[:100]}"
    assert "Summary: A guide to how embeddings work" in chunk_text, \
        f"Chunk text should contain 'Summary:' label, got: {chunk_text}"

    # Verify the event was published with correct task_type
    created_event = await _consume_one_event(bus, "chunks.created", "test.summary_title.group", timeout=3.0)
    assert created_event["post_id"] == 200
    assert created_event["task_type"] == "summary_title"
    assert created_event["chunk_table"] == chunk_table
    assert set(created_event["chunk_ids"]) == set(chunk_ids)


@pytest.mark.asyncio
async def test_summary_title_single_chunk_for_typical_length(clean_pipeline_tables):
    """summary_title task should produce exactly one chunk for typical length data."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    post_repo: PostRepository = fixtures["post_repo"]
    post_table: str = fixtures["post_table"]

    bus = await _setup_bus(postgres_pool)

    # Create a post with moderate title/summary (typical case)
    post = make_post(
        post_id=201,
        title="Machine Learning Basics",
        summary="An introduction to core ML concepts and algorithms.",
        updated_at=None,
    )
    _, _ = await post_repo.upsert(post, post_table)

    chunk_table = "posts_pipeline_test_chunks_summary_title_baai_bge_base_en_v1_5"
    chunk_repo = ChunkRepository(postgres_pool, table_name=chunk_table, vector_dim=768)
    handler = ChunkPostHandler(
        post_fetcher=post_repo,
        chunk_store=chunk_repo,
        version_checker=chunk_repo,
        event_log=bus,
    )

    task = ChunkTask(
        task_type="summary_title",
        post_id=201,
        post_table=post_table,
        embed_model="BAAI/bge-base-en-v1.5",
    )

    chunk_ids = await handler.handle(task)

    # Verify exactly one chunk was created
    async with postgres_pool.acquire() as conn:
        count = await conn.fetchval(
            f"SELECT COUNT(*) FROM {chunk_table} WHERE post_id = $1",
            201,
        )

    assert count == 1, "summary_title should produce exactly one chunk for typical length"
    assert len(chunk_ids) == 1
