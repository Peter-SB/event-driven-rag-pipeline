"""Integration tests for the embed pipeline end-to-end flow.

Tests the complete flow with real infrastructure (Postgres testcontainer,
RabbitMQ testcontainer, live PostgresEventBus, real dispatchers).

Flow under test
---------------
    chunks.created event on event log
        ↓
    ChunkDispatcher.run() (real aio_pika connection)
        ↓  [publishes EmbedTask to RabbitMQ "embedding" exchange]
    EmbedTask on RabbitMQ queue
        ↓
    EmbedHandler.embed_chunks() (real ChunkRepository + mock EmbeddingModel)
        ↓  [fetches chunk texts, encodes vectors, saves to DB]
    Embeddings in Postgres + embedding.completed event on event log

Tested behaviours
-----------------
- Full flow: chunks.created → EmbedTask dispatched → embeddings stored → embedding.completed published
- Mock embedding model returns deterministic 768-dim vectors (no real GPU needed)
- Idempotency: running embed_chunks twice on the same chunks overwrites vectors, emits one event per run
- Consumer offset advances after ChunkDispatcher processes chunks.created event
- EmbedTask carries correct post_id, chunk_ids, chunk_table, and model_name from event
"""
from __future__ import annotations

import asyncio
import json

import aio_pika
import asyncpg
import pytest

from event_driven_rag_service.config import consumer_groups
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.dispatchers.chunk_dispatcher import ChunkDispatcher
from event_driven_rag_service.handlers.embed_handler import EmbedHandler
from event_driven_rag_service.infrastructure.event_bus import PostgresEventBus
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.repository.post_repository import PostRepository
from event_driven_rag_service.tasks.embed_task import EmbedTask
from tests.utils.factories import make_post, make_chunks_created_event

pytestmark = pytest.mark.integration

_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_VECTOR_DIM = 768


# ---------------------------------------------------------------------------
# Mock embedding model — deterministic vectors, no GPU required
# ---------------------------------------------------------------------------

class _MockEmbeddingModel:
    """Deterministic mock: returns a unit-like vector seeded by text length."""

    def __init__(self, dim: int = _VECTOR_DIM) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return _MODEL_NAME

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            seed = len(text) % self._dim
            vec = [0.0] * self._dim
            vec[seed] = 1.0
            vectors.append(vec)
        return vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _setup_bus(pool: asyncpg.Pool) -> PostgresEventBus:
    bus = PostgresEventBus(pool)
    await bus.setup_tables()
    return bus


async def _consume_one_event(
    bus: PostgresEventBus,
    topic: str,
    consumer_group: str,
    timeout: float = 5.0,
) -> dict:
    async def _read_first() -> dict:
        async for event in bus.subscribe(topic, consumer_group=consumer_group):
            return event
        raise RuntimeError(f"No events published on topic {topic!r}")

    return await asyncio.wait_for(_read_first(), timeout=timeout)


async def _consume_one_rmq_message(
    queue: aio_pika.abc.AbstractQueue,
    timeout: float = 5.0,
) -> dict:
    msg = await asyncio.wait_for(queue.get(no_ack=True), timeout=timeout)
    return json.loads(msg.body)


async def _insert_chunks_and_publish_event(
    post_repo: PostRepository,
    chunk_repo: ChunkRepository,
    bus: PostgresEventBus,
    post_id: int,
    post_table: str,
) -> list[str]:
    """Insert a post, create real chunks via ChunkPostHandler, return chunk_ids."""
    from event_driven_rag_service.handlers.chunk_handler import ChunkPostHandler
    from event_driven_rag_service.tasks.chunk_task import ChunkTask

    post = make_post(post_id=post_id, body_text="word " * 200, summary="Summary.", updated_at=None)
    await post_repo.upsert(post, post_table)

    handler = ChunkPostHandler(
        post_fetcher=post_repo,
        chunk_store=chunk_repo,
        version_checker=chunk_repo,
        event_log=bus,
    )
    task = ChunkTask(
        task_type="body",
        post_id=post_id,
        post_table=post_table,
        embed_model=_MODEL_NAME,
    )
    chunk_ids = await handler.handle(task)
    assert len(chunk_ids) > 0, f"Precondition: ChunkPostHandler must create chunks for post_id={post_id}"
    return chunk_ids


# ---------------------------------------------------------------------------
# Full pipeline: chunks.created → ChunkDispatcher → RabbitMQ → EmbedHandler → embedding.completed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_embed_pipeline_chunks_created_to_embedding_completed(clean_pipeline_tables):
    """Full flow: chunks in DB → chunks.created event → dispatch → EmbedHandler → embeddings stored → embedding.completed published."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    post_repo: PostRepository = fixtures["post_repo"]
    post_table: str = fixtures["post_table"]
    chunk_table: str = fixtures["chunk_table"]
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]

    bus = await _setup_bus(postgres_pool)

    # Setup RabbitMQ: declare embedding exchange and model-specific queue
    channel = await rmq_conn.channel()
    embed_cfg = EMBED_CONFIGS["body"]
    exchange = await channel.declare_exchange("embedding", aio_pika.ExchangeType.TOPIC, durable=True)
    routing_key = f"gpu.embed.{embed_cfg.model}"
    queue = await channel.declare_queue(routing_key, exclusive=True, auto_delete=True)
    await queue.bind(exchange, routing_key=routing_key)

    # Create real chunks and drain the chunks.created event so the bus has it
    chunk_repo = ChunkRepository(postgres_pool, table_name=chunk_table, vector_dim=_VECTOR_DIM)
    chunk_ids = await _insert_chunks_and_publish_event(
        post_repo, chunk_repo, bus, post_id=201, post_table=post_table,
    )

    # Run ChunkDispatcher briefly — consumes chunks.created, publishes EmbedTask
    dispatcher = ChunkDispatcher(rmq_conn, bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass  # Expected — dispatcher processed the event and is polling for more

    # Consume EmbedTask from RabbitMQ queue
    task_dict = await _consume_one_rmq_message(queue, timeout=3.0)
    task = EmbedTask.model_validate(task_dict)

    assert task.task_type == "chunk", (
        f"EmbedTask.task_type should be 'chunk', got {task.task_type!r}"
    )
    assert task.post_id == 201, (
        f"EmbedTask.post_id should be 201, got {task.post_id}"
    )
    assert task.chunk_table == chunk_table, (
        f"EmbedTask.chunk_table should be {chunk_table!r}, got {task.chunk_table!r}"
    )
    assert task.model_name == _MODEL_NAME, (
        f"EmbedTask.model_name should be {_MODEL_NAME!r}, got {task.model_name!r}"
    )
    assert set(task.chunk_ids) == set(chunk_ids), (
        f"EmbedTask.chunk_ids should match the created chunk IDs. "
        f"Expected {sorted(chunk_ids)}, got {sorted(task.chunk_ids)}"
    )

    # Run EmbedHandler with mock model
    mock_model = _MockEmbeddingModel(dim=_VECTOR_DIM)
    handler = EmbedHandler(
        chunk_fetcher=chunk_repo,
        embedding_store=chunk_repo,
        event_log=bus,
    )
    ok_tasks, failed_tasks = await handler.embed_chunks([task], model_name=_MODEL_NAME, encoder=mock_model)

    assert len(failed_tasks) == 0, (
        f"EmbedHandler.embed_chunks should not fail any tasks, got {len(failed_tasks)} failures"
    )
    assert len(ok_tasks) == 1, (
        f"EmbedHandler.embed_chunks should return 1 ok task, got {len(ok_tasks)}"
    )

    # Verify embeddings were written to the chunk table
    async with postgres_pool.acquire() as conn:
        embedded_count = await conn.fetchval(
            f"SELECT COUNT(*) FROM {chunk_table} WHERE post_id = $1 AND embedding IS NOT NULL",
            201,
        )
    assert embedded_count == len(chunk_ids), (
        f"All {len(chunk_ids)} chunks should have embeddings written; "
        f"only {embedded_count} have non-NULL embedding"
    )

    # Consume embedding.completed event from the bus
    completed_event = await _consume_one_event(
        bus, "embedding.completed", "test.embed.pipeline.group", timeout=3.0
    )
    assert completed_event["post_id"] == 201, (
        f"embedding.completed post_id should be 201, got {completed_event['post_id']}"
    )
    assert completed_event["chunk_table"] == chunk_table, (
        f"embedding.completed chunk_table should be {chunk_table!r}, got {completed_event['chunk_table']!r}"
    )
    assert completed_event["model_name"] == _MODEL_NAME, (
        f"embedding.completed model_name should be {_MODEL_NAME!r}, got {completed_event['model_name']!r}"
    )
    assert set(completed_event["chunk_ids"]) == set(chunk_ids), (
        f"embedding.completed chunk_ids should match created chunks. "
        f"Expected {sorted(chunk_ids)}, got {sorted(completed_event['chunk_ids'])}"
    )

    await channel.close()


# ---------------------------------------------------------------------------
# EmbedTask carries correct fields from chunks.created event
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_task_fields_match_chunks_created_event(clean_pipeline_tables):
    """ChunkDispatcher must propagate post_id, chunk_ids, chunk_table, model_name correctly."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    post_repo: PostRepository = fixtures["post_repo"]
    post_table: str = fixtures["post_table"]
    chunk_table: str = fixtures["chunk_table"]
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]

    bus = await _setup_bus(postgres_pool)

    # Publish a hand-crafted chunks.created event directly to the bus
    fake_chunk_ids = ["aaaaaaaa-0000-0000-0000-000000000001", "aaaaaaaa-0000-0000-0000-000000000002"]
    chunks_event = make_chunks_created_event(
        post_id=202,
        chunk_ids=fake_chunk_ids,
        task_type="body",
        chunk_table=chunk_table,
        post_table=post_table,
    )
    await bus.publish("chunks.created", chunks_event)

    # Setup RabbitMQ queue for embed tasks
    channel = await rmq_conn.channel()
    embed_cfg = EMBED_CONFIGS["body"]
    exchange = await channel.declare_exchange("embedding", aio_pika.ExchangeType.TOPIC, durable=True)
    routing_key = f"gpu.embed.{embed_cfg.model}"
    queue = await channel.declare_queue(routing_key, exclusive=True, auto_delete=True)
    await queue.bind(exchange, routing_key=routing_key)

    # Run dispatcher
    dispatcher = ChunkDispatcher(rmq_conn, bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    task_dict = await _consume_one_rmq_message(queue, timeout=3.0)
    task = EmbedTask.model_validate(task_dict)

    assert task.post_id == 202, f"EmbedTask.post_id should be 202, got {task.post_id}"
    assert task.post_table == post_table, (
        f"EmbedTask.post_table should be {post_table!r}, got {task.post_table!r}"
    )
    assert task.chunk_table == chunk_table, (
        f"EmbedTask.chunk_table should be {chunk_table!r}, got {task.chunk_table!r}"
    )
    assert set(task.chunk_ids) == set(fake_chunk_ids), (
        f"EmbedTask.chunk_ids should equal the event's chunk_ids. "
        f"Expected {sorted(fake_chunk_ids)}, got {sorted(task.chunk_ids)}"
    )
    assert task.model_name == embed_cfg.model, (
        f"EmbedTask.model_name should be {embed_cfg.model!r} (from EMBED_CONFIGS['body']), "
        f"got {task.model_name!r}"
    )
    assert task.task_type == "chunk", (
        f"EmbedTask.task_type should always be 'chunk' for chunk pipeline, got {task.task_type!r}"
    )

    await channel.close()


# ---------------------------------------------------------------------------
# Idempotency: embedding twice overwrites vectors, emits one event per run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_handler_idempotency_second_run_overwrites_not_duplicates(clean_pipeline_tables):
    """Running EmbedHandler on the same chunks twice should overwrite vectors (not duplicate rows)
    and emit one embedding.completed event per run."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    post_repo: PostRepository = fixtures["post_repo"]
    post_table: str = fixtures["post_table"]
    chunk_table: str = fixtures["chunk_table"]

    bus = await _setup_bus(postgres_pool)

    chunk_repo = ChunkRepository(postgres_pool, table_name=chunk_table, vector_dim=_VECTOR_DIM)
    chunk_ids = await _insert_chunks_and_publish_event(
        post_repo, chunk_repo, bus, post_id=203, post_table=post_table,
    )
    # Drain the chunks.created event so it does not interfere with later assertions
    async with postgres_pool.acquire() as conn:
        _ = await conn.fetch("SELECT * FROM event_log WHERE topic = 'chunks.created'")

    task = EmbedTask(
        task_type="chunk",
        model_name=_MODEL_NAME,
        post_id=203,
        post_table=post_table,
        chunk_ids=chunk_ids,
        chunk_table=chunk_table,
    )

    mock_model = _MockEmbeddingModel(dim=_VECTOR_DIM)
    handler = EmbedHandler(
        chunk_fetcher=chunk_repo,
        embedding_store=chunk_repo,
        event_log=bus,
    )

    # First run
    ok_first, fail_first = await handler.embed_chunks([task], model_name=_MODEL_NAME, encoder=mock_model)
    assert len(fail_first) == 0, (
        f"First embed run should not fail; got {len(fail_first)} failures"
    )

    async with postgres_pool.acquire() as conn:
        count_first = await conn.fetchval(
            f"SELECT COUNT(*) FROM {chunk_table} WHERE post_id = $1 AND embedding IS NOT NULL",
            203,
        )

    # Second run (same task)
    ok_second, fail_second = await handler.embed_chunks([task], model_name=_MODEL_NAME, encoder=mock_model)
    assert len(fail_second) == 0, (
        f"Second embed run should not fail; got {len(fail_second)} failures"
    )

    async with postgres_pool.acquire() as conn:
        count_second = await conn.fetchval(
            f"SELECT COUNT(*) FROM {chunk_table} WHERE post_id = $1 AND embedding IS NOT NULL",
            203,
        )
        total_chunks = await conn.fetchval(
            f"SELECT COUNT(*) FROM {chunk_table} WHERE post_id = $1",
            203,
        )

    assert count_second == count_first, (
        f"Second run should not add new rows: count went from {count_first} to {count_second}"
    )
    assert count_second == total_chunks, (
        f"After two runs all {total_chunks} chunks should have embeddings, "
        f"but only {count_second} do"
    )

    # Verify two embedding.completed events were published (one per run)
    async with postgres_pool.acquire() as conn:
        event_count = await conn.fetchval(
            "SELECT COUNT(*) FROM event_log "
            "WHERE topic = 'embedding.completed' AND payload->>'post_id' = '203'"
        )
    assert event_count == 2, (
        f"Two embed runs should emit two embedding.completed events, got {event_count}"
    )


# ---------------------------------------------------------------------------
# Consumer offset advances after ChunkDispatcher processes chunks.created
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consumer_offset_advances_after_chunk_dispatcher_runs(clean_pipeline_tables):
    """After ChunkDispatcher processes chunks.created, consumer_offsets should be persisted."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    post_repo: PostRepository = fixtures["post_repo"]
    post_table: str = fixtures["post_table"]
    chunk_table: str = fixtures["chunk_table"]
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]

    bus = await _setup_bus(postgres_pool)

    chunk_repo = ChunkRepository(postgres_pool, table_name=chunk_table, vector_dim=_VECTOR_DIM)
    await _insert_chunks_and_publish_event(
        post_repo, chunk_repo, bus, post_id=204, post_table=post_table,
    )

    # Setup RabbitMQ so the dispatcher has somewhere to publish
    channel = await rmq_conn.channel()
    embed_cfg = EMBED_CONFIGS["body"]
    exchange = await channel.declare_exchange("embedding", aio_pika.ExchangeType.TOPIC, durable=True)
    routing_key = f"gpu.embed.{embed_cfg.model}"
    queue = await channel.declare_queue(routing_key, exclusive=True, auto_delete=True)
    await queue.bind(exchange, routing_key=routing_key)

    dispatcher = ChunkDispatcher(rmq_conn, bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    async with postgres_pool.acquire() as conn:
        offset = await conn.fetchval(
            """
            SELECT last_id FROM consumer_offsets
            WHERE consumer_group = $1 AND topic = $2
            """,
            consumer_groups.CHUNKS_CREATED,
            "chunks.created",
        )

    assert offset is not None, (
        "consumer_offsets should have a row for ChunkDispatcher after it processes chunks.created"
    )
    assert offset > 0, (
        f"consumer_offsets.last_id should be positive after processing, got {offset}"
    )

    await channel.close()
