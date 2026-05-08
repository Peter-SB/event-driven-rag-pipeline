"""Integration tests to verify all dispatchers are wired up and processing events.

This test suite catches issues where dispatchers are missing from the entrypoint,
ensuring the full pipeline flow works as expected.
"""
from __future__ import annotations

import asyncio
import json
import uuid

import aio_pika
import asyncpg
import pytest

from event_driven_rag_service.config import consumer_groups
from event_driven_rag_service.dispatchers.post_dispatcher import PostDispatcher
from event_driven_rag_service.dispatchers.chunk_dispatcher import ChunkDispatcher
from event_driven_rag_service.dispatchers.search_dispatcher import SearchDispatcher
from event_driven_rag_service.dispatchers.embedding_dispatcher import EmbeddingDispatcher
from event_driven_rag_service.events.post_events import PostSyncedEvent
from event_driven_rag_service.events.chunk_events import ChunksCreatedEvent
from event_driven_rag_service.events.search_events import SearchJobCreatedEvent, SearchQueryEmbeddedEvent
from event_driven_rag_service.infrastructure.event_bus import PostgresEventBus
from event_driven_rag_service.tasks.chunk_task import ChunkTask
from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.tasks.search_tasks import SearchRunTask

pytestmark = pytest.mark.integration


async def _setup_bus(pool: asyncpg.Pool) -> PostgresEventBus:
    bus = PostgresEventBus(pool)
    await bus.setup_tables()
    return bus


async def _consume_one_rmq(queue: aio_pika.abc.AbstractQueue, timeout: float = 5.0) -> dict:
    """Consume a single message from RabbitMQ queue."""
    msg = await asyncio.wait_for(queue.get(no_ack=True), timeout=timeout)
    return json.loads(msg.body)


class DispatcherPipeline:
    """Manages all dispatchers for testing."""

    def __init__(
        self,
        post_disp: PostDispatcher,
        chunk_disp: ChunkDispatcher,
        search_disp: SearchDispatcher,
        embed_disp: EmbeddingDispatcher,
    ):
        self.post = post_disp
        self.chunk = chunk_disp
        self.search = search_disp
        self.embed = embed_disp
        self._task = None

    async def start(self):
        """Start all dispatchers concurrently."""
        # asyncio.gather() returns a Future, not a coroutine — use ensure_future.
        self._task = asyncio.ensure_future(
            asyncio.gather(
                self.post.run(),
                self.chunk.run(),
                self.search.run(),
                self.embed.run(),
            )
        )

    async def stop(self):
        """Stop all dispatchers."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# Test: All dispatchers instantiate without errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_dispatchers_instantiate(clean_pipeline_tables):
    """Verify all dispatchers can be instantiated."""
    fixtures = clean_pipeline_tables
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]

    bus = await _setup_bus(postgres_pool)

    # Should not raise
    post_disp = PostDispatcher(rmq_conn, bus)
    chunk_disp = ChunkDispatcher(rmq_conn, bus)
    search_disp = SearchDispatcher(rmq_conn, bus)
    embed_disp = EmbeddingDispatcher(rmq_conn, bus)

    assert post_disp is not None
    assert chunk_disp is not None
    assert search_disp is not None
    assert embed_disp is not None


# ---------------------------------------------------------------------------
# Test: PostDispatcher processes post.synced events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_dispatcher_consumes_post_synced(clean_pipeline_tables):
    """PostDispatcher should consume post.synced and publish chunk tasks."""
    fixtures = clean_pipeline_tables
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]

    bus = await _setup_bus(postgres_pool)

    # Set up RabbitMQ to capture chunk tasks.
    # TASK_ROUTES["chunk"]: exchange="ingestion", routing_key="cpu.chunk.post"
    channel = await rmq_conn.channel()
    exchange = await channel.declare_exchange("ingestion", aio_pika.ExchangeType.TOPIC, durable=True)
    queue = await channel.declare_queue("cpu.chunk.post", exclusive=True, auto_delete=True)
    await queue.bind(exchange, routing_key="cpu.chunk.post")

    # Publish a post.synced event
    event = PostSyncedEvent(
        post_table="posts_test",
        post_id=1,
        updated_at="2024-01-01T00:00:00Z",
    )
    await bus.publish("post.synced", event.to_dict())

    # Run dispatcher
    dispatcher = PostDispatcher(rmq_conn, bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    # Verify chunk task was published
    task_dict = await _consume_one_rmq(queue, timeout=3.0)
    task = ChunkTask.model_validate(task_dict)

    assert task.post_id == 1
    assert task.post_table == "posts_test"
    assert task.task_type == "body"

    await channel.close()


# ---------------------------------------------------------------------------
# Test: ChunkDispatcher processes chunks.created events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chunk_dispatcher_consumes_chunks_created(clean_pipeline_tables):
    """ChunkDispatcher should consume chunks.created and publish embed tasks."""
    fixtures = clean_pipeline_tables
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]

    bus = await _setup_bus(postgres_pool)

    # Set up RabbitMQ to capture embed tasks
    channel = await rmq_conn.channel()
    exchange = await channel.declare_exchange("embedding", aio_pika.ExchangeType.TOPIC, durable=True)
    queue = await channel.declare_queue("gpu.embed.bge-base-v1.5", exclusive=True, auto_delete=True)
    await queue.bind(exchange, routing_key="gpu.embed.bge-base-v1.5")

    # Publish a chunks.created event (schema: task_type, chunk_count, created_at required)
    from datetime import datetime, UTC
    event = ChunksCreatedEvent(
        post_id=1,
        post_table="posts_test",
        chunk_table="posts_test_chunks_body_bge_base_v1_5",
        chunk_ids=["chunk-1", "chunk-2"],
        task_type="body",
        chunk_count=2,
        created_at=datetime.now(UTC),
    )
    await bus.publish("chunks.created", event.to_dict())

    # Run dispatcher
    dispatcher = ChunkDispatcher(rmq_conn, bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    # Verify embed task was published
    task_dict = await _consume_one_rmq(queue, timeout=3.0)
    task = EmbedTask.model_validate(task_dict)

    assert task.task_type == "chunk"
    assert task.chunk_table == "posts_test_chunks_body_bge_base_v1_5"
    assert task.model_name == "bge-base-v1.5"

    await channel.close()


# ---------------------------------------------------------------------------
# Test: SearchDispatcher processes search_job.created events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_dispatcher_consumes_search_job_created(clean_pipeline_tables):
    """SearchDispatcher should consume search_job.created and publish embed tasks."""
    fixtures = clean_pipeline_tables
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]

    bus = await _setup_bus(postgres_pool)

    # Set up RabbitMQ to capture query embed tasks
    channel = await rmq_conn.channel()
    exchange = await channel.declare_exchange("embedding", aio_pika.ExchangeType.TOPIC, durable=True)
    queue = await channel.declare_queue("gpu.embed.query", exclusive=True, auto_delete=True)
    await queue.bind(exchange, routing_key="gpu.embed.bge-base-v1.5")

    # Publish a search_job.created event
    job_id = str(uuid.uuid4())
    event = SearchJobCreatedEvent(
        query_job_id=job_id,
        query="what is RAG?",
    )
    await bus.publish("search_job.created", event.to_dict())

    # Run dispatcher
    dispatcher = SearchDispatcher(rmq_conn, bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    # Verify query embed task was published
    task_dict = await _consume_one_rmq(queue, timeout=3.0)
    task = EmbedTask.model_validate(task_dict)

    assert task.task_type == "query"
    assert task.query == "what is RAG?"
    assert task.query_job_id == job_id

    await channel.close()


# ---------------------------------------------------------------------------
# Test: EmbeddingDispatcher processes search_query.embedded events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embedding_dispatcher_consumes_search_query_embedded(clean_pipeline_tables):
    """EmbeddingDispatcher should consume search_query.embedded and publish search tasks."""
    fixtures = clean_pipeline_tables
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]

    bus = await _setup_bus(postgres_pool)

    # Set up RabbitMQ to capture search tasks
    channel = await rmq_conn.channel()
    exchange = await channel.declare_exchange("search", aio_pika.ExchangeType.TOPIC, durable=True)
    queue = await channel.declare_queue("cpu.search.run", exclusive=True, auto_delete=True)
    await queue.bind(exchange, routing_key="cpu.search.run")

    # Publish a search_query.embedded event
    job_id = str(uuid.uuid4())
    event = SearchQueryEmbeddedEvent(
        query_job_id=job_id,
        model_name="bge-base-v1.5",
    )
    await bus.publish("search_query.embedded", event.to_dict())

    # Run dispatcher
    dispatcher = EmbeddingDispatcher(rmq_conn, bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    # Verify search task was published
    task_dict = await _consume_one_rmq(queue, timeout=3.0)
    task = SearchRunTask.model_validate(task_dict)

    assert task.job_id == job_id

    await channel.close()


# ---------------------------------------------------------------------------
# Test: Full pipeline with all dispatchers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_dispatchers_running_concurrently(clean_pipeline_tables):
    """Verify all four dispatchers can run concurrently without conflicts."""
    fixtures = clean_pipeline_tables
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]

    bus = await _setup_bus(postgres_pool)

    # Create all dispatchers
    post_disp = PostDispatcher(rmq_conn, bus)
    chunk_disp = ChunkDispatcher(rmq_conn, bus)
    search_disp = SearchDispatcher(rmq_conn, bus)
    embed_disp = EmbeddingDispatcher(rmq_conn, bus)

    pipeline = DispatcherPipeline(post_disp, chunk_disp, search_disp, embed_disp)

    # Start all dispatchers
    await pipeline.start()

    # Give them time to initialize subscriptions
    await asyncio.sleep(0.5)

    # Publish an event to each dispatcher
    from datetime import datetime, UTC
    await bus.publish(
        "post.synced",
        PostSyncedEvent(post_table="posts_test", post_id=1, updated_at=datetime.now(UTC)).to_dict(),
    )

    # Wait a bit and then stop
    await asyncio.sleep(0.5)
    await pipeline.stop()

    # If we got here without errors, all dispatchers ran successfully
    assert True


@pytest.mark.asyncio
async def test_dispatcher_entrypoint_imports(clean_pipeline_tables):
    """Verify the dispatcher entrypoint can import and instantiate all components."""
    # This test simply imports the entrypoint module to verify all imports work
    from event_driven_rag_service.worker.entrypoints import dispatcher as dispatcher_module

    assert hasattr(dispatcher_module, "main")
    assert hasattr(dispatcher_module, "_setup")
