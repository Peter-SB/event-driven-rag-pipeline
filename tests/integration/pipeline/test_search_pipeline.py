"""Integration tests for the full search pipeline.

Flow under test
---------------
    SearchDispatcher.run()    (search_job.created → EmbedTask on RabbitMQ)
    EmbeddingDispatcher.run() (search_query.embedded → SearchRunTask on RabbitMQ)
    EmbedHandler.embed_query  (encode query, save to SearchJobRepository, emit search_query.embedded)
    SearchHandler.handle()    (read embedding, run ANN search, store results)

Real infrastructure: Postgres testcontainer, RabbitMQ testcontainer.
Mock embedding model: deterministic 768-dim vectors (no GPU needed).
"""
from __future__ import annotations

import asyncio
import json
import uuid

import aio_pika
import asyncpg
import pytest

from event_driven_rag_service.config import consumer_groups
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.dispatchers.search_dispatcher import SearchDispatcher
from event_driven_rag_service.dispatchers.embedding_dispatcher import EmbeddingDispatcher
from event_driven_rag_service.events.search_events import SearchJobCreatedEvent
from event_driven_rag_service.handlers.embed_handler import EmbedHandler
from event_driven_rag_service.handlers.search_handler import SearchHandler
from event_driven_rag_service.infrastructure.event_bus import PostgresEventBus
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.repository.search_job_repository import SearchJobRepository
from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.tasks.search_tasks import SearchRunTask
from tests.utils.factories import make_post, make_chunks_created_event

pytestmark = pytest.mark.integration

_MODEL_NAME = "bge-base-v1.5"
_VECTOR_DIM = 768
_POST_TABLE = "posts_pipeline_search_test"
_CHUNK_TABLE = f"{_POST_TABLE}_chunks_body_bge_base_v1_5"


# ---------------------------------------------------------------------------
# Mock embedding model
# ---------------------------------------------------------------------------

class _MockEmbeddingModel:
    @property
    def name(self) -> str:
        return _MODEL_NAME

    def encode(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            seed = len(text) % _VECTOR_DIM
            vec = [0.0] * _VECTOR_DIM
            vec[seed] = 1.0
            results.append(vec)
        return results


# ---------------------------------------------------------------------------
# Composite embedding store (same as production gpu.py)
# ---------------------------------------------------------------------------

class _CompositeStore:
    def __init__(self, chunk_repo: ChunkRepository, job_repo: SearchJobRepository) -> None:
        self._chunks = chunk_repo
        self._jobs = job_repo

    async def save_batch(self, rows: list) -> None:
        chunk_rows = [r for r in rows if "chunk_id" in r]
        query_rows = [r for r in rows if "query_job_id" in r]
        if chunk_rows:
            await self._chunks.save_batch(chunk_rows)
        for row in query_rows:
            await self._jobs.store_embedding(row["query_job_id"], row["embedding"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _setup_bus(pool: asyncpg.Pool) -> PostgresEventBus:
    bus = PostgresEventBus(pool)
    await bus.setup_tables()
    return bus


async def _consume_one_rmq(queue: aio_pika.abc.AbstractQueue, timeout: float = 5.0) -> dict:
    msg = await asyncio.wait_for(queue.get(no_ack=True), timeout=timeout)
    return json.loads(msg.body)


async def _consume_one_event(bus: PostgresEventBus, topic: str, group: str, timeout: float = 5.0) -> dict:
    async def _read():
        async for event in bus.subscribe(topic, consumer_group=group):
            return event
        raise RuntimeError(f"No event on {topic!r}")
    return await asyncio.wait_for(_read(), timeout=timeout)


async def _seed_chunks_with_embeddings(
    pool: asyncpg.Pool,
    chunk_repo: ChunkRepository,
    post_table: str,
    chunk_table: str,
    n_posts: int = 3,
) -> list[str]:
    """Insert posts + chunks + embeddings directly. Returns chunk_ids."""
    from event_driven_rag_service.repository.post_repository import PostRepository
    from event_driven_rag_service.handlers.chunk_handler import ChunkPostHandler
    from event_driven_rag_service.tasks.chunk_task import ChunkTask
    from tests.utils.factories import FakeEventBus

    post_repo = PostRepository(pool, table_name=post_table)
    await post_repo.ensure_table()

    bus = FakeEventBus()
    handler = ChunkPostHandler(
        post_fetcher=post_repo,
        chunk_store=chunk_repo,
        version_checker=chunk_repo,
        event_log=bus,
    )

    all_chunk_ids: list[str] = []
    model = _MockEmbeddingModel()

    for i in range(1, n_posts + 1):
        post = make_post(
            post_id=i,
            body_text=f"This is post number {i}. " * 30,
            title=f"Post {i}",
        )
        await post_repo.upsert(post, post_table)
        task = ChunkTask(task_type="body", post_id=i, post_table=post_table, embed_model=_MODEL_NAME)
        chunk_ids = await handler.handle(task)
        all_chunk_ids.extend(chunk_ids)

    # Embed all chunks
    pairs = await chunk_repo.fetch_texts(all_chunk_ids, chunk_table)
    texts = [t for _, t in pairs]
    vectors = model.encode(texts)
    embed_rows = [
        {"chunk_id": cid, "model_name": _MODEL_NAME, "embedding": v, "chunk_table": chunk_table}
        for (cid, _), v in zip(pairs, vectors)
    ]
    await chunk_repo.save_batch(embed_rows)

    return all_chunk_ids


# ---------------------------------------------------------------------------
# Test: SearchDispatcher publishes EmbedTask for query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_dispatcher_publishes_embed_task_for_query(clean_pipeline_tables):
    """search_job.created → SearchDispatcher → EmbedTask on gpu.embed.{model}."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]

    bus = await _setup_bus(postgres_pool)

    # Declare embedding exchange + queue (exclusive, auto-delete to avoid test pollution)
    channel = await rmq_conn.channel()
    exchange = await channel.declare_exchange("embedding", aio_pika.ExchangeType.TOPIC, durable=True)
    routing_key = f"gpu.embed.{_MODEL_NAME}"
    queue = await channel.declare_queue(routing_key, exclusive=True, auto_delete=True)
    await queue.bind(exchange, routing_key=routing_key)

    # Publish search_job.created with a query
    job_id = str(uuid.uuid4())
    event = SearchJobCreatedEvent(query_job_id=job_id, query="what is retrieval augmented generation?")
    await bus.publish("search_job.created", event.to_dict())

    # Run SearchDispatcher
    dispatcher = SearchDispatcher(rmq_conn, bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    task_dict = await _consume_one_rmq(queue, timeout=3.0)
    task = EmbedTask.model_validate(task_dict)
    
    assert task.kind == "embed", "Expected task kind 'embed', got %r" % task.kind
    assert task.task_type == "query", "Expected task_type 'query', got %r" % task.task_type
    assert task.query == "what is retrieval augmented generation?"
    assert task.query_job_id == job_id
    assert task.model_name == _MODEL_NAME

    await channel.close()


# ---------------------------------------------------------------------------
# Test: Full search pipeline — query embedded → search run → results stored
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_search_pipeline_query_to_results(clean_pipeline_tables):
    """Full flow: embed query → store in job → search chunks → results persisted."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    post_table: str = fixtures["post_table"]
    chunk_table: str = fixtures["chunk_table"]
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]

    bus = await _setup_bus(postgres_pool)

    # Repositories
    chunk_repo = ChunkRepository(postgres_pool, table_name=chunk_table, vector_dim=_VECTOR_DIM)
    job_repo = SearchJobRepository(postgres_pool)
    await job_repo.ensure_table()

    # Seed chunk data with embeddings
    await _seed_chunks_with_embeddings(postgres_pool, chunk_repo, post_table, chunk_table)

    # Create a search job
    job_id = await job_repo.create_job(
        query="post number 2",
        k=3,
        embedding_profile=_MODEL_NAME,
        chunks_table=chunk_table,
        library_id="pipeline_search_test",
    )

    # Embed the query using the mock model and store via composite store
    composite = _CompositeStore(chunk_repo, job_repo)
    query_row = {
        "query_job_id": job_id,
        "model_name": _MODEL_NAME,
        "embedding": _MockEmbeddingModel().encode(["post number 2"])[0],
    }
    await composite.save_batch([query_row])

    # Verify embedding was stored
    job = await job_repo.get_job(job_id)
    assert job["embedding"] is not None, "Query embedding should be persisted in search_jobs"
    assert len(job["embedding"]) == _VECTOR_DIM

    # Run SearchHandler
    handler = SearchHandler(
        job_store=job_repo,
        chunk_searcher=chunk_repo,
        event_log=bus,
    )
    task = SearchRunTask(job_id=job_id)
    await handler.handle(task)

    # Verify results were stored
    job = await job_repo.get_job(job_id)
    assert job["status"] == "complete", f"Expected 'complete', got {job['status']!r}"
    assert job["results"] is not None
    assert len(job["results"]) <= 3  # at most k=3
    assert len(job["results"]) > 0, "Expected at least one search result"

    for result in job["results"]:
        assert "chunk_id" in result
        assert "post_id" in result
        assert "text" in result
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0

    # Verify search_job.completed event was published
    completed_event = await _consume_one_event(
        bus, "search_job.completed", "test.search.pipeline.group", timeout=3.0
    )
    assert completed_event["query_job_id"] == job_id
    assert completed_event["event_type"] == "search_job.completed"


# ---------------------------------------------------------------------------
# Test: EmbeddingDispatcher publishes SearchRunTask on search_query.embedded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embedding_dispatcher_publishes_search_run_task(clean_pipeline_tables):
    """search_query.embedded → EmbeddingDispatcher → SearchRunTask on cpu.search.run."""
    fixtures = clean_pipeline_tables
    postgres_pool: asyncpg.Pool = fixtures["postgres_pool"]
    rmq_conn: aio_pika.Connection = fixtures["rmq_conn"]

    bus = await _setup_bus(postgres_pool)

    # Declare search exchange + queue (exclusive, auto-delete to avoid test pollution)
    channel = await rmq_conn.channel()
    exchange = await channel.declare_exchange("search", aio_pika.ExchangeType.TOPIC, durable=True)
    queue = await channel.declare_queue("cpu.search.run", exclusive=True, auto_delete=True)
    await queue.bind(exchange, routing_key="cpu.search.run")

    # Publish search_query.embedded
    job_id = str(uuid.uuid4())
    from event_driven_rag_service.events.search_events import SearchQueryEmbeddedEvent
    event = SearchQueryEmbeddedEvent(query_job_id=job_id, model_name=_MODEL_NAME)
    await bus.publish("search_query.embedded", event.to_dict())

    # Run EmbeddingDispatcher
    dispatcher = EmbeddingDispatcher(rmq_conn, bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    task_dict = await _consume_one_rmq(queue, timeout=3.0)
    task = SearchRunTask.model_validate(task_dict)

    assert task.kind == "search_run"
    assert task.job_id == job_id

    await channel.close()
