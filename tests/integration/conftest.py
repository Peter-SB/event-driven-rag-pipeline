"""Integration test fixtures.

Starts real Postgres and RabbitMQ containers via testcontainers.  Each service
gets a random ephemeral host port assigned by the OS, so there is no risk of
clashing with a running Docker Compose stack (which binds fixed ports 5432 /
5672).  testcontainers returns the actual host port via get_connection_url().

Each test gets its own asyncpg pool on the test's own event loop (cheap to
create), avoiding ScopeMismatch with session-scoped async pools.

Requires Docker to be running.  testcontainers pulls images automatically on
first run.

Run only these tests with:
    pytest tests/integration/ -m integration
"""
from __future__ import annotations

from typing import AsyncGenerator

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer
from testcontainers.rabbitmq import RabbitMqContainer


# ---------------------------------------------------------------------------
# Session-scoped containers — Docker starts them once per pytest session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def postgres_testcontainer():
    """Postgres+pgvector container shared across the integration session."""
    with PostgresContainer("ankane/pgvector:latest") as container:
        yield container


@pytest.fixture(scope="session")
def rabbitmq_testcontainer():
    """RabbitMQ container shared across the integration session."""
    with RabbitMqContainer() as container:
        yield container


# ---------------------------------------------------------------------------
# Session-scoped connection URLs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def postgres_testcontainer_url(postgres_testcontainer: PostgresContainer) -> str:
    """asyncpg-compatible DSN for the Postgres testcontainer."""
    return postgres_testcontainer.get_connection_url().replace(
        "postgresql+psycopg2", "postgresql"
    )


@pytest.fixture(scope="session")
def rabbitmq_testcontainer_url(rabbitmq_testcontainer: RabbitMqContainer) -> str:
    """aio_pika-compatible AMQP URL for the RabbitMQ testcontainer.

    RabbitMqContainer only exposes get_connection_params() (pika-specific), so
    we build the URL ourselves from the container's host/port/credentials.
    """
    c = rabbitmq_testcontainer
    host = c.get_container_host_ip()
    port = c.get_exposed_port(c.port)
    vhost = c.vhost.lstrip("/") or ""
    return f"amqp://{c.username}:{c.password}@{host}:{port}/{vhost}"


# ---------------------------------------------------------------------------
# Function-scoped pool — created fresh per test on the test's own event loop
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def postgres_pool(postgres_testcontainer_url: str) -> AsyncGenerator[asyncpg.Pool, None]:
    """asyncpg pool for one test, connected to the shared testcontainer."""
    pool = await asyncpg.create_pool(postgres_testcontainer_url, min_size=1, max_size=3)
    yield pool
    await pool.close()


# ---------------------------------------------------------------------------
# Per-test table isolation helpers
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def clean_event_bus_tables(postgres_pool: asyncpg.Pool):
    """Drop event_log + consumer_offsets before each test so the bus is empty."""
    async with postgres_pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS consumer_offsets")
        await conn.execute("DROP TABLE IF EXISTS event_log")
    yield


@pytest_asyncio.fixture
async def clean_posts_table(postgres_pool: asyncpg.Pool):
    """Ensure a fresh posts table exists and is empty for the test."""
    from event_driven_rag_service.repository.post_repository import PostRepository
    repo = PostRepository(postgres_pool, table_name="test_posts")
    await repo.ensure_table()
    async with postgres_pool.acquire() as conn:
        await conn.execute("TRUNCATE test_posts")
    yield repo
    async with postgres_pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS test_posts")


@pytest_asyncio.fixture
async def clean_chunk_table(postgres_pool: asyncpg.Pool):
    """Create and empty a test chunk table (bge-base-v1.5 dim=768)."""
    from event_driven_rag_service.repository.chunk_repository import ChunkRepository
    table = "test_chunks_body_baai_bge_base_en_v1_5"
    repo = ChunkRepository(postgres_pool, table_name=table, vector_dim=768)
    await repo.ensure_table()
    async with postgres_pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {table}")
    yield repo
    async with postgres_pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {table}")


@pytest_asyncio.fixture
async def clean_pipeline_tables(
    postgres_pool: asyncpg.Pool,
    clean_event_bus_tables,
    rabbitmq_testcontainer_url: str,
):
    """Set up isolated posts + event bus + chunk table; yield live connections for pipeline tests."""
    import aio_pika

    from event_driven_rag_service.repository.post_repository import PostRepository
    from event_driven_rag_service.repository.chunk_repository import ChunkRepository

    POST_TABLE = "posts_pipeline_test"
    CHUNK_TABLE = "posts_pipeline_test_chunks_body_baai_bge_base_en_v1_5"

    post_repo = PostRepository(postgres_pool, table_name=POST_TABLE)
    await post_repo.ensure_table()

    async with postgres_pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {POST_TABLE}")

    rmq_conn = await aio_pika.connect_robust(rabbitmq_testcontainer_url)

    yield {
        "postgres_pool": postgres_pool,
        "post_repo": post_repo,
        "post_table": POST_TABLE,
        "chunk_table": CHUNK_TABLE,
        "rmq_conn": rmq_conn,
    }

    await rmq_conn.close()
    async with postgres_pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {CHUNK_TABLE}")
        await conn.execute(f"DROP TABLE IF EXISTS {POST_TABLE}")
