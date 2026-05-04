"""E2E test fixtures.

Spins up Postgres and RabbitMQ via testcontainers, then wires the FastAPI
app directly (bypassing the lifespan) with real infrastructure so tests can
make genuine HTTP calls through httpx's ASGI transport.

Design
------
Container lifecycle: session-scoped sync fixtures — Docker starts once per run.
Infrastructure lifecycle: function-scoped async fixtures — a fresh pool and
  RabbitMQ connection per test, matching the integration-test pattern.
App state: injected directly into ``app.state`` before each test rather than
  running the full lifespan (avoids an asgi-lifespan dependency while still
  exercising all route logic against live services).

Run with:
    pytest tests/e2e/ -m e2e

Docker must be running.  testcontainers pulls images automatically on first use.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
import asyncpg
import aio_pika
from httpx import AsyncClient, ASGITransport
from testcontainers.postgres import PostgresContainer
from testcontainers.rabbitmq import RabbitMqContainer

from event_driven_rag_service.infrastructure.event_bus import PostgresEventBus
from event_driven_rag_service.infrastructure.task_queue import setup_topology
from event_driven_rag_service.repository.post_repository import PostRepository


pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Session-scoped containers  (start once per pytest run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def postgres_container_e2e():
    """Start a Postgres+pgvector container for the e2e session."""
    with PostgresContainer("ankane/pgvector:latest") as container:
        yield container


@pytest.fixture(scope="session")
def rabbitmq_container_e2e():
    """Start a RabbitMQ container for the e2e session."""
    with RabbitMqContainer("rabbitmq:3-management") as container:
        yield container


@pytest.fixture(scope="session")
def postgres_dsn_e2e(postgres_container_e2e: PostgresContainer) -> str:
    return postgres_container_e2e.get_connection_url().replace(
        "postgresql+psycopg2", "postgresql"
    )


@pytest.fixture(scope="session")
def rabbitmq_url_e2e(rabbitmq_container_e2e: RabbitMqContainer) -> str:
    host = rabbitmq_container_e2e.get_container_host_ip()
    port = rabbitmq_container_e2e.get_exposed_port(5672)
    return f"amqp://guest:guest@{host}:{port}/"


# ---------------------------------------------------------------------------
# Function-scoped infrastructure  (fresh per test, on the test's event loop)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def postgres_pool_e2e(postgres_dsn_e2e: str):
    pool = await asyncpg.create_pool(postgres_dsn_e2e, min_size=1, max_size=3)
    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def rmq_connection_e2e(rabbitmq_url_e2e: str):
    conn = await aio_pika.connect_robust(rabbitmq_url_e2e)
    yield conn
    await conn.close()


# ---------------------------------------------------------------------------
# E2E HTTP client — injects live infrastructure into app.state
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def async_client(
    postgres_pool_e2e: asyncpg.Pool,
    rmq_connection_e2e: aio_pika.Connection,
):
    """httpx AsyncClient wired to the FastAPI ASGI app with real services.

    Sets up RabbitMQ topology, the Postgres event bus, and a dedicated
    ``e2e_posts`` table, then injects them into ``app.state`` so route
    handlers work exactly as in production — minus the lifespan startup.
    """
    from event_driven_rag_service.api.app import app

    # Declare RabbitMQ exchanges and queues
    async with rmq_connection_e2e.channel() as ch:
        await setup_topology(ch)

    # Prepare Postgres event bus
    event_bus = PostgresEventBus(postgres_pool_e2e)
    await event_bus.setup_tables()

    # Prepare post repository (isolated table for e2e tests)
    post_repo = PostRepository(postgres_pool_e2e, table_name="e2e_posts")
    await post_repo.ensure_table()

    # Inject state (route handlers read from request.app.state)
    app.state.pool = postgres_pool_e2e
    app.state.rmq = rmq_connection_e2e
    app.state.event_bus = event_bus
    app.state.post_repo = post_repo

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client

    # Clean up test tables so the next test starts from a blank state
    async with postgres_pool_e2e.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS e2e_posts")
        await conn.execute("DELETE FROM event_log WHERE topic = 'post.synced'")
        await conn.execute("DELETE FROM consumer_offsets WHERE topic = 'post.synced'")
