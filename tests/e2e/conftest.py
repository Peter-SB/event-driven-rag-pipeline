"""E2E test fixtures.

Connects to a running Docker Compose stack via environment variables.
All services must be up before these tests are invoked:

    docker compose up -d
    pytest tests/e2e/ -m e2e
    docker compose down

Connection defaults match the docker-compose.yml service addresses.
Override with environment variables if your stack uses different addresses.

Design
------
Infrastructure: function-scoped async fixtures connecting to real running
  services — no testcontainers, no ASGI transport.  Tests make genuine
  HTTP calls and query Postgres directly to verify pipeline state.

Cleanup: async_client runs a pre-test cleanup so the suite is repeatable
  without restarting the stack between runs.
"""
from __future__ import annotations

import asyncio
import os

import aio_pika
import asyncpg
import httpx
import pytest
import pytest_asyncio


pytestmark = pytest.mark.e2e

# Connection defaults match the docker-compose.yml service names and ports.
_DB_URL = os.getenv("DB_URL", "postgresql://rag:rag@localhost:5432/rag")
_RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://rag:rag@localhost:5672/")
_API_BASE = os.getenv("API_BASE", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Infrastructure connections
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def postgres_pool_e2e():
    """asyncpg pool connected to the Docker Compose Postgres instance."""
    pool = await asyncpg.create_pool(_DB_URL, min_size=1, max_size=3)
    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def rmq_connection_e2e():
    """aio_pika connection to the Docker Compose RabbitMQ instance."""
    conn = await aio_pika.connect_robust(_RABBITMQ_URL)
    yield conn
    await conn.close()


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

_EMBED_QUEUES = [
    "cpu.chunk.post",
    "gpu.embed.bge-base-en-v1.5",
    "gpu.embed.bge-small-en-v1.5",
    "gpu.embed.qwen3-0.6b",
    "cpu.search.run",
]


async def _purge_queues(rmq_url: str) -> None:
    """Purge all work queues so stale tasks from previous tests don't run."""
    conn = await aio_pika.connect_robust(rmq_url)
    try:
        async with conn.channel() as ch:
            for q in _EMBED_QUEUES:
                try:
                    queue = await ch.declare_queue(q, passive=True) # todo: what what passive=True does if queue doesn't exist? 
                    await queue.purge()
                except Exception:
                    pass  # Queue may not exist yet on first run
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def async_client(postgres_pool_e2e: asyncpg.Pool):
    """httpx AsyncClient pointed at the running API service.

    Makes real HTTP requests (not ASGI transport) so requests exercise the
    full network stack, real workers, and the lifespan startup sequence.

    Runs a pre-test cleanup of e2e tables and RabbitMQ queues so the suite
    is repeatable without restarting the Docker Compose stack between runs.
    """
    # Purge RabbitMQ queues FIRST so workers don't process stale tasks
    # that reference tables about to be dropped.
    await _purge_queues(_RABBITMQ_URL)

    async with postgres_pool_e2e.acquire() as conn:
        await conn.execute("DELETE FROM consumer_offsets")
        await conn.execute("DELETE FROM event_log")
        await conn.execute("DROP TABLE IF EXISTS posts_e2e")
        await conn.execute("DROP TABLE IF EXISTS posts_e2e_chunks_body_baai_bge_base_en_v1_5")
        await conn.execute("DROP TABLE IF EXISTS posts_e2e_chunks_title_baai_bge_small_en_v1_5")
        await conn.execute("DROP TABLE IF EXISTS posts_e2e_chunks_summary_title_qwen_qwen3_0_6b")

    async with httpx.AsyncClient(base_url=_API_BASE, timeout=30.0) as client:
        yield client

    # Post-test cleanup: purge queues first, then drop DB state
    await _purge_queues(_RABBITMQ_URL)
    async with postgres_pool_e2e.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS posts_e2e")
        await conn.execute("DROP TABLE IF EXISTS posts_e2e_chunks_body_baai_bge_base_en_v1_5")
        await conn.execute("DROP TABLE IF EXISTS posts_e2e_chunks_title_baai_bge_small_en_v1_5")
        await conn.execute("DROP TABLE IF EXISTS posts_e2e_chunks_summary_title_qwen_qwen3_0_6b")
        await conn.execute("DELETE FROM event_log")
        await conn.execute("DELETE FROM consumer_offsets")
