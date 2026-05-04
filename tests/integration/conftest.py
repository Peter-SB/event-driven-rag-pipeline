"""Integration test fixtures.

Provides a real Postgres database via testcontainers.  The container is
shared across the entire test session (expensive to start), while each test
gets its own asyncpg pool bound to the test's own event loop (cheap to
create).  This avoids the ScopeMismatch that occurs when a session-scoped
async pool is used from a function-scoped event loop.

Requires Docker to be running.  testcontainers pulls the image automatically
on first run.

Run only these tests with:
    pytest tests/integration/ -m integration
"""
from __future__ import annotations

from typing import AsyncGenerator

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer


# ---------------------------------------------------------------------------
# Session-scoped container  (Docker starts once per pytest run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def postgres_container():
    """Start a Postgres+pgvector container once per session."""
    with PostgresContainer("ankane/pgvector:latest") as container:
        yield container


# Expose the raw DSN as a session-scoped *sync* fixture so async fixtures
# at any scope can consume it without loop-scope conflicts.
@pytest.fixture(scope="session")
def postgres_dsn(postgres_container: PostgresContainer) -> str:
    """Return the plain asyncpg-compatible DSN for the test container."""
    return postgres_container.get_connection_url().replace(
        "postgresql+psycopg2", "postgresql"
    )


# ---------------------------------------------------------------------------
# Function-scoped pool  (created fresh per test, on the test's event loop)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def postgres_pool(postgres_dsn: str) -> AsyncGenerator[asyncpg.Pool, None]:
    """asyncpg pool for one test.  Created and closed on the test event loop."""
    pool = await asyncpg.create_pool(postgres_dsn, min_size=1, max_size=3)
    yield pool
    await pool.close()


# ---------------------------------------------------------------------------
# Per-test table isolation helpers
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def clean_event_bus_tables(postgres_pool: asyncpg.Pool):
    """Drop and recreate event_log + consumer_offsets before each test."""
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
    table = "test_chunks_body_bge_base_v1_5"
    repo = ChunkRepository(postgres_pool, table_name=table, vector_dim=768)
    await repo.ensure_table()
    async with postgres_pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {table}")
    yield repo
    async with postgres_pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {table}")
