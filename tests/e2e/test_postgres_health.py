"""Postgres health checks: connectivity, pgvector extension, basic DDL/DML.

Simple smoke test that runs first, fails fast if infrastructure is broken.
"""
import pytest
import asyncpg

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_postgres_is_running(postgres_pool_e2e: asyncpg.Pool):
    """Verify Postgres is reachable and responds."""
    async with postgres_pool_e2e.acquire() as conn:
        result = await conn.fetchval("SELECT 42")
        assert result == 42


@pytest.mark.asyncio
async def test_pgvector_extension_available(postgres_pool_e2e: asyncpg.Pool):
    """Verify pgvector extension is installed."""
    async with postgres_pool_e2e.acquire() as conn:
        result = await conn.fetchval("CREATE EXTENSION IF NOT EXISTS vector")
        # If extension exists, CREATE returns None (idempotent)
        # If installed but schema exists, still OK

        # Verify we can use vector type
        await conn.execute("""
            CREATE TEMP TABLE vector_test (id int, v vector(3))
        """)
        # Temp table cleaned up automatically on disconnect


@pytest.mark.asyncio
async def test_can_create_and_drop_table(postgres_pool_e2e: asyncpg.Pool):
    """Verify basic DDL operations work."""
    async with postgres_pool_e2e.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS test_health_check (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        result = await conn.fetchval("""
            SELECT COUNT(*)::int FROM information_schema.tables
            WHERE table_name = 'test_health_check'
        """)
        assert result == 1, "Test table should exist"

        await conn.execute("DROP TABLE test_health_check")

        result = await conn.fetchval("""
            SELECT COUNT(*)::int FROM information_schema.tables
            WHERE table_name = 'test_health_check'
        """)
        assert result == 0, "Test table should be dropped"
