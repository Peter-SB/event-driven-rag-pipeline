"""Unit tests for PostRepository.

Exercises the in-process logic with a mocked asyncpg pool — no database required.

High-risk behaviours covered
-----------------------------
- ensure_table issues CREATE TABLE and CREATE INDEX as separate execute() calls
  (asyncpg rejects multi-statement strings; this was the root cause of the
  'relation does not exist' error in e2e tests).
- ensure_table lowercases table names before executing SQL.
- upsert returns 'skipped' without executing the INSERT when updated_at is not newer.
- upsert returns 'inserted' when no existing row is found.
- upsert returns 'updated' when a newer updated_at is provided.
"""
from __future__ import annotations

from datetime import datetime, UTC, timedelta
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from event_driven_rag_service.repository.post_repository import PostRepository
from tests.utils.factories import make_post


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool(fetchval_return=None, fetchrow_return=None):
    """Return a mock asyncpg Pool whose acquire() context manager yields a mock conn."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=fetchval_return)
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)

    pool = MagicMock()
    # acquire() is used as an async context manager
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool, conn


# ---------------------------------------------------------------------------
# ensure_table — SQL statement separation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_table_calls_execute_twice():
    """ensure_table must issue exactly two execute() calls: CREATE TABLE + CREATE INDEX.

    asyncpg's execute() does not support semi-colon separated multi-statement strings;
    both statements must be sent individually.
    """
    pool, conn = _make_pool()
    repo = PostRepository(pool, table_name="posts_test")

    await repo.ensure_table()

    assert conn.execute.call_count == 2, (
        f"Expected 2 execute() calls (CREATE TABLE + CREATE INDEX), got {conn.execute.call_count}"
    )


@pytest.mark.asyncio
async def test_ensure_table_first_call_creates_table():
    """First execute() call must contain CREATE TABLE IF NOT EXISTS."""
    pool, conn = _make_pool()
    repo = PostRepository(pool, table_name="posts_test")

    await repo.ensure_table()

    first_sql: str = conn.execute.call_args_list[0].args[0]
    assert "CREATE TABLE IF NOT EXISTS" in first_sql.upper()


@pytest.mark.asyncio
async def test_ensure_table_second_call_creates_index():
    """Second execute() call must contain CREATE INDEX IF NOT EXISTS."""
    pool, conn = _make_pool()
    repo = PostRepository(pool, table_name="posts_test")

    await repo.ensure_table()

    second_sql: str = conn.execute.call_args_list[1].args[0]
    assert "CREATE INDEX IF NOT EXISTS" in second_sql.upper()


@pytest.mark.asyncio
async def test_ensure_table_uses_provided_table_name():
    """Table name passed to ensure_table() appears in both SQL statements."""
    pool, conn = _make_pool()
    repo = PostRepository(pool)

    await repo.ensure_table("posts_mylib")

    first_sql: str = conn.execute.call_args_list[0].args[0]
    second_sql: str = conn.execute.call_args_list[1].args[0]
    assert "posts_mylib" in first_sql
    assert "posts_mylib" in second_sql


@pytest.mark.asyncio
async def test_ensure_table_lowercases_table_name():
    """Table name is lowercased before being used in SQL."""
    pool, conn = _make_pool()
    repo = PostRepository(pool)

    await repo.ensure_table("Posts_Main")

    first_sql: str = conn.execute.call_args_list[0].args[0]
    assert "posts_main" in first_sql
    assert "Posts_Main" not in first_sql


@pytest.mark.asyncio
async def test_ensure_table_no_semicolons_in_individual_statements():
    """Neither SQL statement should contain a semicolon that would indicate bundled statements."""
    pool, conn = _make_pool()
    repo = PostRepository(pool, table_name="posts_test")

    await repo.ensure_table()

    for i, mock_call in enumerate(conn.execute.call_args_list):
        sql: str = mock_call.args[0].strip()
        # Strip trailing semicolons which are fine; check for mid-statement ones
        stripped = sql.rstrip(";")
        assert ";" not in stripped, (
            f"execute() call #{i + 1} contains an embedded semicolon suggesting "
            f"multi-statement bundling: {sql!r}"
        )


# ---------------------------------------------------------------------------
# upsert — freshness signal logic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_returns_inserted_when_no_existing_row():
    """When fetchval returns None (no row), upsert must return 'inserted'."""
    pool, conn = _make_pool(fetchval_return=None)
    repo = PostRepository(pool, table_name="posts_test")
    post = make_post(post_id=1)

    status, prior = await repo.upsert(post)

    assert status == "inserted"
    assert prior is None


@pytest.mark.asyncio
async def test_upsert_returns_skipped_when_existing_updated_at_is_equal():
    """When the stored updated_at equals the incoming value, upsert returns 'skipped'."""
    ts = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    pool, conn = _make_pool(fetchval_return=ts)
    repo = PostRepository(pool, table_name="posts_test")
    post = make_post(post_id=2, updated_at=ts)

    status, prior = await repo.upsert(post)

    assert status == "skipped"
    assert prior == ts


@pytest.mark.asyncio
async def test_upsert_returns_skipped_when_existing_updated_at_is_newer():
    """When the stored updated_at is newer than the incoming value, upsert returns 'skipped'."""
    stored_ts = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
    incoming_ts = stored_ts - timedelta(hours=1)
    pool, conn = _make_pool(fetchval_return=stored_ts)
    repo = PostRepository(pool, table_name="posts_test")
    post = make_post(post_id=3, updated_at=incoming_ts)

    status, prior = await repo.upsert(post)

    assert status == "skipped"


@pytest.mark.asyncio
async def test_upsert_does_not_execute_insert_when_skipped():
    """When result is 'skipped', no INSERT SQL must be executed against the database."""
    ts = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    pool, conn = _make_pool(fetchval_return=ts)
    repo = PostRepository(pool, table_name="posts_test")
    post = make_post(post_id=4, updated_at=ts)

    await repo.upsert(post)

    # Only fetchval was called; execute (the INSERT/UPDATE) must not have been called
    conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_returns_updated_when_incoming_is_newer():
    """When the stored updated_at is older than the incoming, upsert returns 'updated'."""
    stored_ts = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    newer_ts = stored_ts + timedelta(hours=1)
    pool, conn = _make_pool(fetchval_return=stored_ts)
    repo = PostRepository(pool, table_name="posts_test")
    post = make_post(post_id=5, updated_at=newer_ts)

    status, prior = await repo.upsert(post)

    assert status == "updated"
    assert prior == stored_ts


@pytest.mark.asyncio
async def test_upsert_executes_insert_when_row_is_new_or_updated():
    """When not skipped, conn.execute() must be called with the INSERT/UPSERT SQL."""
    pool, conn = _make_pool(fetchval_return=None)
    repo = PostRepository(pool, table_name="posts_test")
    post = make_post(post_id=6)

    await repo.upsert(post)

    conn.execute.assert_called_once()
    sql: str = conn.execute.call_args.args[0]
    assert "INSERT INTO" in sql.upper()
