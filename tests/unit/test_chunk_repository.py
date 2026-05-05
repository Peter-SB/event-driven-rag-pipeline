"""Unit tests for ChunkRepository.

Exercises the in-process logic with a mocked asyncpg pool — no database required.

High-risk behaviours covered
-----------------------------
- ensure_table issues CREATE EXTENSION, CREATE TABLE, and CREATE INDEX as separate
  execute() calls (asyncpg rejects multi-statement strings).
- ensure_table caches seen table names and skips subsequent calls for the same table.
- ensure_table lowercases table names before executing SQL.
- ensure_table with different table names both execute (no false-positive cache hit).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from event_driven_rag_service.repository.chunk_repository import ChunkRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool():
    """Return a mock asyncpg Pool whose acquire() context manager yields a mock conn."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])   # used by read methods

    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool, conn


# ---------------------------------------------------------------------------
# ensure_table — SQL statement separation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chunk_ensure_table_calls_execute_four_times():
    """ensure_table must issue exactly 4 separate execute() calls:
    CREATE EXTENSION + CREATE TABLE + 2 CREATE INDEX statements.
    """
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, table_name="test_chunks", vector_dim=768)

    await repo.ensure_table()

    assert conn.execute.call_count == 4, (
        f"Expected 4 execute() calls, got {conn.execute.call_count}"
    )


@pytest.mark.asyncio
async def test_chunk_ensure_table_creates_extension_first():
    """First execute() call must create the vector extension."""
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, table_name="test_chunks", vector_dim=768)

    await repo.ensure_table()

    first_sql: str = conn.execute.call_args_list[0].args[0]
    assert "create extension if not exists vector" in first_sql.lower()


@pytest.mark.asyncio
async def test_chunk_ensure_table_creates_table_second():
    """Second execute() call must contain CREATE TABLE IF NOT EXISTS."""
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, table_name="test_chunks", vector_dim=768)

    await repo.ensure_table()

    second_sql: str = conn.execute.call_args_list[1].args[0]
    assert "CREATE TABLE IF NOT EXISTS" in second_sql.upper()


@pytest.mark.asyncio
async def test_chunk_ensure_table_creates_indexes_after_table():
    """Third and fourth execute() calls must both contain CREATE INDEX IF NOT EXISTS."""
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, table_name="test_chunks", vector_dim=768)

    await repo.ensure_table()

    for i in [2, 3]:
        sql: str = conn.execute.call_args_list[i].args[0]
        assert "CREATE INDEX IF NOT EXISTS" in sql.upper(), (
            f"Call #{i + 1} is not a CREATE INDEX statement: {sql!r}"
        )


@pytest.mark.asyncio
async def test_chunk_ensure_table_embeds_vector_dim_in_table_sql():
    """The CREATE TABLE SQL must include the configured vector dimension."""
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, table_name="test_chunks", vector_dim=384)

    await repo.ensure_table()

    table_sql: str = conn.execute.call_args_list[1].args[0]
    assert "384" in table_sql


@pytest.mark.asyncio
async def test_chunk_ensure_table_uses_provided_table_name_in_all_sqls():
    """The table name appears in the CREATE TABLE and both CREATE INDEX statements."""
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, table_name="posts_mylib_chunks_body_bge", vector_dim=768)

    await repo.ensure_table()

    for i in [1, 2, 3]:
        sql: str = conn.execute.call_args_list[i].args[0]
        assert "posts_mylib_chunks_body_bge" in sql


@pytest.mark.asyncio
async def test_chunk_ensure_table_lowercases_table_name():
    """Table names are lowercased before being used in SQL statements."""
    pool, conn = _make_pool()
    repo = ChunkRepository(pool)

    await repo.ensure_table("Test_Chunks_Body", vector_dim=768)

    for i in [1, 2, 3]:
        sql: str = conn.execute.call_args_list[i].args[0]
        assert "test_chunks_body" in sql
        assert "Test_Chunks_Body" not in sql


# ---------------------------------------------------------------------------
# ensure_table — seen_tables cache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chunk_ensure_table_second_call_same_table_runs_again():
    """Calling ensure_table twice always runs SQL — CREATE TABLE IF NOT EXISTS is idempotent."""
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, table_name="test_chunks", vector_dim=768)

    await repo.ensure_table()
    first_call_count = conn.execute.call_count  # 4 calls

    await repo.ensure_table()  # also runs SQL (idempotent, tolerates dropped tables)

    assert conn.execute.call_count == first_call_count * 2, (
        "Second call to ensure_table must re-execute SQL (no stale in-memory cache)"
    )


@pytest.mark.asyncio
async def test_chunk_ensure_table_different_tables_both_execute():
    """ensure_table for two different table names must both execute SQL (no false cache hit)."""
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, vector_dim=768)

    await repo.ensure_table("table_alpha")
    after_first = conn.execute.call_count  # 4

    await repo.ensure_table("table_beta")
    after_second = conn.execute.call_count  # 8

    assert after_first == 4
    assert after_second == 8


@pytest.mark.asyncio
async def test_chunk_ensure_table_cache_is_per_repo_instance():
    """Each ChunkRepository instance has its own seen_tables cache."""
    pool, conn = _make_pool()

    repo1 = ChunkRepository(pool, table_name="test_chunks", vector_dim=768)
    repo2 = ChunkRepository(pool, table_name="test_chunks", vector_dim=768)

    await repo1.ensure_table()
    after_repo1 = conn.execute.call_count  # 4

    await repo2.ensure_table()  # separate instance — should execute again
    after_repo2 = conn.execute.call_count  # 8

    assert after_repo1 == 4
    assert after_repo2 == 8, "Second repo instance must not share cache with first"


# ---------------------------------------------------------------------------
# Auto-ensure on reads — bound repos (vector_dim set) call ensure_table first
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_text_hashes_calls_ensure_table_when_bound():
    """get_text_hashes on a bound repo must call ensure_table before the SELECT.

    A bound repo (vector_dim set) auto-creates the table so callers can read
    without having to manually call ensure_table first.
    """
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, table_name="test_chunks", vector_dim=768)

    await repo.get_text_hashes(post_id=1)

    # ensure_table issues 4 execute() calls; the SELECT goes through conn.fetch
    assert conn.execute.call_count == 4, (
        "Expected 4 execute() calls from ensure_table before the SELECT"
    )
    first_sql = conn.execute.call_args_list[0].args[0].lower()
    assert "create extension" in first_sql
    conn.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_get_text_hashes_skips_ensure_when_vector_dim_not_set():
    """An unbound repo (no vector_dim) must NOT call ensure_table on reads.

    Unbound repos are used by workers that supply table_name per call; the
    caller is responsible for table creation.
    """
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, table_name="test_chunks")  # no vector_dim

    await repo.get_text_hashes(post_id=1)

    conn.execute.assert_not_called()
    conn.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_get_text_hashes_always_runs_ensure_table():
    """ensure_table runs on every read to tolerate tables dropped externally."""
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, table_name="test_chunks", vector_dim=768)

    await repo.get_text_hashes(post_id=1)
    after_first = conn.execute.call_count  # 4 from ensure_table

    await repo.get_text_hashes(post_id=2)

    # ensure_table re-runs (idempotent CREATE TABLE IF NOT EXISTS) — 4 more calls
    assert conn.execute.call_count == after_first * 2, (
        "ensure_table must re-execute SQL on every read (no stale cache)"
    )
    assert conn.fetch.call_count == 2  # two SELECT calls, one per read


@pytest.mark.asyncio
async def test_get_chunk_versions_calls_ensure_table_when_bound():
    """get_chunk_versions on a bound repo must call ensure_table before the SELECT."""
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, table_name="test_chunks", vector_dim=768)

    await repo.get_chunk_versions(post_id=1)

    assert conn.execute.call_count == 4
    conn.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_texts_calls_ensure_table_when_bound():
    """fetch_texts on a bound repo must call ensure_table before the SELECT."""
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, table_name="test_chunks", vector_dim=768)

    await repo.fetch_texts(chunk_ids=["some-uuid"])

    assert conn.execute.call_count == 4
    conn.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_reads_all_run_ensure_table_independently():
    """Every read method calls ensure_table — idempotent, no shared cache."""
    pool, conn = _make_pool()
    repo = ChunkRepository(pool, table_name="test_chunks", vector_dim=768)

    await repo.get_text_hashes(post_id=1)    # ensure_table: 4 executes
    after_hashes = conn.execute.call_count   # 4

    await repo.get_chunk_versions(post_id=1)  # ensure_table again: +4
    await repo.fetch_texts(chunk_ids=[])      # ensure_table again: +4

    assert conn.execute.call_count == after_hashes * 3, (
        "Each read method must run ensure_table independently"
    )
