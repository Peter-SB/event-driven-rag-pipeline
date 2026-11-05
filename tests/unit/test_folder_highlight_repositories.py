"""Unit tests for FolderRepository and HighlightRepository.

Exercises the in-process logic with a mocked asyncpg pool — no database required.

High-risk behaviours covered
-----------------------------
- ensure_table issues each statement as a separate execute() call
  (asyncpg rejects multi-statement strings) and lowercases the table name.
- upsert returns 'skipped' without executing the INSERT when updated_at is not newer.
- upsert returns 'inserted' when no row exists and 'updated' when a newer one arrives.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from event_driven_rag_service.repository.folder_repository import FolderRepository
from event_driven_rag_service.repository.highlight_repository import HighlightRepository
from tests.utils.factories import _BASE_TS, make_folder, make_highlight


def _make_pool(fetchval_return=None, fetchrow_return=None):
    """Return a mock asyncpg Pool whose acquire() context manager yields a mock conn."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=fetchval_return)
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)

    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool, conn


# ---------------------------------------------------------------------------
# ensure_table
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_folder_ensure_table_issues_table_then_index():
    pool, conn = _make_pool()
    await FolderRepository(pool).ensure_table("Folders_Main")

    sqls = [c.args[0] for c in conn.execute.call_args_list]
    assert len(sqls) == 2
    assert "CREATE TABLE IF NOT EXISTS folders_main" in sqls[0]
    assert "CREATE INDEX IF NOT EXISTS" in sqls[1]


@pytest.mark.asyncio
async def test_highlight_ensure_table_issues_table_then_two_indexes():
    pool, conn = _make_pool()
    await HighlightRepository(pool).ensure_table("Highlights_Main")

    sqls = [c.args[0] for c in conn.execute.call_args_list]
    assert len(sqls) == 3
    assert "CREATE TABLE IF NOT EXISTS highlights_main" in sqls[0]
    assert "updated_at" in sqls[1]
    assert "post_id" in sqls[2]


# ---------------------------------------------------------------------------
# upsert freshness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_folder_upsert_new_returns_inserted():
    pool, conn = _make_pool(fetchval_return=None)
    status, prior = await FolderRepository(pool).upsert(make_folder(1), "folders_main")

    assert status == "inserted"
    assert prior is None
    conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_folder_upsert_same_updated_at_returns_skipped_without_insert():
    pool, conn = _make_pool(fetchval_return=_BASE_TS)
    status, _ = await FolderRepository(pool).upsert(make_folder(1), "folders_main")

    assert status == "skipped"
    conn.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_folder_upsert_newer_returns_updated():
    pool, conn = _make_pool(fetchval_return=_BASE_TS)
    newer = make_folder(1, updated_at=_BASE_TS + timedelta(minutes=1))
    status, prior = await FolderRepository(pool).upsert(newer, "folders_main")

    assert status == "updated"
    assert prior == _BASE_TS
    conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_highlight_upsert_new_returns_inserted():
    pool, conn = _make_pool(fetchval_return=None)
    status, _ = await HighlightRepository(pool).upsert(make_highlight(1), "highlights_main")

    assert status == "inserted"
    conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_highlight_upsert_older_returns_skipped_without_insert():
    pool, conn = _make_pool(fetchval_return=_BASE_TS + timedelta(minutes=5))
    status, _ = await HighlightRepository(pool).upsert(make_highlight(1), "highlights_main")

    assert status == "skipped"
    conn.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_highlight_upsert_newer_returns_updated():
    pool, _ = _make_pool(fetchval_return=_BASE_TS)
    newer = make_highlight(1, updated_at=_BASE_TS + timedelta(minutes=1))
    status, _ = await HighlightRepository(pool).upsert(newer, "highlights_main")

    assert status == "updated"


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_folder_fetch_returns_none_when_missing():
    pool, _ = _make_pool(fetchrow_return=None)
    assert await FolderRepository(pool).fetch(99, "folders_main") is None


@pytest.mark.asyncio
async def test_bound_repository_uses_default_table_name():
    pool, conn = _make_pool()
    repo = HighlightRepository(pool, table_name="Highlights_Bound")
    await repo.ensure_table()

    assert "highlights_bound" in conn.execute.call_args_list[0].args[0]
