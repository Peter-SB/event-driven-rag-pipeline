"""Integration tests for FolderRepository and HighlightRepository.

Verifies table creation, upsert freshness and soft-delete propagation
against a real Postgres instance (via testcontainers).

Tested behaviours
-----------------
- ensure_table() is idempotent
- First upsert returns "inserted"; same or older updated_at returns "skipped"
- Newer updated_at returns "updated" and persists the change
- A soft delete (newer updated_at, is_deleted=True) reaches the table
- Naive SQLite-style timestamps round-trip as UTC
- Library-scoped tables are isolated from each other
"""
from __future__ import annotations

from datetime import timedelta

import pytest
import pytest_asyncio

from event_driven_rag_service.data_models.folder import Folder
from event_driven_rag_service.repository.folder_repository import FolderRepository
from event_driven_rag_service.repository.highlight_repository import HighlightRepository
from tests.utils.factories import _BASE_TS, make_folder, make_highlight


pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def folder_repo(postgres_pool):
    repo = FolderRepository(postgres_pool, table_name="test_folders")
    await repo.ensure_table()
    async with postgres_pool.acquire() as conn:
        await conn.execute("TRUNCATE test_folders")
    yield repo
    async with postgres_pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS test_folders")


@pytest_asyncio.fixture
async def highlight_repo(postgres_pool):
    repo = HighlightRepository(postgres_pool, table_name="test_highlights")
    await repo.ensure_table()
    async with postgres_pool.acquire() as conn:
        await conn.execute("TRUNCATE test_highlights")
    yield repo
    async with postgres_pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS test_highlights")


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_folder_ensure_table_is_idempotent(folder_repo):
    await folder_repo.ensure_table()


@pytest.mark.asyncio
async def test_folder_insert_then_fetch(folder_repo):
    status, _ = await folder_repo.upsert(make_folder(1, name="Recipes", parentId=None))
    assert status == "inserted"

    stored = await folder_repo.fetch(1)
    assert stored is not None
    assert stored.name == "Recipes"
    assert stored.updated_at == _BASE_TS


@pytest.mark.asyncio
async def test_folder_same_updated_at_is_skipped(folder_repo):
    await folder_repo.upsert(make_folder(1))
    status, _ = await folder_repo.upsert(make_folder(1, name="Renamed"))

    assert status == "skipped"
    assert (await folder_repo.fetch(1)).name == "Folder 1"


@pytest.mark.asyncio
async def test_folder_newer_updated_at_renames(folder_repo):
    await folder_repo.upsert(make_folder(1))
    status, prior = await folder_repo.upsert(
        make_folder(1, name="Renamed", updated_at=_BASE_TS + timedelta(minutes=1))
    )

    assert status == "updated"
    assert prior == _BASE_TS
    assert (await folder_repo.fetch(1)).name == "Renamed"


@pytest.mark.asyncio
async def test_folder_soft_delete_propagates(folder_repo):
    await folder_repo.upsert(make_folder(1))
    await folder_repo.upsert(make_folder(1, updated_at=_BASE_TS + timedelta(minutes=1), isDeleted=True))

    assert (await folder_repo.fetch(1)).is_deleted is True


@pytest.mark.asyncio
async def test_folder_child_can_arrive_before_parent(folder_repo):
    status, _ = await folder_repo.upsert(make_folder(2, parentId=1))
    assert status == "inserted"
    assert (await folder_repo.fetch(2)).parent_id == 1


@pytest.mark.asyncio
async def test_folder_naive_timestamp_round_trips_as_utc(folder_repo):
    folder = Folder(id=5, name="x", createdAt="2024-06-01 12:00:00", updatedAt="2024-06-01 12:00:00")
    await folder_repo.upsert(folder)

    stored = await folder_repo.fetch(5)
    assert stored.updated_at == folder.updated_at
    assert stored.updated_at.utcoffset() == timedelta(0)


@pytest.mark.asyncio
async def test_folder_library_tables_are_isolated(postgres_pool):
    main = FolderRepository(postgres_pool)
    await main.ensure_table("test_folders_a")
    await main.ensure_table("test_folders_b")
    try:
        await main.upsert(make_folder(1), "test_folders_a")
        assert await main.fetch(1, "test_folders_b") is None
    finally:
        async with postgres_pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS test_folders_a, test_folders_b")


# ---------------------------------------------------------------------------
# Highlights
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_highlight_ensure_table_is_idempotent(highlight_repo):
    await highlight_repo.ensure_table()


@pytest.mark.asyncio
async def test_highlight_insert_then_fetch(highlight_repo):
    status, _ = await highlight_repo.upsert(
        make_highlight(1, post_id=42, note="why", startOffset=3, endOffset=9)
    )
    assert status == "inserted"

    stored = await highlight_repo.fetch(1)
    assert stored.post_id == 42
    assert stored.note == "why"
    assert (stored.start_offset, stored.end_offset) == (3, 9)


@pytest.mark.asyncio
async def test_highlight_older_updated_at_is_skipped(highlight_repo):
    await highlight_repo.upsert(make_highlight(1, updated_at=_BASE_TS + timedelta(minutes=5)))
    status, _ = await highlight_repo.upsert(make_highlight(1, text="stale"))

    assert status == "skipped"
    assert (await highlight_repo.fetch(1)).text == "A highlighted passage."


@pytest.mark.asyncio
async def test_highlight_newer_updated_at_updates_note(highlight_repo):
    await highlight_repo.upsert(make_highlight(1))
    status, _ = await highlight_repo.upsert(
        make_highlight(1, updated_at=_BASE_TS + timedelta(minutes=1), note="added later")
    )

    assert status == "updated"
    assert (await highlight_repo.fetch(1)).note == "added later"


@pytest.mark.asyncio
async def test_highlight_soft_delete_propagates(highlight_repo):
    await highlight_repo.upsert(make_highlight(1))
    await highlight_repo.upsert(make_highlight(1, updated_at=_BASE_TS + timedelta(minutes=1), isDeleted=True))

    assert (await highlight_repo.fetch(1)).is_deleted is True


@pytest.mark.asyncio
async def test_highlight_can_arrive_before_its_post(highlight_repo):
    status, _ = await highlight_repo.upsert(make_highlight(1, post_id=9999))
    assert status == "inserted"
