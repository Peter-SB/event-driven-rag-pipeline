"""Integration tests for PostRepository.

Verifies the upsert freshness logic, skip behaviour, and table creation
against a real Postgres instance (via testcontainers).

Tested behaviours
-----------------
- First upsert of a post returns "inserted"
- Re-upsert with same updated_at returns "skipped"
- Re-upsert with newer updated_at returns "updated" and persists changes
- Stale upsert (older updated_at) returns "skipped" and keeps existing row
- fetch() returns None for unknown post_id
- fetch() returns the stored row as a dict
- ensure_table() is idempotent (safe to call multiple times)
"""
from __future__ import annotations

from datetime import datetime, UTC, timedelta

import pytest

from event_driven_rag_service.data_models.post import Post
from tests.utils.factories import make_post


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# ensure_table idempotency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_table_is_idempotent(clean_posts_table):
    """Calling ensure_table twice must not raise."""
    await clean_posts_table.ensure_table()  # second call
    # passes if no exception raised


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_new_post_returns_inserted(clean_posts_table):
    post = make_post(post_id=100)
    status, _ = await clean_posts_table.upsert(post)
    assert status == "inserted"


@pytest.mark.asyncio
async def test_inserted_post_can_be_fetched(clean_posts_table):
    post = make_post(post_id=101)
    await clean_posts_table.upsert(post)
    row = await clean_posts_table.fetch(101)
    assert row is not None
    assert row["post_id"] == 101
    assert row["external_id"] == post.external_id


# ---------------------------------------------------------------------------
# Skip — same or older updated_at
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_same_updated_at_returns_skipped(clean_posts_table):
    post = make_post(post_id=102)
    await clean_posts_table.upsert(post)
    # Re-upsert with identical timestamps
    status, _ = await clean_posts_table.upsert(post)
    assert status == "skipped"


@pytest.mark.asyncio
async def test_upsert_older_updated_at_returns_skipped_and_keeps_original(clean_posts_table):
    ts = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    post = make_post(post_id=103, updated_at=ts)
    await clean_posts_table.upsert(post)

    older_post = make_post(post_id=103, body_text="stale content", updated_at=ts - timedelta(hours=1))
    status, _ = await clean_posts_table.upsert(older_post)

    assert status == "skipped"
    row = await clean_posts_table.fetch(103)
    # Original body must be preserved
    assert row["body_text"] != "stale content"


# ---------------------------------------------------------------------------
# Update — newer updated_at
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_newer_updated_at_returns_updated(clean_posts_table):
    ts = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    post = make_post(post_id=104, updated_at=ts)
    await clean_posts_table.upsert(post)

    newer = make_post(post_id=104, body_text="refreshed content", updated_at=ts + timedelta(hours=1))
    status, _ = await clean_posts_table.upsert(newer)

    assert status == "updated"


@pytest.mark.asyncio
async def test_updated_row_reflects_new_content(clean_posts_table):
    ts = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    await clean_posts_table.upsert(make_post(post_id=105, body_text="old", updated_at=ts))
    await clean_posts_table.upsert(make_post(post_id=105, body_text="new", updated_at=ts + timedelta(minutes=5)))

    row = await clean_posts_table.fetch(105)
    assert row["body_text"] == "new"


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_returns_none_for_unknown_post(clean_posts_table):
    result = await clean_posts_table.fetch(999_999)
    assert result is None


@pytest.mark.asyncio
async def test_fetch_returns_dict_with_all_columns(clean_posts_table):
    post = make_post(post_id=106, summary="My summary")
    await clean_posts_table.upsert(post)

    row = await clean_posts_table.fetch(106)
    assert isinstance(row, Post)
    # Spot-check a few important fields
    assert "post_id" in row
    assert "external_id" in row
    assert row["summary"] == "My summary"


# ---------------------------------------------------------------------------
# Multiple posts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_different_posts_stored_independently(clean_posts_table):
    for i in range(107, 112):
        await clean_posts_table.upsert(make_post(post_id=i))

    for i in range(107, 112):
        row = await clean_posts_table.fetch(i)
        assert row is not None
        assert row["post_id"] == i


# ---------------------------------------------------------------------------
# Library ID table creation (ensure tables exist for different libraries)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_table_creates_table_with_library_id_name(postgres_pool):
    """Calling ensure_table with a library ID table name creates the table."""
    from event_driven_rag_service.repository.post_repository import PostRepository

    repo = PostRepository(postgres_pool)
    table_name = "posts_main"

    # Create the table
    await repo.ensure_table(table_name)

    # Verify the table exists by inserting a post
    post = make_post(post_id=200)
    status, _ = await repo.upsert(post, table_name)
    assert status == "inserted"

    # Verify we can fetch it back
    row = await repo.fetch(200, table_name)
    assert row is not None
    assert row["post_id"] == 200

    # Cleanup
    async with postgres_pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {table_name}")


@pytest.mark.asyncio
async def test_ensure_table_creates_different_library_tables(postgres_pool):
    """ensure_table creates separate tables for different library_id values."""
    from event_driven_rag_service.repository.post_repository import PostRepository

    repo = PostRepository(postgres_pool)

    # Create tables for two different libraries
    await repo.ensure_table("posts_main")
    await repo.ensure_table("posts_work")

    # Insert posts into each table
    post_main = make_post(post_id=301)
    post_work = make_post(post_id=302)

    await repo.upsert(post_main, "posts_main")
    await repo.upsert(post_work, "posts_work")

    # Verify isolation: post_main is only in posts_main, not posts_work
    row_in_main = await repo.fetch(301, "posts_main")
    row_in_work = await repo.fetch(301, "posts_work")

    assert row_in_main is not None
    assert row_in_work is None

    # Verify post_work is only in posts_work
    row_in_work = await repo.fetch(302, "posts_work")
    row_in_main = await repo.fetch(302, "posts_main")

    assert row_in_work is not None
    assert row_in_main is None

    # Cleanup
    async with postgres_pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS posts_main")
        await conn.execute("DROP TABLE IF EXISTS posts_work")


@pytest.mark.asyncio
async def test_ensure_table_multiple_calls_same_table(postgres_pool):
    """Calling ensure_table multiple times for the same library table is safe."""
    from event_driven_rag_service.repository.post_repository import PostRepository

    repo = PostRepository(postgres_pool)
    table_name = "posts_science"

    # Call ensure_table three times for the same table
    await repo.ensure_table(table_name)
    await repo.ensure_table(table_name)
    await repo.ensure_table(table_name)

    # Insert a post to verify table works
    post = make_post(post_id=400)
    status, _ = await repo.upsert(post, table_name)
    assert status == "inserted"

    # Cleanup
    async with postgres_pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {table_name}")


@pytest.mark.asyncio
async def test_ensure_table_creates_with_proper_schema(postgres_pool):
    """ensure_table creates table with all required columns and indexes."""
    from event_driven_rag_service.repository.post_repository import PostRepository

    repo = PostRepository(postgres_pool)
    table_name = "posts_verify_schema"

    await repo.ensure_table(table_name)

    # Query the table schema
    async with postgres_pool.acquire() as conn:
        # Get all columns
        columns = await conn.fetch(
            """SELECT column_name, data_type FROM information_schema.columns
               WHERE table_name = $1 ORDER BY ordinal_position""",
            table_name
        )

        # Verify key columns exist
        column_names = [col["column_name"] for col in columns]
        assert "post_id" in column_names
        assert "external_id" in column_names
        assert "external_source" in column_names
        assert "title" in column_names
        assert "body_text" in column_names
        assert "author" in column_names
        assert "updated_at" in column_names

        # Get indexes
        indexes = await conn.fetch(
            """SELECT indexname FROM pg_indexes
               WHERE tablename = $1""",
            table_name
        )

        index_names = [idx["indexname"] for idx in indexes]
        # Should have primary key index and updated_at index
        assert any("updated_at_idx" in name for name in index_names)

    # Cleanup
    async with postgres_pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {table_name}")
