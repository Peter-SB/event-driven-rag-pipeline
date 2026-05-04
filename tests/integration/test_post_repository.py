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
    assert isinstance(row, dict)
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
