"""Integration tests for PostRepository.fetch().

Tests that fetch() correctly:
- Retrieves a post from the database
- Serializes all fields into a Post data model
- Returns None when no post exists
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from event_driven_rag_service.data_models.post import Post
from event_driven_rag_service.repository.post_repository import PostRepository

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helper: create a known post for testing
# ---------------------------------------------------------------------------

async def _insert_known_post(
    repo: PostRepository,
    post_id: int,
) -> Post:
    """Insert a deterministic test post."""
    now = datetime.now(timezone.utc)
    
    known_post = Post(
        post_id=post_id,
        external_id="test-external-id",
        external_source="reddit",
        external_created_at=now - timedelta(hours=1),
        url="https://example.com/test-post",
        title="Test Title",
        body_text="This is a test body text.",
        author="Test Author",
        subreddit="testsub",
        added_at=now - timedelta(days=2),
        updated_at=now,
        custom_title=None,
        custom_body=None,
        notes="Test notes",
        rating=5.0,
        is_read=True,
        read_at=now - timedelta(minutes=30),
        is_favorite=False,
        is_archived=False,
        queued_at=None,
        is_deleted=False,
        folder_ids=[1, 2],
        extra_fields={"key": "value"},
        body_min_hash="abc123",
        summary="Test summary",
    )
    
    await repo.upsert(known_post)
    return known_post


# ---------------------------------------------------------------------------
# Test: fetch returns correct Post object with all fields
# ---------------------------------------------------------------------------

from datetime import timedelta


@pytest.mark.asyncio
async def test_fetch_returns_correct_post_object(clean_posts_table):
    """fetch() should deserialize DB row into a complete Post model."""
    repo = clean_posts_table  # type: ignore[arg-type]
    
    # Insert known post
    now = datetime.now(timezone.utc)
    known_post = Post(
        post_id=1,
        external_id="ext-001",
        external_source="reddit",
        external_created_at=now - timedelta(hours=2),
        url="https://example.com/first-post",
        title="First Post Title",
        body_text="Body text for first post.",
        author="Author One",
        subreddit="r/test1",
        added_at=now - timedelta(days=5),
        updated_at=now,
        custom_title=None,
        custom_body=None,
        notes=None,
        rating=4.5,
        is_read=True,
        read_at=now - timedelta(hours=10),
        is_favorite=True,
        is_archived=False,
        queued_at=None,
        is_deleted=False,
        folder_ids=[3],
        extra_fields={"tag": "important"},
        body_min_hash="hash123",
        summary="First post summary",
    )
    await repo.upsert(known_post)
    
    # Fetch the post
    fetched = await repo.fetch(1)
    
    assert fetched is not None, "fetch() should return a Post object"
    
    # Verify all fields are correctly deserialized
    assert fetched.post_id == 1
    assert fetched.external_id == "ext-001"
    assert fetched.external_source == "reddit"
    assert fetched.url == "https://example.com/first-post"
    assert fetched.title == "First Post Title"
    assert fetched.body_text == "Body text for first post."
    assert fetched.author == "Author One"
    assert fetched.subreddit == "r/test1"
    assert fetched.rating == 4.5
    assert fetched.is_read is True
    assert fetched.is_favorite is True
    assert fetched.folder_ids == [3]
    assert fetched.extra_fields == {"tag": "important"}
    assert fetched.body_min_hash == "hash123"
    assert fetched.summary == "First post summary"
    
    # Verify timestamps are preserved (within 1 second tolerance)
    assert abs((fetched.updated_at - now).total_seconds()) <= 1
    assert abs((fetched.external_created_at - (now - timedelta(hours=2))).total_seconds()) <= 1


# ---------------------------------------------------------------------------
# Test: fetch returns None for non-existent post
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_none_for_missing_post(clean_posts_table):
    """fetch() should return None when no row exists."""
    repo = clean_posts_table  # type: ignore[arg-type]
    
    result = await repo.fetch(9999)
    
    assert result is None, "fetch() should return None for non-existent post_id"


# ---------------------------------------------------------------------------
# Test: fetch works with explicit table_name parameter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_with_explicit_table_name(clean_posts_table):
    """fetch() should work when table_name is explicitly passed."""
    repo = clean_posts_table  # type: ignore[arg-type]
    
    # Insert a post
    now = datetime.now(timezone.utc)
    test_post = Post(
        post_id=2,
        external_id="ext-002",
        external_source="reddit",
        external_created_at=now - timedelta(hours=1),
        url="https://example.com/second-post",
        title="Second Post",
        body_text="Second post content.",
        author="Author Two",
        subreddit=None,
        added_at=now - timedelta(days=3),
        updated_at=now,
    )
    await repo.upsert(test_post)
    
    # Fetch with explicit table name
    fetched = await repo.fetch(2, table_name="test_posts")
    
    assert fetched is not None
    assert fetched.title == "Second Post"


# ---------------------------------------------------------------------------
# Test: fetch handles empty body_text correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_handles_null_body_text(clean_posts_table):
    """fetch() should handle posts with null/empty body_text."""
    repo = clean_posts_table  # type: ignore[arg-type]
    
    now = datetime.now(timezone.utc)
    empty_body_post = Post(
        post_id=3,
        external_id="ext-003",
        external_source="reddit",
        external_created_at=now - timedelta(hours=1),
        url="https://example.com/empty-body",
        title="Empty Body Test",
        body_text=None,  # Explicitly None
        author="Author Three",
        subreddit=None,
        added_at=now - timedelta(days=2),
        updated_at=now,
    )
    await repo.upsert(empty_body_post)
    
    fetched = await repo.fetch(3)
    
    assert fetched is not None
    assert fetched.body_text is None


# ---------------------------------------------------------------------------
# Test: fetch handles extra_fields as JSON string (from DB)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_parses_extra_fields_json_string(clean_posts_table):
    """fetch() should parse extra_fields from JSON string to dict."""
    repo = clean_posts_table  # type: ignore[arg-type]
    
    now = datetime.now(timezone.utc)
    json_post = Post(
        post_id=4,
        external_id="ext-004",
        external_source="reddit",
        external_created_at=now - timedelta(hours=1),
        url="https://example.com/json-fields",
        title="JSON Fields Test",
        body_text="Test content.",
        author="Author Four",
        subreddit=None,
        added_at=now - timedelta(days=2),
        updated_at=now,
        extra_fields='{"nested": {"key": "value"}}',  # JSON string as stored in DB
    )
    await repo.upsert(json_post)
    
    fetched = await repo.fetch(4)
    
    assert fetched is not None
    assert isinstance(fetched.extra_fields, dict)
    assert fetched.extra_fields == {"nested": {"key": "value"}}
