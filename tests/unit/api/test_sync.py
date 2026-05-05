"""Unit tests for POST /posts/sync endpoint.

Tests the sync_posts route logic in isolation with mocked dependencies.
No database, no RabbitMQ, no I/O — all in-process.

Key scenarios:
  - Single post insert, update, skip
  - Batch operations
  - Error handling
  - Event publishing (fields_changed logic)
  - Response schema
"""
from __future__ import annotations

from datetime import datetime, UTC, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from event_driven_rag_service.api.sync import router as sync_router
from tests.utils.factories import make_post, FakeEventBus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app_with_mocked_state():
    """Create a minimal FastAPI app with mocked post_repo and event_bus."""
    app = FastAPI()
    app.include_router(sync_router)

    # Mock dependencies
    app.state.post_repo = AsyncMock()
    app.state.event_bus = FakeEventBus()
    app.state.seen_post_tables = set()

    return app


@pytest.fixture
def client(app_with_mocked_state):
    """TestClient for the mocked FastAPI app.

    Used as a context manager so Starlette creates ONE anyio portal for the
    entire test — all requests share the same thread and app.state is preserved
    across multiple client.post() calls within the same test.
    """
    with TestClient(app_with_mocked_state) as c:
        yield c


# ---------------------------------------------------------------------------
# Single post operations
# ---------------------------------------------------------------------------

def test_sync_single_post_insert(client):
    """Inserting a new post returns 'inserted' and publishes post.synced event."""
    post = make_post(post_id=1)
    post_repo = client.app.state.post_repo
    event_bus = client.app.state.event_bus

    # Mock: fetch returns None (no existing post), upsert returns ('inserted', version)
    post_repo.fetch.return_value = None
    post_repo.upsert.return_value = ("inserted", 1)

    response = client.post(
        "/posts/sync",
        json={
            "posts": [post.model_dump(by_alias=True, mode='json')],
            "library_id": "main",
        }
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "inserted"
    assert results[0]["post_id"] == 1
    assert results[0]["success"] is True
    # error field is excluded when None due to response_model_exclude_none=True
    assert "error" not in results[0] or results[0]["error"] is None

    # Verify event was published
    events = event_bus.drain_topic("post.synced")
    assert len(events) == 1
    evt = events[0]
    assert evt["post_id"] == 1
    assert evt["post_table"] == "posts_main"
    assert evt["fields_changed"] == [], "Expected all fields changed on initial insert, but got: %s" % evt["fields_changed"]
    assert evt["has_summary"] is True


def test_sync_single_post_skipped(client):
    """Re-syncing with same timestamp returns 'skipped' (no event published)."""
    post = make_post(post_id=2)
    post_repo = client.app.state.post_repo
    event_bus = client.app.state.event_bus

    # Mock: upsert returns ('skipped', None)
    post_repo.upsert.return_value = ("skipped", None)

    response = client.post(
        "/posts/sync",
        json={
            "posts": [post.model_dump(by_alias=True, mode='json')],
            "library_id": "main",
        }
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["status"] == "skipped"
    assert results[0]["success"] is True

    # No event should be published for skipped
    events = event_bus.drain_topic("post.synced")
    assert len(events) == 0


def test_sync_single_post_updated(client):
    """Re-syncing with newer timestamp returns 'updated' and fields_changed populated."""
    post = make_post(post_id=3)
    post_repo = client.app.state.post_repo
    event_bus = client.app.state.event_bus

    # Mock: upsert returns ('updated', 2)
    post_repo.upsert.return_value = ("updated", 2)

    response = client.post(
        "/posts/sync",
        json={
            "posts": [post.model_dump(by_alias=True, mode='json')],
            "library_id": "main",
        }
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["status"] == "updated"
    assert results[0]["success"] is True

    # Event published with fields_changed
    events = event_bus.drain_topic("post.synced")
    assert len(events) == 1
    evt = events[0]
    assert evt["fields_changed"] == [
        "body_text", "custom_body", "summary", "title", "custom_title"
    ]


def test_sync_post_without_summary(client):
    """Syncing a post without summary sets has_summary=False in event."""
    post = make_post(post_id=4, summary=None)
    post_repo = client.app.state.post_repo
    event_bus = client.app.state.event_bus

    post_repo.upsert.return_value = ("inserted", 1)

    response = client.post(
        "/posts/sync",
        json={
            "posts": [post.model_dump(by_alias=True, mode='json')],
            "library_id": "main",
        }
    )

    assert response.status_code == 200
    events = event_bus.drain_topic("post.synced")
    assert len(events) == 1
    assert events[0]["has_summary"] is False


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------

def test_sync_batch_of_posts(client):
    """Syncing multiple posts returns results for each in order."""
    posts = [
        make_post(post_id=10),
        make_post(post_id=11),
        make_post(post_id=12),
    ]
    post_repo = client.app.state.post_repo
    event_bus = client.app.state.event_bus

    # Mock: first inserted, second skipped, third updated
    post_repo.upsert.side_effect = [
        ("inserted", 1),
        ("skipped", None),
        ("updated", 2),
    ]

    response = client.post(
        "/posts/sync",
        json={
            "posts": [p.model_dump(by_alias=True, mode='json') for p in posts],
            "library_id": "main",
        }
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    assert results[0]["post_id"] == 10
    assert results[0]["status"] == "inserted"
    assert results[1]["post_id"] == 11
    assert results[1]["status"] == "skipped"
    assert results[2]["post_id"] == 12
    assert results[2]["status"] == "updated"

    # Should have 2 events (insert + update, not skipped)
    events = event_bus.drain_topic("post.synced")
    assert len(events) == 2
    assert events[0]["post_id"] == 10
    assert events[1]["post_id"] == 12


def test_sync_batch_with_custom_library_id(client):
    """Batch sync with library_id reflects as posts_{library_id} in post_table field."""
    posts = [make_post(post_id=20)]
    post_repo = client.app.state.post_repo
    event_bus = client.app.state.event_bus

    post_repo.upsert.return_value = ("inserted", 1)

    response = client.post(
        "/posts/sync",
        json={
            "posts": [p.model_dump(by_alias=True, mode='json') for p in posts],
            "library_id": "work",
        }
    )

    assert response.status_code == 200
    events = event_bus.drain_topic("post.synced")
    assert len(events) == 1
    assert events[0]["post_table"] == "posts_work"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_sync_post_with_repository_error(client):
    """If upsert raises an exception, result is marked error with message."""
    post = make_post(post_id=30)
    post_repo = client.app.state.post_repo
    event_bus = client.app.state.event_bus

    # Mock: upsert raises an error
    post_repo.upsert.side_effect = ValueError("Database constraint violation")

    response = client.post(
        "/posts/sync",
        json={
            "posts": [post.model_dump(by_alias=True, mode='json')],
            "library_id": "main",
        }
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["status"] == "error"
    assert results[0]["success"] is False
    assert "Database constraint violation" in results[0]["error"]

    # No event published for error
    events = event_bus.drain_topic("post.synced")
    assert len(events) == 0


def test_sync_batch_partial_failure(client):
    """Batch with one failing post still processes others."""
    posts = [
        make_post(post_id=40),
        make_post(post_id=41),
        make_post(post_id=42),
    ]
    post_repo = client.app.state.post_repo
    event_bus = client.app.state.event_bus

    # Mock: insert, error, insert
    post_repo.upsert.side_effect = [
        ("inserted", 1),
        RuntimeError("Connection lost"),
        ("inserted", 1),
    ]

    response = client.post(
        "/posts/sync",
        json={
            "posts": [p.model_dump(by_alias=True, mode='json') for p in posts],
            "library_id": "main",
        }
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["success"] is True
    assert results[0]["status"] == "inserted"
    assert results[1]["success"] is False
    assert results[1]["status"] == "error"
    assert results[2]["success"] is True
    assert results[2]["status"] == "inserted"

    # Only 2 events (the successful ones)
    events = event_bus.drain_topic("post.synced")
    assert len(events) == 2


def test_sync_event_publishing_error_marks_post_as_error(client):
    """If event publishing fails, the post is marked with error status."""
    post = make_post(post_id=50)
    post_repo = client.app.state.post_repo

    # Inject a failing event bus
    failing_bus = AsyncMock()
    failing_bus.publish.side_effect = RuntimeError("Event bus down")
    client.app.state.event_bus = failing_bus

    post_repo.upsert.return_value = ("inserted", 1)

    response = client.post(
        "/posts/sync",
        json={
            "posts": [post.model_dump(by_alias=True, mode='json')],
            "library_id": "main",
        }
    )

    # The entire request fails when event bus fails
    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["status"] == "error"
    assert results[0]["success"] is False
    assert "Event bus down" in results[0]["error"]


# ---------------------------------------------------------------------------
# Event payload structure
# ---------------------------------------------------------------------------

def test_sync_event_payload_has_all_required_fields(client):
    """Event payload includes all fields from PostSyncedEvent schema."""
    post = make_post(post_id=60)
    post_repo = client.app.state.post_repo
    event_bus = client.app.state.event_bus

    post_repo.upsert.return_value = ("inserted", 1)

    client.post(
        "/posts/sync",
        json={
            "posts": [post.model_dump(by_alias=True, mode='json')],
            "library_id": "main",
        }
    )

    events = event_bus.drain_topic("post.synced")
    evt = events[0]

    # Verify all expected fields are present
    assert "post_id" in evt
    assert "post_table" in evt
    assert "has_summary" in evt
    assert "fields_changed" in evt
    assert "updated_at" in evt
    assert "event_type" in evt
    assert "event_id" in evt
    assert "occurred_at" in evt


def test_sync_event_updated_at_matches_post(client):
    """Event updated_at field matches the post's updated_at."""
    ts = datetime(2025, 5, 1, 15, 30, 0, tzinfo=UTC)
    post = make_post(post_id=70, updated_at=ts)
    post_repo = client.app.state.post_repo
    event_bus = client.app.state.event_bus

    post_repo.upsert.return_value = ("inserted", 1)

    client.post(
        "/posts/sync",
        json={
            "posts": [post.model_dump(by_alias=True, mode='json')],
            "library_id": "main",
        }
    )

    events = event_bus.drain_topic("post.synced")
    evt = events[0]
    # isoformat() with UTC gives '+00:00' but may be serialized as 'Z'
    # Check that both represent the same moment
    from datetime import datetime as dt
    parsed = dt.fromisoformat(evt["updated_at"].replace('Z', '+00:00'))
    assert parsed == ts


# ---------------------------------------------------------------------------
# Table creation and library ID isolation
# ---------------------------------------------------------------------------

def test_sync_creates_post_table_on_first_sync_for_library(client):
    """First sync with a library_id calls ensure_table for that library's post table."""
    post = make_post(post_id=100)
    post_repo = client.app.state.post_repo

    post_repo.upsert.return_value = ("inserted", 1)
    post_repo.ensure_table = AsyncMock()

    response = client.post(
        "/posts/sync",
        json={
            "posts": [post.model_dump(by_alias=True, mode='json')],
            "library_id": "main",
        }
    )

    assert response.status_code == 200
    post_repo.ensure_table.assert_called_once_with("posts_main")


def test_sync_does_not_recreate_table_on_second_sync_same_library(client):
    """Second sync with the same library_id does NOT call ensure_table again."""
    post = make_post(post_id=101)
    post_repo = client.app.state.post_repo

    post_repo.upsert.return_value = ("inserted", 1)
    post_repo.ensure_table = AsyncMock()

    # First sync
    response1 = client.post(
        "/posts/sync",
        json={
            "posts": [post.model_dump(by_alias=True, mode='json')],
            "library_id": "main",
        }
    )
    assert response1.status_code == 200
    assert post_repo.ensure_table.call_count == 1

    # Second sync same library — must NOT call ensure_table again
    post2 = make_post(post_id=102)
    response2 = client.post(
        "/posts/sync",
        json={
            "posts": [post2.model_dump(by_alias=True, mode='json')],
            "library_id": "main",
        }
    )
    assert response2.status_code == 200
    assert post_repo.ensure_table.call_count == 1


def test_sync_creates_separate_tables_for_different_libraries(client):
    """Different library_ids trigger ensure_table for separate post tables."""
    post1 = make_post(post_id=110)
    post2 = make_post(post_id=111)
    post_repo = client.app.state.post_repo

    post_repo.upsert.return_value = ("inserted", 1)
    post_repo.ensure_table = AsyncMock()

    client.post(
        "/posts/sync",
        json={"posts": [post1.model_dump(by_alias=True, mode='json')], "library_id": "main"},
    )
    client.post(
        "/posts/sync",
        json={"posts": [post2.model_dump(by_alias=True, mode='json')], "library_id": "work"},
    )

    assert post_repo.ensure_table.call_count == 2
    calls = [call.args[0] for call in post_repo.ensure_table.call_args_list]
    assert "posts_main" in calls
    assert "posts_work" in calls


def test_sync_tracks_created_tables_in_seen_post_tables(client):
    """Verify seen_post_tables is updated after ensure_table succeeds."""
    post = make_post(post_id=120)
    post_repo = client.app.state.post_repo
    seen_tables = client.app.state.seen_post_tables

    post_repo.upsert.return_value = ("inserted", 1)
    assert len(seen_tables) == 0

    client.post(
        "/posts/sync",
        json={"posts": [post.model_dump(by_alias=True, mode='json')], "library_id": "main"},
    )
    assert "posts_main" in seen_tables
    assert len(seen_tables) == 1

    post2 = make_post(post_id=121)
    client.post(
        "/posts/sync",
        json={"posts": [post2.model_dump(by_alias=True, mode='json')], "library_id": "secondary"},
    )
    assert "posts_main" in seen_tables
    assert "posts_secondary" in seen_tables
    assert len(seen_tables) == 2


# ---------------------------------------------------------------------------
# _evaluate_changed_fields — unit tests
# ---------------------------------------------------------------------------

def test_evaluate_changed_fields_returns_empty_list_when_no_existing():
    """When existing is None (first sync), fields_changed must be empty."""
    from event_driven_rag_service.api.sync import _evaluate_changed_fields

    post = make_post(post_id=1)
    result = _evaluate_changed_fields(post, None)
    assert result == []


def test_evaluate_changed_fields_returns_empty_when_nothing_changed():
    """When existing post has identical field values, fields_changed must be empty."""
    from event_driven_rag_service.api.sync import _evaluate_changed_fields

    post = make_post(post_id=1, body_text="same", summary="same", title="same", custom_body="same", custom_title="same")
    existing = make_post(post_id=1, body_text="same", summary="same", title="same", custom_body="same", custom_title="same")
    result = _evaluate_changed_fields(post, existing)
    assert result == []


def test_evaluate_changed_fields_detects_body_text_change():
    from event_driven_rag_service.api.sync import _evaluate_changed_fields

    post = make_post(post_id=1, body_text="new body")
    existing = make_post(post_id=1, body_text="old body")
    result = _evaluate_changed_fields(post, existing)
    assert result == ["body_text"]


def test_evaluate_changed_fields_detects_custom_body_change():
    from event_driven_rag_service.api.sync import _evaluate_changed_fields

    post = make_post(post_id=1, custom_body="new custom")
    existing = make_post(post_id=1, custom_body="old custom")
    result = _evaluate_changed_fields(post, existing)
    assert result == ["custom_body"]


def test_evaluate_changed_fields_detects_summary_change():
    from event_driven_rag_service.api.sync import _evaluate_changed_fields

    post = make_post(post_id=1, summary="new summary")
    existing = make_post(post_id=1, summary="old summary")
    result = _evaluate_changed_fields(post, existing)
    assert result == ["summary"]


def test_evaluate_changed_fields_detects_title_change():
    from event_driven_rag_service.api.sync import _evaluate_changed_fields

    post = make_post(post_id=1, title="new title")
    existing = make_post(post_id=1, title="old title")
    result = _evaluate_changed_fields(post, existing)
    assert result == ["title"]


def test_evaluate_changed_fields_detects_custom_title_change():
    from event_driven_rag_service.api.sync import _evaluate_changed_fields

    post = make_post(post_id=1, custom_title="new custom title")
    existing = make_post(post_id=1, custom_title="old custom title")
    result = _evaluate_changed_fields(post, existing)
    assert result == ["custom_title"]


def test_evaluate_changed_fields_detects_multiple_changes():
    from event_driven_rag_service.api.sync import _evaluate_changed_fields

    post = make_post(post_id=1, body_text="new body", summary="new summary", title="new title")
    existing = make_post(post_id=1, body_text="old body", summary="old summary", title="old title")
    result = _evaluate_changed_fields(post, existing)
    assert sorted(result) == sorted(["body_text", "summary", "title"])


def test_evaluate_changed_fields_treats_none_as_empty_string():
    """None vs empty string should NOT be treated as a change (both are 'empty')."""
    from event_driven_rag_service.api.sync import _evaluate_changed_fields

    post = make_post(post_id=1, summary=None)
    existing = make_post(post_id=1, summary="")
    result = _evaluate_changed_fields(post, existing)
    assert result == []


def test_evaluate_changed_fields_detects_none_to_value_change():
    from event_driven_rag_service.api.sync import _evaluate_changed_fields

    post = make_post(post_id=1, summary="now has summary")
    existing = make_post(post_id=1, summary=None)
    result = _evaluate_changed_fields(post, existing)
    assert result == ["summary"]


def test_evaluate_changed_fields_detects_value_to_none_change():
    from event_driven_rag_service.api.sync import _evaluate_changed_fields

    post = make_post(post_id=1, summary=None)
    existing = make_post(post_id=1, summary="had summary")
    result = _evaluate_changed_fields(post, existing)
    assert result == ["summary"]
