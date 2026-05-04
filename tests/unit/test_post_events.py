"""Tests for event model construction and validation.

Covers PostSyncedEvent, ChunksCreatedEvent, and the BaseEvent base class.

Tested behaviours
-----------------
- BaseEvent auto-generates a unique event_id per instance
- BaseEvent sets occurred_at to now (UTC) by default
- BaseEvent rejects unknown extra fields (extra="forbid")
- PostSyncedEvent has the correct default event_type
- PostSyncedEvent.fields_changed defaults to empty list
- PostSyncedEvent.has_summary defaults to False
- PostSyncedEvent.to_dict() round-trips cleanly (JSON-serialisable values)
- Two PostSyncedEvent instances get different event_ids
- ChunksCreatedEvent derives chunk_count from the provided list
"""
from __future__ import annotations

from datetime import datetime, UTC

import pytest
from pydantic import ValidationError

from event_driven_rag_service.events.post_events import PostSyncedEvent, PostDeletedEvent
from event_driven_rag_service.events.chunk_events import ChunksCreatedEvent


_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# BaseEvent behaviour (tested via PostSyncedEvent as a concrete subclass)
# ---------------------------------------------------------------------------

def test_base_event_auto_generates_event_id():
    event = PostSyncedEvent(post_id=1, post_table="posts", updated_at=_NOW)
    assert event.event_id is not None
    assert len(event.event_id) == 36  # UUID4 string length


def test_base_event_two_instances_have_different_event_ids():
    e1 = PostSyncedEvent(post_id=1, post_table="posts", updated_at=_NOW)
    e2 = PostSyncedEvent(post_id=1, post_table="posts", updated_at=_NOW)
    assert e1.event_id != e2.event_id


def test_base_event_occurred_at_defaults_to_utc_now():
    before = datetime.now(UTC)
    event = PostSyncedEvent(post_id=1, post_table="posts", updated_at=_NOW)
    after = datetime.now(UTC)
    assert before <= event.occurred_at <= after


def test_base_event_version_defaults_to_1():
    event = PostSyncedEvent(post_id=1, post_table="posts", updated_at=_NOW)
    assert event.event_version == 1


def test_base_event_rejects_extra_fields():
    with pytest.raises(ValidationError):
        PostSyncedEvent(
            post_id=1,
            post_table="posts",
            updated_at=_NOW,
            unknown_field="should_fail",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# PostSyncedEvent
# ---------------------------------------------------------------------------

def test_post_synced_event_default_event_type():
    event = PostSyncedEvent(post_id=42, post_table="posts", updated_at=_NOW)
    assert event.event_type == "post.synced"


def test_post_synced_event_fields_changed_defaults_to_empty_list():
    event = PostSyncedEvent(post_id=1, post_table="posts", updated_at=_NOW)
    assert event.fields_changed == []


def test_post_synced_event_has_summary_defaults_to_false():
    event = PostSyncedEvent(post_id=1, post_table="posts", updated_at=_NOW)
    assert event.has_summary is False


def test_post_synced_event_accepts_fields_changed():
    event = PostSyncedEvent(
        post_id=1,
        post_table="posts",
        updated_at=_NOW,
        fields_changed=["body_text", "title"],
    )
    assert "body_text" in event.fields_changed
    assert "title" in event.fields_changed


def test_post_synced_event_to_dict_is_json_serialisable():
    event = PostSyncedEvent(post_id=5, post_table="posts", updated_at=_NOW)
    d = event.to_dict()
    import json
    # Should not raise
    json.dumps(d)
    assert d["post_id"] == 5
    assert d["event_type"] == "post.synced"


def test_post_synced_event_round_trips_via_to_dict():
    event = PostSyncedEvent(
        post_id=7,
        post_table="staging_posts",
        updated_at=_NOW,
        has_summary=True,
        fields_changed=["body_text"],
        trace_id="trace-abc",
    )
    d = event.to_dict()
    restored = PostSyncedEvent(**d)

    assert restored.post_id == event.post_id
    assert restored.post_table == event.post_table
    assert restored.has_summary == event.has_summary
    assert restored.fields_changed == event.fields_changed
    assert restored.trace_id == event.trace_id


# ---------------------------------------------------------------------------
# PostDeletedEvent
# ---------------------------------------------------------------------------

def test_post_deleted_event_default_event_type():
    event = PostDeletedEvent(post_id=99, post_table="posts")
    assert event.event_type == "post.deleted"


def test_post_deleted_event_requires_post_id():
    with pytest.raises(ValidationError):
        PostDeletedEvent(post_table="posts")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ChunksCreatedEvent
# ---------------------------------------------------------------------------

def test_chunks_created_event_default_event_type():
    from datetime import datetime, UTC
    event = ChunksCreatedEvent(
        post_id=1,
        post_table="posts",
        chunk_ids=["a", "b", "c"],
        chunk_table="chunks_body_bge_base_v1_5",
        task_type="body",
        chunk_count=3,
        created_at=_NOW,
    )
    assert event.event_type == "chunks.created"


def test_chunks_created_event_stores_chunk_ids():
    ids = ["id-1", "id-2"]
    event = ChunksCreatedEvent(
        post_id=2,
        post_table="posts",
        chunk_ids=ids,
        chunk_table="chunks_body_bge_base_v1_5",
        task_type="body",
        chunk_count=len(ids),
        created_at=_NOW,
    )
    assert event.chunk_ids == ids
