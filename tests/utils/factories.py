"""Shared test utilities: in-memory fakes and data factories.

Kept in tests/utils/ so they can be imported by both unit and integration
conftest files without any circular dependency.

Nothing in this module does I/O.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, UTC, timedelta
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

from event_driven_rag_service.data_models.chunk import Chunk, ChunkMetadata
from event_driven_rag_service.data_models.post import Post
from event_driven_rag_service.infrastructure.event_bus import EventBusBase


# ---------------------------------------------------------------------------
# In-memory event bus
# ---------------------------------------------------------------------------

class FakeEventBus(EventBusBase):
    """Synchronous in-memory event bus for unit tests.

    Supports multiple topics.  subscribe() yields events that were published
    *before or after* the subscription is created (call drain_topic() to
    inspect published events without consuming them via subscribe).

    Usage in tests
    --------------
        bus = FakeEventBus()
        await bus.publish("post.synced", {"post_id": 1})
        events = bus.drain_topic("post.synced")
        assert len(events) == 1
    """

    def __init__(self) -> None:
        self._queues: dict[str, list[dict]] = defaultdict(list)

    async def publish(self, topic: str, event: dict) -> None:
        self._queues[topic].append(event)

    async def subscribe(  # type: ignore[override]
        self, topic: str, consumer_group: str
    ) -> AsyncIterator[dict]:
        """Yield all events currently queued for the topic, then stop."""
        for event in list(self._queues[topic]):
            yield event

    def drain_topic(self, topic: str) -> list[dict]:
        """Return (and clear) all events on *topic*.  Does not block."""
        events = list(self._queues.get(topic, []))
        self._queues[topic] = []
        return events

    def peek_topic(self, topic: str) -> list[dict]:
        """Return a copy of events on *topic* without clearing."""
        return list(self._queues.get(topic, []))

    def clear(self) -> None:
        self._queues.clear()


# ---------------------------------------------------------------------------
# Data factories
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)


def make_post(
    post_id: int = 1,
    body_text: str = "Hello world this is some body content for testing purposes.",
    summary: str | None = "A short summary.",
    updated_at: datetime | None = None,
    subreddit: str | None = "test",
    **kwargs,
) -> Post:
    """Build a minimal valid Post. Override any field via kwargs."""
    ts = updated_at or _BASE_TS
    return Post(
        id=post_id,
        redditId=f"ext_{post_id}",
        redditCreatedAt=_BASE_TS,
        url=f"https://reddit.com/r/test/comments/{post_id}",
        title="Test Post Title",
        bodyText=body_text,
        author="testuser",
        subreddit=subreddit,
        addedAt=_BASE_TS,
        updatedAt=ts,
        summary=summary,
        **kwargs,
    )


def make_post_synced_event(
    post_id: int = 1,
    has_summary: bool = True,
    fields_changed: list[str] | None = None,
    updated_at: datetime | None = None,
) -> dict:
    """Build a post.synced event payload dict."""
    return {
        "event_id": f"evt-{post_id}",
        "event_type": "post.synced",
        "event_version": 1,
        "occurred_at": (_BASE_TS).isoformat(),
        "post_id": post_id,
        "post_table": "posts",
        "has_summary": has_summary,
        "fields_changed": fields_changed if fields_changed is not None else [],
        "updated_at": (updated_at or _BASE_TS).isoformat(),
        "trace_id": None,
    }


def make_chunk(
    post_id: int = 1,
    chunk_index: int = 0,
    text: str = "This is a sample chunk of text used for testing.",
) -> Chunk:
    """Build a minimal valid Chunk."""
    import hashlib, uuid
    return Chunk(
        id=str(uuid.uuid4()),
        post_id=post_id,
        post_updated_at=_BASE_TS,
        chunk_index=chunk_index,
        text=text,
        metadata=ChunkMetadata(title="Test Post Title", external_id=f"ext_{post_id}"),
        token_count=max(1, round(len(text.split()) * 1.3)),
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
        created_at=_BASE_TS,
    )


def make_chunks_created_event(
    post_id: int = 1,
    chunk_ids: list[str] | None = None,
    task_type: str = "body",
    chunk_table: str = "chunks_body_bge_base_v1_5",
) -> dict:
    """Build a chunks.created event payload dict."""
    return {
        "event_id": f"evt-chunks-{post_id}",
        "post_id": post_id,
        "post_table": "posts",
        "chunk_ids": chunk_ids or ["chunk-uuid-1", "chunk-uuid-2"],
        "chunk_table": chunk_table,
        "chunk_count": len(chunk_ids) if chunk_ids else 2,
        "task_type": task_type,
        "trace_id": None,
        "created_at": _BASE_TS.isoformat(),
    }


# ---------------------------------------------------------------------------
# RabbitMQ fakes
# ---------------------------------------------------------------------------

class FakeMessage:
    """Minimal stand-in for aio_pika.Message, records routing_key and body."""

    def __init__(self, body: bytes) -> None:
        self.body = body


class FakeExchange:
    """Records all published messages for assertion in tests."""

    def __init__(self) -> None:
        self.published: list[tuple[FakeMessage, str]] = []  # (message, routing_key)

    async def publish(self, message: FakeMessage, routing_key: str = "") -> None:
        self.published.append((message, routing_key))

    def messages_for_key(self, key: str) -> list[FakeMessage]:
        return [msg for msg, rk in self.published if rk == key]

    @property
    def all_routing_keys(self) -> list[str]:
        return [rk for _, rk in self.published]
