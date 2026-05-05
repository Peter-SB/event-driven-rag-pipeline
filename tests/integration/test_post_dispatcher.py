"""Integration tests for PostDispatcher.

Verifies that the PostDispatcher correctly reads post.synced events from the
real PostgresEventBus and dispatches the right ChunkTask messages to the exchange.

Scope: real Postgres testcontainer for event storage; FakeExchange to capture
published tasks without requiring a live RabbitMQ broker.

Architecture under test
-----------------------
                           (this test)
  event_log table  ──►  PostgresEventBus.subscribe()
                                │
                    PostDispatcher._dispatch_chunk_tasks()
                                │
                          FakeExchange  ←──  assertions here

Tested behaviours
-----------------
- Body chunk task dispatched after post.synced event is stored in event_log
- Summary_title chunk task dispatched when has_summary=True (first sync)
- No body task dispatched when only summary-related fields changed
- No summary_title task dispatched when has_summary=False
- Consumer offset advances after the event is consumed
"""
from __future__ import annotations

import asyncio
import json

import asyncpg
import pytest

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.dispatchers.post_dispatcher import PostDispatcher
from event_driven_rag_service.infrastructure.event_bus import PostgresEventBus
from event_driven_rag_service.tasks.chunk_task import ChunkTask
from tests.utils.factories import FakeExchange, make_post_synced_event

pytestmark = pytest.mark.integration

_CONSUMER_GROUP = "test.post_dispatcher.integration"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _consume_one(bus: PostgresEventBus, topic: str, consumer_group: str, timeout: float = 5.0) -> dict:
    """Pull exactly one event from the bus, failing the test if none arrives within *timeout*."""
    async def _read_first() -> dict:
        async for event in bus.subscribe(topic, consumer_group=consumer_group):
            return event

    return await asyncio.wait_for(_read_first(), timeout=timeout)


def _tasks_from_exchange(exchange: FakeExchange) -> list[ChunkTask]:
    return [ChunkTask(**json.loads(msg.body)) for msg, _ in exchange.published]


async def _setup_bus(pool: asyncpg.Pool) -> PostgresEventBus:
    bus = PostgresEventBus(pool)
    await bus.setup_tables()
    return bus


# ---------------------------------------------------------------------------
# First sync (empty fields_changed — all fields new)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatcher_dispatches_body_task_on_first_post_sync(
    postgres_pool: asyncpg.Pool, clean_event_bus_tables
):
    """Body chunk task must reach the exchange after a post.synced event is stored."""
    bus = await _setup_bus(postgres_pool)
    event = make_post_synced_event(post_id=2001, has_summary=False, fields_changed=[])
    await bus.publish("post.synced", event)

    consumed = await _consume_one(bus, "post.synced", _CONSUMER_GROUP)

    exchange = FakeExchange()
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = bus
    dispatcher._rmq = None
    await dispatcher._dispatch_chunk_tasks(exchange, consumed)

    tasks = _tasks_from_exchange(exchange)
    body_tasks = [t for t in tasks if t.task_type == "body"]
    assert len(body_tasks) == 1
    assert body_tasks[0].post_id == 2001
    assert body_tasks[0].post_table == make_post_synced_event(post_id=2001)["post_table"]
    assert body_tasks[0].embed_model == EMBED_CONFIGS["body"].model


@pytest.mark.asyncio
async def test_dispatcher_dispatches_both_tasks_on_first_sync_with_summary(
    postgres_pool: asyncpg.Pool, clean_event_bus_tables
):
    """Both body and summary_title tasks dispatched when has_summary=True on first sync."""
    bus = await _setup_bus(postgres_pool)
    event = make_post_synced_event(post_id=2002, has_summary=True, fields_changed=[])
    await bus.publish("post.synced", event)

    consumed = await _consume_one(bus, "post.synced", _CONSUMER_GROUP)

    exchange = FakeExchange()
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = bus
    dispatcher._rmq = None
    await dispatcher._dispatch_chunk_tasks(exchange, consumed)

    task_types = {t.task_type for t in _tasks_from_exchange(exchange)}
    assert "body" in task_types
    assert "summary_title" in task_types


# ---------------------------------------------------------------------------
# Incremental update (non-empty fields_changed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatcher_skips_body_task_when_only_summary_changed(
    postgres_pool: asyncpg.Pool, clean_event_bus_tables
):
    """Body chunks are current when only summary changed — no body task must be dispatched."""
    bus = await _setup_bus(postgres_pool)
    # Only summary changed; body is unchanged
    event = make_post_synced_event(post_id=2003, has_summary=True, fields_changed=["summary"])
    await bus.publish("post.synced", event)

    consumed = await _consume_one(bus, "post.synced", _CONSUMER_GROUP)

    exchange = FakeExchange()
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = bus
    dispatcher._rmq = None
    await dispatcher._dispatch_chunk_tasks(exchange, consumed)

    tasks = _tasks_from_exchange(exchange)
    assert all(t.task_type != "body" for t in tasks), (
        "Body task must not be dispatched when only summary-related fields changed"
    )
    # Summary_title task should still fire (summary changed and has_summary=True)
    assert any(t.task_type == "summary_title" for t in tasks)


@pytest.mark.asyncio
async def test_dispatcher_skips_summary_task_when_has_summary_false(
    postgres_pool: asyncpg.Pool, clean_event_bus_tables
):
    """No summary_title task dispatched when the post has no summary."""
    bus = await _setup_bus(postgres_pool)
    event = make_post_synced_event(post_id=2004, has_summary=False, fields_changed=[])
    await bus.publish("post.synced", event)

    consumed = await _consume_one(bus, "post.synced", _CONSUMER_GROUP)

    exchange = FakeExchange()
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = bus
    dispatcher._rmq = None
    await dispatcher._dispatch_chunk_tasks(exchange, consumed)

    tasks = _tasks_from_exchange(exchange)
    assert all(t.task_type != "summary_title" for t in tasks)


# ---------------------------------------------------------------------------
# Consumer offset
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consumer_offset_advances_after_event_consumed(
    postgres_pool: asyncpg.Pool, clean_event_bus_tables
):
    """After consuming a post.synced event, the consumer offset must be persisted.

    This ensures the dispatcher resumes from the correct position on restart
    and does not re-deliver already-processed events.
    """
    bus = await _setup_bus(postgres_pool)
    event = make_post_synced_event(post_id=2005)
    await bus.publish("post.synced", event)

    await _consume_one(bus, "post.synced", _CONSUMER_GROUP)

    async with postgres_pool.acquire() as conn:
        offset = await conn.fetchval(
            """
            SELECT last_id FROM consumer_offsets
            WHERE  consumer_group = $1
              AND  topic = $2
            """,
            _CONSUMER_GROUP,
            "post.synced",
        )

    assert offset is not None, "Offset row must exist after consuming an event"
    assert offset > 0, "Offset must have advanced past the initial value of 0"


@pytest.mark.asyncio
async def test_second_consume_does_not_redeliver_first_event(
    postgres_pool: asyncpg.Pool, clean_event_bus_tables
):
    """An event consumed once must not appear again when the consumer re-subscribes."""
    bus = await _setup_bus(postgres_pool)
    event = make_post_synced_event(post_id=2006)
    await bus.publish("post.synced", event)

    # Consume the event once, advancing the offset
    await _consume_one(bus, "post.synced", _CONSUMER_GROUP)

    # Publish a second distinct event so the bus has something to deliver
    event2 = make_post_synced_event(post_id=2007)
    await bus.publish("post.synced", event2)

    # Second consume should yield the SECOND event, not the first again
    second_event = await _consume_one(bus, "post.synced", _CONSUMER_GROUP)

    assert second_event["post_id"] == 2007, (
        "Re-subscribing must resume from the stored offset, not replay event 2006"
    )
