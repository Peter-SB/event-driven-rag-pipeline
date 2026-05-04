"""Integration tests for PostgresEventBus.

Verifies the publish → subscribe roundtrip, at-least-once offset tracking,
and consumer group isolation using real Postgres.

Tested behaviours
-----------------
- Published events are visible via subscribe()
- subscribe() yields events in insertion order
- Multiple events on the same topic are all delivered
- Consumer group offset advances so events are not re-delivered on reconnect
- Two different consumer groups on the same topic each receive all events
  (they track independent offsets)
- Events published to different topics do not bleed across topics
"""
from __future__ import annotations

import asyncio

import pytest

from event_driven_rag_service.infrastructure.event_bus import PostgresEventBus


pytestmark = pytest.mark.integration


async def _collect(bus: PostgresEventBus, topic: str, group: str, *, count: int) -> list[dict]:
    """Pull exactly *count* events from *topic* and return them."""
    results = []
    async for event in bus.subscribe(topic, consumer_group=group):
        results.append(event)
        if len(results) >= count:
            break
    return results


# ---------------------------------------------------------------------------
# Basic publish / subscribe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_published_event_is_received_by_subscriber(postgres_pool, clean_event_bus_tables):
    bus = PostgresEventBus(postgres_pool)
    await bus.setup_tables()

    await bus.publish("test.topic", {"value": "hello"})

    events = await _collect(bus, "test.topic", "grp-1", count=1)
    assert len(events) == 1
    assert events[0]["value"] == "hello"


@pytest.mark.asyncio
async def test_events_delivered_in_insertion_order(postgres_pool, clean_event_bus_tables):
    bus = PostgresEventBus(postgres_pool)
    await bus.setup_tables()

    for i in range(5):
        await bus.publish("order.topic", {"seq": i})

    events = await _collect(bus, "order.topic", "grp-order", count=5)
    assert [e["seq"] for e in events] == list(range(5))


@pytest.mark.asyncio
async def test_multiple_events_all_delivered(postgres_pool, clean_event_bus_tables):
    bus = PostgresEventBus(postgres_pool)
    await bus.setup_tables()

    for i in range(10):
        await bus.publish("multi.topic", {"n": i})

    events = await _collect(bus, "multi.topic", "grp-multi", count=10)
    assert len(events) == 10


# ---------------------------------------------------------------------------
# Consumer group isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_consumer_groups_each_receive_all_events(postgres_pool, clean_event_bus_tables):
    """Independent consumer groups must not share offset state."""
    bus = PostgresEventBus(postgres_pool)
    await bus.setup_tables()

    await bus.publish("shared.topic", {"x": 1})
    await bus.publish("shared.topic", {"x": 2})

    group_a = await _collect(bus, "shared.topic", "grp-a", count=2)
    group_b = await _collect(bus, "shared.topic", "grp-b", count=2)

    assert len(group_a) == 2
    assert len(group_b) == 2


@pytest.mark.asyncio
async def test_consumer_group_does_not_redeliver_after_reconnect(postgres_pool, clean_event_bus_tables):
    """Once a consumer group advances its offset, old events must not come back."""
    bus = PostgresEventBus(postgres_pool)
    await bus.setup_tables()

    await bus.publish("offset.topic", {"msg": "first"})
    await bus.publish("offset.topic", {"msg": "second"})

    # First subscriber consumes both events — offset advances to 2
    first_run = await _collect(bus, "offset.topic", "grp-persistent", count=2)
    assert len(first_run) == 2

    # Publish a new event after the first subscription is done
    await bus.publish("offset.topic", {"msg": "third"})

    # Second subscriber using the same group should only see the new event
    second_run = await _collect(bus, "offset.topic", "grp-persistent", count=1)
    assert len(second_run) == 1
    assert second_run[0]["msg"] == "third"


# ---------------------------------------------------------------------------
# Topic isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_events_on_different_topics_do_not_bleed(postgres_pool, clean_event_bus_tables):
    bus = PostgresEventBus(postgres_pool)
    await bus.setup_tables()

    await bus.publish("topic.a", {"from": "a"})
    await bus.publish("topic.b", {"from": "b"})

    events_a = await _collect(bus, "topic.a", "grp-iso", count=1)
    events_b = await _collect(bus, "topic.b", "grp-iso", count=1)

    assert events_a[0]["from"] == "a"
    assert events_b[0]["from"] == "b"
    assert len(events_a) == 1
    assert len(events_b) == 1
