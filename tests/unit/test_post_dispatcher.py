"""Tests for PostDispatcher routing logic.

PostDispatcher reads post.synced events and publishes ChunkTask messages
to RabbitMQ.  These tests use a FakeEventBus and FakeExchange to verify
the routing decisions without any network I/O.

Tested behaviours
-----------------
- Body chunk task always published on post.synced (first sync: empty fields_changed)
- Summary_title chunk task published when has_summary=True and fields changed
- Summary_title task NOT published when has_summary=False
- When fields_changed is non-empty, only affected tasks are dispatched
- body task dispatched when body_text or custom_body is in fields_changed
- Malformed events (missing post_id) are logged and do not raise
- Published tasks carry correct post_id, post_table, and embed_model
"""
from __future__ import annotations

import json

import pytest

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.dispatchers.post_dispatcher import PostDispatcher
from event_driven_rag_service.tasks.chunk_task import ChunkTask
from tests.utils.factories import FakeEventBus, FakeExchange, make_post_synced_event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _published_tasks(exchange: FakeExchange) -> list[ChunkTask]:
    """Deserialise every published message back to a ChunkTask."""
    return [ChunkTask(**json.loads(msg.body)) for msg, _ in exchange.published]


# ---------------------------------------------------------------------------
# First sync (empty fields_changed = "all fields new")
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_body_task_always_published_on_first_sync(fake_bus, fake_exchange):
    """Empty fields_changed means first sync — body chunk task must always fire."""
    event = make_post_synced_event(post_id=1, has_summary=False, fields_changed=[])
    await fake_bus.publish("post.synced", event)

    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus
    dispatcher._rmq = None  # not needed for this path

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    body_tasks = [t for t in tasks if t.task_type == "body"]
    assert len(body_tasks) == 1
    assert body_tasks[0].post_id == 1


@pytest.mark.asyncio
async def test_summary_task_published_on_first_sync_when_has_summary(fake_bus, fake_exchange):
    event = make_post_synced_event(post_id=2, has_summary=True, fields_changed=[])
    await fake_bus.publish("post.synced", event)

    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    task_types = {t.task_type for t in tasks}
    assert "body" in task_types
    assert "summary_title" in task_types


@pytest.mark.asyncio
async def test_summary_task_not_published_when_no_summary(fake_bus, fake_exchange):
    """If has_summary=False the dispatcher must not waste a chunk task on summary_title."""
    event = make_post_synced_event(post_id=3, has_summary=False, fields_changed=[])

    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    assert all(t.task_type != "summary_title" for t in tasks)


# ---------------------------------------------------------------------------
# Incremental update (non-empty fields_changed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_body_task_dispatched_when_body_text_in_fields_changed(fake_bus, fake_exchange):
    event = make_post_synced_event(
        post_id=4, has_summary=False, fields_changed=["body_text"]
    )
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    assert any(t.task_type == "body" for t in tasks)


@pytest.mark.asyncio
async def test_body_task_dispatched_when_custom_body_changes(fake_bus, fake_exchange):
    event = make_post_synced_event(
        post_id=5, has_summary=False, fields_changed=["custom_body"]
    )
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    assert any(t.task_type == "body" for t in tasks)


@pytest.mark.asyncio
async def test_body_task_not_dispatched_when_only_summary_changes(fake_bus, fake_exchange):
    """Only the summary changed — body chunks are up to date, skip body task."""
    event = make_post_synced_event(
        post_id=6, has_summary=True, fields_changed=["summary"]
    )
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    # summary_title may be dispatched, body should NOT be
    assert all(t.task_type != "body" for t in tasks)


@pytest.mark.asyncio
async def test_summary_task_dispatched_when_title_changes(fake_bus, fake_exchange):
    event = make_post_synced_event(
        post_id=7, has_summary=True, fields_changed=["title"]
    )
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    assert any(t.task_type == "summary_title" for t in tasks)


# ---------------------------------------------------------------------------
# Title dispatching tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_title_task_not_published_on_first_sync_without_title_change(fake_bus, fake_exchange):
    """Empty fields_changed new post added, so title task should be published."""
    event = make_post_synced_event(post_id=11, fields_changed=[])
    await fake_bus.publish("post.synced", event)

    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    title_tasks = [t for t in tasks if t.task_type == "title"]
    assert len(title_tasks) == 1, "Title task should be published on first sync even if title not in fields_changed"


@pytest.mark.asyncio
async def test_title_task_dispatched_when_title_field_changes(fake_bus, fake_exchange):
    """When title field changes, title task should be dispatched."""
    event = make_post_synced_event(
        post_id=13, fields_changed=["title"]
    )
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    assert any(t.task_type == "title" for t in tasks)


@pytest.mark.asyncio
async def test_title_task_dispatched_when_custom_title_field_changes(fake_bus, fake_exchange):
    """When custom_title field changes, title task should be dispatched."""
    event = make_post_synced_event(
        post_id=14, fields_changed=["custom_title"]
    )
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    assert any(t.task_type == "title" for t in tasks)


@pytest.mark.asyncio
async def test_title_task_not_dispatched_when_only_body_changes(fake_bus, fake_exchange):
    """When only body changes, title task should not be dispatched."""
    event = make_post_synced_event(
        post_id=15, fields_changed=["body_text"]
    )
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    assert all(t.task_type != "title" for t in tasks)


@pytest.mark.asyncio
async def test_title_task_uses_bge_small_model(fake_bus, fake_exchange):
    """Title task should use bge-small-en-v1.5 model."""
    event = make_post_synced_event(post_id=16, fields_changed=["title"])
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    title_task = next((t for t in tasks if t.task_type == "title"), None)
    assert title_task is not None
    assert title_task.embed_model == EMBED_CONFIGS["title"].model
    assert title_task.embed_model == "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# Published task payload correctness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_published_body_task_has_correct_embed_model(fake_bus, fake_exchange):
    event = make_post_synced_event(post_id=8, has_summary=False, fields_changed=[])
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    body_task = next(t for t in tasks if t.task_type == "body")
    assert body_task.embed_model == EMBED_CONFIGS["body"].model


@pytest.mark.asyncio
async def test_published_task_routing_key_is_cpu_chunk_post(fake_bus, fake_exchange):
    event = make_post_synced_event(post_id=9, has_summary=False, fields_changed=[])
    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    assert fake_exchange.all_routing_keys == ["cpu.chunk.post", "cpu.chunk.post"]


@pytest.mark.asyncio
async def test_published_task_carries_trace_id(fake_bus, fake_exchange):
    event = make_post_synced_event(post_id=10, fields_changed=[])
    event["trace_id"] = "my-trace-42"

    dispatcher = PostDispatcher.__new__(PostDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_chunk_tasks(fake_exchange, event)

    tasks = _published_tasks(fake_exchange)
    assert all(t.trace_id == "my-trace-42" for t in tasks)
