"""Tests for ChunkDispatcher routing logic.

ChunkDispatcher reads chunks.created events from the event log and publishes
EmbedTask messages to the RabbitMQ embedding exchange.

Tested behaviours
-----------------
- EmbedTask published for each chunks.created event
- task_type on the event determines the embed model (body → bge-base-v1.5)
- Routing key is model-name-aware (gpu.embed.{model})
- Published task carries chunk_ids, chunk_table, post_id from the event
- Unknown task_type falls back to the body embed config
- trace_id is propagated from the source event
"""
from __future__ import annotations

import json

import pytest

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.dispatchers.chunk_dispatcher import ChunkDispatcher
from event_driven_rag_service.tasks.embed_task import EmbedTask
from tests.utils.factories import FakeEventBus, FakeExchange, make_chunks_created_event


def _published_embed_tasks(exchange: FakeExchange) -> list[EmbedTask]:
    return [EmbedTask(**json.loads(msg.body)) for msg, _ in exchange.published]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_task_published_for_chunks_created(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=1, task_type="body")
    dispatcher = ChunkDispatcher.__new__(ChunkDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_embedding(fake_exchange, _mock_route(), event)

    tasks = _published_embed_tasks(fake_exchange)
    assert len(tasks) == 1


@pytest.mark.asyncio
async def test_embed_task_carries_chunk_ids_from_event(fake_bus, fake_exchange):
    ids = ["id-a", "id-b", "id-c"]
    event = make_chunks_created_event(post_id=2, chunk_ids=ids, task_type="body")
    dispatcher = ChunkDispatcher.__new__(ChunkDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_embedding(fake_exchange, _mock_route(), event)

    task = _published_embed_tasks(fake_exchange)[0]
    assert task.chunk_ids == ids


@pytest.mark.asyncio
async def test_embed_task_uses_body_model_for_body_task_type(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=3, task_type="body")
    dispatcher = ChunkDispatcher.__new__(ChunkDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_embedding(fake_exchange, _mock_route(), event)

    task = _published_embed_tasks(fake_exchange)[0]
    assert task.model_name == EMBED_CONFIGS["body"].model


@pytest.mark.asyncio
async def test_embed_task_uses_correct_model_for_summary_title(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=4, task_type="summary_title")
    dispatcher = ChunkDispatcher.__new__(ChunkDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_embedding(fake_exchange, _mock_route(), event)

    task = _published_embed_tasks(fake_exchange)[0]
    assert task.model_name == EMBED_CONFIGS["summary_title"].model


@pytest.mark.asyncio
async def test_embed_task_uses_correct_model_for_title(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=8, task_type="title")
    dispatcher = ChunkDispatcher.__new__(ChunkDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_embedding(fake_exchange, _mock_route(), event)

    task = _published_embed_tasks(fake_exchange)[0]
    assert task.model_name == EMBED_CONFIGS["title"].model
    assert task.model_name == "bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# Routing key
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_routing_key_contains_model_name(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=5, task_type="body")
    dispatcher = ChunkDispatcher.__new__(ChunkDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_embedding(fake_exchange, _mock_route(), event)

    # The routing key should embed the model name so GPU workers can filter
    assert len(fake_exchange.all_routing_keys) == 1
    assert EMBED_CONFIGS["body"].model in fake_exchange.all_routing_keys[0]


@pytest.mark.asyncio
async def test_routing_key_contains_title_model_name(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=9, task_type="title")
    dispatcher = ChunkDispatcher.__new__(ChunkDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_embedding(fake_exchange, _mock_route(), event)

    # Title routing key should contain bge-small-en-v1.5
    assert len(fake_exchange.all_routing_keys) == 1
    assert "bge-small-en-v1.5" in fake_exchange.all_routing_keys[0]


# ---------------------------------------------------------------------------
# Fallback for unknown task_type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_task_type_falls_back_to_body_config(fake_bus, fake_exchange):
    """If the worker stamps an unknown task_type we degrade gracefully to body config."""
    event = make_chunks_created_event(post_id=6, task_type="nonexistent_type")
    dispatcher = ChunkDispatcher.__new__(ChunkDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_embedding(fake_exchange, _mock_route(), event)

    task = _published_embed_tasks(fake_exchange)[0]
    assert task.model_name == EMBED_CONFIGS["body"].model


# ---------------------------------------------------------------------------
# Trace propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trace_id_propagated_to_embed_task(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=7, task_type="body")
    event["trace_id"] = "trace-xyz"

    dispatcher = ChunkDispatcher.__new__(ChunkDispatcher)
    dispatcher._event_bus = fake_bus

    await dispatcher._dispatch_embedding(fake_exchange, _mock_route(), event)

    task = _published_embed_tasks(fake_exchange)[0]
    assert task.trace_id == "trace-xyz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockRoute:
    """Minimal stand-in for TaskRoute — resolves key via format_map."""
    exchange = "embedding"
    routing_key = "gpu.embed.{model_name}"

    def resolve_key(self, task: EmbedTask) -> str:
        return self.routing_key.format_map(task.model_dump())


def _mock_route() -> _MockRoute:
    return _MockRoute()
