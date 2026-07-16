"""Tests for ChunkDispatcher routing logic.

ChunkDispatcher reads chunks.created events from the event log and publishes
EmbedTask messages to the RabbitMQ embedding exchange.

Tested behaviours
-----------------
- EmbedTask published for each chunks.created event
- task_type on the event determines the embed model (body → bge-base-v1.5)
- Routing key is the task_type's configured queue (EMBED_CONFIGS[...].queue),
  not derived from the model name — multiple task_types can share one queue
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


def _dispatcher(fake_bus: FakeEventBus) -> ChunkDispatcher:
    dispatcher = ChunkDispatcher.__new__(ChunkDispatcher)
    dispatcher._event_bus = fake_bus
    return dispatcher


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_task_published_for_chunks_created(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=1, task_type="body")

    await _dispatcher(fake_bus)._dispatch_embedding(fake_exchange, event)

    tasks = _published_embed_tasks(fake_exchange)
    assert len(tasks) == 1


@pytest.mark.asyncio
async def test_embed_task_carries_chunk_ids_from_event(fake_bus, fake_exchange):
    ids = ["id-a", "id-b", "id-c"]
    event = make_chunks_created_event(post_id=2, chunk_ids=ids, task_type="body")

    await _dispatcher(fake_bus)._dispatch_embedding(fake_exchange, event)

    task = _published_embed_tasks(fake_exchange)[0]
    assert task.chunk_ids == ids


@pytest.mark.asyncio
async def test_embed_task_uses_body_model_for_body_task_type(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=3, task_type="body")

    await _dispatcher(fake_bus)._dispatch_embedding(fake_exchange, event)

    task = _published_embed_tasks(fake_exchange)[0]
    assert task.model_name == EMBED_CONFIGS["body"].model


@pytest.mark.asyncio
async def test_embed_task_uses_correct_model_for_summary_title(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=4, task_type="summary_title")

    await _dispatcher(fake_bus)._dispatch_embedding(fake_exchange, event)

    task = _published_embed_tasks(fake_exchange)[0]
    assert task.model_name == EMBED_CONFIGS["summary_title"].model


@pytest.mark.asyncio
async def test_embed_task_uses_correct_model_for_title(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=8, task_type="title")

    await _dispatcher(fake_bus)._dispatch_embedding(fake_exchange, event)

    task = _published_embed_tasks(fake_exchange)[0]
    assert task.model_name == EMBED_CONFIGS["title"].model
    assert task.model_name == "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# Routing key
#
# EMBED_CONFIGS[...].queue is the bound RabbitMQ queue (task_queue.py
# BINDINGS). Deriving the routing key from the sanitised *model name* instead
# breaks as soon as a task_type's model string doesn't reduce to the queue's
# suffix — e.g. summary_title's model is "Qwen3-Embedding-0.6B-Q8_0.gguf" but
# it shares the "gpu.embed.qwen3-0.6b" queue with analysis's
# "Qwen/Qwen3-0.6B". A model-name-derived key silently drops every
# summary_title embed task since no queue is bound to
# "gpu.embed.qwen3-embedding-0.6b-q8_0.gguf".
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_routing_key_is_the_configured_queue_for_body(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=5, task_type="body")

    await _dispatcher(fake_bus)._dispatch_embedding(fake_exchange, event)

    assert len(fake_exchange.all_routing_keys) == 1
    assert fake_exchange.all_routing_keys[0] == EMBED_CONFIGS["body"].queue


@pytest.mark.asyncio
async def test_routing_key_is_the_configured_queue_for_title(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=9, task_type="title")

    await _dispatcher(fake_bus)._dispatch_embedding(fake_exchange, event)

    assert len(fake_exchange.all_routing_keys) == 1
    assert fake_exchange.all_routing_keys[0] == EMBED_CONFIGS["title"].queue


@pytest.mark.asyncio
async def test_routing_key_matches_bound_queue_for_every_task_type(fake_bus, fake_exchange):
    dispatcher = _dispatcher(fake_bus)
    for task_type in EMBED_CONFIGS:
        event = make_chunks_created_event(post_id=100, task_type=task_type)

        await dispatcher._dispatch_embedding(fake_exchange, event)

        routing_key = fake_exchange.all_routing_keys[-1]
        assert routing_key == EMBED_CONFIGS[task_type].queue, (
            f"{task_type}: routing key {routing_key!r} does not match the "
            f"bound queue {EMBED_CONFIGS[task_type].queue!r}"
        )


@pytest.mark.asyncio
async def test_summary_title_and_analysis_share_the_same_queue(fake_bus, fake_exchange):
    """Two task_types with different models can be routed to one shared GPU queue."""
    summary_event = make_chunks_created_event(post_id=101, task_type="summary_title")
    analysis_event = make_chunks_created_event(post_id=102, task_type="analysis")

    dispatcher = _dispatcher(fake_bus)
    await dispatcher._dispatch_embedding(fake_exchange, summary_event)
    await dispatcher._dispatch_embedding(fake_exchange, analysis_event)

    assert fake_exchange.all_routing_keys == ["gpu.embed.qwen3-0.6b", "gpu.embed.qwen3-0.6b"]


# ---------------------------------------------------------------------------
# Fallback for unknown task_type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_task_type_falls_back_to_body_config(fake_bus, fake_exchange):
    """If the worker stamps an unknown task_type we degrade gracefully to body config."""
    event = make_chunks_created_event(post_id=6, task_type="nonexistent_type")

    await _dispatcher(fake_bus)._dispatch_embedding(fake_exchange, event)

    task = _published_embed_tasks(fake_exchange)[0]
    assert task.model_name == EMBED_CONFIGS["body"].model


# ---------------------------------------------------------------------------
# Trace propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trace_id_propagated_to_embed_task(fake_bus, fake_exchange):
    event = make_chunks_created_event(post_id=7, task_type="body")
    event["trace_id"] = "trace-xyz"

    await _dispatcher(fake_bus)._dispatch_embedding(fake_exchange, event)

    task = _published_embed_tasks(fake_exchange)[0]
    assert task.trace_id == "trace-xyz"
