"""Unit tests for SearchDispatcher.

Tests the dispatcher's responsibility:
- Consume search_job.created events
- Translate to EmbedTask messages and publish to RabbitMQ
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from event_driven_rag_service.dispatchers.search_dispatcher import SearchDispatcher
from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.events.search_events import SearchJobCreatedEvent


class FakeEventBusSubscription:
    """Minimal async generator for test subscriptions."""

    def __init__(self, events: list[dict]):
        self.events = events
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.events):
            raise StopAsyncIteration
        event = self.events[self.index]
        self.index += 1
        return event


@pytest.mark.asyncio
async def test_search_dispatcher_publishes_embed_task_for_query():
    """search_job.created event → EmbedTask published to gpu.embed.{model} routing key."""
    # Mock RabbitMQ
    mock_channel = AsyncMock()
    mock_exchange = AsyncMock()
    mock_channel.declare_exchange.return_value = mock_exchange

    mock_rmq = AsyncMock()
    mock_rmq.channel.return_value = mock_channel

    # Mock event bus
    job_id = str(uuid.uuid4())
    query_text = "what is retrieval augmented generation?"
    trace_id = "trace-123"

    event = {
        "event_type": "search_job.created",
        "query_job_id": job_id,
        "query": query_text,
        "trace_id": trace_id,
        "embedding_profile": "BAAI/bge-base-en-v1.5",
    }

    mock_bus = MagicMock()
    mock_bus.subscribe = MagicMock(return_value=FakeEventBusSubscription([event]))

    # Run dispatcher and capture published message
    dispatcher = SearchDispatcher(mock_rmq, mock_bus)

    # Run until it processes the event, then timeout
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=1.0)
    except asyncio.TimeoutError:
        pass  # Expected — run() is an infinite loop

    # Verify exchange was declared
    mock_channel.declare_exchange.assert_called_once_with(
        "embedding", "topic", durable=True
    )

    # Verify message was published
    assert mock_exchange.publish.called, "exchange.publish should have been called"

    # Extract the published message
    call_args = mock_exchange.publish.call_args
    message = call_args[0][0]  # First positional arg
    routing_key = call_args[1]["routing_key"]  # Named arg

    # Verify routing key
    assert routing_key == "gpu.embed.bge-base-en-v1.5", f"Expected routing_key 'gpu.embed.bge-base-en-v1.5', got {routing_key!r}"

    # Decode and validate EmbedTask
    task_dict = json.loads(message.body.decode())
    task = EmbedTask.model_validate(task_dict)

    assert task.task_type == "query", f"Expected task_type 'query', got {task.task_type!r}"
    assert task.query == query_text, f"Expected query {query_text!r}, got {task.query!r}"
    assert task.query_job_id == job_id, f"Expected query_job_id {job_id!r}, got {task.query_job_id!r}"
    assert task.model_name == "BAAI/bge-base-en-v1.5", f"Expected model_name 'bge-base-v1.5', got {task.model_name!r}"
    assert task.trace_id == trace_id, f"Expected trace_id {trace_id!r}, got {task.trace_id!r}"


@pytest.mark.asyncio
async def test_search_dispatcher_handles_missing_trace_id():
    """search_job.created without trace_id should not crash."""
    # Mock RabbitMQ
    mock_channel = AsyncMock()
    mock_exchange = AsyncMock()
    mock_channel.declare_exchange.return_value = mock_exchange

    mock_rmq = AsyncMock()
    mock_rmq.channel.return_value = mock_channel

    # Mock event bus — event without trace_id
    job_id = str(uuid.uuid4())
    event = {
        "event_type": "search_job.created",
        "query_job_id": job_id,
        "query": "test query",
        "embedding_profile": "BAAI/bge-base-en-v1.5",
    }

    mock_bus = MagicMock()
    mock_bus.subscribe = MagicMock(return_value=FakeEventBusSubscription([event]))

    # Run dispatcher
    dispatcher = SearchDispatcher(mock_rmq, mock_bus)

    try:
        await asyncio.wait_for(dispatcher.run(), timeout=1.0)
    except asyncio.TimeoutError:
        pass

    # Verify message was published (trace_id=None is OK)
    assert mock_exchange.publish.called

    call_args = mock_exchange.publish.call_args
    message = call_args[0][0]
    task_dict = json.loads(message.body.decode())
    task = EmbedTask.model_validate(task_dict)

    assert task.trace_id is None, f"Expected trace_id=None, got {task.trace_id!r}"


@pytest.mark.asyncio
async def test_search_dispatcher_subscribes_to_correct_topic():
    """SearchDispatcher should subscribe to 'search_job.created' topic."""
    mock_rmq = AsyncMock()
    mock_channel = AsyncMock()
    mock_rmq.channel.return_value = mock_channel

    mock_bus = MagicMock()
    mock_bus.subscribe = MagicMock(return_value=FakeEventBusSubscription([]))

    dispatcher = SearchDispatcher(mock_rmq, mock_bus)

    try:
        await asyncio.wait_for(dispatcher.run(), timeout=0.5)
    except asyncio.TimeoutError:
        pass

    # Verify subscription
    mock_bus.subscribe.assert_called_once()
    call_args = mock_bus.subscribe.call_args

    topic = call_args[0][0]
    assert topic == "search_job.created", f"Expected topic 'search_job.created', got {topic!r}"

    # Check consumer group
    kwargs = call_args[1]
    assert "consumer_group" in kwargs, "consumer_group should be specified"


async def _run_dispatcher_with_event(event: dict) -> tuple[str, dict]:
    """Helper: run dispatcher with a single event, return (routing_key, EmbedTask dict)."""
    mock_channel = AsyncMock()
    mock_exchange = AsyncMock()
    mock_channel.declare_exchange.return_value = mock_exchange
    mock_rmq = AsyncMock()
    mock_rmq.channel.return_value = mock_channel

    mock_bus = MagicMock()
    mock_bus.subscribe = MagicMock(return_value=FakeEventBusSubscription([event]))

    dispatcher = SearchDispatcher(mock_rmq, mock_bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=1.0)
    except asyncio.TimeoutError:
        pass

    call_args = mock_exchange.publish.call_args
    routing_key = call_args[1]["routing_key"]
    task_dict = json.loads(call_args[0][0].body.decode())
    return routing_key, task_dict


@pytest.mark.asyncio
async def test_search_dispatcher_routes_title_model_to_small_queue():
    """embedding_profile=bge-small (384-dim) must route to the small-model queue."""
    event = {
        "event_type": "search_job.created",
        "query_job_id": str(uuid.uuid4()),
        "query": "title search query",
        "embedding_profile": "BAAI/bge-small-en-v1.5",
    }
    routing_key, task_dict = await _run_dispatcher_with_event(event)

    assert routing_key == "gpu.embed.bge-small-en-v1.5", routing_key
    assert task_dict["model_name"] == "BAAI/bge-small-en-v1.5"


@pytest.mark.asyncio
async def test_search_dispatcher_routes_body_model_to_base_queue():
    """embedding_profile=bge-base (768-dim) must route to the base-model queue."""
    event = {
        "event_type": "search_job.created",
        "query_job_id": str(uuid.uuid4()),
        "query": "body search query",
        "embedding_profile": "BAAI/bge-base-en-v1.5",
    }
    routing_key, task_dict = await _run_dispatcher_with_event(event)

    assert routing_key == "gpu.embed.bge-base-en-v1.5", routing_key
    assert task_dict["model_name"] == "BAAI/bge-base-en-v1.5"


@pytest.mark.asyncio
async def test_search_dispatcher_skips_event_with_unknown_profile():
    """An unrecognised embedding_profile must be skipped — no task published."""
    mock_channel = AsyncMock()
    mock_exchange = AsyncMock()
    mock_channel.declare_exchange.return_value = mock_exchange
    mock_rmq = AsyncMock()
    mock_rmq.channel.return_value = mock_channel

    event = {
        "event_type": "search_job.created",
        "query_job_id": str(uuid.uuid4()),
        "query": "unknown model query",
        "embedding_profile": "unknown/model-xyz",
    }
    mock_bus = MagicMock()
    mock_bus.subscribe = MagicMock(return_value=FakeEventBusSubscription([event]))

    dispatcher = SearchDispatcher(mock_rmq, mock_bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=1.0)
    except asyncio.TimeoutError:
        pass

    mock_exchange.publish.assert_not_called()
