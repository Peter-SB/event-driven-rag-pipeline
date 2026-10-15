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

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.config.reranking_config import RERANK_CONFIGS
from event_driven_rag_service.dispatchers.search_dispatcher import SearchDispatcher
from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.tasks.search_tasks import SearchFinalizeTask, SearchRankTask
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
        "embedding_profile": EMBED_CONFIGS["body"].primary,
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

    # Verify the embedding exchange was declared (both the job-created loop and
    # the candidates-retrieved loop declare exchanges on startup — run() now
    # gathers both subscriptions concurrently).
    mock_channel.declare_exchange.assert_any_call(
        "embedding", "topic", durable=True
    )

    # Verify message was published
    assert mock_exchange.publish.called, "exchange.publish should have been called"

    # Extract the published message
    call_args = mock_exchange.publish.call_args
    message = call_args[0][0]  # First positional arg
    routing_key = call_args[1]["routing_key"]  # Named arg

    # Verify routing key — must be the config's bound queue, not a literal
    # string, so this test doesn't silently pass after a model rename.
    expected_queue = EMBED_CONFIGS["body"].queue
    assert routing_key == expected_queue, f"Expected routing_key {expected_queue!r}, got {routing_key!r}"

    # Decode and validate EmbedTask
    task_dict = json.loads(message.body.decode())
    task = EmbedTask.model_validate(task_dict)

    assert task.task_type == "query", f"Expected task_type 'query', got {task.task_type!r}"
    assert task.query == query_text, f"Expected query {query_text!r}, got {task.query!r}"
    assert task.query_job_id == job_id, f"Expected query_job_id {job_id!r}, got {task.query_job_id!r}"
    assert task.model_name == EMBED_CONFIGS["body"].primary, f"Expected model_name {EMBED_CONFIGS['body'].primary!r}, got {task.model_name!r}"
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
        "embedding_profile": EMBED_CONFIGS["body"].primary,
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
async def test_search_dispatcher_subscribes_to_both_topics():
    """SearchDispatcher.run() should gather both subscriptions concurrently:
    'search_job.created' and 'search_candidates.retrieved'."""
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

    assert mock_bus.subscribe.call_count == 2
    topics = {call.args[0] for call in mock_bus.subscribe.call_args_list}
    assert topics == {"search_job.created", "search_candidates.retrieved"}

    for call in mock_bus.subscribe.call_args_list:
        assert "consumer_group" in call.kwargs, "consumer_group should be specified"


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
        "embedding_profile": EMBED_CONFIGS["title"].primary,
    }
    routing_key, task_dict = await _run_dispatcher_with_event(event)

    assert routing_key == EMBED_CONFIGS["title"].queue, routing_key
    assert task_dict["model_name"] == EMBED_CONFIGS["title"].primary


@pytest.mark.asyncio
async def test_search_dispatcher_routes_body_model_to_base_queue():
    """embedding_profile=bge-base (768-dim) must route to the base-model queue."""
    event = {
        "event_type": "search_job.created",
        "query_job_id": str(uuid.uuid4()),
        "query": "body search query",
        "embedding_profile": EMBED_CONFIGS["body"].primary,
    }
    routing_key, task_dict = await _run_dispatcher_with_event(event)

    assert routing_key == EMBED_CONFIGS["body"].queue, routing_key
    assert task_dict["model_name"] == EMBED_CONFIGS["body"].primary


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


# ---------------------------------------------------------------------------
# search_candidates.retrieved → SearchRankTask (rerank) or
#                                SearchFinalizeTask (no rerank)
# ---------------------------------------------------------------------------

async def _run_dispatcher_with_candidates_event(event: dict) -> tuple[str, str, dict]:
    """Helper: run dispatcher with one search_candidates.retrieved event.
    Returns (exchange_name, routing_key, task_dict) for whichever exchange
    actually got a publish call."""
    exchanges: dict[str, AsyncMock] = {}

    def _declare_exchange(name, *_args, **_kwargs):
        if name not in exchanges:
            exchanges[name] = AsyncMock()
        return exchanges[name]

    mock_channel = AsyncMock()
    mock_channel.declare_exchange.side_effect = _declare_exchange
    mock_rmq = AsyncMock()
    mock_rmq.channel.return_value = mock_channel

    def _subscribe(topic, **_kwargs):
        events = [event] if topic == "search_candidates.retrieved" else []
        return FakeEventBusSubscription(events)

    mock_bus = MagicMock()
    mock_bus.subscribe = MagicMock(side_effect=_subscribe)

    dispatcher = SearchDispatcher(mock_rmq, mock_bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=1.0)
    except asyncio.TimeoutError:
        pass

    for name, exchange in exchanges.items():
        if exchange.publish.called:
            call_args = exchange.publish.call_args
            routing_key = call_args[1]["routing_key"]
            task_dict = json.loads(call_args[0][0].body.decode())
            return name, routing_key, task_dict

    raise AssertionError("no exchange received a publish call")


@pytest.mark.asyncio
async def test_search_dispatcher_routes_rerank_true_to_gpu_rerank_queue():
    """search_candidates.retrieved with rerank=True must publish a
    SearchRankTask to the profile's RERANK_CONFIGS queue on the embedding
    exchange — not cpu.search.finalize."""
    event = {
        "event_type": "search_candidates.retrieved",
        "query_job_id": "job-1",
        "candidates": [{"chunk_id": "c1", "score": 0.9}],
        "k": 5,
        "rerank": True,
        "rerank_profile": "qwen3-reranker-0.6b",
    }
    exchange_name, routing_key, task_dict = await _run_dispatcher_with_candidates_event(event)

    expected_cfg = RERANK_CONFIGS["qwen3-reranker-0.6b"]
    assert exchange_name == "embedding"
    assert routing_key == expected_cfg.queue

    task = SearchRankTask.model_validate(task_dict)
    assert task.job_id == "job-1"
    assert task.rerank_profile == "qwen3-reranker-0.6b"
    assert len(task.candidates) == 1
    assert task.candidates[0].chunk_id == "c1"
    assert task.candidates[0].score == 0.9


@pytest.mark.asyncio
async def test_search_dispatcher_routes_rerank_false_to_finalize_queue():
    """search_candidates.retrieved with rerank=False must publish a
    SearchFinalizeTask to cpu.search.finalize on the search exchange."""
    event = {
        "event_type": "search_candidates.retrieved",
        "query_job_id": "job-2",
        "candidates": [{"chunk_id": "c1", "score": 0.9}, {"chunk_id": "c2", "score": 0.7}],
        "k": 2,
        "rerank": False,
        "rerank_profile": None,
    }
    exchange_name, routing_key, task_dict = await _run_dispatcher_with_candidates_event(event)

    assert exchange_name == "search"
    assert routing_key == "cpu.search.finalize"

    task = SearchFinalizeTask.model_validate(task_dict)
    assert task.job_id == "job-2"
    assert len(task.candidates) == 2


@pytest.mark.asyncio
async def test_search_dispatcher_defaults_rerank_profile_to_default():
    """rerank=True with no rerank_profile should route via RERANK_CONFIGS['default']."""
    event = {
        "event_type": "search_candidates.retrieved",
        "query_job_id": "job-3",
        "candidates": [],
        "k": 3,
        "rerank": True,
        "rerank_profile": None,
    }
    exchange_name, routing_key, task_dict = await _run_dispatcher_with_candidates_event(event)

    assert routing_key == RERANK_CONFIGS["default"].queue


@pytest.mark.asyncio
async def test_search_dispatcher_skips_unknown_rerank_profile():
    """An unrecognised rerank_profile must be skipped — no task published, no crash."""
    exchanges: dict[str, AsyncMock] = {}

    def _declare_exchange(name, *_args, **_kwargs):
        if name not in exchanges:
            exchanges[name] = AsyncMock()
        return exchanges[name]

    mock_channel = AsyncMock()
    mock_channel.declare_exchange.side_effect = _declare_exchange
    mock_rmq = AsyncMock()
    mock_rmq.channel.return_value = mock_channel

    event = {
        "event_type": "search_candidates.retrieved",
        "query_job_id": "job-4",
        "candidates": [],
        "k": 3,
        "rerank": True,
        "rerank_profile": "no-such-profile",
    }
    def _subscribe(topic, **_kwargs):
        events = [event] if topic == "search_candidates.retrieved" else []
        return FakeEventBusSubscription(events)

    mock_bus = MagicMock()
    mock_bus.subscribe = MagicMock(side_effect=_subscribe)

    dispatcher = SearchDispatcher(mock_rmq, mock_bus)
    try:
        await asyncio.wait_for(dispatcher.run(), timeout=1.0)
    except asyncio.TimeoutError:
        pass

    for exchange in exchanges.values():
        exchange.publish.assert_not_called()
