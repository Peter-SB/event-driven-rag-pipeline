"""Unit tests verifying that GpuEmbedWorker._process_batch processes query tasks
before chunk tasks, regardless of the order they arrive in the batch.

No real GPU, RabbitMQ, or Postgres needed — only the worker's _process_batch
logic and a mock EmbedHandler that records call order.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.worker.gpu_worker import GpuEmbedWorker, WorkerMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(task_type: str, idx: int) -> EmbedTask:
    """Create a minimal EmbedTask of the given type."""
    if task_type == "chunk":
        return EmbedTask(
            task_type="chunk",
            model_name="BAAI/bge-base-en-v1.5",
            post_id=idx,
            post_table="posts_test",
            chunk_ids=[f"chunk-{idx}"],
            chunk_table="posts_test_chunks_body_baai_bge_base_en_v1_5",
        )
    return EmbedTask(
        task_type="query",
        model_name="BAAI/bge-base-en-v1.5",
        query=f"query text {idx}",
        query_job_id=f"job-{idx}",
    )


def _make_msg(task: EmbedTask, delivery_tag: int) -> WorkerMessage:
    return WorkerMessage(task=task, delivery_tag=delivery_tag)


class _MockModel:
    """Minimal duck-type for EmbeddingModel."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 3 for _ in texts]


class _CallRecorder:
    """Records (task_type, call_index) so we can verify ordering."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed_query(self, task: EmbedTask, model_name: str, model: Any) -> bool:
        self.calls.append("query")
        return True

    async def embed_chunks(
        self, tasks: list[EmbedTask], model_name: str, model: Any
    ) -> tuple[list[EmbedTask], list[EmbedTask]]:
        self.calls.extend(["chunk"] * len(tasks))
        return tasks, []


def _make_worker(recorder: _CallRecorder) -> GpuEmbedWorker:
    loop = asyncio.new_event_loop()
    worker = GpuEmbedWorker(
        rabbitmq_url="amqp://localhost",
        model_queues=[("BAAI/bge-base-en-v1.5", "gpu.embed.bge-base-en-v1.5")],
        model_loader=lambda name: _MockModel(),
        handler=recorder,  # type: ignore[arg-type]
        loop=loop,
        max_batch=32,
    )
    worker._model = _MockModel()
    worker._current_model_name = "BAAI/bge-base-en-v1.5"
    return worker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_query_tasks_processed_before_chunk_tasks_mixed_batch():
    """A mixed batch of chunk+query tasks should call embed_query first."""
    recorder = _CallRecorder()
    worker = _make_worker(recorder)

    batch = [
        _make_msg(_make_task("chunk", 1), delivery_tag=1),
        _make_msg(_make_task("query", 2), delivery_tag=2),
        _make_msg(_make_task("chunk", 3), delivery_tag=3),
        _make_msg(_make_task("query", 4), delivery_tag=4),
    ]

    ok, failed = worker._process_batch(batch, "BAAI/bge-base-en-v1.5")

    assert len(ok) == 4
    assert len(failed) == 0
    # First two calls must be query, last two chunk
    assert recorder.calls[:2] == ["query", "query"]
    assert recorder.calls[2:] == ["chunk", "chunk"]


def test_query_only_batch_processed_correctly():
    """A batch of only query tasks should all be processed as queries."""
    recorder = _CallRecorder()
    worker = _make_worker(recorder)

    batch = [_make_msg(_make_task("query", i), delivery_tag=i) for i in range(3)]
    ok, failed = worker._process_batch(batch, "BAAI/bge-base-en-v1.5")

    assert len(ok) == 3
    assert len(failed) == 0
    assert all(c == "query" for c in recorder.calls)


def test_chunk_only_batch_processed_correctly():
    """A batch of only chunk tasks should all be processed as chunks."""
    recorder = _CallRecorder()
    worker = _make_worker(recorder)

    batch = [_make_msg(_make_task("chunk", i), delivery_tag=i) for i in range(4)]
    ok, failed = worker._process_batch(batch, "BAAI/bge-base-en-v1.5")

    assert len(ok) == 4
    assert len(failed) == 0
    assert all(c == "chunk" for c in recorder.calls)


def test_query_tasks_processed_before_chunks_even_when_chunks_arrive_first():
    """Order in the batch should not matter — queries always run before chunks."""
    recorder = _CallRecorder()
    worker = _make_worker(recorder)

    # All chunks first, then one query at the end
    batch = [
        _make_msg(_make_task("chunk", 1), delivery_tag=1),
        _make_msg(_make_task("chunk", 2), delivery_tag=2),
        _make_msg(_make_task("chunk", 3), delivery_tag=3),
        _make_msg(_make_task("query", 4), delivery_tag=4),
    ]

    ok, failed = worker._process_batch(batch, "BAAI/bge-base-en-v1.5")

    assert len(ok) == 4
    assert recorder.calls[0] == "query", "query must be first even though it was last in the batch"
    assert recorder.calls[1:] == ["chunk", "chunk", "chunk"]


def test_failed_query_ends_up_in_failed_list():
    """embed_query returning False should place the message in the failed list."""

    class _FailingRecorder(_CallRecorder):
        async def embed_query(self, task, model_name, model):
            self.calls.append("query")
            return False  # signal failure

    recorder = _FailingRecorder()
    worker = _make_worker(recorder)

    batch = [
        _make_msg(_make_task("query", 1), delivery_tag=1),
        _make_msg(_make_task("chunk", 2), delivery_tag=2),
    ]

    ok, failed = worker._process_batch(batch, "BAAI/bge-base-en-v1.5")

    assert len(failed) == 1
    assert failed[0].delivery_tag == 1
    assert len(ok) == 1
    assert ok[0].delivery_tag == 2
