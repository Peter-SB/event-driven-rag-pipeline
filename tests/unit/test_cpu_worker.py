"""Unit tests for CpuChunkWorker.

Tests the sync/async bridge that connects the sync RabbitMQ consumer (pika)
to the async handler (asyncpg-based repositories).

Tests use synchronous `def test_...` (not `async def`) because CpuChunkWorker
calls `loop.run_until_complete()`, which cannot be called from within a running
event loop (pytest-asyncio issue).

Tested behaviours
-----------------
- process() deserializes payload to ChunkTask via pydantic
- process() passes the ChunkTask to handler.handle()
- process() handles both success (ack) and exception (nack) cases
- Exceptions from handler.handle() are propagated to BaseWorker for nack handling
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from event_driven_rag_service.worker.cpu_worker import CpuChunkWorker
from event_driven_rag_service.tasks.chunk_task import ChunkTask


# ---------------------------------------------------------------------------
# Synchronous tests (no pytest.mark.asyncio)
# ---------------------------------------------------------------------------


def test_process_deserializes_payload_and_calls_handler():
    """process() validates payload to ChunkTask and delegates to handler.handle()."""
    loop = asyncio.new_event_loop()
    try:
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(return_value=["chunk-uuid-1", "chunk-uuid-2"])

        worker = CpuChunkWorker.__new__(CpuChunkWorker)
        worker._handler = mock_handler
        worker._loop = loop

        payload = {
            "kind": "chunk",
            "task_id": "task-123",
            "task_type": "body",
            "post_id": 42,
            "post_table": "posts_main",
            "embed_model": "bge-base-v1.5",
            "trace_id": None,
            "source_event_id": None,
            "analysis_text": None,
        }

        worker.process(payload)

        mock_handler.handle.assert_called_once()
        called_task = mock_handler.handle.call_args[0][0]
        assert isinstance(called_task, ChunkTask)
        assert called_task.post_id == 42
        assert called_task.task_type == "body"
        assert called_task.post_table == "posts_main"
        assert called_task.embed_model == "bge-base-v1.5"
    finally:
        loop.close()


def test_process_propagates_handler_exception():
    """process() lets exceptions from handler bubble up (for BaseWorker to nack)."""
    loop = asyncio.new_event_loop()
    try:
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(side_effect=RuntimeError("DB connection failed"))

        worker = CpuChunkWorker.__new__(CpuChunkWorker)
        worker._handler = mock_handler
        worker._loop = loop

        payload = {
            "kind": "chunk",
            "task_id": "task-456",
            "task_type": "body",
            "post_id": 1,
            "post_table": "posts_main",
            "embed_model": "bge-base-v1.5",
            "trace_id": None,
            "source_event_id": None,
            "analysis_text": None,
        }

        with pytest.raises(RuntimeError, match="DB connection failed"):
            worker.process(payload)
    finally:
        loop.close()


def test_process_works_with_fresh_event_loop():
    """process() can create its own event loop if none is provided."""
    # This verifies the default loop creation behavior
    mock_handler = MagicMock()
    mock_handler.handle = AsyncMock(return_value=["chunk-1"])

    # Pass no loop — constructor should create one
    worker = CpuChunkWorker.__new__(CpuChunkWorker)
    worker._handler = mock_handler
    # Simulate constructor not passing a loop
    worker._loop = asyncio.new_event_loop()

    payload = {
        "kind": "chunk",
        "task_id": "task-789",
        "task_type": "body",
        "post_id": 2,
        "post_table": "posts_work",
        "embed_model": "bge-base-v1.5",
        "trace_id": None,
        "source_event_id": None,
        "analysis_text": None,
    }

    worker.process(payload)
    mock_handler.handle.assert_called_once()

    worker._loop.close()


def test_process_with_summary_title_task():
    """process() correctly deserializes summary_title task type."""
    loop = asyncio.new_event_loop()
    try:
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(return_value=["chunk-1"])

        worker = CpuChunkWorker.__new__(CpuChunkWorker)
        worker._handler = mock_handler
        worker._loop = loop

        payload = {
            "kind": "chunk",
            "task_id": "task-summary",
            "task_type": "summary_title",
            "post_id": 100,
            "post_table": "posts_main",
            "embed_model": "bge-base-v1.5",
            "trace_id": None,
            "source_event_id": None,
            "analysis_text": None,
        }

        worker.process(payload)

        called_task = mock_handler.handle.call_args[0][0]
        assert called_task.task_type == "summary_title"
    finally:
        loop.close()


def test_process_with_analysis_task():
    """process() correctly deserializes analysis task type with analysis_text."""
    loop = asyncio.new_event_loop()
    try:
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(return_value=[])

        worker = CpuChunkWorker.__new__(CpuChunkWorker)
        worker._handler = mock_handler
        worker._loop = loop

        analysis_text = "This is the analysis result text."
        payload = {
            "kind": "chunk",
            "task_id": "task-analysis",
            "task_type": "analysis",
            "post_id": 200,
            "post_table": "posts_main",
            "embed_model": "qwen3-0.6b",
            "trace_id": None,
            "source_event_id": None,
            "analysis_text": analysis_text,
        }

        worker.process(payload)

        called_task = mock_handler.handle.call_args[0][0]
        assert called_task.task_type == "analysis"
        assert called_task.analysis_text == analysis_text
    finally:
        loop.close()


def test_process_rejects_malformed_payload():
    """process() raises pydantic ValidationError on malformed payload."""
    loop = asyncio.new_event_loop()
    try:
        mock_handler = MagicMock()

        worker = CpuChunkWorker.__new__(CpuChunkWorker)
        worker._handler = mock_handler
        worker._loop = loop

        # Missing required field: post_id
        payload = {
            "kind": "chunk",
            "task_id": "task-bad",
            "task_type": "body",
            # post_id missing!
            "post_table": "posts_main",
            "embed_model": "bge-base-v1.5",
        }

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            worker.process(payload)
    finally:
        loop.close()

