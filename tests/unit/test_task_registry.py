"""Tests for task models and the task registry.

Covers ChunkTask, EmbedTask construction, the TASK_ROUTES lookup, and
the parse_task() discriminated-union deserialiser used by workers.

Tested behaviours
-----------------
- ChunkTask.chunk_table_name() produces expected table name format
- TASK_ROUTES["embed"] has no static routing_key (must be resolved from
  EMBED_CONFIGS[...].queue by the dispatcher, not derived from model_name)
- parse_task() returns correct concrete type based on ``kind`` field
- parse_task() raises ValidationError for unknown or missing kind
- BaseTask auto-generates a unique task_id per instance
- model_dump_json() round-trips cleanly back through parse_task()
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from event_driven_rag_service.tasks.chunk_task import ChunkTask
from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.tasks.registry import TASK_ROUTES, parse_task


# ---------------------------------------------------------------------------
# ChunkTask
# ---------------------------------------------------------------------------

def test_chunk_task_table_name_body():
    task = ChunkTask(task_type="body", post_id=1, post_table="posts_main", embed_model="BAAI/bge-base-en-v1.5")
    assert task.chunk_table_name() == "posts_main_chunks_body_baai_bge_base_en_v1_5"


def test_chunk_task_table_name_summary_title():
    task = ChunkTask(task_type="summary_title", post_id=1, post_table="posts_main", embed_model="BAAI/bge-base-en-v1.5")
    assert task.chunk_table_name() == "posts_main_chunks_summary_title_baai_bge_base_en_v1_5"


def test_chunk_task_table_name_analysis_with_different_model():
    task = ChunkTask(task_type="analysis", post_id=1, post_table="posts_main", embed_model="Qwen/Qwen3-0.6B")
    assert task.chunk_table_name() == "posts_main_chunks_analysis_qwen_qwen3_0_6b"


def test_chunk_task_kind_is_always_chunk():
    task = ChunkTask(task_type="body", post_id=99, post_table="posts", embed_model="BAAI/bge-base-en-v1.5")
    assert task.kind == "chunk"


def test_chunk_task_gets_unique_task_ids():
    t1 = ChunkTask(task_type="body", post_id=1, post_table="posts", embed_model="BAAI/bge-base-en-v1.5")
    t2 = ChunkTask(task_type="body", post_id=1, post_table="posts", embed_model="BAAI/bge-base-en-v1.5")
    # Each instance must have its own UUID, not a shared class-level value
    assert t1.task_id != t2.task_id


# ---------------------------------------------------------------------------
# EmbedTask
# ---------------------------------------------------------------------------

def test_embed_task_kind_is_always_embed():
    task = EmbedTask(task_type="chunk", model_name="BAAI/bge-base-en-v1.5", chunk_ids=["a", "b"], chunk_table="chunks_body_baai_bge_base_en_v1_5")
    assert task.kind == "embed"


def test_embed_route_has_no_static_routing_key():
    """The embed route's queue depends on task_type (EMBED_CONFIGS[...].queue),
    not on a model-name-derived template — see ChunkDispatcher/SearchDispatcher."""
    assert TASK_ROUTES["embed"].routing_key is None


# ---------------------------------------------------------------------------
# TASK_ROUTES
# ---------------------------------------------------------------------------

def test_task_routes_contains_chunk_and_embed():
    assert "chunk" in TASK_ROUTES
    assert "embed" in TASK_ROUTES


def test_chunk_route_points_to_ingestion_exchange():
    assert TASK_ROUTES["chunk"].exchange == "ingestion"


def test_chunk_route_routing_key():
    assert TASK_ROUTES["chunk"].routing_key == "cpu.chunk.post"


# ---------------------------------------------------------------------------
# parse_task — discriminated union deserialiser
# ---------------------------------------------------------------------------

def test_parse_task_returns_chunk_task_for_kind_chunk():
    payload = {
        "kind": "chunk",
        "task_type": "body",
        "post_id": 42,
        "post_table": "posts",
        "embed_model": "BAAI/bge-base-en-v1.5",
    }
    task = parse_task(payload)
    assert isinstance(task, ChunkTask)
    assert task.post_id == 42


def test_parse_task_returns_embed_task_for_kind_embed():
    payload = {
        "kind": "embed",
        "task_type": "chunk",
        "model_name": "BAAI/bge-base-en-v1.5",
        "chunk_ids": ["id1", "id2"],
        "chunk_table": "chunks_body_baai_bge_base_en_v1_5",
    }
    task = parse_task(payload)
    assert isinstance(task, EmbedTask)
    assert task.model_name == "BAAI/bge-base-en-v1.5"


def test_parse_task_raises_on_unknown_kind():
    with pytest.raises(ValidationError):
        parse_task({"kind": "totally_unknown"})


def test_parse_task_raises_on_missing_kind():
    with pytest.raises(ValidationError):
        parse_task({"task_type": "body", "post_id": 1})


def test_chunk_task_round_trips_through_json():
    original = ChunkTask(
        task_type="body",
        post_id=7,
        post_table="posts",
        embed_model="BAAI/bge-base-en-v1.5",
        trace_id="trace-abc",
    )
    restored = parse_task(json.loads(original.model_dump_json()))
    assert isinstance(restored, ChunkTask)
    assert restored.post_id == original.post_id
    assert restored.trace_id == original.trace_id
    assert restored.task_id == original.task_id
