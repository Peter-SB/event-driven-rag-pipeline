"""Integration tests: FallbackEmbeddingModel wired into EmbedHandler against real Postgres.

Uses httpx.MockTransport to stand in for the remote LM Studio endpoint (no
real network/hardware needed here — see test_remote_embedding_live.py for the
opt-in test against a genuine LM Studio instance) while exercising the real
ChunkRepository persistence path via a Postgres testcontainer.

Tested behaviours
------------------
- A healthy remote endpoint's vectors are the ones actually persisted to Postgres.
- A remote failure mid-batch falls back to the local model and the batch still
  succeeds — chunks end up with vectors, nothing is lost or requeued as failed.
"""
from __future__ import annotations

import httpx
import pytest
import asyncpg

from event_driven_rag_service.handlers.embed_handler import EmbedHandler
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.worker.remote_embedding import (
    FallbackEmbeddingModel,
    RemoteEmbeddingModel,
)
from tests.utils.factories import FakeEventBus, make_chunk

pytestmark = pytest.mark.integration

_MODEL_NAME = "Qwen3-Embedding-0.6B-Q8_0.gguf"
_DIM = 1024


class _FakeLocalModel:
    """Stand-in for the local SentenceTransformer model — records if it was used."""

    def __init__(self, name: str, dim: int) -> None:
        self._name = name
        self._dim = dim
        self.encode_calls = 0

    @property
    def name(self) -> str:
        return self._name

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.encode_calls += 1
        return [[0.5] * self._dim for _ in texts]


def _remote_client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://lm-studio.test/v1", transport=httpx.MockTransport(handler))


def _loaded_state_client() -> httpx.Client:
    """load_client stub reporting the target model as already loaded, so tests
    that only care about the /embeddings call don't need to also mock the
    native LM Studio load API."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": _MODEL_NAME, "state": "loaded"}]})

    return httpx.Client(base_url="http://lm-studio.test", transport=httpx.MockTransport(handler))


def _build_model(handler, local) -> FallbackEmbeddingModel:
    client = _remote_client(handler)
    remote = RemoteEmbeddingModel(client, _MODEL_NAME, load_client=_loaded_state_client())
    return FallbackEmbeddingModel(remote, local, "lm-studio.test")


@pytest.mark.asyncio
async def test_embed_chunks_persists_vectors_from_healthy_remote(postgres_pool: asyncpg.Pool):
    table_name = "chunks_remote_pipeline_healthy"
    repo = ChunkRepository(postgres_pool, table_name=table_name, vector_dim=_DIM)
    await repo.ensure_table()

    chunk = make_chunk(post_id=1, chunk_index=0, text="Title: remote pipeline test")
    await repo.bulk_insert([chunk])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [0.9] * _DIM}]})

    local = _FakeLocalModel(_MODEL_NAME, _DIM)
    model = _build_model(handler, local)

    embed_handler = EmbedHandler(chunk_fetcher=repo, embedding_store=repo, event_log=FakeEventBus())
    task = EmbedTask(
        task_id="t1",
        task_type="chunk",
        model_name=_MODEL_NAME,
        post_id=1,
        post_table="posts_test",
        chunk_ids=[chunk.id],
        chunk_table=table_name,
    )

    ok, failed = await embed_handler.embed_chunks([task], _MODEL_NAME, model)

    assert failed == []
    assert len(ok) == 1
    assert local.encode_calls == 0, "remote was healthy — local model should never run"

    async with postgres_pool.acquire() as conn:
        row = await conn.fetchrow(f"SELECT embedding FROM {table_name} WHERE id = $1", chunk.id)
    assert row["embedding"] is not None
    assert "0.9" in str(row["embedding"])


@pytest.mark.asyncio
async def test_embed_chunks_falls_back_to_local_when_remote_down(postgres_pool: asyncpg.Pool):
    table_name = "chunks_remote_pipeline_fallback"
    repo = ChunkRepository(postgres_pool, table_name=table_name, vector_dim=_DIM)
    await repo.ensure_table()

    chunk = make_chunk(post_id=2, chunk_index=0, text="Title: fallback pipeline test")
    await repo.bulk_insert([chunk])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    local = _FakeLocalModel(_MODEL_NAME, _DIM)
    model = _build_model(handler, local)

    embed_handler = EmbedHandler(chunk_fetcher=repo, embedding_store=repo, event_log=FakeEventBus())
    task = EmbedTask(
        task_id="t2",
        task_type="chunk",
        model_name=_MODEL_NAME,
        post_id=2,
        post_table="posts_test",
        chunk_ids=[chunk.id],
        chunk_table=table_name,
    )

    ok, failed = await embed_handler.embed_chunks([task], _MODEL_NAME, model)

    assert failed == [], "batch should still succeed via the local fallback"
    assert len(ok) == 1
    assert local.encode_calls == 1

    async with postgres_pool.acquire() as conn:
        row = await conn.fetchrow(f"SELECT embedding FROM {table_name} WHERE id = $1", chunk.id)
    assert row["embedding"] is not None
    assert "0.5" in str(row["embedding"])
