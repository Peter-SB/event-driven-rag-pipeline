"""Integration tests for SearchJobRepository against a real Postgres testcontainer."""
from __future__ import annotations

import pytest
import asyncpg

from event_driven_rag_service.repository.search_job_repository import SearchJobRepository

pytestmark = pytest.mark.integration

_CHUNKS_TABLE = "posts_test_chunks_body_baai_bge_base_en_v1_5"
_EMBEDDING_PROFILE = "BAAI/bge-base-en-v1.5"
_LIBRARY_ID = "testlib"


@pytest.fixture
async def job_repo(postgres_pool: asyncpg.Pool):
    """Fresh SearchJobRepository with a clean table for each test."""
    repo = SearchJobRepository(postgres_pool)
    await repo.ensure_table()
    # Truncate so each test starts from an empty table
    async with postgres_pool.acquire() as conn:
        await conn.execute("TRUNCATE search_jobs")
    yield repo
    async with postgres_pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS search_jobs")


# ---------------------------------------------------------------------------
# ensure_table
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_table_is_idempotent(postgres_pool: asyncpg.Pool):
    """ensure_table should not raise when called multiple times."""
    repo = SearchJobRepository(postgres_pool)
    await repo.ensure_table()
    await repo.ensure_table()  # second call must not fail

    async with postgres_pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'search_jobs'"
        )
    assert count == 1

    async with postgres_pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS search_jobs")


# ---------------------------------------------------------------------------
# create_job / get_job
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_job_returns_uuid_string(job_repo: SearchJobRepository):
    job_id = await job_repo.create_job(
        query="test query",
        k=5,
        embedding_profile=_EMBEDDING_PROFILE,
        chunks_table=_CHUNKS_TABLE,
        library_id=_LIBRARY_ID,
    )
    assert isinstance(job_id, str)
    import uuid
    uuid.UUID(job_id)  # must be valid UUID


@pytest.mark.asyncio
async def test_get_job_returns_correct_fields(job_repo: SearchJobRepository):
    job_id = await job_repo.create_job(
        query="What is RAG?",
        k=10,
        embedding_profile=_EMBEDDING_PROFILE,
        chunks_table=_CHUNKS_TABLE,
        library_id=_LIBRARY_ID,
    )
    job = await job_repo.get_job(job_id)

    assert job is not None
    assert str(job["id"]) == job_id
    assert job["status"] == "embedding"
    assert job["query"] == "What is RAG?"
    assert job["k"] == 10
    assert job["embedding_profile"] == _EMBEDDING_PROFILE
    assert job["chunks_table"] == _CHUNKS_TABLE
    assert job["library_id"] == _LIBRARY_ID
    assert job["embedding"] is None
    assert job["results"] is None
    assert job["error"] is None


@pytest.mark.asyncio
async def test_get_job_returns_none_for_missing_id(job_repo: SearchJobRepository):
    result = await job_repo.get_job("00000000-0000-0000-0000-000000000000")
    assert result is None


# ---------------------------------------------------------------------------
# store_embedding
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_embedding_persists_vector(job_repo: SearchJobRepository):
    job_id = await job_repo.create_job(
        query="embed me", k=5, embedding_profile=_EMBEDDING_PROFILE,
        chunks_table=_CHUNKS_TABLE, library_id=_LIBRARY_ID,
    )
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    await job_repo.store_embedding(job_id, embedding)

    job = await job_repo.get_job(job_id)
    assert job["embedding"] is not None
    stored = job["embedding"]
    assert len(stored) == 5
    assert all(abs(a - b) < 1e-6 for a, b in zip(stored, embedding))


# ---------------------------------------------------------------------------
# save_batch (EmbeddingStore protocol)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_batch_handles_query_row(job_repo: SearchJobRepository):
    """save_batch with QueryEmbeddingRow should store the embedding."""
    job_id = await job_repo.create_job(
        query="batch embed test", k=3, embedding_profile=_EMBEDDING_PROFILE,
        chunks_table=_CHUNKS_TABLE, library_id=_LIBRARY_ID,
    )
    embedding = [float(i) / 100 for i in range(10)]
    await job_repo.save_batch([
        {"query_job_id": job_id, "model_name": _EMBEDDING_PROFILE, "embedding": embedding}
    ])

    job = await job_repo.get_job(job_id)
    assert job["embedding"] is not None
    assert len(job["embedding"]) == 10


@pytest.mark.asyncio
async def test_save_batch_ignores_chunk_rows(job_repo: SearchJobRepository):
    """save_batch should silently ignore rows with chunk_id (those go to ChunkRepository)."""
    await job_repo.save_batch([
        {"chunk_id": "abc", "model_name": "BAAI/bge-base-en-v1.5", "embedding": [0.1], "chunk_table": "t"}
    ])
    # No exception raised — chunk rows are quietly ignored


# ---------------------------------------------------------------------------
# mark_searching / complete_job / fail_job
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mark_searching_transitions_status(job_repo: SearchJobRepository):
    job_id = await job_repo.create_job(
        query="q", k=5, embedding_profile=_EMBEDDING_PROFILE,
        chunks_table=_CHUNKS_TABLE, library_id=_LIBRARY_ID,
    )
    await job_repo.mark_searching(job_id)

    job = await job_repo.get_job(job_id)
    assert job["status"] == "searching"


@pytest.mark.asyncio
async def test_complete_job_stores_results_and_sets_status(job_repo: SearchJobRepository):
    job_id = await job_repo.create_job(
        query="q", k=5, embedding_profile=_EMBEDDING_PROFILE,
        chunks_table=_CHUNKS_TABLE, library_id=_LIBRARY_ID,
    )
    results = [{"chunk_id": "c1", "post_id": 1, "text": "hello", "metadata": None, "score": 0.9}]
    await job_repo.complete_job(job_id, results)

    job = await job_repo.get_job(job_id)
    assert job["status"] == "complete"
    assert job["results"] is not None
    assert len(job["results"]) == 1
    assert job["results"][0]["chunk_id"] == "c1"
    assert job["completed_at"] is not None


@pytest.mark.asyncio
async def test_fail_job_stores_error_and_sets_status(job_repo: SearchJobRepository):
    job_id = await job_repo.create_job(
        query="q", k=5, embedding_profile=_EMBEDDING_PROFILE,
        chunks_table=_CHUNKS_TABLE, library_id=_LIBRARY_ID,
    )
    await job_repo.fail_job(job_id, "something went wrong")

    job = await job_repo.get_job(job_id)
    assert job["status"] == "failed"
    assert job["error"] == "something went wrong"
    assert job["completed_at"] is not None
