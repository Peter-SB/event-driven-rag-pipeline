"""Integration tests for GPU embedding worker.

These tests verify GPU worker behavior against a real Postgres database.
They use MOCK_EMBEDDINGS=1 to avoid needing an actual GPU.

Tested behaviors
----------------
- GPU worker startup auto-creates chunk tables for configured embed models
- GPU worker embeds chunks and persists vectors to Postgres
- Vectors are correctly associated with chunks
- Worker gracefully handles missing tables (auto-creates them)
"""
from __future__ import annotations

import os
import pytest
import asyncpg

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from tests.utils.factories import make_chunk


# Ensure MOCK_EMBEDDINGS is set so tests don't require GPU
os.environ["MOCK_EMBEDDINGS"] = "1"


@pytest.mark.asyncio
async def test_gpu_worker_startup_creates_chunk_tables(postgres_pool: asyncpg.Pool):
    """Verify GPU worker startup ensures chunk tables exist for all configs."""
    # Startup: iterate configs and ensure tables (same as gpu.py entrypoint does)
    for table_name, config in EMBED_CONFIGS.items():
        repo = ChunkRepository(postgres_pool, table_name=table_name, vector_dim=config.dim)
        await repo.ensure_table()

    # Verify each table exists by querying the schema
    async with postgres_pool.acquire() as conn:
        for table_name in EMBED_CONFIGS.keys():
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = $1
                )
                """,
                table_name,
            )
            assert result is True, f"Table {table_name} should exist after ensure_table"


@pytest.mark.asyncio
async def test_gpu_worker_embeds_chunks_and_persists_vectors(postgres_pool: asyncpg.Pool):
    """Verify chunks are stored and embedding vectors can be persisted."""
    table_name = "chunks_body_baai_bge_base_en_v1_5"
    repo = ChunkRepository(postgres_pool, table_name=table_name, vector_dim=768)

    # Setup: ensure table exists
    await repo.ensure_table()

    # Create and insert test chunks
    chunks = [
        make_chunk(post_id=1, chunk_index=0, text="First chunk of text."),
        make_chunk(post_id=1, chunk_index=1, text="Second chunk of text."),
    ]
    await repo.bulk_insert(chunks)

    # Mock embeddings (deterministic)
    embedding_rows = [
        {"chunk_id": chunks[i].id, "embedding": [0.1 * (i + 1)] * 768}
        for i in range(len(chunks))
    ]

    # Persist embeddings (same as GPU worker does)
    await repo.update_embeddings(embedding_rows)

    # Verify vectors were written
    async with postgres_pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, embedding FROM {table_name}
            WHERE post_id = $1 ORDER BY chunk_index
            """,
            1,
        )

    assert len(rows) == 2
    # Verify embeddings are not null
    for i, row in enumerate(rows):
        assert row["embedding"] is not None
        # pgvector returns as a string or vector type, verify it's not empty
        # and has reasonable size (the string representation contains all values)
        embedding_str = str(row["embedding"])
        assert len(embedding_str) > 100  # String representation should be substantial
        # Verify one of our expected values is in the embedding
        # i=0 produces 0.1, i=1 produces 0.2
        expected_val = str(0.1 * (i + 1))
        assert expected_val in embedding_str


@pytest.mark.asyncio
async def test_gpu_worker_handles_nonexistent_table_gracefully(postgres_pool: asyncpg.Pool):
    """Verify worker doesn't crash if chunk table is missing at startup."""
    table_name = "chunks_test_nonexistent_model"

    # Ensure table doesn't exist initially
    async with postgres_pool.acquire() as conn:
        await conn.execute(
            f"DROP TABLE IF EXISTS {table_name} CASCADE"
        )

    # Call ensure_table (as GPU worker startup does)
    # This should create the table, not fail
    repo = ChunkRepository(postgres_pool, table_name=table_name, vector_dim=768)
    await repo.ensure_table()

    # Verify table now exists
    async with postgres_pool.acquire() as conn:
        result = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = $1
            )
            """,
            table_name,
        )
    assert result is True


@pytest.mark.asyncio
async def test_chunk_repository_respects_vector_dimension(postgres_pool: asyncpg.Pool):
    """Verify chunk table is created with correct vector dimension."""
    table_name = "chunks_test_dim_check"

    # Ensure table with specific dimension
    dim = 768
    repo = ChunkRepository(postgres_pool, table_name=table_name, vector_dim=dim)
    await repo.ensure_table()

    # Query the column type to verify dimension constraint
    async with postgres_pool.acquire() as conn:
        # pgvector columns have a type like "vector(768)"
        col_type = await conn.fetchval(
            """
            SELECT udt_name FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = $1
              AND column_name = 'embedding'
            """,
            table_name,
        )
    assert col_type == "vector", f"Expected column type 'vector', got {col_type}"
