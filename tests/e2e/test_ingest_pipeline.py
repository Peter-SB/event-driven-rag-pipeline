"""End-to-end tests for the complete ingest pipeline.

These tests verify the full flow: POST /sync → chunks → embeddings.
They require docker-compose to be running with MOCK_EMBEDDINGS=1 on GPU worker.

Tested flow
-----------
1. API receives POST /sync with a list of posts
2. PostDispatcher reads post.synced events and queues chunk tasks
3. CpuChunkWorker processes chunk tasks and creates chunks
4. ChunkDispatcher reads chunks.created events and queues embed tasks
5. GpuEmbedWorker processes embed tasks and persists vectors
6. Embeddings appear in Postgres chunk tables
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, UTC

import pytest
import asyncpg
import httpx

from event_driven_rag_service.data_models.post import Post
from event_driven_rag_service.repository.chunk_repository import ChunkRepository

logger = logging.getLogger(__name__)

# Ensure MOCK_EMBEDDINGS is set for GPU worker
os.environ["MOCK_EMBEDDINGS"] = "1"


@pytest.mark.asyncio
async def test_sync_post_triggers_chunk_and_embed_pipeline(
    postgres_pool_e2e: asyncpg.Pool,
    async_client: httpx.AsyncClient,
):
    """
    Full pipeline: POST /sync → chunks → embeddings.

    Flow:
    1. POST /posts/sync with a new post
    2. Poll until chunks are created (up to 10 seconds)
    3. Verify chunks exist with correct structure
    4. Poll until embeddings are written (up to 10 seconds)
    5. Verify embeddings exist and are non-null vectors
    """
    post_id = 100  # Use unique ID to avoid test interference

    # Step 1: Create and sync a post
    sync_payload = {
        "posts": [
            {
                "id": post_id,
                "redditId": f"reddit_{post_id}",
                "externalSource": "reddit",
                "redditCreatedAt": datetime.now(UTC).isoformat(),
                "url": f"https://reddit.com/r/test/comments/{post_id}",
                "title": "Test Post for E2E Pipeline",
                "bodyText": (
                    "This is a test post with enough content to be chunked. "
                    "We need multiple sentences so the boundary chunker "
                    "produces more than one chunk. "
                    "This helps verify the full pipeline. "
                    "Another paragraph here. " * 5
                ),
                "author": "test_user",
                "addedAt": datetime.now(UTC).isoformat(),
                "updatedAt": datetime.now(UTC).isoformat(),
            }
        ],
        "library_id": "e2e",
    }

    response = await async_client.post("/posts/sync", json=sync_payload)
    assert response.status_code == 200
    sync_result = response.json()
    assert len(sync_result["results"]) == 1
    assert sync_result["results"][0]["success"] is True
    logger.info(f"POST /sync succeeded for post_id={post_id}")

    # Step 1.2: Verify post exists in DB (optional sanity check)
    async with postgres_pool_e2e.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT post_id, title FROM posts_e2e WHERE post_id = $1", post_id
        )
        assert row is not None, "Post should be inserted into DB"
        assert row["post_id"] == post_id
        assert row["title"] == sync_payload["posts"][0]["title"]

    # Step 2.1: Check chunk table exists (it may be created lazily by the dispatcher)
    chunk_table = "posts_e2e_chunks_body_bge_base_v1_5"
    table_exists = False
    for attempt in range(20):
        async with postgres_pool_e2e.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*)::int FROM information_schema.tables WHERE table_name = $1",
                chunk_table,
            )
            if count == 1:
                table_exists = True
                break

        await asyncio.sleep(0.5)
    else:
        logger.error(f"Chunk table {chunk_table} does not exist after 10s")
        async with postgres_pool_e2e.acquire() as conn:
            chunk_tables = await conn.fetch(
                "SELECT table_name FROM information_schema.tables"
            )
        logger.error(f"Existing tables: {[row['table_name'] for row in chunk_tables]}")
    
    assert table_exists, f"Chunk table {chunk_table} does not exist after 10s"

    # Step 2.2: Poll for chunks to be created (up to 10 seconds)
    chunks = []
    for attempt in range(20):  # 20 attempts * 0.5s = 10s max wait
        async with postgres_pool_e2e.acquire() as conn:
            try:
                rows = await conn.fetch(
                    f"""
                    SELECT post_id, text FROM {chunk_table}
                    WHERE post_id = $1 ORDER BY chunk_index
                    """,
                    post_id,
                )
                if rows:
                    chunks = rows
                    break
            except asyncpg.exceptions.UndefinedTableError:
                # Table might not exist yet if dispatcher hasn't run
                pass

        await asyncio.sleep(0.5)

    assert len(chunks) > 0, f"No chunks found after 10s for post_id={post_id}"
    logger.info(f"Found {len(chunks)} chunks for post_id={post_id}")

    # Step 3: Verify chunks have valid structure
    chunk_ids = [c["id"] for c in chunks]
    assert all(isinstance(cid, str) for cid in chunk_ids)
    assert all(c["text"] and len(c["text"]) > 0 for c in chunks)
    logger.info(f"Chunk validation passed: {len(chunk_ids)} chunks with valid structure")

    # Step 4: Poll for embeddings to be written (up to 10 seconds)
    embeddings = []
    for attempt in range(20):  # 20 attempts * 0.5s = 10s max wait
        async with postgres_pool_e2e.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, embedding FROM {chunk_table}
                WHERE post_id = $1 AND embedding IS NOT NULL
                """,
                post_id,
            )
            if len(rows) == len(chunk_ids):
                # All chunks have embeddings
                embeddings = rows
                break

        await asyncio.sleep(0.5)

    assert (
        len(embeddings) == len(chunk_ids)
    ), f"Expected {len(chunk_ids)} embeddings, found {len(embeddings)}"
    logger.info(f"All {len(embeddings)} chunks have embeddings")

    # Step 5: Verify embeddings are valid vectors
    repo = ChunkRepository(postgres_pool_e2e)
    embedding_dim = 768  # bge-base-v1.5 dimension
    for row in embeddings:
        embedding = row["embedding"]
        assert embedding is not None, "Embedding should not be null"
        assert len(embedding) == embedding_dim, (
            f"Embedding dimension should be {embedding_dim}, got {len(embedding)}"
        )
        assert all(isinstance(v, float) for v in embedding), "Embedding should contain floats"

    logger.info("E2E pipeline test completed successfully")


@pytest.mark.asyncio
async def test_multiple_posts_sync_concurrently(
    postgres_pool_e2e: asyncpg.Pool,
    async_client: httpx.AsyncClient,
):
    """
    Verify pipeline handles multiple posts synced simultaneously.

    Posts should be chunked and embedded independently, without interference.
    """
    post_ids = [200, 201, 202]

    # Sync all posts at once
    posts_payload = []
    for pid in post_ids:
        posts_payload.append(
            {
                "id": pid,
                "redditId": f"reddit_{pid}",
                "externalSource": "reddit",
                "redditCreatedAt": datetime.now(UTC).isoformat(),
                "url": f"https://reddit.com/r/test/comments/{pid}",
                "title": f"Post {pid}",
                "bodyText": f"Content for post {pid}. " * 10,
                "author": "test_user",
                "addedAt": datetime.now(UTC).isoformat(),
                "updatedAt": datetime.now(UTC).isoformat(),
            }
        )

    response = await async_client.post(
        "/posts/sync", json={"posts": posts_payload, "library_id": "e2e"}
    )
    assert response.status_code == 200
    logger.info(f"Synced {len(post_ids)} posts concurrently")

    # Verify all posts were chunked independently
    chunk_table = "posts_e2e_chunks_body_bge_base_v1_5"
    for pid in post_ids:
        chunks = []
        for attempt in range(20):
            try:
                async with postgres_pool_e2e.acquire() as conn:
                    rows = await conn.fetch(
                        f"SELECT COUNT(*) as cnt FROM {chunk_table} WHERE post_id = $1",
                        pid,
                    )
                    if rows[0]["cnt"] > 0:
                        chunks = rows
                        break
            except asyncpg.exceptions.UndefinedTableError:
                pass

            await asyncio.sleep(0.5)

        assert len(chunks) > 0, f"No chunks for post_id={pid} after 10s"

    logger.info("Multiple posts concurrent sync completed successfully")
