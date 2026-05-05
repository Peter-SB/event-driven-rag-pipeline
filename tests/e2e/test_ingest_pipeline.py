"""E2E tests for the complete ingest pipeline.

Verifies the full pipeline against the running Docker Compose stack:

    POST /posts/sync → post.synced event → chunk tasks → chunks in Postgres
                     → embed tasks → embedding vectors in Postgres

Requires the full stack running with MOCK_EMBEDDINGS=1 on the GPU worker
(set in docker-compose.yml) so tests complete in seconds without real GPU:

    docker compose up -d
    pytest tests/e2e/ -m e2e

Tested flow
-----------
1. POST /posts/sync — API persists post and publishes post.synced
2. PostDispatcher reads event and queues chunk tasks via RabbitMQ
3. CpuChunkWorker processes tasks and writes chunks to Postgres
4. ChunkDispatcher reads chunks.created event and queues embed tasks
5. GpuEmbedWorker embeds (mock) and persists vectors to chunk table

Event assertions
----------------
At each pipeline stage we verify the corresponding event was published
to the event_log with the correct payload structure and content.
"""
from __future__ import annotations

import asyncio
import json
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def fetch_events(
    pool: asyncpg.Pool,
    topic: str,
    post_id: int,
    timeout: float = 15.0,
    interval: float = 0.5,
) -> list[dict]:
    """Poll the event_log table for events matching *topic* and *post_id*.

    Returns all matching events (oldest first) once at least one is found,
    or raises AssertionError after *timeout* seconds.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT payload FROM event_log
                WHERE topic = $1
                  AND payload->>'post_id' = $2::text
                ORDER BY id ASC
                """,
                topic,
                str(post_id),
            )
        if rows:
            return [json.loads(r["payload"]) for r in rows]
        await asyncio.sleep(interval)

    raise AssertionError(
        f"No {topic} events found for post_id={post_id} after {timeout}s"
    )


async def fetch_events_raw(
    pool: asyncpg.Pool,
    topic: str,
    timeout: float = 15.0,
    interval: float = 0.5,
) -> list[dict]:
    """Poll the event_log table for all events matching *topic*.

    Returns all matching events (oldest first) once at least one is found,
    or raises AssertionError after *timeout* seconds.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT payload FROM event_log
                WHERE topic = $1
                ORDER BY id ASC
                """,
                topic,
            )
        if rows:
            return [json.loads(r["payload"]) for r in rows]
        await asyncio.sleep(interval)

    raise AssertionError(f"No {topic} events found after {timeout}s")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_post_triggers_chunk_and_embed_pipeline(
    postgres_pool_e2e: asyncpg.Pool,
    async_client: httpx.AsyncClient,
):
    """
    Full pipeline: POST /sync → chunks → embeddings.

    Flow:
    1. POST /posts/sync with a new post
    2. Verify post.synced event was published with correct payload
    3. Poll until chunks are created (up to 10 seconds)
    4. Verify chunks.created event was published with correct payload
    5. Verify chunks exist with correct structure
    6. Poll until embeddings are written (up to 10 seconds)
    7. Verify embedding.completed event was published with correct payload
    8. Verify embeddings exist and are non-null vectors
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
    assert sync_result["results"][0]["success"] is True, f"POST /posts/sync should succeed, got response: {sync_result}"
    logger.info(f"POST /sync succeeded for post_id={post_id}")

    # Step 1.2: Verify post exists in DB (optional sanity check)
    async with postgres_pool_e2e.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT post_id, title FROM posts_e2e WHERE post_id = $1", post_id
        )
        assert row is not None, "Post should be inserted into DB"
        assert row["post_id"] == post_id
        assert row["title"] == sync_payload["posts"][0]["title"]

    # Step 2: Verify post.synced event was published
    post_synced_events = await fetch_events(
        postgres_pool_e2e, "post.synced", post_id, timeout=5.0
    )
    assert len(post_synced_events) >= 1, "Expected at least one post.synced event"
    ps_event = post_synced_events[0]
    assert ps_event["event_type"] == "post.synced"
    assert ps_event["post_id"] == post_id
    assert ps_event["post_table"] == "posts_e2e"
    assert ps_event["has_summary"] is False
    assert ps_event["fields_changed"] == [], (
        "New post insert should have empty fields_changed (all fields new)"
    )
    assert "event_id" in ps_event, "post.synced should carry an event_id"
    assert "occurred_at" in ps_event, "post.synced should carry occurred_at"
    assert "updated_at" in ps_event, "post.synced should carry updated_at"
    logger.info(
        "post.synced event verified: post_id=%s fields_changed=%s",
        ps_event["post_id"],
        ps_event["fields_changed"],
    )

    # Step 3.1: Check chunk table exists (it may be created lazily by the dispatcher)
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
            assert table_exists, f"Chunk table {chunk_table} does not exist after 10s, existing tables: {[row['table_name'] for row in chunk_tables]}"
        logger.error(f"Existing tables: {[row['table_name'] for row in chunk_tables]}")
    
    assert table_exists, f"Chunk table {chunk_table} does not exist after 10s"

    # Step 3.2: Poll for chunks to be created (up to 10 seconds)
    chunks = []
    for attempt in range(20):  # 20 attempts * 0.5s = 10s max wait
        async with postgres_pool_e2e.acquire() as conn:
            try:
                rows = await conn.fetch(
                    f"""
                    SELECT * FROM {chunk_table}
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

    # Step 4: Verify chunks.created event was published
    chunks_created_events = await fetch_events(
        postgres_pool_e2e, "chunks.created", post_id, timeout=5.0
    )
    assert len(chunks_created_events) >= 1, (
        "Expected at least one chunks.created event"
    )
    cc_event = chunks_created_events[0]
    assert cc_event["event_type"] == "chunks.created"
    assert cc_event["post_id"] == post_id
    assert cc_event["post_table"] == "posts_e2e"
    assert cc_event["chunk_table"] == chunk_table
    assert cc_event["task_type"] == "body"
    assert cc_event["chunk_count"] == len(chunks)
    assert len(cc_event["chunk_ids"]) == len(chunks)
    assert all(isinstance(cid, str) for cid in cc_event["chunk_ids"]), (
        "chunk_ids should all be strings"
    )
    assert "event_id" in cc_event, "chunks.created should carry an event_id"
    assert "created_at" in cc_event, "chunks.created should carry created_at"
    logger.info(
        "chunks.created event verified: post_id=%s chunk_count=%s task_type=%s",
        cc_event["post_id"],
        cc_event["chunk_count"],
        cc_event["task_type"],
    )

    # Step 5: Verify chunks have valid structure
    chunk_ids = [str(c["id"]) for c in chunks]
    assert all(isinstance(cid, str) for cid in chunk_ids), f"Chunk IDs should be strings, got {chunk_ids!r}, types: {[type(cid) for cid in chunk_ids]}"
    assert all(c["text"] and len(c["text"]) > 0 for c in chunks), f"Chunk texts should be non-empty, got {[c['text'] for c in chunks]!r}"
    logger.info(f"Chunk validation passed: {len(chunk_ids)} chunks with valid structure")

    # Step 6: Poll for embeddings to be written (up to 10 seconds)
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

    # Step 7: Verify embedding.completed event was published
    embed_completed_events = await fetch_events(
        postgres_pool_e2e, "embedding.completed", post_id, timeout=5.0
    )
    assert len(embed_completed_events) >= 1, (
        "Expected at least one embedding.completed event"
    )
    ec_event = embed_completed_events[0]
    assert ec_event["event_type"] == "embedding.completed"
    assert ec_event["post_id"] == post_id
    assert ec_event["post_table"] == "posts_e2e"
    assert ec_event["chunk_table"] == chunk_table
    assert ec_event["model_name"] == "bge-base-v1.5"
    assert len(ec_event["chunk_ids"]) == len(chunk_ids)
    assert all(isinstance(cid, str) for cid in ec_event["chunk_ids"]), (
        "chunk_ids in embedding.completed should all be strings"
    )
    assert "event_id" in ec_event, "embedding.completed should carry an event_id"
    logger.info(
        "embedding.completed event verified: post_id=%s model=%s chunk_count=%s",
        ec_event["post_id"],
        ec_event["model_name"],
        len(ec_event["chunk_ids"]),
    )

    # Step 8: Verify embeddings are valid vectors
    embedding_dim = 768  # bge-base-v1.5 dimension
    for row in embeddings:
        raw = row["embedding"]
        assert raw is not None, "Embedding should not be null"
        # asyncpg returns pgvector columns as a string "[f1,f2,...]" — parse it
        import json
        embedding = json.loads(raw) if isinstance(raw, str) else list(raw)
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
    Each post should produce its own set of events.
    """
    post_ids = [200, 201, 202]
    embedding_dim = 768
    chunk_table = "posts_e2e_chunks_body_bge_base_v1_5"

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

    # Verify each post produced a post.synced event
    for pid in post_ids:
        events = await fetch_events(
            postgres_pool_e2e, "post.synced", pid, timeout=5.0
        )
        assert len(events) >= 1, f"No post.synced event for post_id={pid}"
        event = events[0]
        assert event["post_id"] == pid
        assert event["post_table"] == "posts_e2e"
        assert event["fields_changed"] == [], (
            f"New post {pid} should have empty fields_changed"
        )
        logger.info("post.synced event verified for post_id=%s", pid)

    # Give the dispatcher a moment to pick up the new events after cleanup
    await asyncio.sleep(1.0)

    # Verify all posts were chunked independently
    for pid in post_ids:
        chunks = []
        for attempt in range(30):  # 30 * 0.5s = 15s max wait
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

        assert len(chunks) > 0, f"No chunks for post_id={pid} after 15s"

    # Verify each post produced a chunks.created event
    for pid in post_ids:
        events = await fetch_events(
            postgres_pool_e2e, "chunks.created", pid, timeout=5.0
        )
        assert len(events) >= 1, f"No chunks.created event for post_id={pid}"
        event = events[0]
        assert event["post_id"] == pid
        assert event["post_table"] == "posts_e2e"
        assert event["chunk_table"] == chunk_table
        assert event["task_type"] == "body"
        assert event["chunk_count"] > 0
        logger.info(
            "chunks.created event verified for post_id=%s (count=%s)",
            pid,
            event["chunk_count"],
        )

    # Verify each post produced an embedding.completed event
    for pid in post_ids:
        events = await fetch_events(
            postgres_pool_e2e, "embedding.completed", pid, timeout=5.0
        )
        assert len(events) >= 1, f"No embedding.completed event for post_id={pid}"
        event = events[0]
        assert event["post_id"] == pid
        assert event["post_table"] == "posts_e2e"
        assert event["chunk_table"] == chunk_table
        assert event["model_name"] == "bge-base-v1.5"
        assert len(event["chunk_ids"]) > 0
        logger.info(
            "embedding.completed event verified for post_id=%s (model=%s)",
            pid,
            event["model_name"],
        )

    logger.info("Multiple posts concurrent sync completed successfully")
