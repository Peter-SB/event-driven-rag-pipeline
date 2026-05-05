"""E2E tests for the search pipeline against the running Docker Compose stack.

Requires the full stack running (including all dispatchers and workers):

    docker compose up -d
    pytest tests/e2e/ -m e2e

IMPORTANT: The dispatcher service MUST be running and must include:
    - PostDispatcher (post.synced → chunk tasks)
    - ChunkDispatcher (chunks.created → embed tasks)
    - SearchDispatcher (search_job.created → embed tasks)
    - EmbeddingDispatcher (search_query.embedded → search tasks)

Also required:
    - CpuChunkWorker (chunking)
    - GpuEmbedWorker (embeddings)
    - CpuSearchWorker (search execution)

Test strategy
-------------
1. Seed chunk data by syncing posts via POST /posts/sync and waiting for
   embeddings to be written (the ingest pipeline must be running).
2. POST /search with a query → get job_id.
3. Poll GET /search/{job_id} until status is 'complete' or 'failed'.
4. Assert results are present and structurally valid.

The search results use MOCK_EMBEDDINGS=1, so vectors are deterministic
(not semantically meaningful) — we only verify the pipeline runs end-to-end
and returns valid structured results.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, UTC

import asyncpg
import httpx
import pytest

logger = logging.getLogger(__name__)

os.environ.setdefault("MOCK_EMBEDDINGS", "1")

_E2E_LIBRARY = "searchtest"
_CHUNK_TABLE = f"posts_{_E2E_LIBRARY}_chunks_body_bge_base_v1_5"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _wait_for_embeddings(
    pool: asyncpg.Pool,
    chunk_table: str,
    post_ids: list[int],
    timeout: float = 30.0,
    interval: float = 0.5,
) -> None:
    """Block until all posts have at least one embedded chunk in *chunk_table*."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            async with pool.acquire() as conn:
                count = await conn.fetchval(
                    f"""
                    SELECT COUNT(DISTINCT post_id) FROM {chunk_table}
                    WHERE post_id = ANY($1::int[]) AND embedding IS NOT NULL
                    """,
                    post_ids,
                )
            if count == len(post_ids):
                return
        except Exception:
            pass
        await asyncio.sleep(interval)

    raise AssertionError(
        f"Embeddings not ready after {timeout}s for post_ids={post_ids} in {chunk_table}"
    )


async def _poll_search_job(
    client: httpx.AsyncClient,
    job_id: str,
    timeout: float = 30.0,
    interval: float = 0.5,
) -> dict:
    """Poll GET /search/{job_id} until terminal status, then return the response body."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/search/{job_id}")
        assert resp.status_code == 200, f"GET /search/{job_id} returned {resp.status_code}"
        body = resp.json()
        if body["status"] in ("complete", "failed"):
            return body
        await asyncio.sleep(interval)

    raise AssertionError(f"Search job {job_id} did not complete after {timeout}s")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def search_e2e_client(postgres_pool_e2e: asyncpg.Pool):
    """httpx client + pre-test cleanup of search E2E tables."""
    async with postgres_pool_e2e.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS posts_{_E2E_LIBRARY}")
        await conn.execute(f"DROP TABLE IF EXISTS {_CHUNK_TABLE}")
        await conn.execute("DELETE FROM search_jobs WHERE chunks_table = $1", _CHUNK_TABLE)
        await conn.execute(
            "DELETE FROM event_log WHERE payload->>'post_table' = $1",
            f"posts_{_E2E_LIBRARY}",
        )

    _API_BASE = os.getenv("API_BASE", "http://localhost:8000")
    async with httpx.AsyncClient(base_url=_API_BASE, timeout=60.0) as client:
        yield client

    async with postgres_pool_e2e.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS posts_{_E2E_LIBRARY}")
        await conn.execute(f"DROP TABLE IF EXISTS {_CHUNK_TABLE}")
        await conn.execute("DELETE FROM search_jobs WHERE chunks_table = $1", _CHUNK_TABLE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_endpoint_creates_job_and_returns_202(search_e2e_client):
    """POST /search should immediately return 202 with a job_id."""
    response = await search_e2e_client.post(
        "/search/",
        json={
            "query": "machine learning retrieval",
            "chunk_type": "body",
            "k": 5,
            "library_id": _E2E_LIBRARY,
        },
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert "job_id" in body
    assert isinstance(body["job_id"], str)
    logger.info("Search job created: %s", body["job_id"])


@pytest.mark.asyncio
async def test_search_job_polling_returns_status(search_e2e_client):
    """GET /search/{job_id} should return status while job is pending."""
    post_resp = await search_e2e_client.post(
        "/search/",
        json={"query": "test", "chunk_type": "body", "k": 5, "library_id": _E2E_LIBRARY},
    )
    job_id = post_resp.json()["job_id"]

    get_resp = await search_e2e_client.get(f"/search/{job_id}")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["job_id"] == job_id
    assert body["status"] in ("embedding", "searching", "complete", "failed")
    logger.info("Job %s status: %s", job_id, body["status"])


@pytest.mark.asyncio
async def test_full_search_pipeline_with_seeded_chunks(
    postgres_pool_e2e: asyncpg.Pool,
    search_e2e_client,
):
    """Full E2E: seed posts → wait for embeddings → search → verify results.

    Flow:
    1. Sync 3 posts via POST /posts/sync
    2. Wait for chunks and embeddings to appear in Postgres (ingest pipeline)
    3. POST /search with a query
    4. Poll GET /search/{job_id} until complete (search pipeline)
    5. Verify results: count, structure, score range
    """
    post_ids = [9001, 9002, 9003]

    # Step 1: Sync posts
    posts_payload = [
        {
            "id": pid,
            "redditId": f"reddit_search_{pid}",
            "externalSource": "reddit",
            "redditCreatedAt": datetime.now(UTC).isoformat(),
            "url": f"https://reddit.com/r/test/comments/search_{pid}",
            "title": f"Search Test Post {pid}",
            "bodyText": (
                f"This is test post {pid} for search pipeline validation. "
                "Retrieval augmented generation combines large language models "
                "with a vector database for semantic search. "
                "The embedding model converts text to vectors. " * 5
            ),
            "author": "test_user",
            "addedAt": datetime.now(UTC).isoformat(),
            "updatedAt": datetime.now(UTC).isoformat(),
        }
        for pid in post_ids
    ]

    sync_resp = await search_e2e_client.post(
        "/posts/sync",
        json={"posts": posts_payload, "library_id": _E2E_LIBRARY},
    )
    assert sync_resp.status_code == 200
    for result in sync_resp.json()["results"]:
        assert result["success"] is True, f"Sync failed for post: {result}"
    logger.info("Synced %d posts for library=%s", len(post_ids), _E2E_LIBRARY)

    # Step 2: Wait for the ingest pipeline to produce embeddings
    await _wait_for_embeddings(postgres_pool_e2e, _CHUNK_TABLE, post_ids, timeout=30.0)
    logger.info("Embeddings ready for all %d posts in %s", len(post_ids), _CHUNK_TABLE)

    # Step 3: POST /search
    search_resp = await search_e2e_client.post(
        "/search/",
        json={
            "query": "retrieval augmented generation vector database",
            "chunk_type": "body",
            "k": 5,
            "library_id": _E2E_LIBRARY,
        },
    )
    assert search_resp.status_code == 202, search_resp.text
    job_id = search_resp.json()["job_id"]
    logger.info("Search job created: %s", job_id)

    # Step 3.1: Check event log for search_job.created event
    # async with postgres_pool_e2e.acquire() as conn:
    #     row = await conn.fetchrow(
    #         "SELECT * FROM event_log WHERE topic = 'search_job.created' AND payload->>'job_id' = $1",
    #         job_id,
    #     )
    # assert row is not None, "Expected search_job.created event in event_log"
    # payload = json.loads(row["payload"])

    # # 3.2: Verify event payload structure
    # assert payload["event_type"] == "search_job.created"
    # assert payload["payload"]["job_id"] == job_id
    # assert payload["payload"]["query"] == "retrieval augmented generation vector database"

    # Step 4: Poll until complete
    result_body = await _poll_search_job(search_e2e_client, job_id, timeout=20.0)

    assert result_body["status"] == "complete", (
        f"Search job failed: error={result_body.get('error')}"
    )
    assert result_body["results"] is not None
    assert len(result_body["results"]) > 0, "Expected at least one search result"
    assert len(result_body["results"]) <= 5  # at most k=5

    # Step 5: Verify result structure
    for item in result_body["results"]:
        assert "chunk_id" in item, "Result should have chunk_id"
        assert "post_id" in item, "Result should have post_id"
        assert "text" in item, "Result should have text"
        assert "score" in item, "Result should have score"
        assert isinstance(item["score"], float), f"Score should be float, got {type(item['score'])}"
        assert 0.0 <= item["score"] <= 1.0, f"Score out of range: {item['score']}"
        assert item["post_id"] in post_ids, f"Result post_id {item['post_id']} not in seeded posts"
        assert len(item["text"]) > 0, "Result text should be non-empty"

    logger.info(
        "Search pipeline E2E complete: %d results, top score=%.4f",
        len(result_body["results"]),
        result_body["results"][0]["score"],
    )


@pytest.mark.asyncio
async def test_search_unknown_job_id_returns_404(search_e2e_client):
    """GET /search/{job_id} with a non-existent ID should return 404."""
    response = await search_e2e_client.get("/search/00000000-0000-0000-0000-000000000099")
    assert response.status_code == 404
