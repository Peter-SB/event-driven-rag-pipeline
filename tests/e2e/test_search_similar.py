"""E2E test for POST /search/similar against the running Docker Compose stack.

Unlike the async search pipeline tests, this endpoint is synchronous — it
reads pre-existing embeddings from Postgres and runs ANN search inline, so no
worker polling is needed.

Test strategy
-------------
1. Create the body chunk table for library "similartest" directly via SQL
   (skips the full ingest pipeline — we seed pre-computed vectors).
2. Insert three posts with known embeddings:
   - Source post (5001): 3 body chunks at unit axes 0, 1, 2
                          → averaged query = [1/3, 1/3, 1/3, 0, ...]
   - Close neighbour (5002): one chunk at normalised [1,1,1,0,...] → sim ≈ 1.0
   - Far neighbour   (5003): one chunk at unit axis 3               → sim = 0.0
3. POST /search/similar for post 5001, chunk_type=body.
4. Assert:
   - Source post 5001 is not in results (excluded)
   - Close neighbour 5002 appears first
   - Far neighbour 5003 has a lower score than the close neighbour

Run with:
    docker compose up -d
    pytest tests/e2e/test_search_similar.py -m e2e -v
"""
from __future__ import annotations

import math
import os
import uuid
from typing import AsyncGenerator

import asyncpg
import httpx
import pytest
import pytest_asyncio

from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.utils.build_table_names import build_chunk_table_name


pytestmark = pytest.mark.e2e

_DB_URL = os.getenv("DB_URL", "postgresql://rag:rag@localhost:5432/rag")
_API_BASE = os.getenv("API_BASE", "http://localhost:8000")

_LIBRARY_ID = "similartest"
_CHUNK_TYPE = "body"
_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
_DIM = 768
_CHUNK_TABLE = build_chunk_table_name(f"posts_{_LIBRARY_ID}", _CHUNK_TYPE, _EMBED_MODEL)

# Post IDs chosen to be large and unlikely to collide with other test data
_SRC_POST_ID = 5001
_CLOSE_POST_ID = 5002
_FAR_POST_ID = 5003


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _unit(dim: int, axis: int) -> list[float]:
    v = [0.0] * dim
    v[axis] = 1.0
    return v


def _normalised(*components: float, dim: int) -> list[float]:
    v = list(components) + [0.0] * (dim - len(components))
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _vec_str(v: list[float]) -> str:
    return "[" + ",".join(str(x) for x in v) + "]"


async def _insert_chunk(conn: asyncpg.Connection, table: str, post_id: int, idx: int, emb: list[float]) -> None:
    await conn.execute(
        f"""
        INSERT INTO {table}
            (id, post_id, chunk_index, text, embedding, metadata, token_count, text_hash, post_updated_at, created_at)
        VALUES ($1::uuid, $2, $3, $4, $5::vector, $6::jsonb, $7, $8, now(), now())
        ON CONFLICT (id) DO NOTHING
        """,
        str(uuid.uuid4()), post_id, idx,
        f"e2e similar test chunk post={post_id} idx={idx}",
        _vec_str(emb),
        '{"title": "E2E Similar Test", "external_id": "ext_e2e"}',
        10, str(uuid.uuid4()),
    )


# ---------------------------------------------------------------------------
# Fixture: create table + seed data, clean up after
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def similar_e2e_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Set up chunk table with pre-computed embeddings, yield HTTP client, clean up."""
    pool = await asyncpg.create_pool(_DB_URL, min_size=1, max_size=3)

    # Create the chunk table (idempotent)
    repo = ChunkRepository(pool, table_name=_CHUNK_TABLE, vector_dim=_DIM)
    await repo.ensure_table()

    # Clear any residual data from previous runs
    async with pool.acquire() as conn:
        await conn.execute(f"DELETE FROM {_CHUNK_TABLE} WHERE post_id = ANY($1::int[])", [_SRC_POST_ID, _CLOSE_POST_ID, _FAR_POST_ID])

    # Seed pre-computed vectors
    async with pool.acquire() as conn:
        # Source post: 3 body chunks at orthogonal unit axes → average = [1/3, 1/3, 1/3, 0...]
        for idx in range(3):
            await _insert_chunk(conn, _CHUNK_TABLE, _SRC_POST_ID, idx, _unit(_DIM, idx))

        # Close neighbour: one chunk aligned with the average → cosine_sim ≈ 1.0
        await _insert_chunk(conn, _CHUNK_TABLE, _CLOSE_POST_ID, 0, _normalised(1, 1, 1, dim=_DIM))

        # Far neighbour: one chunk orthogonal to the average → cosine_sim = 0.0
        await _insert_chunk(conn, _CHUNK_TABLE, _FAR_POST_ID, 0, _unit(_DIM, 3))

    async with httpx.AsyncClient(base_url=_API_BASE, timeout=30.0) as client:
        yield client

    # Cleanup: remove seeded rows
    async with pool.acquire() as conn:
        await conn.execute(f"DELETE FROM {_CHUNK_TABLE} WHERE post_id = ANY($1::int[])", [_SRC_POST_ID, _CLOSE_POST_ID, _FAR_POST_ID])
    await pool.close()


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_find_similar_body_chunks_returns_correct_neighbour(similar_e2e_client: httpx.AsyncClient):
    """POST /search/similar for a body post returns the nearest individual chunk.

    Verifies:
    - 200 response with correct schema
    - Source post 5001 excluded from results
    - Close neighbour 5002 ranked before far neighbour 5003
    - Results are individual chunks (not aggregated by post)
    - chunks_averaged == 3 (all source body chunks were averaged)
    """
    response = await similar_e2e_client.post(
        "/search/similar",
        json={
            "post_id": _SRC_POST_ID,
            "chunk_type": _CHUNK_TYPE,
            "library_id": _LIBRARY_ID,
            "k": 10,
        },
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    body = response.json()

    # Schema checks
    assert body["post_id"] == _SRC_POST_ID
    assert body["chunk_type"] == _CHUNK_TYPE
    assert body["chunks_averaged"] == 3, (
        f"Expected 3 body chunks averaged, got {body['chunks_averaged']}"
    )
    assert isinstance(body["results"], list)

    # Source post must be excluded
    returned_post_ids = [r["post_id"] for r in body["results"]]
    assert _SRC_POST_ID not in returned_post_ids, (
        f"Source post {_SRC_POST_ID} should not appear in results"
    )

    # Both neighbours must be present
    assert _CLOSE_POST_ID in returned_post_ids, f"Close neighbour {_CLOSE_POST_ID} missing from results"
    assert _FAR_POST_ID in returned_post_ids, f"Far neighbour {_FAR_POST_ID} missing from results"

    # Close neighbour must rank before far neighbour
    close_idx = returned_post_ids.index(_CLOSE_POST_ID)
    far_idx = returned_post_ids.index(_FAR_POST_ID)
    assert close_idx < far_idx, (
        f"Expected close neighbour (post {_CLOSE_POST_ID}) before far (post {_FAR_POST_ID}), "
        f"got order: {returned_post_ids}"
    )

    # Score sanity: close neighbour should have a high similarity score
    close_result = next(r for r in body["results"] if r["post_id"] == _CLOSE_POST_ID)
    far_result = next(r for r in body["results"] if r["post_id"] == _FAR_POST_ID)
    assert close_result["score"] > far_result["score"], (
        f"Close neighbour score {close_result['score']:.4f} should exceed far {far_result['score']:.4f}"
    )
    assert close_result["score"] > 0.99, (
        f"Close neighbour cosine_sim should be ~1.0, got {close_result['score']:.6f}"
    )
    assert far_result["score"] < 0.01, (
        f"Far neighbour cosine_sim should be ~0.0, got {far_result['score']:.6f}"
    )

    # Results are individual chunks, not aggregated by post
    for item in body["results"]:
        assert "chunk_id" in item
        assert "post_id" in item
        assert "text" in item
        assert "score" in item
        assert 0.0 <= item["score"] <= 1.0
