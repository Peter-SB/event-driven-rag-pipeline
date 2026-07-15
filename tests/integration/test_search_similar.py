"""Integration tests for the /search/similar repository layer.

Verifies get_post_embeddings and search_nearest (with exclude_post_id) against
real Postgres+pgvector via testcontainers.  Covers all three chunk types that
have different embedding dimensions.

Vector design (see scripts/generate_test_vectors.py for verification):
    source chunks (body):   unit axes 0, 1, 2  →  average = [1/3, 1/3, 1/3, 0...]
    close neighbour:        normalised [1,1,1,0...] →  cosine_sim(avg, close) = 1.0
    far neighbour:          unit axis 3            →  cosine_sim(avg, far)   = 0.0

For title / summary_title, the source post has one chunk (unit axis 0).
The close neighbour sits at normalised [1,1,...] and the far neighbour at unit axis 1.

Tested behaviours
-----------------
- get_post_embeddings returns all stored vectors for a post
- get_post_embeddings returns [] when post has no embedded chunks
- search_nearest with exclude_post_id omits source-post chunks from results
- Body: averaging 3 chunks produces the correct query vector, closest chunk wins
- Title (dim=384): single chunk used as query, closest chunk wins
- summary_title (dim=1024): single chunk used as query, closest chunk wins
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, UTC
from typing import AsyncGenerator

import asyncpg
import pytest
import pytest_asyncio

from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from tests.utils.factories import make_chunk


pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Vector helpers (mirrors scripts/generate_test_vectors.py)
# ---------------------------------------------------------------------------

def _unit(dim: int, axis: int) -> list[float]:
    v = [0.0] * dim
    v[axis] = 1.0
    return v


def _normalised(*components: float, dim: int) -> list[float]:
    """Build a normalised vector: first len(components) dims set, rest zero."""
    v = list(components) + [0.0] * (dim - len(components))
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _average(vecs: list[list[float]]) -> list[float]:
    n, d = len(vecs), len(vecs[0])
    return [sum(v[i] for v in vecs) / n for i in range(d)]


def _vec_str(v: list[float]) -> str:
    return "[" + ",".join(str(x) for x in v) + "]"


async def _insert_with_embedding(
    pool: asyncpg.Pool,
    table: str,
    post_id: int,
    chunk_index: int,
    embedding: list[float],
) -> str:
    """Insert a chunk row with an embedding in a single SQL statement."""
    chunk_id = str(uuid.uuid4())
    text = f"chunk text for post {post_id} index {chunk_index}"
    text_hash = str(uuid.uuid4())  # unique per row; content doesn't matter for vector tests
    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            INSERT INTO {table}
                (id, post_id, chunk_index, text, embedding, metadata, token_count, text_hash, post_updated_at, created_at)
            VALUES ($1::uuid, $2, $3, $4, $5::vector, $6::jsonb, $7, $8, now(), now())
            """,
            chunk_id, post_id, chunk_index, text, _vec_str(embedding),
            '{"title": "T"}', 10, text_hash,
        )
    return chunk_id


# ---------------------------------------------------------------------------
# Additional chunk-table fixtures (title dim=384, summary_title dim=1024)
# The existing clean_chunk_table fixture covers body (dim=768).
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def title_chunk_table(postgres_pool: asyncpg.Pool) -> AsyncGenerator[ChunkRepository, None]:
    """Fresh title chunk table (BAAI/bge-small-en-v1.5, dim=384)."""
    table = "test_similar_chunks_title_baai_bge_small_en_v1_5"
    repo = ChunkRepository(postgres_pool, table_name=table, vector_dim=384)
    await repo.ensure_table()
    async with postgres_pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {table}")
    yield repo
    async with postgres_pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {table}")


@pytest_asyncio.fixture
async def summary_title_chunk_table(postgres_pool: asyncpg.Pool) -> AsyncGenerator[ChunkRepository, None]:
    """Fresh summary_title chunk table (Qwen/Qwen3-0.6B, dim=1024)."""
    table = "test_similar_chunks_summary_title_qwen_qwen3_0_6b"
    repo = ChunkRepository(postgres_pool, table_name=table, vector_dim=1024)
    await repo.ensure_table()
    async with postgres_pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {table}")
    yield repo
    async with postgres_pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {table}")


# ---------------------------------------------------------------------------
# get_post_embeddings tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_post_embeddings_returns_all_vectors(clean_chunk_table, postgres_pool):
    """get_post_embeddings returns one vector per embedded chunk for the post."""
    table = clean_chunk_table.table_name
    dim = 768
    embeddings_in = [_unit(dim, 0), _unit(dim, 1), _unit(dim, 2)]

    for idx, emb in enumerate(embeddings_in):
        await _insert_with_embedding(postgres_pool, table, post_id=1, chunk_index=idx, embedding=emb)

    result = await clean_chunk_table.get_post_embeddings(1, table)

    assert len(result) == 3
    # Each returned vector has the correct dimension
    for vec in result:
        assert len(vec) == dim


@pytest.mark.asyncio
async def test_get_post_embeddings_returns_empty_when_no_embeddings(clean_chunk_table):
    """get_post_embeddings returns [] for a post with no embedded chunks."""
    table = clean_chunk_table.table_name
    # Insert a chunk but do NOT set the embedding (embedding = NULL via bulk_insert)
    chunk = make_chunk(post_id=2, chunk_index=0, text="unembed chunk for testing here")
    await clean_chunk_table.bulk_insert([chunk])

    result = await clean_chunk_table.get_post_embeddings(2, table)
    assert result == []


@pytest.mark.asyncio
async def test_get_post_embeddings_raises_chunk_table_not_found_for_missing_table(postgres_pool):
    """get_post_embeddings must raise ChunkTableNotFoundError (not a raw asyncpg error)
    when the chunk table hasn't been created yet — e.g. a library that was never synced."""
    from event_driven_rag_service.exceptions import ChunkTableNotFoundError

    repo = ChunkRepository(postgres_pool)
    with pytest.raises(ChunkTableNotFoundError):
        await repo.get_post_embeddings(1, "posts_neversynced_chunks_body_baai_bge_base_en_v1_5")


@pytest.mark.asyncio
async def test_get_post_embeddings_only_returns_target_post(clean_chunk_table, postgres_pool):
    """get_post_embeddings does not return vectors belonging to other posts."""
    table = clean_chunk_table.table_name
    dim = 768
    await _insert_with_embedding(postgres_pool, table, post_id=10, chunk_index=0, embedding=_unit(dim, 0))
    await _insert_with_embedding(postgres_pool, table, post_id=11, chunk_index=0, embedding=_unit(dim, 1))

    result_10 = await clean_chunk_table.get_post_embeddings(10, table)
    result_11 = await clean_chunk_table.get_post_embeddings(11, table)

    assert len(result_10) == 1
    assert len(result_11) == 1


# ---------------------------------------------------------------------------
# search_nearest exclude_post_id tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_nearest_excludes_source_post(clean_chunk_table, postgres_pool):
    """search_nearest with exclude_post_id must not return chunks from that post."""
    table = clean_chunk_table.table_name
    dim = 768
    src_id = str(uuid.uuid4())
    other_id = str(uuid.uuid4())

    # Source post chunk — identical vector to query (would be top hit without exclusion)
    src_emb = _unit(dim, 0)
    async with postgres_pool.acquire() as conn:
        await conn.execute(
            f"INSERT INTO {table} (id, post_id, chunk_index, text, embedding, metadata, token_count, text_hash, post_updated_at, created_at) VALUES ($1::uuid, $2, $3, $4, $5::vector, $6::jsonb, $7, $8, now(), now())",
            src_id, 20, 0, "source chunk", _vec_str(src_emb), '{"title":"T"}', 5, str(uuid.uuid4()),
        )

    # Other post chunk — slightly different direction
    other_emb = _normalised(1, 1, dim=dim)
    async with postgres_pool.acquire() as conn:
        await conn.execute(
            f"INSERT INTO {table} (id, post_id, chunk_index, text, embedding, metadata, token_count, text_hash, post_updated_at, created_at) VALUES ($1::uuid, $2, $3, $4, $5::vector, $6::jsonb, $7, $8, now(), now())",
            other_id, 21, 0, "other chunk", _vec_str(other_emb), '{"title":"T"}', 5, str(uuid.uuid4()),
        )

    results = await clean_chunk_table.search_nearest(table, src_emb, k=10, exclude_post_id=20)

    returned_post_ids = {r["post_id"] for r in results}
    assert 20 not in returned_post_ids, "Source post should be excluded from results"
    assert 21 in returned_post_ids


# ---------------------------------------------------------------------------
# Body: 3 source chunks averaged → nearest individual chunk wins
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_body_nearest_chunk_matches_averaged_query(clean_chunk_table, postgres_pool):
    """Averaging body chunks and searching should rank the close neighbour first.

    Source post: 3 chunks at unit axes 0, 1, 2 → average = [1/3, 1/3, 1/3, ...]
    Close neighbour (post 2): normalised [1, 1, 1, 0...] → cosine_sim = 1.0
    Far neighbour  (post 3): unit axis 3               → cosine_sim = 0.0
    """
    table = clean_chunk_table.table_name
    dim = 768

    # Source post (post_id=100): 3 body chunks with distinct unit-vector embeddings
    src_chunks = [_unit(dim, i) for i in range(3)]
    for idx, emb in enumerate(src_chunks):
        await _insert_with_embedding(postgres_pool, table, post_id=100, chunk_index=idx, embedding=emb)

    # Close neighbour (post_id=101): single chunk whose direction matches the average
    close_emb = _normalised(1, 1, 1, dim=dim)
    await _insert_with_embedding(postgres_pool, table, post_id=101, chunk_index=0, embedding=close_emb)

    # Far neighbour (post_id=102): single chunk orthogonal to the average
    far_emb = _unit(dim, 3)
    await _insert_with_embedding(postgres_pool, table, post_id=102, chunk_index=0, embedding=far_emb)

    # Compute averaged query (same logic as the endpoint)
    query_vec = _average(src_chunks)

    results = await clean_chunk_table.search_nearest(table, query_vec, k=5, exclude_post_id=100)

    assert len(results) >= 2
    post_ids_in_order = [r["post_id"] for r in results]
    # Close neighbour must rank before far neighbour
    assert post_ids_in_order.index(101) < post_ids_in_order.index(102), (
        f"Expected post 101 before 102, got order: {post_ids_in_order}"
    )
    # Results are individual chunks, not aggregated
    for r in results:
        assert "chunk_id" not in r, "Results should be individual chunks (id key), not aggregated"
        assert "id" in r


@pytest.mark.asyncio
async def test_body_search_returns_individual_chunks_not_posts(clean_chunk_table, postgres_pool):
    """search_nearest returns one row per chunk, not one row per post."""
    table = clean_chunk_table.table_name
    dim = 768
    query = _unit(dim, 0)

    # Insert 3 chunks for post 200 and 2 chunks for post 201
    for idx in range(3):
        await _insert_with_embedding(postgres_pool, table, post_id=200, chunk_index=idx, embedding=query)
    for idx in range(2):
        await _insert_with_embedding(postgres_pool, table, post_id=201, chunk_index=idx, embedding=_unit(dim, 1))

    results = await clean_chunk_table.search_nearest(table, query, k=10, exclude_post_id=None)

    # Should return 5 individual chunk rows, not 2 post-level rows
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Title (dim=384): single chunk used as query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_title_single_chunk_nearest_search(title_chunk_table, postgres_pool):
    """Title uses a single source chunk directly as the query vector.

    Source post (post 300): unit axis 0 → [1, 0, 0, ...]
    Close neighbour (post 301): normalised [1, 1, 0, ...] → higher cosine_sim
    Far neighbour  (post 302): unit axis 2 → lower cosine_sim
    """
    table = title_chunk_table.table_name
    dim = 384

    src_emb = _unit(dim, 0)
    close_emb = _normalised(1, 1, dim=dim)
    far_emb = _unit(dim, 2)

    await _insert_with_embedding(postgres_pool, table, post_id=300, chunk_index=0, embedding=src_emb)
    await _insert_with_embedding(postgres_pool, table, post_id=301, chunk_index=0, embedding=close_emb)
    await _insert_with_embedding(postgres_pool, table, post_id=302, chunk_index=0, embedding=far_emb)

    # Title endpoint passes the single embedding directly (average of one vector = itself)
    query_vec = _average([src_emb])
    assert query_vec == src_emb

    results = await title_chunk_table.search_nearest(table, query_vec, k=5, exclude_post_id=300)

    post_ids = [r["post_id"] for r in results]
    assert 300 not in post_ids, "Source post must be excluded"
    assert post_ids.index(301) < post_ids.index(302), (
        f"Expected post 301 before 302, got: {post_ids}"
    )


# ---------------------------------------------------------------------------
# summary_title (dim=1024): single chunk used as query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_title_single_chunk_nearest_search(summary_title_chunk_table, postgres_pool):
    """summary_title uses a single source chunk directly as the query vector.

    Source post (post 400): unit axis 0
    Close neighbour (post 401): normalised [1, 1, 0, ...] → higher cosine_sim
    Far neighbour  (post 402): unit axis 2 → lower cosine_sim
    """
    table = summary_title_chunk_table.table_name
    dim = 1024

    src_emb = _unit(dim, 0)
    close_emb = _normalised(1, 1, dim=dim)
    far_emb = _unit(dim, 2)

    await _insert_with_embedding(postgres_pool, table, post_id=400, chunk_index=0, embedding=src_emb)
    await _insert_with_embedding(postgres_pool, table, post_id=401, chunk_index=0, embedding=close_emb)
    await _insert_with_embedding(postgres_pool, table, post_id=402, chunk_index=0, embedding=far_emb)

    query_vec = src_emb
    results = await summary_title_chunk_table.search_nearest(table, query_vec, k=5, exclude_post_id=400)

    post_ids = [r["post_id"] for r in results]
    assert 400 not in post_ids
    assert post_ids.index(401) < post_ids.index(402), (
        f"Expected post 401 before 402, got: {post_ids}"
    )
