"""Integration tests for ChunkRepository.

Verifies bulk_insert, idempotency via text_hash, embedding updates, and
stale chunk deletion against real Postgres+pgvector.

Tested behaviours
-----------------
- bulk_insert persists chunks with correct fields
- bulk_insert is idempotent (ON CONFLICT id DO NOTHING)
- get_text_hashes returns {hash: id} for all stored chunks of a post_id
- get_text_hashes returns empty dict when no chunks exist
- update_embeddings fills the embedding column
- fetch_texts returns (id, text) pairs for given chunk_ids
- delete_stale_chunks removes only chunks older than the cutoff
"""
from __future__ import annotations

from datetime import datetime, UTC, timedelta

import pytest

from tests.utils.factories import make_chunk


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# bulk_insert
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bulk_insert_persists_chunks(clean_chunk_table):
    chunks = [make_chunk(post_id=1, chunk_index=i, text=f"unique chunk text number {i} for testing purposes here") for i in range(3)]
    await clean_chunk_table.bulk_insert(chunks)

    # Verify via get_text_hashes — all three hashes should be retrievable
    hashes = await clean_chunk_table.get_text_hashes(1)
    assert len(hashes) == 3


@pytest.mark.asyncio
async def test_bulk_insert_is_idempotent(clean_chunk_table):
    """Inserting the same chunks twice must not create duplicate rows."""
    chunks = [make_chunk(post_id=2, chunk_index=0)]
    await clean_chunk_table.bulk_insert(chunks)
    await clean_chunk_table.bulk_insert(chunks)  # second insert same rows

    hashes = await clean_chunk_table.get_text_hashes(2)
    assert len(hashes) == 1  # still exactly one row


@pytest.mark.asyncio
async def test_bulk_insert_empty_list_is_safe(clean_chunk_table):
    """Calling bulk_insert([]) must not raise."""
    await clean_chunk_table.bulk_insert([])


# ---------------------------------------------------------------------------
# get_text_hashes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_text_hashes_returns_empty_for_unknown_post(clean_chunk_table):
    hashes = await clean_chunk_table.get_text_hashes(999)
    assert hashes == {}


@pytest.mark.asyncio
async def test_get_text_hashes_maps_hash_to_chunk_id(clean_chunk_table):
    chunk = make_chunk(post_id=3, chunk_index=0)
    await clean_chunk_table.bulk_insert([chunk])

    hashes = await clean_chunk_table.get_text_hashes(3)
    assert chunk.text_hash in hashes
    assert hashes[chunk.text_hash] == chunk.id


@pytest.mark.asyncio
async def test_get_text_hashes_only_returns_hashes_for_requested_post(clean_chunk_table):
    c1 = make_chunk(post_id=4, chunk_index=0, text="post four text alpha beta gamma delta epsilon")
    c2 = make_chunk(post_id=5, chunk_index=0, text="post five completely different text here now")
    await clean_chunk_table.bulk_insert([c1, c2])

    hashes_4 = await clean_chunk_table.get_text_hashes(4)
    hashes_5 = await clean_chunk_table.get_text_hashes(5)

    assert c1.text_hash in hashes_4
    assert c2.text_hash not in hashes_4

    assert c2.text_hash in hashes_5
    assert c1.text_hash not in hashes_5


# ---------------------------------------------------------------------------
# fetch_texts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_texts_returns_id_text_pairs(clean_chunk_table):
    chunk = make_chunk(post_id=6, text="unique text for fetching purposes only here")
    await clean_chunk_table.bulk_insert([chunk])

    pairs = await clean_chunk_table.fetch_texts([chunk.id], clean_chunk_table.table_name)
    assert len(pairs) == 1
    chunk_id, text = pairs[0]
    assert chunk_id == chunk.id
    assert text == chunk.text


@pytest.mark.asyncio
async def test_fetch_texts_returns_empty_for_unknown_ids(clean_chunk_table):
    import uuid
    nonexistent = str(uuid.uuid4())  # valid UUID format, just not in the DB
    pairs = await clean_chunk_table.fetch_texts([nonexistent], clean_chunk_table.table_name)
    assert pairs == []


# ---------------------------------------------------------------------------
# update_embeddings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_embeddings_fills_vector_column(clean_chunk_table):
    chunk = make_chunk(post_id=7, text="text to embed for testing vector storage here")
    await clean_chunk_table.bulk_insert([chunk])

    fake_vector = [0.1] * 768
    await clean_chunk_table.update_embeddings([{
        "chunk_id": chunk.id,
        "embedding": fake_vector,
        "chunk_table": clean_chunk_table.table_name,
    }])

    # Read back and verify embedding is stored
    async with clean_chunk_table._pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT embedding FROM {clean_chunk_table.table_name} WHERE id = $1::uuid",
            chunk.id,
        )
    assert row is not None
    assert row["embedding"] is not None


@pytest.mark.asyncio
async def test_update_embeddings_empty_list_is_safe(clean_chunk_table):
    await clean_chunk_table.update_embeddings([])


# ---------------------------------------------------------------------------
# delete_stale_chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_stale_chunks_removes_old_rows(clean_chunk_table):
    import uuid, hashlib
    from event_driven_rag_service.data_models.chunk import Chunk, ChunkMetadata

    old_ts = datetime(2024, 1, 1, tzinfo=UTC)
    new_ts = datetime(2024, 6, 1, tzinfo=UTC)
    cutoff = datetime(2024, 3, 1, tzinfo=UTC)

    # Build two chunks manually with different post_updated_at values
    def _make(text: str, ts: datetime) -> Chunk:
        return Chunk(
            id=str(uuid.uuid4()),
            post_id=8,
            post_updated_at=ts,
            chunk_index=0,
            text=text,
            metadata=ChunkMetadata(title="T"),
            token_count=5,
            text_hash=hashlib.sha256(text.encode()).hexdigest(),
            created_at=ts,
        )

    old_chunk = _make("old chunk text data content", old_ts)
    new_chunk = _make("new chunk text data content", new_ts)
    await clean_chunk_table.bulk_insert([old_chunk, new_chunk])

    deleted = await clean_chunk_table.delete_stale_chunks(8, cutoff)
    assert deleted == 1

    remaining = await clean_chunk_table.get_text_hashes(8)
    assert new_chunk.text_hash in remaining
    assert old_chunk.text_hash not in remaining
