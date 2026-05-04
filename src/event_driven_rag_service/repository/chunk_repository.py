"""Async Postgres repository for chunks and their embeddings.

Chunks and embeddings share the same row (one table per (field, model) pair).
The two-step lifecycle:
  1. CpuChunkWorker inserts chunk rows with embedding = NULL.
  2. GpuEmbedWorker updates rows with the computed embedding vectors.

Table name format: ``chunks_{field}_{model}``  e.g. ``chunks_body_bge_base_v1_5``
Use :func:`build_chunk_table_name` to derive the name.

idempotency
-----------
To avoid re-chunking unchanged text, the worker checks whether a row with the
same (post_id, text_hash) already exists in the table.  If it does and the
``post_updated_at`` is still current, the chunk is skipped.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import asyncpg

from event_driven_rag_service.data_models.chunk import Chunk

logger = logging.getLogger(__name__)


_CREATE_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {table} (
    id              UUID         PRIMARY KEY,
    post_id         INTEGER      NOT NULL,
    chunk_index     INTEGER      NOT NULL,
    text            TEXT         NOT NULL,
    embedding       vector({dim}),        -- NULL until GpuEmbedWorker fills it in
    metadata        JSONB,
    token_count     INTEGER,
    text_hash       TEXT,
    post_updated_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {table}_post_id_idx    ON {table} (post_id);
CREATE INDEX IF NOT EXISTS {table}_text_hash_idx  ON {table} (post_id, text_hash);
"""

_CREATE_HNSW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw
ON {table}
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);
"""


class ChunkRepository:
    """Async Postgres repository for text chunks and their embeddings.

    Each instance is bound to a single table (one per field+model combination).
    Call :meth:`ensure_table` once at startup to create the table if needed.

    Args:
        pool:       asyncpg connection pool.
        table_name: fully-qualified table name (from :func:`build_chunk_table_name`).
        vector_dim: dimension of the embedding vectors stored in this table.
    """

    def __init__(self, pool: asyncpg.Pool, table_name: str, vector_dim: int) -> None:
        self._pool = pool
        self.table_name = table_name.lower()
        self.vector_dim = vector_dim

    async def ensure_table(self) -> None:
        """Idempotently create the chunk table, indexes, and pgvector extension."""
        sql = _CREATE_TABLE_SQL.format(table=self.table_name, dim=self.vector_dim)
        async with self._pool.acquire() as conn:
            await conn.execute(sql)
        logger.info("ChunkRepository: table '%s' ready (dim=%d)", self.table_name, self.vector_dim)

    async def create_hnsw_index(self) -> None:
        """Create the HNSW vector index.  Run once after initial bulk load."""
        sql = _CREATE_HNSW_INDEX_SQL.format(table=self.table_name)
        async with self._pool.acquire() as conn:
            await conn.execute(sql)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get_text_hashes(self, post_id: int) -> dict[str, str]:
        """Return {text_hash: chunk_id} for all stored chunks of *post_id*.

        Used by CpuChunkWorker to skip chunks whose text is unchanged,
        avoiding unnecessary re-chunking and re-embedding.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, text_hash FROM {self.table_name} WHERE post_id = $1",
                post_id,
            )
        return {row["text_hash"]: str(row["id"]) for row in rows if row["text_hash"]}

    async def get_chunk_versions(self, post_id: int) -> dict[str, datetime | None]:
        """Return {chunk_id: post_updated_at} for all stored chunks of *post_id*."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, post_updated_at FROM {self.table_name} WHERE post_id = $1",
                post_id,
            )
        return {str(row["id"]): row["post_updated_at"] for row in rows}

    async def fetch_texts(
        self, chunk_ids: list[str], table: str
    ) -> list[tuple[str, str]]:
        """Return (chunk_id, text) pairs for the given IDs (in any order).

        Used by GpuEmbedWorker to retrieve the text it needs to embed.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, text FROM {table} WHERE id = ANY($1::uuid[])",
                chunk_ids,
            )
        return [(str(row["id"]), row["text"]) for row in rows]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def bulk_insert(self, chunks: list[Chunk]) -> None:
        """Insert a batch of chunks.  Skips rows that already exist (by id)."""
        if not chunks:
            return
        rows = [
            (
                c.id,
                c.post_id,
                c.chunk_index,
                c.text,
                c.metadata.model_dump_json(),
                c.token_count,
                c.text_hash,
                c.post_updated_at,
            )
            for c in chunks
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self.table_name}
                    (id, post_id, chunk_index, text, metadata, token_count, text_hash, post_updated_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)
                ON CONFLICT (id) DO NOTHING
                """,
                rows,
            )

    async def update_embeddings(self, rows: list[dict[str, Any]]) -> None:
        """Store computed embedding vectors for a batch of chunk rows.

        Each dict must have: ``chunk_id``, ``embedding`` (list[float]),
        ``chunk_table`` (ignored here — the caller routes to the right repo).
        """
        if not rows:
            return
        # asyncpg doesn't have a native pgvector codec, so we serialize the
        # embedding list to a pgvector-compatible string and cast it in SQL.
        params = [
            ("[" + ",".join(str(float(x)) for x in r["embedding"]) + "]", r["chunk_id"])
            for r in rows
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                f"UPDATE {self.table_name} SET embedding = $1::vector WHERE id = $2::uuid",
                params,
            )

    async def bump_chunk_version(self, chunk_id: str, post_updated_at: datetime) -> None:
        """Advance post_updated_at without re-embedding (text unchanged)."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {self.table_name} SET post_updated_at = $1 WHERE id = $2::uuid",
                post_updated_at,
                chunk_id,
            )

    async def delete_stale_chunks(self, post_id: int, keep_from: datetime) -> int:
        """Delete chunks for *post_id* older than *keep_from*.  Returns row count."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.table_name}
                WHERE post_id = $1
                  AND (post_updated_at IS NULL OR post_updated_at < $2)
                """,
                post_id,
                keep_from,
            )
        # asyncpg returns "DELETE N" as a string
        count = int(result.split()[-1])
        return count


