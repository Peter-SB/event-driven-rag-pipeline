"""Async Postgres repository for chunks and their embeddings.

Chunks and embeddings share the same row (one table per library + field + model).
The two-step lifecycle:
  1. CpuChunkWorker inserts chunk rows with embedding = NULL.
  2. GpuEmbedWorker updates rows with the computed embedding vectors.

Table name format: ``posts_{id}_chunks_{field}_{model}``
Example: ``posts_main_chunks_body_bge_base_v1_5``
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
from typing import Any, Optional

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

    Can be used in two modes:
    - Unbound (production): pass table_name explicitly to each method.
    - Bound (tests/fixtures): pass table_name + vector_dim to __init__; methods use them as defaults.

    Args:
        pool: asyncpg connection pool.
        table_name: Optional default table name.
        vector_dim: Optional default vector dimension (required when table_name is bound).
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        table_name: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ) -> None:
        self._pool = pool
        self._table_name = table_name
        self._vector_dim = vector_dim
        self._seen_tables: set[str] = set()  # Cache to avoid repeated CREATE TABLE IF NOT EXISTS

    @property
    def table_name(self) -> Optional[str]:
        return self._table_name

    async def ensure_table(self, table_name: Optional[str] = None, vector_dim: Optional[int] = None) -> None:
        """Idempotently create a chunk table, indexes, and pgvector extension.

        Args:
            table_name: The target table (e.g., "posts_main_chunks_body_bge_base_v1_5").
                        Falls back to the bound table_name if not provided.
            vector_dim: Dimension of the embedding vectors for this table.
                        Falls back to the bound vector_dim if not provided.
        """
        table_name = (table_name or self._table_name).lower()
        vector_dim = vector_dim or self._vector_dim
        if table_name in self._seen_tables:
            return
        sql = _CREATE_TABLE_SQL.format(table=table_name, dim=vector_dim)
        async with self._pool.acquire() as conn:
            await conn.execute(sql)
        self._seen_tables.add(table_name)
        logger.info("ChunkRepository: table '%s' ready (dim=%d)", table_name, vector_dim)

    async def create_hnsw_index(self, table_name: str) -> None:
        """Create the HNSW vector index.  Run once after initial bulk load.

        Args:
            table_name: The target table.
        """
        table_name = table_name.lower()
        sql = _CREATE_HNSW_INDEX_SQL.format(table=table_name)
        async with self._pool.acquire() as conn:
            await conn.execute(sql)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get_text_hashes(self, post_id: int, table_name: Optional[str] = None) -> dict[str, str]:
        """Return {text_hash: chunk_id} for all stored chunks of *post_id*.

        Used by CpuChunkWorker to skip chunks whose text is unchanged,
        avoiding unnecessary re-chunking and re-embedding.

        Args:
            post_id: The post ID to query.
            table_name: The target table. Falls back to bound table_name if not provided.
        """
        table_name = (table_name or self._table_name).lower()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, text_hash FROM {table_name} WHERE post_id = $1",
                post_id,
            )
        return {row["text_hash"]: str(row["id"]) for row in rows if row["text_hash"]}

    async def get_chunk_versions(self, post_id: int, table_name: Optional[str] = None) -> dict[str, datetime | None]:
        """Return {chunk_id: post_updated_at} for all stored chunks of *post_id*.

        Args:
            post_id: The post ID to query.
            table_name: The target table. Falls back to bound table_name if not provided.
        """
        table_name = (table_name or self._table_name).lower()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, post_updated_at FROM {table_name} WHERE post_id = $1",
                post_id,
            )
        return {str(row["id"]): row["post_updated_at"] for row in rows}

    async def fetch_texts(
        self, chunk_ids: list[str], table_name: Optional[str] = None
    ) -> list[tuple[str, str]]:
        """Return (chunk_id, text) pairs for the given IDs (in any order).

        Used by GpuEmbedWorker to retrieve the text it needs to embed.

        Args:
            chunk_ids: List of chunk IDs to fetch.
            table_name: The target table. Falls back to bound table_name if not provided.
        """
        table_name = (table_name or self._table_name).lower()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, text FROM {table_name} WHERE id = ANY($1::uuid[])",
                chunk_ids,
            )
        return [(str(row["id"]), row["text"]) for row in rows]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def bulk_insert(self, chunks: list[Chunk], table_name: Optional[str] = None) -> None:
        """Insert a batch of chunks.  Skips rows that already exist (by id).

        Args:
            chunks: List of Chunk objects to insert.
            table_name: The target table. Falls back to bound table_name if not provided.
        """
        if not chunks:
            return
        table_name = (table_name or self._table_name).lower()
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
                INSERT INTO {table_name}
                    (id, post_id, chunk_index, text, metadata, token_count, text_hash, post_updated_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)
                ON CONFLICT (id) DO NOTHING
                """,
                rows,
            )

    async def update_embeddings(self, rows: list[dict[str, Any]], table_name: Optional[str] = None) -> None:
        """Store computed embedding vectors for a batch of chunk rows.

        Args:
            rows: Each dict must have: ``chunk_id``, ``embedding`` (list[float]).
            table_name: The target table. Falls back to bound table_name if not provided.
        """
        if not rows:
            return
        table_name = (table_name or self._table_name).lower()
        # asyncpg doesn't have a native pgvector codec, so we serialize the
        # embedding list to a pgvector-compatible string and cast it in SQL.
        params = [
            ("[" + ",".join(str(float(x)) for x in r["embedding"]) + "]", r["chunk_id"])
            for r in rows
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                f"UPDATE {table_name} SET embedding = $1::vector WHERE id = $2::uuid",
                params,
            )

    async def save_batch(self, rows: list[Any]) -> None:
        """EmbeddingStore protocol implementation.

        Routes chunk embedding rows to their target ``chunk_table``.
        Query embedding rows (keyed by ``query_job_id``) are ignored here —
        those require a separate search-query repository.

        Each chunk row must have: ``chunk_id``, ``embedding``, ``chunk_table``.
        """
        from collections import defaultdict
        by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if "chunk_id" in row:
                table = row.get("chunk_table")
                if table:
                    by_table[table].append(row)
        for table, table_rows in by_table.items():
            table = table.lower()
            params = [
                ("[" + ",".join(str(float(x)) for x in r["embedding"]) + "]", r["chunk_id"])
                for r in table_rows
            ]
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    f"UPDATE {table} SET embedding = $1::vector WHERE id = $2::uuid",
                    params,
                )

    async def bump_chunk_version(self, chunk_id: str, post_updated_at: datetime, table_name: Optional[str] = None) -> None:
        """Advance post_updated_at without re-embedding (text unchanged).

        Args:
            chunk_id: The chunk ID to update.
            post_updated_at: The new post_updated_at timestamp.
            table_name: The target table. Falls back to bound table_name if not provided.
        """
        table_name = (table_name or self._table_name).lower()
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {table_name} SET post_updated_at = $1 WHERE id = $2::uuid",
                post_updated_at,
                chunk_id,
            )

    async def delete_stale_chunks(self, post_id: int, keep_from: datetime, table_name: Optional[str] = None) -> int:
        """Delete chunks for *post_id* older than *keep_from*.  Returns row count.

        Args:
            post_id: The post ID.
            keep_from: Keep chunks with post_updated_at >= this timestamp.
            table_name: The target table. Falls back to bound table_name if not provided.
        """
        table_name = (table_name or self._table_name).lower()
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {table_name}
                WHERE post_id = $1
                  AND (post_updated_at IS NULL OR post_updated_at < $2)
                """,
                post_id,
                keep_from,
            )
        # asyncpg returns "DELETE N" as a string
        count = int(result.split()[-1])
        return count


