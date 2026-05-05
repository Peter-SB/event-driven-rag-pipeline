"""Async Postgres repository for search jobs.

Each job progresses through:  embedding → searching → complete | failed

The ``search_jobs`` table is created on first use via :meth:`ensure_table`.

Query embeddings are written by GpuEmbedWorker via the :meth:`save_batch`
method (EmbeddingStore protocol), after which the EmbeddingDispatcher fires
a SearchRunTask that triggers search execution.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import asyncpg

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS search_jobs (
    id              UUID         PRIMARY KEY,
    status          TEXT         NOT NULL DEFAULT 'embedding',
    library_id      TEXT         NOT NULL,
    query           TEXT         NOT NULL,
    k               INTEGER      NOT NULL DEFAULT 10,
    embedding_profile TEXT       NOT NULL,
    chunks_table    TEXT         NOT NULL,
    embedding       TEXT,
    results         JSONB,
    error           TEXT,
    created_at      TIMESTAMPTZ  DEFAULT now(),
    completed_at    TIMESTAMPTZ
);
"""


class SearchJobRepository:
    """Async Postgres repository for search job records."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def ensure_table(self) -> None:
        """Idempotently create the search_jobs table."""
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)
        logger.info("SearchJobRepository: search_jobs table ready")

    async def create_job(
        self,
        query: str,
        k: int,
        embedding_profile: str,
        chunks_table: str,
        library_id: str,
    ) -> str:
        """Insert a new search job with status='embedding'. Returns the UUID job_id."""
        job_id = str(uuid.uuid4())
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO search_jobs
                    (id, status, library_id, query, k, embedding_profile, chunks_table)
                VALUES ($1::uuid, 'embedding', $2, $3, $4, $5, $6)
                """,
                job_id, library_id, query, k, embedding_profile, chunks_table,
            )
        logger.debug("SearchJobRepository: created job %s library=%s", job_id, library_id)
        return job_id

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a job by id. Returns None if not found."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM search_jobs WHERE id = $1::uuid", job_id
            )
        if row is None:
            return None
        result = dict(row)
        # Parse embedding from TEXT "[f1,f2,...]" → list[float]
        if result.get("embedding"):
            result["embedding"] = json.loads(result["embedding"])
        # Parse results from JSONB
        if result.get("results") and isinstance(result["results"], str):
            result["results"] = json.loads(result["results"])
        return result

    async def store_embedding(self, job_id: str, embedding: list[float]) -> None:
        """Save the query embedding for a job (positive slot)."""
        embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE search_jobs SET embedding = $1 WHERE id = $2::uuid",
                embedding_str, job_id,
            )
        logger.debug("SearchJobRepository: stored embedding for job %s", job_id)

    async def mark_searching(self, job_id: str) -> None:
        """Transition job to status='searching'."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE search_jobs SET status = 'searching' WHERE id = $1::uuid",
                job_id,
            )

    async def complete_job(self, job_id: str, results: List[Dict]) -> None:
        """Save results and transition job to status='complete'."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE search_jobs
                SET status = 'complete', results = $1::jsonb, completed_at = now()
                WHERE id = $2::uuid
                """,
                json.dumps(results), job_id,
            )
        logger.info("SearchJobRepository: job %s complete (%d results)", job_id, len(results))

    async def fail_job(self, job_id: str, error: str) -> None:
        """Record an error and transition job to status='failed'."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE search_jobs
                SET status = 'failed', error = $1, completed_at = now()
                WHERE id = $2::uuid
                """,
                error, job_id,
            )
        logger.warning("SearchJobRepository: job %s failed: %s", job_id, error)

    # ------------------------------------------------------------------
    # EmbeddingStore protocol — routes QueryEmbeddingRow to store_embedding
    # ------------------------------------------------------------------

    async def save_batch(self, rows: list[Any]) -> None:
        """EmbeddingStore protocol: persist query embedding rows.

        Handles QueryEmbeddingRow dicts (keyed by ``query_job_id``).
        Chunk rows (``chunk_id``) are silently ignored — those go to ChunkRepository.
        """
        for row in rows:
            if "query_job_id" in row:
                await self.store_embedding(row["query_job_id"], row["embedding"])


__all__ = ["SearchJobRepository"]
