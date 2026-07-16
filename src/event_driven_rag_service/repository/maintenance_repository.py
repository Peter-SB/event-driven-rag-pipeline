"""Maintenance repository — read-only queries for operational tooling.

Single responsibility: surface the data the RequeueService needs to detect
un-embedded chunks.  No writes happen here.
"""
from __future__ import annotations

import logging

import asyncpg

logger = logging.getLogger(__name__)


class MaintenanceRepository:
    """Async Postgres repository for maintenance / operational queries.

    Args:
        pool: asyncpg connection pool shared with the application.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def list_chunk_tables(self) -> list[str]:
        """Return the names of all chunk tables that exist in the public schema.

        Chunk tables match the pattern ``posts_%_chunks_%``.  Any table whose
        name matches this pattern — including hand-created or legacy tables —
        will be returned.  The caller is responsible for deciding which of these
        to act on.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name LIKE 'posts_%_chunks_%'
                ORDER BY table_name
                """
            )
        return [row["table_name"] for row in rows]

    async def fetch_unembedded_chunks(
        self, table_name: str
    ) -> list[tuple[str, int]]:
        """Return ``(chunk_id, post_id)`` pairs for rows with no embedding.

        Only rows where ``embedding IS NULL`` are returned; already-embedded
        rows are never touched.  This makes the query safe to call at any time,
        even while the GPU worker is actively processing.

        Args:
            table_name: The chunk table to inspect (e.g.
                ``posts_main_chunks_body_baai_bge_base_en_v1_5``).

        Returns:
            A list of ``(chunk_id, post_id)`` tuples, possibly empty.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, post_id FROM {table_name} WHERE embedding IS NULL"  # noqa: S608
            )
        return [(str(row["id"]), row["post_id"]) for row in rows]
