"""Async Postgres repository for highlights.

Uses asyncpg.  Callers inject a connection pool created at startup.
Mirrors PostRepository: one table per library (``highlights_{library_id}``),
``updated_at`` as the freshness signal.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple

import asyncpg

from event_driven_rag_service.data_models.highlight import Highlight

logger = logging.getLogger(__name__)


_UPSERT_SQL = """
INSERT INTO {table} (
    highlight_id, post_id, text, note, start_offset, end_offset,
    created_at, updated_at, is_deleted
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
ON CONFLICT (highlight_id) DO UPDATE SET
    post_id      = EXCLUDED.post_id,
    text         = EXCLUDED.text,
    note         = EXCLUDED.note,
    start_offset = EXCLUDED.start_offset,
    end_offset   = EXCLUDED.end_offset,
    created_at   = EXCLUDED.created_at,
    updated_at   = EXCLUDED.updated_at,
    is_deleted   = EXCLUDED.is_deleted
WHERE {table}.updated_at < EXCLUDED.updated_at
"""


class HighlightRepository:
    """Async Postgres repository for highlights.

    Can be used unbound (pass table_name to each method) or bound
    (pass table_name to __init__; methods use it as the default).
    """

    def __init__(self, pool: asyncpg.Pool, table_name: Optional[str] = None) -> None:
        self._pool = pool
        self._table_name = table_name

    @property
    def table_name(self) -> Optional[str]:
        return self._table_name

    async def ensure_table(self, table_name: Optional[str] = None) -> None:
        """Idempotently create a highlights table and its indexes.

        ``post_id`` is deliberately not a foreign key: highlights may sync
        before their post, and posts are soft-deleted rather than removed.
        """
        table_name = (table_name or self._table_name).lower()

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            highlight_id  INTEGER      PRIMARY KEY,
            post_id       INTEGER      NOT NULL,
            text          TEXT         NOT NULL,
            note          TEXT,
            start_offset  INTEGER,
            end_offset    INTEGER,
            created_at    TIMESTAMPTZ  NOT NULL,
            updated_at    TIMESTAMPTZ  NOT NULL,
            is_deleted    BOOLEAN      NOT NULL DEFAULT FALSE
        )
        """
        updated_at_index_sql = f"CREATE INDEX IF NOT EXISTS {table_name}_updated_at_idx ON {table_name} (updated_at)"
        post_id_index_sql = f"CREATE INDEX IF NOT EXISTS {table_name}_post_id_idx ON {table_name} (post_id)"

        async with self._pool.acquire() as conn:
            await conn.execute(create_table_sql)
            await conn.execute(updated_at_index_sql)
            await conn.execute(post_id_index_sql)

    async def fetch(self, highlight_id: int, table_name: Optional[str] = None) -> Optional[Highlight]:
        """Return a single highlight, or None if not found."""
        table_name = (table_name or self._table_name).lower()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {table_name} WHERE highlight_id = $1",
                highlight_id,
            )
        return Highlight(**dict(row)) if row else None

    async def upsert(self, highlight: Highlight, table_name: Optional[str] = None) -> Tuple[str, Optional[datetime]]:
        """Insert or update a highlight using updated_at as the freshness signal.

        Returns:
            (status, prior_updated_at) where status is one of
            "inserted", "updated" (incoming updated_at is newer) or "skipped".
        """
        table_name = (table_name or self._table_name).lower()

        async with self._pool.acquire() as conn:
            existing_updated_at: Optional[datetime] = await conn.fetchval(
                f"SELECT updated_at FROM {table_name} WHERE highlight_id = $1",
                highlight.highlight_id,
            )

            if existing_updated_at is not None and existing_updated_at >= highlight.updated_at:
                return "skipped", existing_updated_at

            await conn.execute(
                _UPSERT_SQL.format(table=table_name),
                highlight.highlight_id, highlight.post_id, highlight.text, highlight.note,
                highlight.start_offset, highlight.end_offset,
                highlight.created_at, highlight.updated_at, highlight.is_deleted,
            )

        status = "inserted" if existing_updated_at is None else "updated"
        return status, existing_updated_at
