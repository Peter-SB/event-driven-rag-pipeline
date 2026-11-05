"""Async Postgres repository for folders.

Uses asyncpg.  Callers inject a connection pool created at startup.
Mirrors PostRepository: one table per library (``folders_{library_id}``),
``updated_at`` as the freshness signal.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple

import asyncpg

from event_driven_rag_service.data_models.folder import Folder

logger = logging.getLogger(__name__)


_UPSERT_SQL = """
INSERT INTO {table} (
    folder_id, name, parent_id, created_at, updated_at, is_deleted
) VALUES ($1,$2,$3,$4,$5,$6)
ON CONFLICT (folder_id) DO UPDATE SET
    name       = EXCLUDED.name,
    parent_id  = EXCLUDED.parent_id,
    created_at = EXCLUDED.created_at,
    updated_at = EXCLUDED.updated_at,
    is_deleted = EXCLUDED.is_deleted
WHERE {table}.updated_at < EXCLUDED.updated_at
"""


class FolderRepository:
    """Async Postgres repository for folders.

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
        """Idempotently create a folders table and its index.

        ``parent_id`` is deliberately not a foreign key: a batch may arrive
        child-first, and soft-deleted parents must stay referenceable.
        """
        table_name = (table_name or self._table_name).lower()

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            folder_id   INTEGER      PRIMARY KEY,
            name        TEXT         NOT NULL,
            parent_id   INTEGER,
            created_at  TIMESTAMPTZ  NOT NULL,
            updated_at  TIMESTAMPTZ  NOT NULL,
            is_deleted  BOOLEAN      NOT NULL DEFAULT FALSE
        )
        """
        index_sql = f"CREATE INDEX IF NOT EXISTS {table_name}_updated_at_idx ON {table_name} (updated_at)"

        async with self._pool.acquire() as conn:
            await conn.execute(create_table_sql)
            await conn.execute(index_sql)

    async def fetch(self, folder_id: int, table_name: Optional[str] = None) -> Optional[Folder]:
        """Return a single folder, or None if not found."""
        table_name = (table_name or self._table_name).lower()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {table_name} WHERE folder_id = $1",
                folder_id,
            )
        return Folder(**dict(row)) if row else None

    async def upsert(self, folder: Folder, table_name: Optional[str] = None) -> Tuple[str, Optional[datetime]]:
        """Insert or update a folder using updated_at as the freshness signal.

        Returns:
            (status, prior_updated_at) where status is one of
            "inserted", "updated" (incoming updated_at is newer) or "skipped".
        """
        table_name = (table_name or self._table_name).lower()

        async with self._pool.acquire() as conn:
            existing_updated_at: Optional[datetime] = await conn.fetchval(
                f"SELECT updated_at FROM {table_name} WHERE folder_id = $1",
                folder.folder_id,
            )

            if existing_updated_at is not None and existing_updated_at >= folder.updated_at:
                return "skipped", existing_updated_at

            await conn.execute(
                _UPSERT_SQL.format(table=table_name),
                folder.folder_id, folder.name, folder.parent_id,
                folder.created_at, folder.updated_at, folder.is_deleted,
            )

        status = "inserted" if existing_updated_at is None else "updated"
        return status, existing_updated_at
