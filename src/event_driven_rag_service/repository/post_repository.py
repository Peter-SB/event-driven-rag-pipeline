"""Async Postgres repository for posts.

Uses asyncpg.  Callers inject a connection pool created at startup.
All public methods are async and acquire/release pool connections internally.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional, Tuple

import asyncpg

from event_driven_rag_service.data_models.post import Post

logger = logging.getLogger(__name__)




_UPSERT_SQL = """
INSERT INTO {table} (
    post_id, external_id, external_source, external_created_at,
    url, title, body_text, author, subreddit,
    added_at, updated_at,
    custom_title, custom_body, notes, rating,
    is_read, read_at, is_favorite, is_archived,
    queued_at, is_deleted, folder_ids,
    extra_fields, body_min_hash, summary
) VALUES (
    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23::jsonb,$24,$25
)
ON CONFLICT (post_id) DO UPDATE SET
    external_id          = EXCLUDED.external_id,
    external_source      = EXCLUDED.external_source,
    external_created_at  = EXCLUDED.external_created_at,
    url                  = EXCLUDED.url,
    title                = EXCLUDED.title,
    body_text            = EXCLUDED.body_text,
    author               = EXCLUDED.author,
    subreddit            = EXCLUDED.subreddit,
    updated_at           = EXCLUDED.updated_at,
    custom_title         = EXCLUDED.custom_title,
    custom_body          = EXCLUDED.custom_body,
    notes                = EXCLUDED.notes,
    rating               = EXCLUDED.rating,
    is_read              = EXCLUDED.is_read,
    read_at              = EXCLUDED.read_at,
    is_favorite          = EXCLUDED.is_favorite,
    is_archived          = EXCLUDED.is_archived,
    queued_at            = EXCLUDED.queued_at,
    is_deleted           = EXCLUDED.is_deleted,
    folder_ids           = EXCLUDED.folder_ids,
    extra_fields         = EXCLUDED.extra_fields::jsonb,
    body_min_hash        = EXCLUDED.body_min_hash,
    summary              = EXCLUDED.summary
WHERE {table}.updated_at < EXCLUDED.updated_at
"""


class PostRepository:
    """Async Postgres repository for posts.

    Can be used in two modes:
    - Unbound (production): pass table_name explicitly to each method.
    - Bound (tests/fixtures): pass table_name to __init__; methods use it as default.

    Args:
        pool: asyncpg connection pool.
        table_name: Optional default table name (binds this repo to one table).
    """

    def __init__(self, pool: asyncpg.Pool, table_name: Optional[str] = None) -> None:
        self._pool = pool
        self._table_name = table_name

    @property
    def table_name(self) -> Optional[str]:
        return self._table_name

    async def ensure_table(self, table_name: Optional[str] = None) -> None:
        """Idempotently create a posts table and indexes.

        Args:
            table_name: The target table to create (e.g., "posts_main").
                        Falls back to the bound table_name if not provided.
        """
        table_name = (table_name or self._table_name).lower()
        
        # Create table
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            post_id              INTEGER      PRIMARY KEY,
            external_id          TEXT         NOT NULL,
            external_source      TEXT         NOT NULL DEFAULT 'reddit',
            external_created_at  TIMESTAMPTZ  NOT NULL,
            url                  TEXT         NOT NULL,
            title                TEXT         NOT NULL,
            body_text            TEXT,
            author               TEXT         NOT NULL,
            subreddit            TEXT,
            added_at             TIMESTAMPTZ  NOT NULL,
            updated_at           TIMESTAMPTZ  NOT NULL,
            custom_title         TEXT,
            custom_body          TEXT,
            notes                TEXT,
            rating               DOUBLE PRECISION,
            is_read              BOOLEAN      NOT NULL DEFAULT FALSE,
            read_at              TIMESTAMPTZ,
            is_favorite          BOOLEAN      NOT NULL DEFAULT FALSE,
            is_archived          BOOLEAN      NOT NULL DEFAULT FALSE,
            queued_at            TIMESTAMPTZ,
            is_deleted           BOOLEAN      NOT NULL DEFAULT FALSE,
            folder_ids           INTEGER[]    NOT NULL DEFAULT ARRAY[]::INTEGER[],
            extra_fields         JSONB,
            body_min_hash        TEXT,
            summary              TEXT,
            embedded_at          TIMESTAMPTZ
        )
        """
        
        # Create index
        index_sql = f"CREATE INDEX IF NOT EXISTS {table_name}_updated_at_idx ON {table_name} (updated_at)"
        
        async with self._pool.acquire() as conn:
            await conn.execute(create_table_sql)
            await conn.execute(index_sql)

    async def fetch(self, post_id: int, table_name: Optional[str] = None) -> Optional[Post]:
        """Return a single post object, or None if not found.

        Args:
            post_id: The post ID to fetch.
            table_name: The target table (e.g., "posts_main").
                        Falls back to the bound table_name if not provided.
        """
        table_name = (table_name or self._table_name).lower()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {table_name} WHERE post_id = $1",
                post_id,
            )
        return Post(**dict(row)) if row else None

    async def upsert(self, post: Post, table_name: Optional[str] = None) -> Tuple[str, Optional[datetime]]:
        """Insert or update a post using updated_at as the freshness signal.

        Args:
            post: The post to upsert.
            table_name: The target table (e.g., "posts_main").

        Returns:
            (status, prior_updated_at) where status is one of:
              - "inserted"  — first time this post_id was seen
              - "updated"   — row exists and incoming updated_at is newer
              - "skipped"   — row exists but incoming updated_at is not newer
        """
        table_name = (table_name or self._table_name).lower()
        extra_json = json.dumps(post.extra_fields) if post.extra_fields else None

        async with self._pool.acquire() as conn:
            # Check if a newer-or-equal row already exists
            existing_updated_at: Optional[datetime] = await conn.fetchval(
                f"SELECT updated_at FROM {table_name} WHERE post_id = $1",
                post.post_id,
            )

            if existing_updated_at is not None and existing_updated_at >= post.updated_at:
                return "skipped", existing_updated_at

            sql = _UPSERT_SQL.format(table=table_name)
            await conn.execute(
                sql,
                post.post_id, post.external_id, post.external_source, post.external_created_at,
                post.url, post.title, post.body_text, post.author, post.subreddit,
                post.added_at, post.updated_at,
                post.custom_title, post.custom_body, post.notes, post.rating,
                post.is_read, post.read_at, post.is_favorite, post.is_archived,
                post.queued_at, post.is_deleted, post.folder_ids,
                extra_json, post.body_min_hash, post.summary,
            )

        status = "inserted" if existing_updated_at is None else "updated"
        return status, existing_updated_at

    async def mark_embedded(self, post_id: int, embedded_at: datetime, table_name: Optional[str] = None) -> None:
        """Record when a post's chunks were last fully embedded.

        Args:
            post_id: The post ID to update.
            embedded_at: The timestamp of embedding completion.
            table_name: The target table (e.g., "posts_main").
                        Falls back to the bound table_name if not provided.
        """
        table_name = (table_name or self._table_name).lower()
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {table_name} SET embedded_at = $1 WHERE post_id = $2",
                embedded_at,
                post_id,
            )
