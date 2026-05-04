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


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
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
);
CREATE INDEX IF NOT EXISTS {table}_updated_at_idx ON {table} (updated_at);
"""

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

    Args:
        pool:       asyncpg connection pool.
        table_name: target table (defaults to "posts").
    """

    def __init__(self, pool: asyncpg.Pool, table_name: str = "posts") -> None:
        self._pool = pool
        self.table_name = table_name.lower()

    async def ensure_table(self) -> None:
        """Idempotently create the posts table and indexes."""
        sql = _CREATE_TABLE_SQL.format(table=self.table_name)
        async with self._pool.acquire() as conn:
            await conn.execute(sql)

    async def fetch(self, post_id: int) -> Optional[dict[str, Any]]:
        """Return a single post row as a plain dict, or None if not found."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self.table_name} WHERE post_id = $1",
                post_id,
            )
        return dict(row) if row else None

    async def upsert(self, post: Post) -> Tuple[str, Optional[datetime]]:
        """Insert or update a post using updated_at as the freshness signal.

        Returns:
            (status, prior_updated_at) where status is one of:
              - "inserted"  — first time this post_id was seen
              - "updated"   — row exists and incoming updated_at is newer
              - "skipped"   — row exists but incoming updated_at is not newer
        """
        extra_json = json.dumps(post.extra_fields) if post.extra_fields else None

        async with self._pool.acquire() as conn:
            # Check if a newer-or-equal row already exists
            existing_updated_at: Optional[datetime] = await conn.fetchval(
                f"SELECT updated_at FROM {self.table_name} WHERE post_id = $1",
                post.post_id,
            )

            if existing_updated_at is not None and existing_updated_at >= post.updated_at:
                return "skipped", existing_updated_at

            sql = _UPSERT_SQL.format(table=self.table_name)
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

    async def mark_embedded(self, post_id: int, embedded_at: datetime) -> None:
        """Record when a post's chunks were last fully embedded."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {self.table_name} SET embedded_at = $1 WHERE post_id = $2",
                embedded_at,
                post_id,
            )

        try:
            return self._upsert_post_attempt(post)
        except psycopg2.errors.UndefinedTable:
            logger.warning(
                "Table %s missing during upsert for post_id=%s — running _ensure_table and retrying",
                self.table_name,
                post.post_id,
            )
            self._ensure_table()
            return self._upsert_post_attempt(post)

    def _upsert_post_attempt(self, post: Post) -> Tuple[str, datetime | None]:
        """Single upsert attempt; callers handle UndefinedTable for retry logic."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                values = self._serialize_post(post)
                logger.debug(
                    "Upserting post_id=%s into %s with updated_at=%s",
                    post.post_id,
                    self.table_name,
                    post.updated_at,
                )
                cur.execute(self._upsert_sql(), values)

                row = cur.fetchone()
                if row is None:
                    # No row returned means "ON CONFLICT DO UPDATE's WHERE" failed, ie skipped
                    status = "skipped"
                    stored_at = self._fetch_updated_at(conn, post.post_id)
                else:
                    inserted_flag, stored_at = row
                    status = "inserted" if inserted_flag else "updated"
            conn.commit()
            logger.info(
                "Post %s %s in %s (stored updated_at=%s)",
                post.post_id,
                status,
                self.table_name,
                stored_at,
            )
            return status, stored_at
        except Exception:
            conn.rollback()
            logger.exception(
                "Failed to upsert post_id=%s into %s", post.post_id, self.table_name
            )
            raise
        finally:
            conn.close()

    def fetch_post(self, post_id: int) -> Optional[Post]:
        """Fetch a post by id, or None if missing."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    sql.SQL("SELECT * FROM {table} WHERE post_id = %s").format(
                        table=sql.Identifier(self.table_name)
                    ),
                    (post_id,),
                )
                row = cur.fetchone()
                logger.debug("Fetched post_id=%s from %s: found=%s", post_id, self.table_name, bool(row))
                return Post.model_validate(dict(row)) if row else None
        finally:
            conn.close()

    def fetch_posts_by_ids(self, post_ids: list) -> Dict[int, Post]:
        """Fetch multiple posts by id in a single query. Returns {post_id: Post}."""
        if not post_ids:
            return {}
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    sql.SQL("SELECT * FROM {table} WHERE post_id = ANY(%s)").format(
                        table=sql.Identifier(self.table_name)
                    ),
                    (list(post_ids),),
                )
                rows = cur.fetchall()
                return {row["post_id"]: Post.model_validate(dict(row)) for row in rows}
        finally:
            conn.close()



    def mark_post_embedded(self, post_id: int) -> None:
        """Mark a post as embedded using its updated_at timestamp."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("UPDATE {table} SET embedded_at = updated_at WHERE post_id = %s")
                        .format(table=sql.Identifier(self.table_name)),(post_id,))
            conn.commit()
            logger.info("Marked post_id=%s as embedded in %s", post_id, self.table_name)
        except Exception:
            conn.rollback()
            logger.exception("Failed to mark post_id=%s embedded in %s", post_id, self.table_name)
            raise
        finally:
            conn.close()

    def iter_all_posts(self, batch_size: int = 100) -> Iterator[Post]:
        """Iterate all non-deleted posts from the table in batches."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                query = sql.SQL(
                    "SELECT * FROM {table} WHERE is_deleted = FALSE ORDER BY post_id"
                ).format(table=sql.Identifier(self.table_name))
                cur.execute(query)

                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        break
                    for row in rows:
                        yield Post.model_validate(dict(row))
        finally:
            conn.close()

    def _serialize_post(self, post: Post) -> Tuple:
        """Serialize a Post into a values tuple ordered by COLUMNS."""
        extra_fields = post.extra_fields
        if isinstance(extra_fields, (dict, list)):
            extra_fields = json.dumps(extra_fields)

        return (
            post.post_id,
            post.reddit_id,
            post.url,
            post.title,
            post.body_text,
            post.author,
            post.subreddit,
            post.reddit_created_at,
            post.added_at,
            post.updated_at,
            post.custom_title,
            post.custom_body,
            post.notes,
            post.rating,
            post.is_read,
            post.read_at,
            post.is_favorite,
            post.is_archived,
            post.is_deleted,
            post.queued_at,
            post.folder_ids,
            extra_fields,
            post.body_min_hash,
            post.summary,
        )

    def _ensure_table(self):
        """ makes sure the post table exists"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql.SQL(CREATE_TABLE_SQL).format(
                    table=sql.Identifier(self.table_name)
                ))
                cur.execute(sql.SQL(CREATE_UPDATED_AT_INDEX_SQL).format(
                    table=sql.Identifier(self.table_name),
                    index_name=sql.Identifier(f"{self.table_name}_updated_at_idx"),
                ))
                if not self._column_exists(cur, "is_deleted"):
                    cur.execute(
                        sql.SQL(
                            "ALTER TABLE {table} ADD COLUMN is_deleted BOOLEAN NOT NULL DEFAULT FALSE"
                        ).format(table=sql.Identifier(self.table_name))
                    )
                if not self._column_exists(cur, "read_at"):
                    cur.execute(
                        sql.SQL("ALTER TABLE {table} ADD COLUMN read_at TIMESTAMP")
                        .format(table=sql.Identifier(self.table_name))
                    )
                if not self._column_exists(cur, "is_archived"):
                    cur.execute(
                        sql.SQL(
                            "ALTER TABLE {table} ADD COLUMN is_archived BOOLEAN NOT NULL DEFAULT FALSE"
                        ).format(table=sql.Identifier(self.table_name))
                    )
                if not self._column_exists(cur, "queued_at"):
                    cur.execute(
                        sql.SQL("ALTER TABLE {table} ADD COLUMN queued_at TIMESTAMP")
                        .format(table=sql.Identifier(self.table_name))
                    )
                if not self._column_exists(cur, "folder_ids"):
                    cur.execute(
                        sql.SQL(
                            "ALTER TABLE {table} ADD COLUMN folder_ids INTEGER[] NOT NULL DEFAULT ARRAY[]::INTEGER[]"
                        ).format(table=sql.Identifier(self.table_name))
                    )
            conn.commit()
            logger.debug("Ensured table and index exist for %s", self.table_name)
        except psycopg2.errors.UniqueViolation:
            # Race condition: two concurrent requests tried to CREATE TABLE at the same time.
            # The other request won; our transaction is aborted — rollback and retry once so
            # the IF NOT EXISTS path runs cleanly now that the table exists.
            conn.rollback()
            logger.debug(
                "Concurrent table creation detected for %s; retrying _ensure_table",
                self.table_name,
            )
            conn.close()
            self._ensure_table()
            return
        except Exception:
            conn.rollback()
            logger.exception("Error ensuring post table %s exists", self.table_name)
            raise
        finally:
            conn.close()

    def _upsert_sql(self):
        assignments = [
            sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(col))
            for col in COLUMNS
            if col != "post_id"
        ]
        return sql.SQL(
            """
            INSERT INTO {table} ({columns}) VALUES ({placeholders})
            ON CONFLICT (post_id) DO UPDATE SET
                {assignments}
            WHERE EXCLUDED.updated_at > {table}.updated_at
            RETURNING (xmax = 0) AS inserted, updated_at
            """
        ).format(
            table=sql.Identifier(self.table_name),
            columns=sql.SQL(", ").join(sql.Identifier(c) for c in COLUMNS),
            placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in COLUMNS),
            assignments=sql.SQL(", ").join(assignments),
        )

    def _fetch_updated_at(self, conn, post_id: int) -> datetime | None:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT updated_at FROM {table} WHERE post_id = %s").format(
                table=sql.Identifier(self.table_name)),
                (post_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0] if isinstance(row, (list, tuple)) else row

    def _column_exists(self, cur, column_name: str) -> bool:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
              AND column_name = %s
            """,
            (self.table_name, column_name),
        )
        return cur.fetchone() is not None

    @staticmethod
    def _get_conn():
        logger.debug(
            "Opening Postgres connection to host=%s port=%s db=%s",
            DB_TARGET.hostname,
            DB_TARGET.port,
            DB_TARGET.path.lstrip("/"),
        )
        return psycopg2.connect(POSTGRES_URI)
