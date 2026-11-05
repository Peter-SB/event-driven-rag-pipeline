"""Read-only SQL for folder suggestion.

Every method is a SELECT. Table names are built by the caller from validated
inputs (``validate_library_id`` + ``build_chunk_table_name``), never from user text.

pgvector's ``vector`` type has no HNSW index above 2000 dimensions, so for wide
models (Qwen3-4B = 2560) these queries are exact sequential scans. That is fine
for a personal library (thousands of rows); see docs/FolderSuggestions.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import asyncpg
import numpy as np

from event_driven_rag_service.exceptions import ChunkTableNotFoundError


@dataclass(frozen=True)
class CandidateRow:
    post_id: int
    title: str
    subreddit: str | None
    url: str | None
    embedding: np.ndarray


def parse_vector(raw: str | list | tuple | np.ndarray) -> np.ndarray:
    """pgvector text form ``[0.1,0.2,...]`` -> float32 array."""
    if isinstance(raw, str):
        return np.fromstring(raw.strip()[1:-1], dtype=np.float32, sep=",")
    return np.asarray(raw, dtype=np.float32)


class FolderSuggestionRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def folder_post_ids(self, posts_table: str, folder_id: int) -> list[int]:
        """Non-deleted posts that belong to *folder_id*."""
        async with self._pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    f"SELECT post_id FROM {posts_table.lower()} "
                    "WHERE $1 = ANY(folder_ids) AND NOT COALESCE(is_deleted, FALSE) "
                    "ORDER BY post_id",
                    folder_id,
                )
            except asyncpg.exceptions.UndefinedTableError:
                raise ChunkTableNotFoundError(posts_table)
        return [r["post_id"] for r in rows]

    async def folder_embeddings(self, chunks_table: str, post_ids: list[int]) -> dict[int, np.ndarray]:
        """One vector per post (mean of its embedded chunks). Posts with none are absent."""
        async with self._pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    f"SELECT post_id, avg(embedding)::text AS embedding FROM {chunks_table.lower()} "
                    "WHERE post_id = ANY($1::int[]) AND embedding IS NOT NULL GROUP BY post_id",
                    post_ids,
                )
            except asyncpg.exceptions.UndefinedTableError:
                raise ChunkTableNotFoundError(chunks_table)
        return {r["post_id"]: parse_vector(r["embedding"]) for r in rows}

    async def nearest_candidates(
        self,
        chunks_table: str,
        posts_table: str,
        query: np.ndarray,
        limit: int,
        exclude_post_ids: list[int],
    ) -> list[CandidateRow]:
        """Nearest embedded chunks to *query* (cosine), skipping deleted and excluded posts."""
        vec = "[" + ",".join(str(float(x)) for x in query) + "]"
        async with self._pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    f"""
                    SELECT c.post_id, c.embedding::text AS embedding, p.title, p.subreddit, p.url
                    FROM {chunks_table.lower()} c
                    JOIN {posts_table.lower()} p ON p.post_id = c.post_id
                    WHERE c.embedding IS NOT NULL
                      AND NOT COALESCE(p.is_deleted, FALSE)
                      AND NOT (c.post_id = ANY($2::int[]))
                    ORDER BY c.embedding <=> $1::vector
                    LIMIT $3
                    """,
                    vec,
                    exclude_post_ids,
                    limit,
                )
            except asyncpg.exceptions.UndefinedTableError:
                raise ChunkTableNotFoundError(chunks_table)
        return [
            CandidateRow(
                post_id=r["post_id"],
                title=r["title"] or "",
                subreddit=r["subreddit"],
                url=r["url"],
                embedding=parse_vector(r["embedding"]),
            )
            for r in rows
        ]

    async def post_titles(self, posts_table: str, post_ids: list[int]) -> dict[int, str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT post_id, title FROM {posts_table.lower()} WHERE post_id = ANY($1::int[])",
                post_ids,
            )
        return {r["post_id"]: r["title"] or "" for r in rows}


VERDICTS = ("very_no", "no", "yes", "very_yes")
POSITIVE_VERDICTS = ("yes", "very_yes")
NEGATIVE_VERDICTS = ("very_no", "no")


class FeedbackRepository:
    """Explicit four-level verdicts on suggestions, per (library, folder).

    ``very_no`` / ``no`` / ``yes`` / ``very_yes``. This is the only write path in the
    folder-suggestion feature. Absence of a row means *unrated*, never "no": a story
    missing from a folder is not a negative. Reads tolerate the table not existing yet
    so a read-only database still works.
    """

    TABLE = "folder_suggestion_feedback"
    _schema_ready = False  # per process: the migration below runs once

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def verdicts(self, library_id: str, folder_key: str) -> dict[int, str]:
        """``{post_id: verdict}`` — empty if nothing rated or the table is absent."""
        async with self._pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    f"SELECT post_id, verdict FROM {self.TABLE} WHERE library_id = $1 AND folder_key = $2",
                    library_id,
                    folder_key,
                )
            except (asyncpg.exceptions.UndefinedTableError, asyncpg.exceptions.UndefinedColumnError):
                return {}
        return {r["post_id"]: r["verdict"] for r in rows}

    async def _ensure_schema(self, conn) -> None:
        """Create the table, or bring an older one up to date.

        The first version was named ``recommendation_feedback`` with a ``playlist_key``
        column and only allowed ``yes`` / ``no``; existing rows stay valid.
        """
        if FeedbackRepository._schema_ready:
            return
        await conn.execute(
            f"""
            DO $$ BEGIN
                IF to_regclass('recommendation_feedback') IS NOT NULL
                   AND to_regclass('{self.TABLE}') IS NULL THEN
                    ALTER TABLE recommendation_feedback RENAME TO {self.TABLE};
                END IF;
                IF EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_name = '{self.TABLE}' AND column_name = 'playlist_key') THEN
                    ALTER TABLE {self.TABLE} RENAME COLUMN playlist_key TO folder_key;
                END IF;
            END $$;
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                library_id TEXT NOT NULL,
                folder_key TEXT NOT NULL,
                post_id    INTEGER NOT NULL,
                verdict    TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (library_id, folder_key, post_id)
            );
            DO $$ DECLARE c record; BEGIN
                FOR c IN SELECT conname FROM pg_constraint
                         WHERE conrelid = '{self.TABLE}'::regclass AND contype = 'c' LOOP
                    EXECUTE format('ALTER TABLE {self.TABLE} DROP CONSTRAINT %I', c.conname);
                END LOOP;
            END $$;
            ALTER TABLE {self.TABLE} ADD CONSTRAINT {self.TABLE}_verdict_check
                CHECK (verdict IN ({", ".join(f"'{v}'" for v in VERDICTS)}));
            """
        )
        FeedbackRepository._schema_ready = True

    async def set_verdict(self, library_id: str, folder_key: str, post_id: int, verdict: str) -> None:
        """Record a verdict; ``clear`` removes the rating (back to unrated)."""
        if verdict != "clear" and verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS} or 'clear', got {verdict!r}")
        async with self._pool.acquire() as conn:
            await self._ensure_schema(conn)
            if verdict == "clear":
                await conn.execute(
                    f"DELETE FROM {self.TABLE} WHERE library_id=$1 AND folder_key=$2 AND post_id=$3",
                    library_id, folder_key, post_id,
                )
            else:
                await conn.execute(
                    f"""
                    INSERT INTO {self.TABLE} (library_id, folder_key, post_id, verdict)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (library_id, folder_key, post_id)
                    DO UPDATE SET verdict = EXCLUDED.verdict, created_at = now()
                    """,
                    library_id, folder_key, post_id, verdict,
                )
