"""Repository for persisting and managing async search jobs.

The ``search_jobs`` table is created on first use (``_ensure_table``), mirroring
the ``PGVecStore._ensure_table_and_index`` pattern used elsewhere.

Each job tracks:
- The query texts and their embeddings (stored by the worker once embedded)
- Search parameters (k, negative_weight, aggregation_strategy, ...)
- Status progression: embedding → searching → complete | failed
"""
import json
import uuid
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

from semantic_search_service.config import POSTGRES_URI

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS search_jobs (
    id UUID PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'embedding',
    positive_query TEXT NOT NULL,
    negative_query TEXT,
    positive_embedding FLOAT[],
    negative_embedding FLOAT[],
    k INTEGER NOT NULL DEFAULT 10,
    negative_weight FLOAT NOT NULL DEFAULT 0.3,
    aggregation_strategy TEXT NOT NULL DEFAULT 'max',
    embedding_profile TEXT NOT NULL,
    posts_table TEXT NOT NULL,
    chunks_table TEXT NOT NULL,
    unique_posts BOOLEAN NOT NULL DEFAULT FALSE,
    include_text BOOLEAN NOT NULL DEFAULT TRUE,
    results JSONB,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ
);
"""


class SearchJobRepository:
    """Manages search job records in Postgres."""

    def __init__(self):
        self._ensure_table()

    def _get_conn(self):
        return psycopg2.connect(POSTGRES_URI)

    def _ensure_table(self) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
            conn.commit()
        finally:
            conn.close()

    def create_job(
        self,
        positive_query: str,
        negative_query: Optional[str],
        k: int,
        negative_weight: float,
        aggregation_strategy: str,
        embedding_profile: str,
        posts_table: str,
        chunks_table: str,
        unique_posts: bool,
        include_text: bool,
    ) -> str:
        """Insert a new search job with status='embedding'. Returns the job_id UUID."""
        job_id = str(uuid.uuid4())
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO search_jobs (
                        id, status, positive_query, negative_query, k,
                        negative_weight, aggregation_strategy, embedding_profile,
                        posts_table, chunks_table, unique_posts, include_text
                    ) VALUES (%s, 'embedding', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job_id, positive_query, negative_query, k,
                        negative_weight, aggregation_strategy, embedding_profile,
                        posts_table, chunks_table, unique_posts, include_text,
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a job by id. Returns None if not found."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM search_jobs WHERE id = %s", (job_id,))
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            conn.close()

    def store_embedding(self, job_id: str, slot: str, embedding: List[float]) -> bool:
        """Atomically store an embedding for the given slot ('positive' or 'negative').

        Returns True when all required embeddings are now present, indicating
        that the search execution should be triggered. Uses a single
        ``UPDATE ... RETURNING`` to avoid a separate readiness query.
        """
        col = sql.Identifier(f"{slot}_embedding")
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("""
                        UPDATE search_jobs
                        SET {col} = %s
                        WHERE id = %s
                        RETURNING
                            positive_embedding IS NOT NULL AND
                            (negative_query IS NULL OR negative_embedding IS NOT NULL) AS ready
                    """).format(col=col),
                    (embedding, job_id),
                )
                row = cur.fetchone()
            conn.commit()
            return bool(row and row[0])
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def mark_searching(self, job_id: str) -> None:
        """Transition job to status='searching'."""
        self._set_status(job_id, "searching")

    def complete_job(self, job_id: str, results: List[Dict]) -> None:
        """Save results and transition job to status='complete'."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE search_jobs
                    SET status = 'complete', results = %s, completed_at = now()
                    WHERE id = %s
                    """,
                    (json.dumps(results), job_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def fail_job(self, job_id: str, error: str) -> None:
        """Record an error and transition job to status='failed'."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE search_jobs
                    SET status = 'failed', error = %s, completed_at = now()
                    WHERE id = %s
                    """,
                    (error, job_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _set_status(self, job_id: str, status: str) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE search_jobs SET status = %s WHERE id = %s",
                    (status, job_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


__all__ = ["SearchJobRepository"]
