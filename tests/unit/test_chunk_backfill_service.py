"""Unit tests for ChunkBackfillService.

All I/O is replaced with in-process fakes - no database, no RabbitMQ.

Tested behaviours
------------------
- requeue_missing_chunks() publishes one summary_title ChunkTask per post
  with a summary
- posts without a summary are never surfaced by the reader, so no task is
  published for them
- multiple post tables -> tasks published for each
- empty database (no post tables) -> zero counts, no errors
- published tasks use EMBED_CONFIGS["summary_title"].primary as embed_model
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.services.chunk_backfill_service import ChunkBackfillService
from event_driven_rag_service.tasks.chunk_task import ChunkTask


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakePostSummaryReader:
    """In-memory fake for PostSummaryReader protocol."""

    def __init__(
        self,
        tables: list[str] | None = None,
        posts_with_summary: dict[str, list[int]] | None = None,
    ) -> None:
        self._tables = tables or []
        self._posts = posts_with_summary or {}

    async def list_post_tables(self) -> list[str]:
        return list(self._tables)

    async def fetch_posts_with_summary(self, table_name: str) -> list[int]:
        return list(self._posts.get(table_name, []))


@dataclass
class FakeChunkTaskPublisher:
    """In-memory fake for ChunkTaskPublisher protocol."""

    published: list[ChunkTask] = field(default_factory=list)

    async def publish(self, task: ChunkTask) -> None:
        self.published.append(task)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(
    tables: list[str] | None = None,
    posts_with_summary: dict[str, list[int]] | None = None,
) -> tuple[ChunkBackfillService, FakeChunkTaskPublisher]:
    reader = FakePostSummaryReader(tables, posts_with_summary)
    publisher = FakeChunkTaskPublisher()
    service = ChunkBackfillService(reader=reader, publisher=publisher)
    return service, publisher


# ---------------------------------------------------------------------------
# Tests - empty / no-work scenarios
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_post_tables_returns_zero_counts():
    service, publisher = _make_service(tables=[], posts_with_summary={})

    result = await service.requeue_missing_chunks()

    assert result.posts_scanned == 0
    assert result.tasks_published == 0
    assert result.tables_scanned == 0
    assert publisher.published == []


@pytest.mark.asyncio
async def test_table_with_no_summaries_produces_no_tasks():
    service, publisher = _make_service(
        tables=["posts_main"],
        posts_with_summary={"posts_main": []},
    )

    result = await service.requeue_missing_chunks()

    assert result.tables_scanned == 1
    assert result.posts_scanned == 0
    assert result.tasks_published == 0
    assert publisher.published == []


# ---------------------------------------------------------------------------
# Tests - task publishing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_post_with_summary_produces_one_task():
    service, publisher = _make_service(
        tables=["posts_main"],
        posts_with_summary={"posts_main": [42]},
    )

    result = await service.requeue_missing_chunks()

    assert result.posts_scanned == 1
    assert result.tasks_published == 1
    assert len(publisher.published) == 1

    task = publisher.published[0]
    assert task.task_type == "summary_title"
    assert task.post_id == 42
    assert task.post_table == "posts_main"
    assert task.embed_model == EMBED_CONFIGS["summary_title"].primary


@pytest.mark.asyncio
async def test_multiple_posts_in_one_table_produce_separate_tasks():
    service, publisher = _make_service(
        tables=["posts_main"],
        posts_with_summary={"posts_main": [1, 2, 3]},
    )

    result = await service.requeue_missing_chunks()

    assert result.posts_scanned == 3
    assert result.tasks_published == 3
    post_ids = {t.post_id for t in publisher.published}
    assert post_ids == {1, 2, 3}


@pytest.mark.asyncio
async def test_multiple_tables_all_get_tasks():
    service, publisher = _make_service(
        tables=["posts_main", "posts_work"],
        posts_with_summary={
            "posts_main": [1],
            "posts_work": [2, 3],
        },
    )

    result = await service.requeue_missing_chunks()

    assert result.tables_scanned == 2
    assert result.posts_scanned == 3
    assert result.tasks_published == 3

    post_tables = {t.post_table for t in publisher.published}
    assert post_tables == {"posts_main", "posts_work"}
