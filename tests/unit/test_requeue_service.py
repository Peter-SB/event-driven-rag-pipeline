"""Unit tests for RequeueService.

All I/O is replaced with in-process fakes — no database, no RabbitMQ.

Tested behaviours
-----------------
- requeue_missing_embeddings() publishes one task per post_id per table
- chunks with embeddings are not requeued (fetch_unembedded_chunks returns [])
- multiple posts in one table → one task each
- multiple tables → tasks published for each
- unrecognised chunk table → skipped, counted in tables_skipped
- result totals are accumulated correctly across tables
- grouping: chunks from the same post go into a single task
- empty database (no chunk tables) → zero counts, no errors
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.services.requeue_service import RequeueService
from event_driven_rag_service.tasks.embed_task import EmbedTask
from event_driven_rag_service.utils.build_table_names import build_chunk_table_name


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeChunkTableReader:
    """In-memory fake for ChunkTableReader protocol."""

    def __init__(
        self,
        tables: list[str] | None = None,
        unembedded_by_table: dict[str, list[tuple[str, int]]] | None = None,
    ) -> None:
        self._tables = tables or []
        self._unembedded = unembedded_by_table or {}

    async def list_chunk_tables(self) -> list[str]:
        return list(self._tables)

    async def fetch_unembedded_chunks(self, table_name: str) -> list[tuple[str, int]]:
        return list(self._unembedded.get(table_name, []))


@dataclass
class FakeEmbedTaskPublisher:
    """In-memory fake for EmbedTaskPublisher protocol."""

    published: list[tuple[EmbedTask, str]] = field(default_factory=list)

    async def publish(self, task: EmbedTask, routing_key: str) -> None:
        self.published.append((task, routing_key))

    def tasks_for_key(self, key: str) -> list[EmbedTask]:
        return [t for t, rk in self.published if rk == key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BODY_CFG = EMBED_CONFIGS["body"]
_BODY_TABLE = build_chunk_table_name("posts_main", "body", _BODY_CFG.model)


def _make_service(
    tables: list[str] | None = None,
    unembedded: dict[str, list[tuple[str, int]]] | None = None,
) -> tuple[RequeueService, FakeChunkTableReader, FakeEmbedTaskPublisher]:
    reader = FakeChunkTableReader(tables, unembedded)
    publisher = FakeEmbedTaskPublisher()
    service = RequeueService(reader=reader, publisher=publisher)
    return service, reader, publisher


# ---------------------------------------------------------------------------
# Tests — empty / no-work scenarios
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_chunk_tables_returns_zero_counts():
    service, _, publisher = _make_service(tables=[], unembedded={})

    result = await service.requeue_missing_embeddings()

    assert result.requeued_chunks == 0
    assert result.tasks_published == 0
    assert result.tables_scanned == 0
    assert result.tables_skipped == 0
    assert publisher.published == []


@pytest.mark.asyncio
async def test_table_with_no_missing_embeddings_produces_no_tasks():
    service, _, publisher = _make_service(
        tables=[_BODY_TABLE],
        unembedded={_BODY_TABLE: []},
    )

    result = await service.requeue_missing_embeddings()

    assert result.requeued_chunks == 0
    assert result.tasks_published == 0
    assert publisher.published == []


# ---------------------------------------------------------------------------
# Tests — task publishing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_chunk_produces_one_task():
    service, _, publisher = _make_service(
        tables=[_BODY_TABLE],
        unembedded={_BODY_TABLE: [("chunk-1", 42)]},
    )

    result = await service.requeue_missing_embeddings()

    assert result.requeued_chunks == 1
    assert result.tasks_published == 1
    assert len(publisher.published) == 1

    task, routing_key = publisher.published[0]
    assert routing_key == _BODY_CFG.queue
    assert task.model_name == _BODY_CFG.model
    assert task.post_id == 42
    assert task.chunk_ids == ["chunk-1"]
    assert task.chunk_table == _BODY_TABLE
    assert task.post_table == "posts_main"


@pytest.mark.asyncio
async def test_chunks_from_same_post_grouped_into_one_task():
    service, _, publisher = _make_service(
        tables=[_BODY_TABLE],
        unembedded={
            _BODY_TABLE: [("chunk-1", 10), ("chunk-2", 10), ("chunk-3", 10)]
        },
    )

    result = await service.requeue_missing_embeddings()

    assert result.tasks_published == 1
    assert result.requeued_chunks == 3
    task, _ = publisher.published[0]
    assert set(task.chunk_ids) == {"chunk-1", "chunk-2", "chunk-3"}


@pytest.mark.asyncio
async def test_chunks_from_different_posts_produce_separate_tasks():
    service, _, publisher = _make_service(
        tables=[_BODY_TABLE],
        unembedded={
            _BODY_TABLE: [("chunk-1", 1), ("chunk-2", 2), ("chunk-3", 1)]
        },
    )

    result = await service.requeue_missing_embeddings()

    assert result.tasks_published == 2
    assert result.requeued_chunks == 3

    post_ids = {t.post_id for t, _ in publisher.published}
    assert post_ids == {1, 2}

    task_for_1 = next(t for t, _ in publisher.published if t.post_id == 1)
    assert set(task_for_1.chunk_ids) == {"chunk-1", "chunk-3"}


@pytest.mark.asyncio
async def test_multiple_tables_all_get_tasks():
    title_cfg = EMBED_CONFIGS["title"]
    title_table = build_chunk_table_name("posts_main", "title", title_cfg.model)

    service, _, publisher = _make_service(
        tables=[_BODY_TABLE, title_table],
        unembedded={
            _BODY_TABLE: [("b-1", 1)],
            title_table: [("t-1", 1)],
        },
    )

    result = await service.requeue_missing_embeddings()

    assert result.tables_scanned == 2
    assert result.tasks_published == 2
    assert result.requeued_chunks == 2

    routing_keys = {rk for _, rk in publisher.published}
    assert _BODY_CFG.queue in routing_keys
    assert title_cfg.queue in routing_keys


# ---------------------------------------------------------------------------
# Tests — result accounting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unrecognised_table_is_skipped_and_counted():
    service, _, publisher = _make_service(
        tables=["posts_main_chunks_unknown_model"],
        unembedded={},
    )

    result = await service.requeue_missing_embeddings()

    assert result.tables_skipped == 1
    assert result.tables_scanned == 1
    assert result.requeued_chunks == 0
    assert "posts_main_chunks_unknown_model" in result.skipped_table_names
    assert publisher.published == []


@pytest.mark.asyncio
async def test_totals_accumulate_across_tables():
    title_cfg = EMBED_CONFIGS["title"]
    title_table = build_chunk_table_name("posts_work", "title", title_cfg.model)

    service, _, publisher = _make_service(
        tables=[_BODY_TABLE, title_table],
        unembedded={
            _BODY_TABLE: [("b-1", 1), ("b-2", 2)],   # 2 posts → 2 tasks
            title_table: [("t-1", 3)],                 # 1 post  → 1 task
        },
    )

    result = await service.requeue_missing_embeddings()

    assert result.requeued_chunks == 3
    assert result.tasks_published == 3
    assert result.tables_skipped == 0
