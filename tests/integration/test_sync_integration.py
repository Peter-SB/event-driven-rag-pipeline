"""Integration tests for the sync API endpoint: POST /posts/sync.

Tests the first step of the pipeline via real HTTP requests:
    POST /posts/sync → post row in Postgres + post.synced event in event_log

Uses a real Postgres container (via testcontainers) and the FastAPI ASGI app
wired with real infrastructure — no RabbitMQ needed because the sync route
only touches PostRepository and the event bus.

Run with:
    pytest tests/integration/ -m integration
"""
from __future__ import annotations

import json
from datetime import datetime, UTC, timedelta

import asyncpg
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from event_driven_rag_service.infrastructure.event_bus import PostgresEventBus
from event_driven_rag_service.repository.post_repository import PostRepository
from tests.utils.factories import make_post

pytestmark = pytest.mark.integration

_LIBRARY_ID = "inttest"
_POST_TABLE = f"posts_{_LIBRARY_ID}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def sync_client(postgres_pool: asyncpg.Pool, clean_event_bus_tables):
    """httpx AsyncClient wired to the FastAPI app with real Postgres.

    Injects PostRepository and PostgresEventBus into app.state so the sync
    route runs exactly as in production — no RabbitMQ required because the
    sync endpoint only publishes to the event log.
    """
    from event_driven_rag_service.api.app import app

    event_bus = PostgresEventBus(postgres_pool)
    await event_bus.setup_tables()

    app.state.pool = postgres_pool
    app.state.event_bus = event_bus
    app.state.post_repo = PostRepository(postgres_pool)
    app.state.seen_post_tables = set()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client

    async with postgres_pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {_POST_TABLE}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync_payload(*post_ids: int, updated_at: datetime | None = None, **post_kwargs) -> dict:
    posts = [
        make_post(post_id=pid, updated_at=updated_at, **post_kwargs).model_dump(by_alias=True, mode="json")
        for pid in post_ids
    ]
    return {"posts": posts, "library_id": _LIBRARY_ID}


async def _read_events(pool: asyncpg.Pool, topic: str = "post.synced") -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT payload FROM event_log WHERE topic = $1 ORDER BY id",
            topic,
        )
    return [json.loads(r["payload"]) for r in rows]


async def _fetch_post(pool: asyncpg.Pool, post_id: int) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT * FROM {_POST_TABLE} WHERE post_id = $1", post_id
        )
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sync_new_post_persists_row_in_database(postgres_pool, sync_client):
    """POST /posts/sync must write the post row to the library's Postgres table."""
    post = make_post(post_id=6001)
    await sync_client.post("/posts/sync", json=_sync_payload(6001))

    row = await _fetch_post(postgres_pool, 6001)
    assert row is not None
    assert row["post_id"] == 6001
    assert row["body_text"] == post.body_text


@pytest.mark.asyncio
async def test_sync_new_post_emits_post_synced_event(postgres_pool, sync_client):
    """POST /posts/sync must append one post.synced event to event_log."""
    await sync_client.post("/posts/sync", json=_sync_payload(6002))

    events = await _read_events(postgres_pool)
    assert len(events) == 1
    evt = events[0]
    assert evt["event_type"] == "post.synced"
    assert evt["post_id"] == 6002
    assert evt["post_table"] == _POST_TABLE
    assert evt["fields_changed"] == []   # empty = all fields new on first sync


@pytest.mark.asyncio
async def test_sync_duplicate_post_emits_no_second_event(postgres_pool, sync_client):
    """Re-syncing a post with the same updated_at must not emit a second event."""
    payload = _sync_payload(6003)
    await sync_client.post("/posts/sync", json=payload)   # first → inserted
    response = await sync_client.post("/posts/sync", json=payload)  # duplicate → skipped

    assert response.json()["results"][0]["status"] == "skipped"
    events = await _read_events(postgres_pool)
    assert len(events) == 1   # only the first sync produced an event


@pytest.mark.asyncio
async def test_sync_updated_post_emits_event_with_fields_changed(postgres_pool, sync_client):
    """Re-syncing with a fresher updated_at must emit a second event with fields_changed set."""
    base_ts = datetime(2024, 3, 1, 10, 0, tzinfo=UTC)
    newer_ts = base_ts + timedelta(hours=1)

    await sync_client.post("/posts/sync", json=_sync_payload(6004, updated_at=base_ts))
    await sync_client.post("/posts/sync", json=_sync_payload(6004, updated_at=newer_ts))

    events = await _read_events(postgres_pool)
    assert len(events) == 2
    update_evt = events[1]
    assert update_evt["post_id"] == 6004
    assert "body_text" in update_evt["fields_changed"]
