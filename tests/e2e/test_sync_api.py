"""E2E tests for the sync API endpoint.

Makes real HTTP calls to the API server running in the Docker Compose stack.
Verifies the first step of the pipeline from the outside:

    POST /posts/sync → 200 response + correct status per post

Requires the full stack to be up:
    docker compose up -d
    pytest tests/e2e/ -m e2e

Marked @pytest.mark.e2e so they are excluded from the default run.
"""
from __future__ import annotations

from datetime import datetime, UTC, timedelta

import pytest
from httpx import AsyncClient

from tests.utils.factories import make_post


pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync_payload(*post_ids: int, updated_at: datetime | None = None, **post_kwargs) -> dict:
    """Build a SyncRequest JSON body for one or more posts."""
    posts = [
        make_post(post_id=pid, updated_at=updated_at, **post_kwargs).model_dump(by_alias=True, mode='json')
        for pid in post_ids
    ]
    return {"posts": posts, "library_id": "e2e"}


# ---------------------------------------------------------------------------
# Liveness probe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_endpoint_returns_ok(async_client: AsyncClient):
    """GET /health must respond 200 and report the service as healthy."""
    response = await async_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Single-post ingest
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sync_new_post_returns_inserted(async_client: AsyncClient):
    """First ingest of a post must be persisted and reported as 'inserted'."""
    json_payload = _sync_payload(1001)
    response = await async_client.post("/posts/sync", json=json_payload)

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    result = results[0]
    assert result["status"] == "inserted", f"Expected status 'inserted', {result!r}"
    assert result["success"] is True
    assert result["post_id"] == 1001


@pytest.mark.asyncio
async def test_sync_duplicate_post_returns_skipped(async_client: AsyncClient):
    """Re-syncing a post with the same updated_at must be a no-op ('skipped')."""
    payload = _sync_payload(1002)
    await async_client.post("/posts/sync", json=payload)     # first ingest
    response = await async_client.post("/posts/sync", json=payload)  # duplicate

    result = response.json()["results"][0]
    assert result["status"] == "skipped", f"Expected status 'skipped', {result!r}"
    assert result["success"] is True


@pytest.mark.asyncio
async def test_sync_post_with_newer_timestamp_returns_updated(async_client: AsyncClient):
    """Re-syncing with a fresher updated_at must update the row and return 'updated'."""
    base_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    newer_time = base_time + timedelta(hours=1)

    await async_client.post("/posts/sync", json=_sync_payload(1003, updated_at=base_time))
    response = await async_client.post("/posts/sync", json=_sync_payload(1003, updated_at=newer_time))

    result = response.json()["results"][0]
    assert result["status"] == "updated", f"Expected status 'updated', {result!r}"
    assert result["success"] is True


# ---------------------------------------------------------------------------
# Batch ingest
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sync_batch_returns_inserted_for_all_new_posts(async_client: AsyncClient):
    """A batch of unseen posts must all be inserted and returned in order."""
    payload = _sync_payload(2001, 2002, 2003)
    response = await async_client.post("/posts/sync", json=payload)

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    assert all(r["status"] == "inserted" for r in results)
    assert [r["post_id"] for r in results] == [2001, 2002, 2003]
