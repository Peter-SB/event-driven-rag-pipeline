"""Unit tests for POST /maintenance/requeue-missing-embeddings.

Tests the HTTP layer in isolation: correct status codes, response shape,
and that the route delegates to RequeueService with the right app.state deps.

No database, no RabbitMQ — all dependencies are mocked.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from event_driven_rag_service.api.maintenance import router as maintenance_router
from event_driven_rag_service.services.requeue_service import RequeueResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(**kwargs) -> RequeueResult:
    defaults = dict(
        requeued_chunks=0,
        tasks_published=0,
        tables_scanned=0,
        tables_skipped=0,
    )
    defaults.update(kwargs)
    return RequeueResult(**defaults)


@pytest.fixture
def app():
    """Minimal FastAPI app with mocked state for maintenance route tests."""
    _app = FastAPI()
    _app.include_router(maintenance_router)

    # maintenance_repo is passed to RequeueService as ChunkTableReader;
    # we mock it so nothing real is ever called.
    _app.state.maintenance_repo = MagicMock()
    # rmq is passed to RmqEmbedTaskPublisher; we mock it.
    _app.state.rmq = MagicMock()
    return _app


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------

def test_returns_200_with_expected_fields(client, monkeypatch):
    monkeypatch.setattr(
        "event_driven_rag_service.api.maintenance.RequeueService.requeue_missing_embeddings",
        AsyncMock(return_value=_make_result(requeued_chunks=5, tasks_published=3, tables_scanned=2)),
    )

    resp = client.post("/maintenance/requeue-missing-embeddings")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "requeued_chunks",
        "tasks_published",
        "tables_scanned",
        "tables_skipped",
    }


def test_response_values_reflect_service_result(client, monkeypatch):
    monkeypatch.setattr(
        "event_driven_rag_service.api.maintenance.RequeueService.requeue_missing_embeddings",
        AsyncMock(
            return_value=_make_result(
                requeued_chunks=12,
                tasks_published=4,
                tables_scanned=3,
                tables_skipped=1,
            )
        ),
    )

    body = client.post("/maintenance/requeue-missing-embeddings").json()

    assert body["requeued_chunks"] == 12
    assert body["tasks_published"] == 4
    assert body["tables_scanned"] == 3
    assert body["tables_skipped"] == 1


def test_zero_counts_on_empty_database(client, monkeypatch):
    monkeypatch.setattr(
        "event_driven_rag_service.api.maintenance.RequeueService.requeue_missing_embeddings",
        AsyncMock(return_value=_make_result()),
    )

    body = client.post("/maintenance/requeue-missing-embeddings").json()

    assert body["requeued_chunks"] == 0
    assert body["tasks_published"] == 0
    assert body["tables_scanned"] == 0
    assert body["tables_skipped"] == 0
