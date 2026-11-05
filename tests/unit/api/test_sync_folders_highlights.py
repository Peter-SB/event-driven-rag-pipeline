"""Unit tests for POST /folders/sync and POST /highlights/sync.

Tests the route logic in isolation with mocked repositories — no database, no I/O.

Key scenarios:
  - Per-item results and statuses
  - Table name is derived from library_id and created lazily
  - A failing item is reported without failing the batch
  - library_id validation and empty batches
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from event_driven_rag_service.api.sync_folders import router as folders_router
from event_driven_rag_service.api.sync_highlights import router as highlights_router
from tests.utils.factories import make_folder, make_highlight


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(folders_router)
    app.include_router(highlights_router)
    app.state.folder_repo = AsyncMock()
    app.state.highlight_repo = AsyncMock()
    with TestClient(app) as c:
        yield c


def _wire(model) -> dict:
    return model.model_dump(by_alias=True, mode="json")


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

def test_sync_folders_reports_per_folder_status(client):
    repo = client.app.state.folder_repo
    repo.upsert.side_effect = [("inserted", None), ("skipped", None)]

    response = client.post("/folders/sync", json={
        "folders": [_wire(make_folder(1)), _wire(make_folder(2))],
        "library_id": "main",
    })

    assert response.status_code == 200
    assert response.json()["results"] == [
        {"folder_id": 1, "status": "inserted", "success": True},
        {"folder_id": 2, "status": "skipped", "success": True},
    ]


def test_sync_folders_uses_library_scoped_table(client):
    repo = client.app.state.folder_repo
    repo.upsert.return_value = ("inserted", None)

    client.post("/folders/sync", json={"folders": [_wire(make_folder(1))], "library_id": "work"})

    repo.ensure_table.assert_awaited_once_with("folders_work")
    assert repo.upsert.await_args.args[1] == "folders_work"


def test_sync_folders_failed_item_does_not_fail_batch(client):
    repo = client.app.state.folder_repo
    repo.upsert.side_effect = [RuntimeError("boom"), ("inserted", None)]

    response = client.post("/folders/sync", json={
        "folders": [_wire(make_folder(1)), _wire(make_folder(2))],
        "library_id": "main",
    })

    results = response.json()["results"]
    assert response.status_code == 200
    assert results[0] == {"folder_id": 1, "status": "error", "success": False, "error": "boom"}
    assert results[1]["success"] is True


def test_sync_folders_rejects_invalid_library_id(client):
    response = client.post("/folders/sync", json={"folders": [], "library_id": "Bad-Id"})
    assert response.status_code == 422


def test_sync_folders_requires_library_id(client):
    response = client.post("/folders/sync", json={"folders": []})
    assert response.status_code == 422


def test_sync_folders_empty_batch_returns_empty_results(client):
    response = client.post("/folders/sync", json={"folders": [], "library_id": "main"})
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_sync_folders_accepts_soft_deleted_folder(client):
    repo = client.app.state.folder_repo
    repo.upsert.return_value = ("updated", None)

    response = client.post("/folders/sync", json={
        "folders": [_wire(make_folder(1, isDeleted=True))],
        "library_id": "main",
    })

    assert response.status_code == 200
    assert repo.upsert.await_args.args[0].is_deleted is True


# ---------------------------------------------------------------------------
# Highlights
# ---------------------------------------------------------------------------

def test_sync_highlights_reports_per_highlight_status(client):
    repo = client.app.state.highlight_repo
    repo.upsert.side_effect = [("inserted", None), ("updated", None)]

    response = client.post("/highlights/sync", json={
        "highlights": [_wire(make_highlight(1)), _wire(make_highlight(2))],
        "library_id": "main",
    })

    assert response.status_code == 200
    assert response.json()["results"] == [
        {"highlight_id": 1, "status": "inserted", "success": True},
        {"highlight_id": 2, "status": "updated", "success": True},
    ]


def test_sync_highlights_uses_library_scoped_table(client):
    repo = client.app.state.highlight_repo
    repo.upsert.return_value = ("inserted", None)

    client.post("/highlights/sync", json={"highlights": [_wire(make_highlight(1))], "library_id": "work"})

    repo.ensure_table.assert_awaited_once_with("highlights_work")
    assert repo.upsert.await_args.args[1] == "highlights_work"


def test_sync_highlights_failed_item_does_not_fail_batch(client):
    repo = client.app.state.highlight_repo
    repo.upsert.side_effect = [("inserted", None), RuntimeError("boom")]

    response = client.post("/highlights/sync", json={
        "highlights": [_wire(make_highlight(1)), _wire(make_highlight(2))],
        "library_id": "main",
    })

    results = response.json()["results"]
    assert results[0]["success"] is True
    assert results[1] == {"highlight_id": 2, "status": "error", "success": False, "error": "boom"}


def test_sync_highlights_rejects_invalid_library_id(client):
    response = client.post("/highlights/sync", json={"highlights": [], "library_id": "1bad"})
    assert response.status_code == 422


def test_sync_highlights_rejects_missing_text(client):
    bad = _wire(make_highlight(1))
    del bad["text"]
    response = client.post("/highlights/sync", json={"highlights": [bad], "library_id": "main"})
    assert response.status_code == 422
