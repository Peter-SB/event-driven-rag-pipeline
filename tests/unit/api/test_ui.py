"""Unit tests for the HTMX search UI endpoints (/ui/*).

Uses a minimal FastAPI app with in-memory fakes — no Postgres, no RabbitMQ,
no lifespan.  Asserts on HTTP status codes and key HTML fragments.
"""
from __future__ import annotations

from datetime import datetime, UTC
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from event_driven_rag_service.api.ui import router as ui_router
from tests.utils.factories import FakeEventBus


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeChunkRepo:
    """Reports whether the chunk table exists — controls the pre-search validation."""

    def __init__(self, exists: bool = True) -> None:
        self.exists = exists
        self.checked_tables: list[str] = []

    async def table_exists(self, table_name: str) -> bool:
        self.checked_tables.append(table_name)
        return self.exists


class FakeSearchJobRepo:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._counter = 0

    async def ensure_table(self) -> None:
        pass

    async def create_job(
        self, query: str, k: int, embedding_profile: str, chunks_table: str, library_id: str = ""
    ) -> str:
        self._counter += 1
        job_id = f"fake-{self._counter:04d}"
        self._jobs[job_id] = {
            "id": job_id,
            "status": "embedding",
            "library_id": library_id,
            "query": query,
            "k": k,
            "embedding_profile": embedding_profile,
            "chunks_table": chunks_table,
            "embedding": None,
            "results": None,
            "error": None,
            "created_at": datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            "completed_at": None,
        }
        return job_id

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def set_status(self, job_id: str, status: str, **extra) -> None:
        self._jobs[job_id]["status"] = status
        self._jobs[job_id].update(extra)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_repo():
    return FakeSearchJobRepo()


@pytest.fixture
def client(fake_repo):
    app = FastAPI()
    app.include_router(ui_router)
    app.state.search_job_repo = fake_repo
    app.state.event_bus = FakeEventBus()
    app.state.pool = MagicMock()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def fake_chunk_repo():
    """By default, the chunk table always exists — most tests aren't about validation."""
    fake = FakeChunkRepo(exists=True)
    with patch("event_driven_rag_service.api.ui.ChunkRepository", return_value=fake):
        yield fake


# ---------------------------------------------------------------------------
# GET /ui/search — search page
# ---------------------------------------------------------------------------

def test_search_page_returns_200(client):
    resp = client.get("/ui/search")
    assert resp.status_code == 200


def test_search_page_contains_form(client):
    resp = client.get("/ui/search")
    assert "hx-post" in resp.text
    assert 'name="query"' in resp.text


def test_search_page_lists_chunk_types(client):
    resp = client.get("/ui/search")
    assert "body" in resp.text
    assert "title" in resp.text


def test_search_page_has_jobs_table(client):
    resp = client.get("/ui/search")
    assert "jobs-body" in resp.text


# ---------------------------------------------------------------------------
# POST /ui/search/submit — create job, return row partial
# ---------------------------------------------------------------------------

def test_submit_search_returns_200_with_table_row(client):
    resp = client.post(
        "/ui/search/submit",
        data={"query": "machine learning", "library_id": "main", "chunk_type": "body", "k": "10"},
    )
    assert resp.status_code == 200
    assert "<tr" in resp.text


def test_submit_search_row_contains_query(client):
    resp = client.post(
        "/ui/search/submit",
        data={"query": "deep learning", "library_id": "main", "chunk_type": "body", "k": "5"},
    )
    assert "deep learning" in resp.text


def test_submit_search_row_shows_embedding_status(client):
    resp = client.post(
        "/ui/search/submit",
        data={"query": "test", "library_id": "main", "chunk_type": "body", "k": "10"},
    )
    assert "embedding" in resp.text
    assert "badge-embedding" in resp.text


def test_submit_search_row_has_poll_trigger(client):
    """Rows for in-progress jobs must carry htmx polling attributes."""
    resp = client.post(
        "/ui/search/submit",
        data={"query": "test", "library_id": "main", "chunk_type": "body", "k": "10"},
    )
    assert "hx-trigger" in resp.text
    assert "hx-get" in resp.text


def test_submit_search_emits_event(client):
    client.post(
        "/ui/search/submit",
        data={"query": "rag pipeline", "library_id": "lib1", "chunk_type": "body", "k": "3"},
    )
    bus: FakeEventBus = client.app.state.event_bus
    events = bus.drain_topic("search_job.created")
    assert len(events) == 1
    assert events[0]["query"] == "rag pipeline"


def test_submit_search_invalid_library_id_returns_422(client):
    resp = client.post(
        "/ui/search/submit",
        data={"query": "test", "library_id": "INVALID", "chunk_type": "body", "k": "10"},
    )
    assert resp.status_code == 422


def test_submit_search_invalid_chunk_type_returns_422(client):
    resp = client.post(
        "/ui/search/submit",
        data={"query": "test", "library_id": "main", "chunk_type": "nonexistent", "k": "10"},
    )
    assert resp.status_code == 422


def test_submit_search_missing_chunk_table_returns_404(client, fake_chunk_repo):
    """Submitting a search for a library/chunk_type that was never synced should 404
    up front rather than creating a job that fails later."""
    fake_chunk_repo.exists = False
    resp = client.post(
        "/ui/search/submit",
        data={"query": "test", "library_id": "neversynced", "chunk_type": "body", "k": "10"},
    )
    assert resp.status_code == 404
    assert "neversynced" in resp.text


def test_submit_search_missing_chunk_table_does_not_create_job(client, fake_repo, fake_chunk_repo):
    fake_chunk_repo.exists = False
    client.post(
        "/ui/search/submit",
        data={"query": "test", "library_id": "neversynced", "chunk_type": "body", "k": "10"},
    )
    assert len(fake_repo._jobs) == 0


def test_submit_search_creates_job_in_repo(client, fake_repo):
    client.post(
        "/ui/search/submit",
        data={"query": "stored query", "library_id": "test", "chunk_type": "body", "k": "7"},
    )
    assert len(fake_repo._jobs) == 1
    job = next(iter(fake_repo._jobs.values()))
    assert job["query"] == "stored query"
    assert job["k"] == 7
    assert job["library_id"] == "test"


# ---------------------------------------------------------------------------
# GET /ui/search/{job_id}/status — job row polling partial
# ---------------------------------------------------------------------------

def _create_job(client, query="test query"):
    resp = client.post(
        "/ui/search/submit",
        data={"query": query, "library_id": "main", "chunk_type": "body", "k": "10"},
    )
    # extract job_id from the tr id attribute
    import re
    m = re.search(r'id="job-([^"]+)"', resp.text)
    assert m, f"No job id in row: {resp.text}"
    return m.group(1)


def test_status_returns_200_for_known_job(client):
    job_id = _create_job(client)
    resp = client.get(f"/ui/search/{job_id}/status")
    assert resp.status_code == 200


def test_status_row_shows_current_status(client, fake_repo):
    job_id = _create_job(client)
    fake_repo.set_status(job_id, "searching")

    resp = client.get(f"/ui/search/{job_id}/status")
    assert "searching" in resp.text
    assert "badge-searching" in resp.text


def test_status_row_still_polls_when_pending(client):
    job_id = _create_job(client)
    resp = client.get(f"/ui/search/{job_id}/status")
    assert "hx-trigger" in resp.text


def test_status_row_drops_poll_trigger_when_complete(client, fake_repo):
    job_id = _create_job(client)
    fake_repo.set_status(
        job_id, "complete",
        results=[{"chunk_id": "c1", "post_id": 1, "text": "hi", "metadata": None, "score": 0.9}],
    )
    resp = client.get(f"/ui/search/{job_id}/status")
    assert "badge-complete" in resp.text
    assert "hx-trigger" not in resp.text


def test_status_complete_row_is_clickable(client, fake_repo):
    job_id = _create_job(client)
    fake_repo.set_status(
        job_id, "complete",
        results=[{"chunk_id": "c1", "post_id": 1, "text": "hi", "metadata": None, "score": 0.9}],
    )
    resp = client.get(f"/ui/search/{job_id}/status")
    assert "clickable" in resp.text
    assert f"/ui/search/{job_id}" in resp.text


def test_status_shows_result_count_when_complete(client, fake_repo):
    job_id = _create_job(client)
    fake_repo.set_status(
        job_id, "complete",
        results=[
            {"chunk_id": "c1", "post_id": 1, "text": "a", "metadata": None, "score": 0.9},
            {"chunk_id": "c2", "post_id": 2, "text": "b", "metadata": None, "score": 0.8},
        ],
    )
    resp = client.get(f"/ui/search/{job_id}/status")
    assert "2" in resp.text


def test_status_returns_error_row_for_missing_job(client):
    resp = client.get("/ui/search/nonexistent-id/status")
    assert "not found" in resp.text.lower()


# ---------------------------------------------------------------------------
# GET /ui/search/{job_id}/poll — results content partial
# ---------------------------------------------------------------------------

def test_poll_returns_spinner_when_embedding(client):
    job_id = _create_job(client)
    resp = client.get(f"/ui/search/{job_id}/poll")
    assert resp.status_code == 200
    assert "spinner" in resp.text or "Processing" in resp.text


def test_poll_continues_polling_while_pending(client):
    job_id = _create_job(client)
    resp = client.get(f"/ui/search/{job_id}/poll")
    assert "hx-trigger" in resp.text


def test_poll_returns_cards_when_complete(client, fake_repo):
    job_id = _create_job(client)
    fake_repo.set_status(
        job_id, "complete",
        results=[{
            "chunk_id": "abc", "post_id": 42, "text": "hello world", "metadata": None, "score": 0.95,
        }],
    )
    resp = client.get(f"/ui/search/{job_id}/poll")
    assert resp.status_code == 200
    assert "result-card" in resp.text
    assert "hello world" in resp.text


def test_poll_stops_polling_when_complete(client, fake_repo):
    job_id = _create_job(client)
    fake_repo.set_status(
        job_id, "complete",
        results=[{"chunk_id": "c1", "post_id": 1, "text": "x", "metadata": None, "score": 0.9}],
    )
    resp = client.get(f"/ui/search/{job_id}/poll")
    assert "hx-trigger" not in resp.text


def test_poll_shows_error_when_failed(client, fake_repo):
    job_id = _create_job(client)
    fake_repo.set_status(job_id, "failed", error="pgvector index not ready")
    resp = client.get(f"/ui/search/{job_id}/poll")
    assert "failed" in resp.text.lower() or "error" in resp.text.lower()
    assert "pgvector index not ready" in resp.text


def test_poll_shows_metadata_title_in_card(client, fake_repo):
    job_id = _create_job(client)
    fake_repo.set_status(
        job_id, "complete",
        results=[{
            "chunk_id": "c1",
            "post_id": 7,
            "text": "some text here",
            "metadata": {"title": "My Article Title", "external_id": "ext7"},
            "score": 0.88,
        }],
    )
    resp = client.get(f"/ui/search/{job_id}/poll")
    assert "My Article Title" in resp.text


def test_poll_truncates_long_text(client, fake_repo):
    """Text longer than 300 chars should show 'Load more' button."""
    long_text = "word " * 100  # 500 chars
    job_id = _create_job(client)
    fake_repo.set_status(
        job_id, "complete",
        results=[{"chunk_id": "c1", "post_id": 1, "text": long_text, "metadata": None, "score": 0.7}],
    )
    resp = client.get(f"/ui/search/{job_id}/poll")
    assert "Load more" in resp.text
    assert "text-preview" in resp.text
    assert "text-full" in resp.text


def test_poll_no_load_more_for_short_text(client, fake_repo):
    """Text under 300 chars should not show a 'Load more' button."""
    job_id = _create_job(client)
    fake_repo.set_status(
        job_id, "complete",
        results=[{"chunk_id": "c1", "post_id": 1, "text": "short text", "metadata": None, "score": 0.9}],
    )
    resp = client.get(f"/ui/search/{job_id}/poll")
    assert "Load more" not in resp.text


# ---------------------------------------------------------------------------
# GET /ui/search/{job_id} — full results page
# ---------------------------------------------------------------------------

def test_results_page_returns_200(client):
    job_id = _create_job(client, query="neural networks")
    resp = client.get(f"/ui/search/{job_id}")
    assert resp.status_code == 200


def test_results_page_shows_query(client):
    job_id = _create_job(client, query="transformer architecture")
    resp = client.get(f"/ui/search/{job_id}")
    assert "transformer architecture" in resp.text


def test_results_page_shows_status_badge(client, fake_repo):
    job_id = _create_job(client)
    fake_repo.set_status(
        job_id, "complete",
        results=[{"chunk_id": "c1", "post_id": 1, "text": "hi", "metadata": None, "score": 0.9}],
    )
    resp = client.get(f"/ui/search/{job_id}")
    assert "badge-complete" in resp.text


def test_results_page_has_back_link(client):
    job_id = _create_job(client)
    resp = client.get(f"/ui/search/{job_id}")
    assert "/ui/search" in resp.text
    assert "Back" in resp.text


def test_results_page_returns_404_for_missing_job(client):
    resp = client.get("/ui/search/does-not-exist")
    assert resp.status_code == 404


def test_results_page_shows_score(client, fake_repo):
    job_id = _create_job(client)
    fake_repo.set_status(
        job_id, "complete",
        results=[{"chunk_id": "c1", "post_id": 1, "text": "test", "metadata": None, "score": 0.9123}],
    )
    resp = client.get(f"/ui/search/{job_id}")
    assert "0.9123" in resp.text
