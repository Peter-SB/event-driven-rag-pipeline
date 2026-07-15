"""Integration tests for the HTMX search UI against a real Postgres testcontainer.

Verifies that:
- submit_search creates a real search_jobs row
- job_status reflects live DB state
- poll_results swaps to result cards when job is completed in DB

Uses the postgres_pool fixture from integration/conftest.py.
Run with: pytest tests/integration/test_ui_integration.py -m integration
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from event_driven_rag_service.api.ui import router as ui_router
from event_driven_rag_service.repository.search_job_repository import SearchJobRepository
from tests.utils.factories import FakeEventBus

pytestmark = pytest.mark.integration

_EMBEDDING_PROFILE = "BAAI/bge-base-en-v1.5"
_CHUNKS_TABLE = "posts_main_chunks_body_baai_bge_base_en_v1_5"


# ---------------------------------------------------------------------------
# Fixture: real FastAPI app with real SearchJobRepository + fake event bus
# ---------------------------------------------------------------------------

@pytest.fixture
async def ui_client(postgres_pool):
    """TestClient wired to a real SearchJobRepository and an in-memory event bus."""
    repo = SearchJobRepository(postgres_pool)
    await repo.ensure_table()

    # Clean slate for each test
    async with postgres_pool.acquire() as conn:
        await conn.execute("TRUNCATE search_jobs")

    app = FastAPI()
    app.include_router(ui_router)
    app.state.search_job_repo = repo
    app.state.event_bus = FakeEventBus()

    with TestClient(app) as client:
        yield client, repo

    async with postgres_pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS search_jobs")


# ---------------------------------------------------------------------------
# Submit creates a real DB row
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_persists_job_to_postgres(ui_client):
    client, repo = ui_client

    resp = client.post(
        "/ui/search/submit",
        data={"query": "what is pgvector", "library_id": "main", "chunk_type": "body", "k": "5"},
    )
    assert resp.status_code == 200
    assert "<tr" in resp.text

    # Verify the row exists in real Postgres
    import re
    m = re.search(r'id="job-([^"]+)"', resp.text)
    assert m, "No job id found in response HTML"
    job_id = m.group(1)

    job = await repo.get_job(job_id)
    assert job is not None
    assert job["query"] == "what is pgvector"
    assert job["k"] == 5
    assert job["status"] == "embedding"
    assert job["library_id"] == "main"


@pytest.mark.asyncio
async def test_submit_emits_search_job_created_event(ui_client):
    client, repo = ui_client

    client.post(
        "/ui/search/submit",
        data={"query": "event emission check", "library_id": "lib2", "chunk_type": "body", "k": "3"},
    )
    bus: FakeEventBus = client.app.state.event_bus
    events = bus.drain_topic("search_job.created")
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == "search_job.created"
    assert ev["query"] == "event emission check"


# ---------------------------------------------------------------------------
# Status polling reflects live DB state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status_updates_when_job_transitions_to_searching(ui_client):
    client, repo = ui_client

    # Create job via UI
    post_resp = client.post(
        "/ui/search/submit",
        data={"query": "transition test", "library_id": "main", "chunk_type": "body", "k": "10"},
    )
    import re
    job_id = re.search(r'id="job-([^"]+)"', post_resp.text).group(1)

    # Transition in real DB
    await repo.mark_searching(job_id)

    # Status endpoint should reflect new state
    resp = client.get(f"/ui/search/{job_id}/status")
    assert resp.status_code == 200
    assert "searching" in resp.text


@pytest.mark.asyncio
async def test_status_reflects_complete_state_from_db(ui_client):
    client, repo = ui_client

    post_resp = client.post(
        "/ui/search/submit",
        data={"query": "completion test", "library_id": "main", "chunk_type": "body", "k": "10"},
    )
    import re
    job_id = re.search(r'id="job-([^"]+)"', post_resp.text).group(1)

    results = [
        {"chunk_id": "c1", "post_id": 1, "text": "result text", "metadata": None, "score": 0.92}
    ]
    await repo.complete_job(job_id, results)

    resp = client.get(f"/ui/search/{job_id}/status")
    assert "complete" in resp.text
    assert "badge-complete" in resp.text
    assert "1" in resp.text  # result count


# ---------------------------------------------------------------------------
# Poll endpoint reflects job lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_poll_shows_spinner_while_embedding(ui_client):
    client, repo = ui_client

    post_resp = client.post(
        "/ui/search/submit",
        data={"query": "poll spinner test", "library_id": "main", "chunk_type": "body", "k": "5"},
    )
    import re
    job_id = re.search(r'id="job-([^"]+)"', post_resp.text).group(1)

    resp = client.get(f"/ui/search/{job_id}/poll")
    assert resp.status_code == 200
    assert "Processing" in resp.text or "spinner" in resp.text
    # Must continue polling (job still pending)
    assert "hx-trigger" in resp.text


@pytest.mark.asyncio
async def test_poll_shows_result_cards_when_complete(ui_client):
    client, repo = ui_client

    post_resp = client.post(
        "/ui/search/submit",
        data={"query": "poll results test", "library_id": "main", "chunk_type": "body", "k": "5"},
    )
    import re
    job_id = re.search(r'id="job-([^"]+)"', post_resp.text).group(1)

    results = [
        {"chunk_id": "abc", "post_id": 99, "text": "the answer is 42", "metadata": None, "score": 0.99}
    ]
    await repo.complete_job(job_id, results)

    resp = client.get(f"/ui/search/{job_id}/poll")
    assert resp.status_code == 200
    assert "result-card" in resp.text
    assert "the answer is 42" in resp.text
    # Must not continue polling
    assert "hx-trigger" not in resp.text


@pytest.mark.asyncio
async def test_poll_shows_error_when_failed(ui_client):
    client, repo = ui_client

    post_resp = client.post(
        "/ui/search/submit",
        data={"query": "fail test", "library_id": "main", "chunk_type": "body", "k": "5"},
    )
    import re
    job_id = re.search(r'id="job-([^"]+)"', post_resp.text).group(1)

    await repo.fail_job(job_id, "embedding model unavailable")

    resp = client.get(f"/ui/search/{job_id}/poll")
    assert resp.status_code == 200
    assert "embedding model unavailable" in resp.text
    assert "hx-trigger" not in resp.text


# ---------------------------------------------------------------------------
# Full results page
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_results_page_renders_complete_job(ui_client):
    client, repo = ui_client

    post_resp = client.post(
        "/ui/search/submit",
        data={"query": "full page test", "library_id": "main", "chunk_type": "body", "k": "5"},
    )
    import re
    job_id = re.search(r'id="job-([^"]+)"', post_resp.text).group(1)

    results = [
        {
            "chunk_id": "r1",
            "post_id": 5,
            "text": "result from postgres",
            "metadata": {"title": "Postgres Result", "external_id": "ext5"},
            "score": 0.88,
        }
    ]
    await repo.complete_job(job_id, results)

    resp = client.get(f"/ui/search/{job_id}")
    assert resp.status_code == 200
    assert "full page test" in resp.text
    assert "Postgres Result" in resp.text
    assert "result from postgres" in resp.text
    assert "0.8800" in resp.text
