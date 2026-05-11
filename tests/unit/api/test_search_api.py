"""Unit tests for POST /search and GET /search/{job_id} endpoints.

Uses a minimal FastAPI app with fakes — no real Postgres, no RabbitMQ,
no lifespan.  Matches the pattern used by tests/unit/api/test_sync.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from event_driven_rag_service.api.search import router as search_router
from tests.utils.factories import FakeEventBus


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSearchJobRepo:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self._jobs: dict[str, dict] = {}

    async def ensure_table(self) -> None:
        pass

    async def create_job(
        self, query: str, k: int, embedding_profile: str, chunks_table: str, library_id: str = ""
    ) -> str:
        job_id = f"fake-job-{len(self.created) + 1}"
        job = {
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
            "created_at": None,
            "completed_at": None,
        }
        self.created.append(job)
        self._jobs[job_id] = job
        return job_id

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def app_with_fakes():
    test_app = FastAPI()
    test_app.include_router(search_router)
    test_app.state.search_job_repo = FakeSearchJobRepo()
    test_app.state.event_bus = FakeEventBus()
    return test_app


@pytest.fixture
def client(app_with_fakes):
    with TestClient(app_with_fakes) as c:
        yield c


# ---------------------------------------------------------------------------
# POST /search tests
# ---------------------------------------------------------------------------

def test_create_search_returns_202_with_job_id(client):
    """POST /search should create a job and return job_id with 202 Accepted."""
    response = client.post(
        "/search/",
        json={
            "query": "What is machine learning?",
            "chunk_type": "body",
            "k": 5,
            "library_id": "main",
        },
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert "job_id" in body
    assert body["job_id"] == "fake-job-1"


def test_create_search_emits_search_job_created_event(client):
    """POST /search should publish a search_job.created event with query and job_id."""
    client.post(
        "/search/",
        json={
            "query": "test query",
            "chunk_type": "body",
            "k": 10,
            "library_id": "main",
        },
    )
    bus: FakeEventBus = client.app.state.event_bus
    events = bus.drain_topic("search_job.created")
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == "search_job.created"
    assert ev["query"] == "test query"
    assert ev["query_job_id"] == "fake-job-1"


def test_create_search_stores_correct_job_params(client):
    """POST /search should persist query, k, chunk_table, and embedding_profile."""
    client.post(
        "/search/",
        json={
            "query": "deep learning",
            "chunk_type": "body",
            "k": 7,
            "library_id": "work",
        },
    )
    repo: FakeSearchJobRepo = client.app.state.search_job_repo
    assert len(repo.created) == 1
    job = repo.created[0]
    assert job["query"] == "deep learning"
    assert job["k"] == 7
    assert job["chunks_table"] == "posts_work_chunks_body_baai_bge_base_en_v1_5"
    assert job["embedding_profile"] == "BAAI/bge-base-en-v1.5"


def test_create_search_invalid_chunk_type_returns_422(client):
    """POST /search with an unknown chunk_type should return 422."""
    response = client.post(
        "/search/",
        json={
            "query": "test",
            "chunk_type": "nonexistent_type",
            "k": 5,
            "library_id": "main",
        },
    )
    assert response.status_code == 422


def test_create_search_invalid_library_id_returns_422(client):
    """POST /search with an invalid library_id should return 422."""
    response = client.post(
        "/search/",
        json={
            "query": "test",
            "chunk_type": "body",
            "k": 5,
            "library_id": "INVALID-ID",
        },
    )
    assert response.status_code == 422


def test_create_search_k_zero_returns_422(client):
    """POST /search with k=0 should return 422."""
    response = client.post(
        "/search/",
        json={"query": "test", "chunk_type": "body", "k": 0, "library_id": "main"},
    )
    assert response.status_code == 422


def test_create_search_k_over_max_returns_422(client):
    """POST /search with k>100 should return 422."""
    response = client.post(
        "/search/",
        json={"query": "test", "chunk_type": "body", "k": 101, "library_id": "main"},
    )
    assert response.status_code == 422


def test_create_search_title_chunk_type_uses_correct_table(client):
    """POST /search with chunk_type=title should route to title chunk table."""
    client.post(
        "/search/",
        json={"query": "test", "chunk_type": "title", "k": 5, "library_id": "work"},
    )
    repo: FakeSearchJobRepo = client.app.state.search_job_repo
    job = repo.created[0]
    assert job["chunks_table"] == "posts_work_chunks_title_baai_bge_small_en_v1_5"
    assert job["embedding_profile"] == "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# GET /search/{job_id} tests
# ---------------------------------------------------------------------------

def test_get_search_result_embedding_status(client):
    """GET /search/{job_id} should return status=embedding while pipeline runs."""
    post_resp = client.post(
        "/search/",
        json={"query": "test", "chunk_type": "body", "k": 5, "library_id": "main"},
    )
    job_id = post_resp.json()["job_id"]

    get_resp = client.get(f"/search/{job_id}")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["job_id"] == job_id
    assert body["status"] == "embedding"
    assert body["results"] is None


def test_get_search_result_complete_with_results(client):
    """GET /search/{job_id} should return results list when status is complete."""
    post_resp = client.post(
        "/search/",
        json={"query": "test", "chunk_type": "body", "k": 5, "library_id": "main"},
    )
    job_id = post_resp.json()["job_id"]

    repo: FakeSearchJobRepo = client.app.state.search_job_repo
    repo._jobs[job_id]["status"] = "complete"
    repo._jobs[job_id]["results"] = [
        {"chunk_id": "abc", "post_id": 1, "text": "chunk text", "metadata": None, "score": 0.88}
    ]

    get_resp = client.get(f"/search/{job_id}")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["status"] == "complete"
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["chunk_id"] == "abc"
    assert result["post_id"] == 1
    assert result["score"] == 0.88


def test_get_search_result_failed_with_error(client):
    """GET /search/{job_id} should return error when status is failed."""
    post_resp = client.post(
        "/search/",
        json={"query": "test", "chunk_type": "body", "k": 5, "library_id": "main"},
    )
    job_id = post_resp.json()["job_id"]

    repo: FakeSearchJobRepo = client.app.state.search_job_repo
    repo._jobs[job_id]["status"] = "failed"
    repo._jobs[job_id]["error"] = "pgvector table not found"

    get_resp = client.get(f"/search/{job_id}")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["status"] == "failed"
    assert body["error"] == "pgvector table not found"


def test_get_search_result_not_found_returns_404(client):
    """GET /search/{job_id} with unknown job_id should return 404."""
    response = client.get("/search/nonexistent-job-id")
    assert response.status_code == 404
