"""Unit tests for POST /search and GET /search/{job_id} endpoints.

Uses a minimal FastAPI app with fakes — no real Postgres, no RabbitMQ,
no lifespan.  Matches the pattern used by tests/unit/api/test_sync.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from event_driven_rag_service.api.search import router as search_router
from event_driven_rag_service.infrastructure.rerank_probe import Availability
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
        self.created: list[dict] = []
        self._jobs: dict[str, dict] = {}

    async def ensure_table(self) -> None:
        pass

    async def create_job(
        self,
        query: str,
        k: int,
        embedding_profile: str,
        chunks_table: str,
        library_id: str = "",
        rerank: bool = False,
        rerank_profile: Optional[str] = None,
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
            "rerank": rerank,
            "rerank_profile": rerank_profile,
            "embedding": None,
            "results": None,
            "error": None,
            "error_code": None,
            "created_at": None,
            "completed_at": None,
        }
        self.created.append(job)
        self._jobs[job_id] = job
        return job_id

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)


class FakeRerankProbe:
    """Answers the remote-reranker availability check POST /search makes up front."""

    def __init__(self, ok: bool = True, reason: str = "The GPU PC is not reachable right now.") -> None:
        self.availability = Availability(ok, "" if ok else reason)
        self.checked: list[str] = []

    async def check(self, cfg) -> Availability:
        self.checked.append(cfg.primary)
        return self.availability


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def app_with_fakes():
    test_app = FastAPI()
    test_app.include_router(search_router)
    test_app.state.search_job_repo = FakeSearchJobRepo()
    test_app.state.event_bus = FakeEventBus()
    test_app.state.pool = MagicMock()
    test_app.state.rerank_probe = FakeRerankProbe()
    return test_app


@pytest.fixture
def client(app_with_fakes):
    with TestClient(app_with_fakes) as c:
        yield c


@pytest.fixture(autouse=True)
def fake_chunk_repo():
    """By default, the chunk table always exists — most tests aren't about validation."""
    fake = FakeChunkRepo(exists=True)
    with patch("event_driven_rag_service.api.search.ChunkRepository", return_value=fake):
        yield fake


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
    """POST /search with k>999 should return 422."""
    response = client.post(
        "/search/",
        json={"query": "test", "chunk_type": "body", "k": 1000, "library_id": "main"},
    )
    assert response.status_code == 422


def test_create_search_k_defaults_to_99_and_allows_999(client):
    client.post("/search/", json={"query": "test", "chunk_type": "body", "library_id": "main"})
    r = client.post(
        "/search/",
        json={"query": "test", "chunk_type": "body", "k": 999, "library_id": "main"},
    )
    assert r.status_code == 202
    created = client.app.state.search_job_repo.created
    assert [j["k"] for j in created] == [99, 999]


def test_create_search_missing_chunk_table_returns_404(client, fake_chunk_repo):
    """POST /search should 404 up front when the chunk table was never created,
    instead of creating a job that will fail later in the async pipeline."""
    fake_chunk_repo.exists = False
    response = client.post(
        "/search/",
        json={"query": "test", "chunk_type": "body", "k": 5, "library_id": "neversynced"},
    )
    assert response.status_code == 404
    assert "neversynced" in response.json()["detail"]


def test_create_search_missing_chunk_table_does_not_create_job(client, fake_chunk_repo):
    """A validation failure must not create a job or publish an event."""
    fake_chunk_repo.exists = False
    client.post(
        "/search/",
        json={"query": "test", "chunk_type": "body", "k": 5, "library_id": "neversynced"},
    )
    repo: FakeSearchJobRepo = client.app.state.search_job_repo
    assert repo.created == []
    bus: FakeEventBus = client.app.state.event_bus
    assert bus.drain_topic("search_job.created") == []


def test_create_search_checks_correct_table_name(client, fake_chunk_repo):
    """The validation check must use the same table name the job will be created with."""
    client.post(
        "/search/",
        json={"query": "test", "chunk_type": "title", "k": 5, "library_id": "work"},
    )
    assert fake_chunk_repo.checked_tables == ["posts_work_chunks_title_baai_bge_small_en_v1_5"]


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
# rerank option
# ---------------------------------------------------------------------------

RERANK_BODY = {"query": "test", "chunk_type": "body", "k": 5, "library_id": "main", "rerank": True}


def test_search_without_rerank_never_probes_the_remote(client):
    client.post("/search/", json={**RERANK_BODY, "rerank": False})
    assert client.app.state.rerank_probe.checked == []
    assert client.app.state.search_job_repo.created[0]["rerank"] is False


def test_rerank_search_when_remote_available_creates_a_reranking_job(client):
    response = client.post("/search/", json=RERANK_BODY)

    assert response.status_code == 202, response.text
    job = client.app.state.search_job_repo.created[0]
    assert job["rerank"] is True
    assert job["rerank_profile"] == "default"
    [event] = client.app.state.event_bus.drain_topic("search_job.created")
    assert event["rerank"] is True
    assert event["rerank_profile"] == "default"
    assert client.app.state.rerank_probe.checked == ["Qwen/Qwen3-Reranker-4B"]


def test_rerank_search_honours_a_named_profile(client):
    client.post("/search/", json={**RERANK_BODY, "rerank_profile": "qwen3-reranker-0.6b"})
    assert client.app.state.search_job_repo.created[0]["rerank_profile"] == "qwen3-reranker-0.6b"
    assert client.app.state.rerank_probe.checked == ["Qwen/Qwen3-Reranker-0.6B"]


def test_rerank_search_when_remote_unavailable_returns_503_with_code(client):
    client.app.state.rerank_probe = FakeRerankProbe(ok=False)

    response = client.post("/search/", json=RERANK_BODY)

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "rerank_unavailable"
    assert detail["message"] == "The GPU PC is not reachable right now."
    assert detail["fallback"] == "semantic"


def test_unavailable_rerank_creates_no_job_and_no_event(client):
    client.app.state.rerank_probe = FakeRerankProbe(ok=False)
    client.post("/search/", json=RERANK_BODY)

    assert client.app.state.search_job_repo.created == []
    assert client.app.state.event_bus.drain_topic("search_job.created") == []


def test_unknown_rerank_profile_returns_422(client):
    response = client.post("/search/", json={**RERANK_BODY, "rerank_profile": "nope"})
    assert response.status_code == 422
    assert client.app.state.search_job_repo.created == []


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


def test_get_search_result_exposes_error_code(client):
    """A job that failed mid-way because the reranker went away carries the same
    code as the up-front 503, so the client handles both alike."""
    job_id = client.post(
        "/search/",
        json={"query": "test", "chunk_type": "body", "k": 5, "library_id": "main"},
    ).json()["job_id"]
    job = client.app.state.search_job_repo._jobs[job_id]
    job["status"] = "failed"
    job["error"] = "The reranker is not available right now."
    job["error_code"] = "rerank_unavailable"

    body = client.get(f"/search/{job_id}").json()
    assert body["status"] == "failed"
    assert body["error_code"] == "rerank_unavailable"


def test_get_search_result_not_found_returns_404(client):
    """GET /search/{job_id} with unknown job_id should return 404."""
    response = client.get("/search/nonexistent-job-id")
    assert response.status_code == 404


def test_get_search_result_parses_json_encoded_results_string(client):
    """Some repository implementations return results as a raw JSON string
    (e.g. from a jsonb column read without decoding) — the route must parse it."""
    post_resp = client.post(
        "/search/",
        json={"query": "test", "chunk_type": "body", "k": 5, "library_id": "main"},
    )
    job_id = post_resp.json()["job_id"]

    repo: FakeSearchJobRepo = client.app.state.search_job_repo
    repo._jobs[job_id]["status"] = "complete"
    repo._jobs[job_id]["results"] = (
        '[{"chunk_id": "abc", "post_id": 1, "text": "chunk text", "metadata": null, "score": 0.5}]'
    )

    get_resp = client.get(f"/search/{job_id}")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["status"] == "complete"
    assert len(body["results"]) == 1
    assert body["results"][0]["chunk_id"] == "abc"
