"""Unit tests for POST /search/similar endpoint.

Uses a minimal FastAPI app with a patched ChunkRepository — no real Postgres.
Matches the pattern used by tests/unit/api/test_search_api.py.

Tested behaviours
-----------------
- 422 on bad library_id / chunk_type / k values
- 404 when no embeddings exist for the source post
- 200 with SimilarResponse schema on success
- Body chunk type: all source-post chunks are averaged into the query vector
- Title / summary_title: single chunk embedding used directly (no averaging)
- Source post is excluded from results via exclude_post_id
- chunks_table is derived correctly for each chunk type
"""
from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from event_driven_rag_service.api.search import router as search_router
from event_driven_rag_service.exceptions import ChunkTableNotFoundError


# ---------------------------------------------------------------------------
# Fake ChunkRepository — records calls, returns configurable data
# ---------------------------------------------------------------------------

class FakeChunkRepo:
    """Records calls to get_post_embeddings and search_nearest for assertion."""

    def __init__(
        self,
        embeddings: list[list[float]] | None = None,
        search_results: list[dict[str, Any]] | None = None,
    ) -> None:
        self.embeddings = embeddings or []
        self.search_results = search_results or []
        # Call records
        self.get_embeddings_calls: list[tuple[int, str]] = []
        self.search_nearest_calls: list[tuple[str, list[float], int, int | None]] = []

    async def get_post_embeddings(self, post_id: int, table_name: str) -> list[list[float]]:
        self.get_embeddings_calls.append((post_id, table_name))
        return self.embeddings

    async def search_nearest(
        self,
        table_name: str,
        query_vector: list[float],
        k: int,
        exclude_post_id: int | None = None,
    ) -> list[dict[str, Any]]:
        self.search_nearest_calls.append((table_name, query_vector, k, exclude_post_id))
        return self.search_results


def _make_result(post_id: int = 2, score: float = 0.9) -> dict[str, Any]:
    return {"id": "chunk-abc", "post_id": post_id, "text": "some text", "metadata": {}, "score": score}


# ---------------------------------------------------------------------------
# App fixture — pool is a MagicMock because ChunkRepository is patched out
# ---------------------------------------------------------------------------

@pytest.fixture
def app_with_fakes():
    test_app = FastAPI()
    test_app.include_router(search_router)
    test_app.state.pool = MagicMock()
    return test_app


@pytest.fixture
def client(app_with_fakes):
    with TestClient(app_with_fakes) as c:
        yield c


def _similar_post(client, fake_repo, **overrides):
    """POST /search/similar with a patched ChunkRepository."""
    payload = {
        "post_id": 1,
        "chunk_type": "body",
        "library_id": "main",
        "k": 5,
        **overrides,
    }
    with patch("event_driven_rag_service.api.search.ChunkRepository", return_value=fake_repo):
        return client.post("/search/similar", json=payload)


# ---------------------------------------------------------------------------
# Validation (422)
# ---------------------------------------------------------------------------

def test_invalid_library_id_returns_422(client):
    """library_id must match ^[a-z][a-z0-9_]*$."""
    fake = FakeChunkRepo(embeddings=[[0.1]])
    resp = _similar_post(client, fake, library_id="INVALID-ID")
    assert resp.status_code == 422


def test_invalid_chunk_type_returns_422(client):
    """chunk_type must be in EMBED_CONFIGS."""
    fake = FakeChunkRepo(embeddings=[[0.1]])
    resp = _similar_post(client, fake, chunk_type="nonexistent")
    assert resp.status_code == 422


def test_k_zero_returns_422(client):
    fake = FakeChunkRepo(embeddings=[[0.1]])
    resp = _similar_post(client, fake, k=0)
    assert resp.status_code == 422


def test_k_over_max_returns_422(client):
    fake = FakeChunkRepo(embeddings=[[0.1]])
    resp = _similar_post(client, fake, k=101)
    assert resp.status_code == 422


def test_missing_post_id_returns_422(client):
    fake = FakeChunkRepo(embeddings=[[0.1]])
    with patch("event_driven_rag_service.api.search.ChunkRepository", return_value=fake):
        resp = client.post("/search/similar", json={"chunk_type": "body", "library_id": "main"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 404 when post has no embeddings
# ---------------------------------------------------------------------------

def test_no_embeddings_returns_404(client):
    """404 when get_post_embeddings returns empty list."""
    fake = FakeChunkRepo(embeddings=[])
    resp = _similar_post(client, fake, post_id=99)
    assert resp.status_code == 404
    assert "99" in resp.json()["detail"]


class _MissingTableRepo:
    """Simulates ChunkRepository hitting a chunk table that was never created.

    ChunkRepository.get_post_embeddings translates asyncpg.UndefinedTableError
    into ChunkTableNotFoundError — this fake models that contract directly.
    """

    async def get_post_embeddings(self, post_id: int, table_name: str) -> list[list[float]]:
        raise ChunkTableNotFoundError(table_name)

    async def search_nearest(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("search_nearest should not be called when the chunk table is missing")


def test_missing_chunk_table_returns_404_not_500(client):
    """A library/chunk_type that was never synced should return 404, not an unhandled 500."""
    resp = _similar_post(client, _MissingTableRepo(), library_id="neversynced")
    assert resp.status_code == 404
    assert "neversynced" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 200 response schema
# ---------------------------------------------------------------------------

def test_returns_200_with_similar_response_schema(client):
    """Successful request should return SimilarResponse schema."""
    fake = FakeChunkRepo(
        embeddings=[[1.0, 0.0, 0.0]],
        search_results=[_make_result(post_id=2, score=0.95)],
    )
    resp = _similar_post(client, fake)
    assert resp.status_code == 200
    body = resp.json()
    assert body["post_id"] == 1
    assert body["chunk_type"] == "body"
    assert body["chunks_averaged"] == 1
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["chunk_id"] == "chunk-abc"
    assert result["post_id"] == 2
    assert result["score"] == 0.95


def test_returns_empty_results_list_when_no_neighbours(client):
    """If search returns no results, results should be [] not null."""
    fake = FakeChunkRepo(embeddings=[[1.0, 0.0]], search_results=[])
    resp = _similar_post(client, fake)
    assert resp.status_code == 200
    assert resp.json()["results"] == []


# ---------------------------------------------------------------------------
# Body chunk type: multiple source chunks are averaged
# ---------------------------------------------------------------------------

def test_body_averages_all_source_chunks(client):
    """Body chunk type must average ALL source-post embeddings into one query vector."""
    # Three orthogonal unit vectors along axes 0, 1, 2 (dim=3 for simplicity)
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0]
    emb_c = [0.0, 0.0, 1.0]
    fake = FakeChunkRepo(
        embeddings=[emb_a, emb_b, emb_c],
        search_results=[_make_result()],
    )
    _similar_post(client, fake, chunk_type="body")

    assert len(fake.search_nearest_calls) == 1
    _, query_vector, _, _ = fake.search_nearest_calls[0]

    expected = [1 / 3, 1 / 3, 1 / 3]
    assert len(query_vector) == 3
    for got, exp in zip(query_vector, expected):
        assert abs(got - exp) < 1e-9, f"query_vector mismatch: {query_vector} != {expected}"


def test_body_chunks_averaged_count_reflected_in_response(client):
    """chunks_averaged in response equals the number of source-post chunks."""
    fake = FakeChunkRepo(
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        search_results=[],
    )
    resp = _similar_post(client, fake, chunk_type="body")
    assert resp.json()["chunks_averaged"] == 3


# ---------------------------------------------------------------------------
# Title / summary_title: single chunk used directly (no aggregation)
# ---------------------------------------------------------------------------

def test_title_uses_single_chunk_as_query(client):
    """Title has one chunk — query vector equals that embedding exactly."""
    emb = [0.6, 0.8, 0.0]
    fake = FakeChunkRepo(embeddings=[emb], search_results=[_make_result()])
    _similar_post(client, fake, chunk_type="title")

    _, query_vector, _, _ = fake.search_nearest_calls[0]
    assert query_vector == emb


def test_summary_title_uses_single_chunk_as_query(client):
    """summary_title has one chunk — query vector equals that embedding exactly."""
    emb = [0.0, 1.0, 0.0]
    fake = FakeChunkRepo(embeddings=[emb], search_results=[_make_result()])
    _similar_post(client, fake, chunk_type="summary_title")

    _, query_vector, _, _ = fake.search_nearest_calls[0]
    assert query_vector == emb


def test_title_chunks_averaged_is_one(client):
    """Title always reports chunks_averaged=1 and uses only the first embedding."""
    fake = FakeChunkRepo(embeddings=[[1.0, 0.0]], search_results=[])
    resp = _similar_post(client, fake, chunk_type="title")
    assert resp.json()["chunks_averaged"] == 1


def test_title_uses_first_embedding_when_multiple_stored(client):
    """For non-body types, only the first stored embedding is used as the query vector."""
    emb_first = [1.0, 0.0, 0.0]
    emb_second = [0.0, 1.0, 0.0]
    fake = FakeChunkRepo(embeddings=[emb_first, emb_second], search_results=[_make_result()])
    _similar_post(client, fake, chunk_type="title")

    _, query_vector, _, _ = fake.search_nearest_calls[0]
    assert query_vector == emb_first


# ---------------------------------------------------------------------------
# Source post excluded from results
# ---------------------------------------------------------------------------

def test_source_post_excluded_from_search(client):
    """search_nearest must be called with exclude_post_id = request post_id."""
    fake = FakeChunkRepo(embeddings=[[1.0, 0.0]], search_results=[])
    _similar_post(client, fake, post_id=42)

    _, _, _, exclude_post_id = fake.search_nearest_calls[0]
    assert exclude_post_id == 42


# ---------------------------------------------------------------------------
# Table name derived correctly for each chunk type
# ---------------------------------------------------------------------------

def test_body_uses_correct_chunk_table(client):
    """Body chunk type should derive table posts_main_chunks_body_baai_bge_base_en_v1_5."""
    fake = FakeChunkRepo(embeddings=[[1.0]], search_results=[])
    _similar_post(client, fake, chunk_type="body", library_id="main")

    post_id, table_name = fake.get_embeddings_calls[0]
    assert table_name == "posts_main_chunks_body_baai_bge_base_en_v1_5"


def test_title_uses_correct_chunk_table(client):
    """Title chunk type should derive table posts_work_chunks_title_baai_bge_small_en_v1_5."""
    fake = FakeChunkRepo(embeddings=[[1.0]], search_results=[])
    _similar_post(client, fake, chunk_type="title", library_id="work")

    _, table_name = fake.get_embeddings_calls[0]
    assert table_name == "posts_work_chunks_title_baai_bge_small_en_v1_5"


def test_summary_title_uses_correct_chunk_table(client):
    """summary_title should derive table posts_lib_chunks_summary_title_qwen3_embedding_0_6b_q8_0_gguf."""
    fake = FakeChunkRepo(embeddings=[[1.0]], search_results=[])
    _similar_post(client, fake, chunk_type="summary_title", library_id="lib")

    _, table_name = fake.get_embeddings_calls[0]
    assert table_name == "posts_lib_chunks_summary_title_qwen3_embedding_0_6b_q8_0_gguf"
