"""Unit tests for /recommend/* and the /ui/recommend pages — repositories patched out."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from event_driven_rag_service.api import recommend as recommend_api
from event_driven_rag_service.api import ui as ui_api
from event_driven_rag_service.exceptions import ChunkTableNotFoundError
from event_driven_rag_service.services.playlist_recommendation import (
    Because,
    EmptyPlaylistError,
    Recommendation,
    RecommendResult,
    Theme,
)


def make_result() -> RecommendResult:
    b = Because(1, "playlist story", 0.8)
    return RecommendResult(
        playlist_size=3,
        embedded=3,
        missing_post_ids=[],
        silhouette=None,
        candidates_considered=40,
        themes=[Theme(0, 3, [b])],
        items=[Recommendation(1, 77, "A <b>title</b>", "sub", None, 0.77, 0, [b])],
    )


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(recommend_api.router)
    app.include_router(ui_api.router)
    app.state.pool = MagicMock()
    run = AsyncMock(return_value=("folder:1", make_result()))
    monkeypatch.setattr(recommend_api, "run_recommendation", run)
    monkeypatch.setattr(ui_api, "run_recommendation", run)
    feedback = MagicMock()
    feedback.set_verdict = AsyncMock()
    monkeypatch.setattr(recommend_api, "FeedbackRepository", lambda pool: feedback)
    monkeypatch.setattr(ui_api, "FeedbackRepository", lambda pool: feedback)
    c = TestClient(app)
    c.run, c.feedback = run, feedback
    return c


BODY = {"library_id": "main", "folder_id": 1}


def test_playlist_returns_ranked_items(client):
    r = client.post("/recommend/playlist", json={**BODY, "k": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["playlist_key"] == "folder:1"
    assert data["items"][0]["post_id"] == 77 and data["items"][0]["because"][0]["post_id"] == 1
    assert client.run.await_args.kwargs["params"].k == 5


@pytest.mark.parametrize("body", [
    {"library_id": "main"},                                       # no source
    {"library_id": "main", "folder_id": 1, "post_ids": [1]},      # two sources
    {"library_id": "Bad Lib", "folder_id": 1},                    # bad library
    {"library_id": "main", "folder_id": 1, "chunk_type": "nope"}, # bad chunk type
    {"library_id": "main", "folder_id": 1, "k": 0},
    {"library_id": "main", "folder_id": 1, "diversity": 1.5},
])
def test_invalid_requests_are_422(client, body):
    assert client.post("/recommend/playlist", json=body).status_code == 422


def test_missing_table_is_404_and_empty_playlist_is_422(client):
    client.run.side_effect = ChunkTableNotFoundError("t")
    assert client.post("/recommend/playlist", json=BODY).status_code == 404
    client.run.side_effect = EmptyPlaylistError("nothing embedded")
    r = client.post("/recommend/playlist", json=BODY)
    assert r.status_code == 422 and "nothing embedded" in r.json()["detail"]


def test_playlist_key_is_stable_for_the_same_ids_in_any_order():
    k = recommend_api.make_playlist_key
    assert k(3, None) == "folder:3"
    assert k(None, [3, 1, 2]) == k(None, [1, 2, 3, 3]) != k(None, [1, 2])


def test_feedback_endpoint_records_verdict(client):
    r = client.post("/recommend/feedback", json={
        "library_id": "main", "playlist_key": "folder:1", "post_id": 77, "verdict": "no"})
    assert r.status_code == 204
    client.feedback.set_verdict.assert_awaited_once_with("main", "folder:1", 77, "no")
    assert client.post("/recommend/feedback", json={
        "library_id": "main", "playlist_key": "folder:1", "post_id": 77, "verdict": "maybe"}).status_code == 422


def test_ui_page_and_results_escape_titles_and_show_feedback_buttons(client):
    assert client.get("/ui/recommend").status_code == 200
    r = client.post("/ui/recommend/submit", data={"library_id": "main", "folder_id": "1"})
    assert r.status_code == 200
    assert "A &lt;b&gt;title&lt;/b&gt;" in r.text and "<b>title</b>" not in r.text
    assert 'value="yes"' in r.text and 'value="no"' in r.text
    assert client.run.await_args.kwargs["folder_id"] == 1


def test_ui_needs_a_source_and_explicit_ids_override_folder(client):
    assert client.post("/ui/recommend/submit", data={"library_id": "main"}).status_code == 422
    client.post("/ui/recommend/submit", data={"library_id": "main", "folder_id": "1", "post_ids": "5, 9 12"})
    kw = client.run.await_args.kwargs
    assert kw["post_ids"] == [5, 9, 12] and kw["folder_id"] is None


def test_ui_feedback_returns_card_state_and_survives_a_read_only_db(client):
    form = {"library_id": "main", "playlist_key": "folder:1", "post_id": "77", "title": "T"}
    ok = client.post("/ui/recommend/feedback", data={**form, "verdict": "yes"})
    assert ok.status_code == 200 and "positives" in ok.text
    client.feedback.set_verdict.side_effect = RuntimeError("cannot execute CREATE TABLE in a read-only transaction")
    bad = client.post("/ui/recommend/feedback", data={**form, "verdict": "no"})
    assert bad.status_code == 200 and "Could not save feedback" in bad.text
