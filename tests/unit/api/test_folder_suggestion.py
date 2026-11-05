"""Unit tests for /folder-suggestions/* and the /ui/folder-suggestions pages — repositories patched out."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from event_driven_rag_service.api import folder_suggestion as suggestion_api
from event_driven_rag_service.api import ui as ui_api
from event_driven_rag_service.exceptions import ChunkTableNotFoundError
from event_driven_rag_service.services.folder_suggestion import (
    Because,
    EmptyFolderError,
    Suggestion,
    SuggestResult,
    Theme,
)


def make_result() -> SuggestResult:
    b = Because(1, "folder story", 0.8)
    return SuggestResult(
        folder_size=3,
        embedded=3,
        missing_post_ids=[],
        silhouette=None,
        candidates_considered=40,
        themes=[Theme(0, 3, [b])],
        items=[Suggestion(1, 77, "A <b>title</b>", "sub", None, 0.77, 0, [b])],
    )


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(suggestion_api.router)
    app.include_router(ui_api.router)
    app.state.pool = MagicMock()
    run = AsyncMock(return_value=("folder:1", make_result()))
    monkeypatch.setattr(suggestion_api, "run_suggestion", run)
    monkeypatch.setattr(ui_api, "run_suggestion", run)
    feedback = MagicMock()
    feedback.set_verdict = AsyncMock()
    monkeypatch.setattr(suggestion_api, "FeedbackRepository", lambda pool: feedback)
    monkeypatch.setattr(ui_api, "FeedbackRepository", lambda pool: feedback)
    c = TestClient(app)
    c.run, c.feedback = run, feedback
    return c


BODY = {"library_id": "main", "folder_id": 1}


def test_folder_returns_ranked_items(client):
    r = client.post("/folder-suggestions", json={**BODY, "k": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["folder_key"] == "folder:1"
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
    assert client.post("/folder-suggestions", json=body).status_code == 422


def test_missing_table_is_404_and_empty_folder_is_422(client):
    client.run.side_effect = ChunkTableNotFoundError("t")
    assert client.post("/folder-suggestions", json=BODY).status_code == 404
    client.run.side_effect = EmptyFolderError("nothing embedded")
    r = client.post("/folder-suggestions", json=BODY)
    assert r.status_code == 422 and "nothing embedded" in r.json()["detail"]


def test_folder_key_is_stable_for_the_same_ids_in_any_order():
    k = suggestion_api.make_folder_key
    assert k(3, None) == "folder:3"
    assert k(None, [3, 1, 2]) == k(None, [1, 2, 3, 3]) != k(None, [1, 2])


def test_feedback_endpoint_records_verdict(client):
    r = client.post("/folder-suggestions/feedback", json={
        "library_id": "main", "folder_key": "folder:1", "post_id": 77, "verdict": "no"})
    assert r.status_code == 204
    client.feedback.set_verdict.assert_awaited_once_with("main", "folder:1", 77, "no")
    for verdict in ("very_no", "yes", "very_yes", "clear"):
        ok = client.post("/folder-suggestions/feedback", json={
            "library_id": "main", "folder_key": "folder:1", "post_id": 77, "verdict": verdict})
        assert ok.status_code == 204
    assert client.post("/folder-suggestions/feedback", json={
        "library_id": "main", "folder_key": "folder:1", "post_id": 77, "verdict": "maybe"}).status_code == 422


def test_ui_page_and_results_escape_titles_and_show_feedback_buttons(client):
    assert client.get("/ui/folder-suggestions").status_code == 200
    r = client.post("/ui/folder-suggestions/submit", data={"library_id": "main", "folder_id": "1"})
    assert r.status_code == 200
    assert "A &lt;b&gt;title&lt;/b&gt;" in r.text and "<b>title</b>" not in r.text
    assert all(f'value="{v}"' in r.text for v in ("very_no", "no", "yes", "very_yes"))
    assert client.run.await_args.kwargs["folder_id"] == 1


def test_ui_needs_a_source_and_explicit_ids_override_folder(client):
    assert client.post("/ui/folder-suggestions/submit", data={"library_id": "main"}).status_code == 422
    client.post("/ui/folder-suggestions/submit", data={"library_id": "main", "folder_id": "1", "post_ids": "5, 9 12"})
    kw = client.run.await_args.kwargs
    assert kw["post_ids"] == [5, 9, 12] and kw["folder_id"] is None


def test_ui_feedback_returns_card_state_and_survives_a_read_only_db(client):
    form = {"library_id": "main", "folder_key": "folder:1", "post_id": "77", "title": "T"}
    ok = client.post("/ui/folder-suggestions/feedback", data={**form, "verdict": "yes"})
    assert ok.status_code == 200 and "positives" in ok.text
    client.feedback.set_verdict.side_effect = RuntimeError("cannot execute CREATE TABLE in a read-only transaction")
    bad = client.post("/ui/folder-suggestions/feedback", data={**form, "verdict": "no"})
    assert bad.status_code == 200 and "Could not save feedback" in bad.text
