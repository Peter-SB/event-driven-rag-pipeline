"""Tests for the remote-reranking fallback backend (worker/reranking.py) —
the rerank analogue of test_remote_embedding.py.

Uses httpx.MockTransport to simulate remote_embedder_service's /rerank
endpoint — no real network or GPU required. Covers the behaviours that
differ meaningfully from embedding (score() response shape, re-alignment to
input order) plus a representative slice of the shared fallback ladder —
the exhaustive per-step coverage already lives in test_remote_embedding.py
against the identical ladder logic.
"""
from __future__ import annotations

import json
import httpx
import pytest

from event_driven_rag_service.worker.reranking import (
    FallbackRerankModel,
    RemoteRerankModel,
    build_fallback_reranker,
)


class FakeLocalReranker:
    def __init__(self, name: str = "local-reranker") -> None:
        self._name = name
        self.score_calls: list[tuple[str, list[str]]] = []

    @property
    def name(self) -> str:
        return self._name

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.score_calls.append((query, documents))
        return [0.0 for _ in documents]


def _client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://gaming-pc:1234/v1", transport=httpx.MockTransport(handler))


def _load_client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://gaming-pc:1234", transport=httpx.MockTransport(handler))


def _model_state_handler(state: str, load_calls: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v0/models":
            return httpx.Response(200, json={"data": [{"id": "some-reranker", "state": state}]})
        if request.url.path == "/api/v1/models/load":
            if load_calls is not None:
                load_calls.append(request)
            return httpx.Response(200, json={"model": "some-reranker", "status": "loaded"})
        raise AssertionError(f"unexpected request: {request.url.path}")

    return handler


def test_remote_score_re_aligns_to_document_order():
    """/rerank returns {index, score} pairs (server may reorder by score) —
    the client must re-align them back to the caller's `documents` order."""

    def handler(request: httpx.Request) -> httpx.Response:
        # Server returns doc[2] first (highest score), doc[0] last — out of order.
        return httpx.Response(200, json={"results": [
            {"index": 2, "score": 0.9},
            {"index": 0, "score": 0.1},
            {"index": 1, "score": 0.5},
        ]})

    remote = RemoteRerankModel(_client(handler), "some-reranker")
    scores = remote.score("query", ["doc0", "doc1", "doc2"])

    assert scores == [0.1, 0.5, 0.9]


def test_fallback_uses_remote_when_healthy_and_model_loaded():
    def rerank_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0, "score": 0.8}]})

    client = _client(rerank_handler)
    load_client = _load_client(_model_state_handler("loaded"))
    remote = RemoteRerankModel(client, "some-reranker", load_client=load_client)
    local = FakeLocalReranker()

    model = FallbackRerankModel(remote, local, "gaming-pc:1234")
    scores = model.score("query", ["doc0"])

    assert scores == [0.8]
    assert local.score_calls == []


def test_fallback_falls_back_to_local_when_endpoint_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = _client(handler)
    remote = RemoteRerankModel(client, "some-reranker")
    local = FakeLocalReranker()

    model = FallbackRerankModel(remote, local, "gaming-pc:1234")
    scores = model.score("query", ["doc0", "doc1"])

    assert scores == [0.0, 0.0]
    assert local.score_calls == [("query", ["doc0", "doc1"])]


def test_fallback_falls_back_to_local_when_model_not_available():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(200, json={"data": []})
        if request.url.path == "/api/v0/models":
            return httpx.Response(200, json={"data": []})
        raise AssertionError(f"unexpected request: {request.url.path}")

    client = _client(handler)
    load_client = _load_client(handler)
    remote = RemoteRerankModel(client, "some-reranker", load_client=load_client)
    local = FakeLocalReranker()

    model = FallbackRerankModel(remote, local, "gaming-pc:1234")
    scores = model.score("query", ["doc0"])

    assert scores == [0.0]
    assert local.score_calls == [("query", ["doc0"])]


def test_fallback_loads_model_then_uses_remote_when_not_yet_loaded():
    load_calls: list = []

    def rerank_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0, "score": 0.7}]})

    client = _client(rerank_handler)
    load_client = _load_client(_model_state_handler("not-loaded", load_calls))
    remote = RemoteRerankModel(client, "some-reranker", load_client=load_client)
    local = FakeLocalReranker()

    model = FallbackRerankModel(remote, local, "gaming-pc:1234")
    scores = model.score("query", ["doc0"])

    assert len(load_calls) == 1
    assert scores == [0.7]
    assert local.score_calls == []


def test_fallback_falls_back_to_local_on_remote_rerank_failure():
    def rerank_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = _client(rerank_handler)
    load_client = _load_client(_model_state_handler("loaded"))
    remote = RemoteRerankModel(client, "some-reranker", load_client=load_client)
    local = FakeLocalReranker()

    model = FallbackRerankModel(remote, local, "gaming-pc:1234")
    scores = model.score("query", ["doc0", "doc1"])

    assert scores == [0.0, 0.0]
    assert local.score_calls == [("query", ["doc0", "doc1"])]


def test_remote_score_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    remote = RemoteRerankModel(_client(handler), "some-reranker")
    with pytest.raises(httpx.HTTPStatusError):
        remote.score("query", ["doc0"])


def test_load_request_carries_the_configured_quantization():
    load_calls: list = []
    load_client = _load_client(_model_state_handler("not-loaded", load_calls))
    remote = RemoteRerankModel(
        _client(lambda r: httpx.Response(200, json={"results": []})),
        "some-reranker",
        load_client=load_client,
        quantization="int4",
    )

    assert remote.ensure_model_loaded() is True
    assert len(load_calls) == 1
    assert json.loads(load_calls[0].content) == {"model": "some-reranker", "quantization": "int4"}


def test_build_fallback_reranker_wires_remote():
    """Smoke test for the factory used by worker/entrypoints/gpu.py."""
    local = FakeLocalReranker("local-only-reranker")

    model = build_fallback_reranker(
        local=local,
        remote_model_name="local-only-reranker",
        base_url="http://127.0.0.1:1",  # nothing listens here — connection refused
        api_key="",
        timeout_s=1.0,
        health_path="/models",
    )

    assert isinstance(model, FallbackRerankModel)
    assert model.name == "local-only-reranker"

    scores = model.score("query", ["doc0"])
    assert scores == [0.0]
    assert local.score_calls == [("query", ["doc0"])]
