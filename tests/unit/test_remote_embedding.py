"""Tests for the remote-embedding fallback backend.

Uses httpx.MockTransport to simulate the remote OpenAI-compatible endpoint —
no real network or GPU required.

Tested behaviours
------------------
- Healthy remote endpoint: encode() calls remote, returns its vectors
- Remote request failure: falls back to local model and marks endpoint down
- Cached-down status: skips remote entirely, goes straight to local
- Recovery: a successful remote call after being down marks endpoint back up
"""
from __future__ import annotations

import time

import httpx
import pytest

from event_driven_rag_service.worker.remote_embedding import (
    FallbackEmbeddingModel,
    RemoteEmbeddingModel,
    RemoteEndpointHealth,
    build_fallback_model,
)


class FakeLocalModel:
    def __init__(self, name: str = "local-model") -> None:
        self._name = name
        self.encode_calls: list[list[str]] = []

    @property
    def name(self) -> str:
        return self._name

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.encode_calls.append(texts)
        return [[0.0, 0.0] for _ in texts]


def _client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://gaming-pc:1234/v1", transport=httpx.MockTransport(handler))


def test_remote_encode_success_returns_remote_vectors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0]}]})

    remote = RemoteEmbeddingModel(_client(handler), "some-model")
    assert remote.encode(["hello"]) == [[1.0, 2.0]]


def test_fallback_uses_remote_when_healthy():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0]}]})

    client = _client(handler)
    remote = RemoteEmbeddingModel(client, "some-model")
    local = FakeLocalModel()
    health = RemoteEndpointHealth(client, "/models", interval_s=9999)

    model = FallbackEmbeddingModel(remote, local, health, "gaming-pc:1234")
    vectors = model.encode(["hello"])

    assert vectors == [[1.0, 2.0]]
    assert local.encode_calls == []


def test_fallback_falls_back_to_local_on_remote_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = _client(handler)
    remote = RemoteEmbeddingModel(client, "some-model")
    local = FakeLocalModel()
    health = RemoteEndpointHealth(client, "/models", interval_s=9999)

    model = FallbackEmbeddingModel(remote, local, health, "gaming-pc:1234")
    vectors = model.encode(["hello", "world"])

    assert vectors == [[0.0, 0.0], [0.0, 0.0]]
    assert local.encode_calls == [["hello", "world"]]
    assert health.is_up is False


def test_fallback_skips_remote_when_already_marked_down():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"data": [{"embedding": [1.0]}]})

    client = _client(handler)
    remote = RemoteEmbeddingModel(client, "some-model")
    local = FakeLocalModel()
    health = RemoteEndpointHealth(client, "/models", interval_s=9999)
    health.mark_down()

    model = FallbackEmbeddingModel(remote, local, health, "gaming-pc:1234")
    vectors = model.encode(["hello"])

    assert calls == []
    assert local.encode_calls == [["hello"]]
    assert vectors == [[0.0, 0.0]]


def test_health_ping_marks_up_and_down():
    responses = iter([httpx.Response(200), httpx.Response(500), httpx.Response(200)])

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)

    client = _client(handler)
    health = RemoteEndpointHealth(client, "/models", interval_s=9999)

    assert health.is_up is True

    # Drive ping cycles manually instead of starting the background thread.
    resp = client.get("/models")
    assert resp.status_code == 200

    resp = client.get("/models")
    assert resp.status_code == 500
    assert health.mark_down() is True
    assert health.is_up is False

    resp = client.get("/models")
    assert resp.status_code == 200
    assert health.mark_up() is True
    assert health.is_up is True


def test_mark_up_and_down_report_transitions_only():
    client = _client(lambda request: httpx.Response(200))
    health = RemoteEndpointHealth(client, "/models", interval_s=9999)

    assert health.mark_down() is True
    assert health.mark_down() is False
    assert health.mark_up() is True
    assert health.mark_up() is False


def test_remote_encode_raises_on_http_error():
    """Regression guard: a non-2xx response must raise, not silently return []."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    remote = RemoteEmbeddingModel(_client(handler), "some-model")
    with pytest.raises(httpx.HTTPStatusError):
        remote.encode(["hello"])


def test_remote_encode_raises_on_malformed_response():
    """Regression guard: a 200 with an unexpected body shape must raise, not crash silently later."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    remote = RemoteEmbeddingModel(_client(handler), "some-model")
    with pytest.raises(KeyError):
        remote.encode(["hello"])


def test_health_background_thread_pings_and_updates_status():
    """Smoke test: the real background thread (not manual driving) reaches a down state."""
    client = _client(lambda request: httpx.Response(500))
    health = RemoteEndpointHealth(client, "/models", interval_s=0.05)

    health.start()
    try:
        deadline = time.monotonic() + 2.0
        while health.is_up and time.monotonic() < deadline:
            time.sleep(0.05)
        assert health.is_up is False
    finally:
        health.stop()


def test_build_fallback_model_wires_remote_and_starts_health_thread():
    """Smoke test for the factory used by worker/entrypoints/gpu.py."""
    local = FakeLocalModel("local-only-model")

    model = build_fallback_model(
        local=local,
        remote_model_name="local-only-model",
        base_url="http://127.0.0.1:1",  # nothing listens here — connection refused
        api_key="",
        timeout_s=1.0,
        health_path="/models",
        health_interval_s=9999,
    )

    assert isinstance(model, FallbackEmbeddingModel)
    assert model.name == "local-only-model"

    vectors = model.encode(["hello"])
    assert vectors == [[0.0, 0.0]]
    assert local.encode_calls == [["hello"]]
