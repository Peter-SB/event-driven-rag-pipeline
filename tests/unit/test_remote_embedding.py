"""Tests for the remote-embedding fallback backend.

Uses httpx.MockTransport to simulate the remote OpenAI-compatible endpoint —
no real network or GPU required.

Tested behaviours
------------------
- Healthy remote endpoint + model loaded: encode() calls remote, returns its vectors
- Unreachable endpoint: falls back to local without touching model-state endpoints
- Endpoint healthy but model not present on the server: falls back to local,
  without treating the endpoint itself as down
- Endpoint healthy, model present but not loaded: a load is requested, then encode()
  proceeds remotely
- Model present but load request fails: falls back to local
- Remote /embeddings request failure: falls back to local model
- No load_client configured: availability/load-state checks pass through
  optimistically and /embeddings is left to reveal the truth
"""
from __future__ import annotations

import httpx
import pytest

from event_driven_rag_service.worker.remote_embedding import (
    FallbackEmbeddingModel,
    RemoteEmbeddingModel,
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


def _load_client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://gaming-pc:1234", transport=httpx.MockTransport(handler))


def _model_state_handler(state: str, load_calls: list | None = None):
    """Build a load_client handler serving /api/v0/models with a fixed state
    and, if load_calls is given, recording any /api/v1/models/load requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v0/models":
            return httpx.Response(200, json={"data": [{"id": "some-model", "state": state}]})
        if request.url.path == "/api/v1/models/load":
            if load_calls is not None:
                load_calls.append(request)
            return httpx.Response(200, json={"model": "some-model", "status": "loaded"})
        raise AssertionError(f"unexpected request: {request.url.path}")

    return handler


def test_remote_encode_success_returns_remote_vectors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0]}]})

    remote = RemoteEmbeddingModel(_client(handler), "some-model")
    assert remote.encode(["hello"]) == [[1.0, 2.0]]


def test_fallback_uses_remote_when_healthy_and_model_loaded():
    def embed_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0]}]})

    client = _client(embed_handler)
    load_client = _load_client(_model_state_handler("loaded"))
    remote = RemoteEmbeddingModel(client, "some-model", load_client=load_client)
    local = FakeLocalModel()

    model = FallbackEmbeddingModel(remote, local, "gaming-pc:1234")
    vectors = model.encode(["hello"])

    assert vectors == [[1.0, 2.0]]
    assert local.encode_calls == []


def test_fallback_falls_back_to_local_when_endpoint_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = _client(handler)
    remote = RemoteEmbeddingModel(client, "some-model")
    local = FakeLocalModel()

    model = FallbackEmbeddingModel(remote, local, "gaming-pc:1234")
    vectors = model.encode(["hello", "world"])

    assert vectors == [[0.0, 0.0], [0.0, 0.0]]
    assert local.encode_calls == [["hello", "world"]]


def test_fallback_falls_back_to_local_when_model_not_available():
    """Endpoint itself is healthy, but the model isn't on the server at all —
    this must fall back to local without erroring on the health check."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(200, json={"data": []})
        if request.url.path == "/api/v0/models":
            return httpx.Response(200, json={"data": []})
        raise AssertionError(f"unexpected request: {request.url.path}")

    client = _client(handler)
    load_client = _load_client(handler)
    remote = RemoteEmbeddingModel(client, "some-model", load_client=load_client)
    local = FakeLocalModel()

    model = FallbackEmbeddingModel(remote, local, "gaming-pc:1234")
    vectors = model.encode(["hello"])

    assert vectors == [[0.0, 0.0]]
    assert local.encode_calls == [["hello"]]


def test_fallback_loads_model_then_uses_remote_when_not_yet_loaded():
    """Model is known to the server but not loaded — a load request must be
    sent, and the batch should still end up embedded remotely."""
    load_calls: list = []

    def embed_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0]}]})

    client = _client(embed_handler)
    load_client = _load_client(_model_state_handler("not-loaded", load_calls))
    remote = RemoteEmbeddingModel(client, "some-model", load_client=load_client)
    local = FakeLocalModel()

    model = FallbackEmbeddingModel(remote, local, "gaming-pc:1234")
    vectors = model.encode(["hello"])

    assert len(load_calls) == 1
    assert vectors == [[1.0, 2.0]]
    assert local.encode_calls == []


def test_fallback_falls_back_to_local_when_load_request_fails():
    def load_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v0/models":
            return httpx.Response(200, json={"data": [{"id": "some-model", "state": "not-loaded"}]})
        if request.url.path == "/api/v1/models/load":
            return httpx.Response(500)
        raise AssertionError(f"unexpected request: {request.url.path}")

    def embed_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("remote /embeddings should never be called when load fails")

    client = _client(embed_handler)
    load_client = _load_client(load_handler)
    remote = RemoteEmbeddingModel(client, "some-model", load_client=load_client)
    local = FakeLocalModel()

    model = FallbackEmbeddingModel(remote, local, "gaming-pc:1234")
    vectors = model.encode(["hello"])

    assert vectors == [[0.0, 0.0]]
    assert local.encode_calls == [["hello"]]


def test_fallback_falls_back_to_local_on_remote_embed_failure():
    def embed_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = _client(embed_handler)
    load_client = _load_client(_model_state_handler("loaded"))
    remote = RemoteEmbeddingModel(client, "some-model", load_client=load_client)
    local = FakeLocalModel()

    model = FallbackEmbeddingModel(remote, local, "gaming-pc:1234")
    vectors = model.encode(["hello", "world"])

    assert vectors == [[0.0, 0.0], [0.0, 0.0]]
    assert local.encode_calls == [["hello", "world"]]


def test_ensure_model_loaded_uses_state_field_not_mere_presence():
    """Regression guard: GET /models (OpenAI-compatible) lists every model LM
    Studio has *downloaded*, regardless of load state — a model can appear
    there while unloaded. With a load_client configured, availability must be
    decided from /api/v0/models' `state` field instead, or a genuinely
    unloaded model would be reported as available and never get a load
    request sent."""
    load_calls: list = []

    def embed_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0]}]})

    client = _client(embed_handler)
    load_client = _load_client(_model_state_handler("not-loaded", load_calls))

    remote = RemoteEmbeddingModel(client, "some-model", load_client=load_client)
    assert remote.ensure_model_loaded() is True
    assert len(load_calls) == 1


def test_ensure_model_loaded_skips_load_when_state_is_loaded():
    """The mirror case: /api/v0/models reports state=loaded, so no load call
    should be sent even though this is exactly the same model id as the
    "not loaded" test above."""
    load_calls: list = []

    client = _client(lambda request: httpx.Response(200, json={"data": [{"embedding": [1.0]}]}))
    load_client = _load_client(_model_state_handler("loaded", load_calls))

    remote = RemoteEmbeddingModel(client, "some-model", load_client=load_client)
    assert remote.ensure_model_loaded() is True
    assert load_calls == []


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


def test_is_model_available_optimistic_without_load_client():
    """No load_client configured — availability/load-state can't be checked,
    so both pass through optimistically and /embeddings is left to reveal
    the truth (matches pre-existing behaviour for callers that don't wire
    up the native LM Studio load API)."""
    remote = RemoteEmbeddingModel(_client(lambda r: httpx.Response(200)), "some-model")
    assert remote.is_model_available() is True
    assert remote.ensure_model_loaded() is True


def test_build_fallback_model_wires_remote():
    """Smoke test for the factory used by worker/entrypoints/gpu.py."""
    local = FakeLocalModel("local-only-model")

    model = build_fallback_model(
        local=local,
        remote_model_name="local-only-model",
        base_url="http://127.0.0.1:1",  # nothing listens here — connection refused
        api_key="",
        timeout_s=1.0,
        health_path="/models",
    )

    assert isinstance(model, FallbackEmbeddingModel)
    assert model.name == "local-only-model"

    vectors = model.encode(["hello"])
    assert vectors == [[0.0, 0.0]]
    assert local.encode_calls == [["hello"]]
