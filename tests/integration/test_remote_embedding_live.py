"""Opt-in integration test against a real LM Studio (or other OpenAI-compatible) endpoint.

Requires a live endpoint reachable at EMBED_REMOTE_URL, loaded with the
summary_title model (see config/embedding_config.py: Qwen3-Embedding-0.6B-Q8_0.gguf).
Skipped automatically when EMBED_REMOTE_URL is unset — not part of CI, opt-in
for verifying real hardware (e.g. a homelab gaming PC running LM Studio).

Once EMBED_REMOTE_URL is set, this test assumes the endpoint is genuinely
reachable and has that model loaded — a failure here points at
infrastructure/LM Studio configuration, not application code (same
"assume it's on" convention as tests/e2e/test_postgres_health.py).

Run with:
    EMBED_REMOTE_URL=http://192.168.1.50:1234/v1 \\
        pytest tests/integration/test_remote_embedding_live.py -v -m remote_embedding
"""
from __future__ import annotations

import os
import time

import httpx
import pytest

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.config.settings import settings
from event_driven_rag_service.worker.entrypoints import gpu as gpu_entrypoint
from event_driven_rag_service.worker.remote_embedding import (
    FallbackEmbeddingModel,
    RemoteEmbeddingModel,
    build_fallback_model,
)

pytestmark = [
    pytest.mark.remote_embedding,
    pytest.mark.skipif(
        not os.getenv("EMBED_REMOTE_URL"),
        reason="EMBED_REMOTE_URL not set — point it at a live LM Studio (or other "
        "OpenAI-compatible) endpoint to run this opt-in test",
    ),
]

_REMOTE_URL = os.getenv("EMBED_REMOTE_URL", "")
_SUMMARY_TITLE_CONFIG = EMBED_CONFIGS["summary_title"]


@pytest.fixture
def live_client() -> httpx.Client:
    client = httpx.Client(base_url=_REMOTE_URL, timeout=30.0)
    yield client
    client.close()


def _root_url(base_url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


@pytest.fixture
def load_client() -> httpx.Client:
    client = httpx.Client(base_url=_root_url(_REMOTE_URL), timeout=120.0)
    yield client
    client.close()


def test_live_endpoint_is_reachable(live_client: httpx.Client):
    """Pre-flight smoke check: the configured endpoint responds to the health path."""
    resp = live_client.get("/models")
    assert resp.status_code == 200


def test_live_endpoint_embeds_summary_title_with_qwen3_gguf(live_client: httpx.Client):
    """Verify LM Studio actually produces embeddings using the summary_title (Qwen3 gguf) model."""
    remote_model_name = _SUMMARY_TITLE_CONFIG.remote_model or _SUMMARY_TITLE_CONFIG.model
    remote = RemoteEmbeddingModel(live_client, remote_model_name)

    texts = [
        "Title: Deep Dive into Transformer Architecture",
        "Title: Understanding Advanced Machine Learning Concepts",
    ]
    vectors = remote.encode(texts)

    assert len(vectors) == len(texts)
    for vector in vectors:
        assert len(vector) == _SUMMARY_TITLE_CONFIG.dim, (
            f"Expected {_SUMMARY_TITLE_CONFIG.dim}-dim vectors from "
            f"{remote_model_name}, got {len(vector)}"
        )
        assert all(isinstance(v, (int, float)) for v in vector)
    # Different inputs should not collapse to identical vectors.
    assert vectors[0] != vectors[1]


def test_live_fallback_model_prefers_remote_and_never_touches_local():
    """FallbackEmbeddingModel should use the live remote endpoint without calling local."""

    class _UnusedLocalModel:
        @property
        def name(self) -> str:
            return _SUMMARY_TITLE_CONFIG.model

        def encode(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("local model should not run while the remote endpoint is healthy")

    model = build_fallback_model(
        local=_UnusedLocalModel(),
        remote_model_name=_SUMMARY_TITLE_CONFIG.remote_model or _SUMMARY_TITLE_CONFIG.model,
        base_url=_REMOTE_URL,
        api_key=os.getenv("EMBED_REMOTE_API_KEY", ""),
        timeout_s=30.0,
        health_path="/models",
    )

    vectors = model.encode(["Title: quick smoke check"])

    assert len(vectors) == 1
    assert len(vectors[0]) == _SUMMARY_TITLE_CONFIG.dim
    assert model.name == _SUMMARY_TITLE_CONFIG.model


def _is_model_loaded(load_client: httpx.Client, model_name: str) -> bool:
    """Check actual load state via /api/v0/models (has a real `state` field) —
    NOT the OpenAI-compatible /models list, which lists every model LM Studio
    has *downloaded* regardless of whether it's loaded into memory (see
    remote_embedding.py's RemoteEmbeddingModel docstring for why that
    distinction matters — it's the root cause this whole file is guarding)."""
    resp = load_client.get("/api/v0/models")
    resp.raise_for_status()
    entry = next((m for m in resp.json().get("data", []) if m.get("id") == model_name), None)
    return bool(entry) and entry.get("state") == "loaded"


def test_live_program_can_load_the_needed_model(
    live_client: httpx.Client, load_client: httpx.Client
):
    """Exercise the actual "is the model loaded, and if not, load it" flow against
    the real LM Studio server — this is the path that goes wrong silently in
    production when the target model isn't loaded on the box.

    This always drives a real load-state check (and a real
    POST /api/v1/models/load if the model isn't loaded), rather than only
    asserting against whatever happens to already be resident.
    """
    remote_model_name = _SUMMARY_TITLE_CONFIG.remote_model or _SUMMARY_TITLE_CONFIG.model
    remote = RemoteEmbeddingModel(live_client, remote_model_name, load_client=load_client)

    # This is expected to trigger a model load if it isn't already loaded, and
    # must not silently swallow a real failure to load the model.
    assert remote.ensure_model_loaded() is True

    assert _is_model_loaded(load_client, remote_model_name), (
        f"Expected {remote_model_name!r} to be loaded on the LM Studio server "
        f"after ensure_model_loaded()"
    )

    # The model being loaded should mean a real encode() call now succeeds too.
    vectors = remote.encode(["Title: post-load smoke check"])
    assert len(vectors[0]) == _SUMMARY_TITLE_CONFIG.dim


def test_live_program_loads_model_when_not_already_loaded(
    live_client: httpx.Client, load_client: httpx.Client
):
    """Force the "model isn't loaded yet" precondition and confirm the program
    actually recovers by sending a load request — not just the happy path
    where the model happens to already be resident.

    This is the scenario the health-loop review was about: a fresh LM Studio
    box (or one that only has *some* models loaded) must still end up serving
    the request via an explicit ``POST /api/v1/models/load``, not fail silently.
    """
    remote_model_name = _SUMMARY_TITLE_CONFIG.remote_model or _SUMMARY_TITLE_CONFIG.model

    unload_resp = load_client.post(
        "/api/v1/models/unload", json={"instance_id": remote_model_name}
    )
    if unload_resp.status_code in (404, 501):
        pytest.skip(
            "This LM Studio server doesn't expose POST /api/v1/models/unload — "
            "can't force the 'not loaded' precondition to test the load-request path"
        )
    unload_resp.raise_for_status()

    # Unloading isn't necessarily instantaneous — give it a moment, then
    # confirm the model is genuinely absent before testing the recovery path.
    for _ in range(10):
        if not _is_model_loaded(load_client, remote_model_name):
            break
        time.sleep(0.5)
    assert not _is_model_loaded(load_client, remote_model_name), (
        f"Expected {remote_model_name!r} to be unloaded after POST /api/v1/models/unload "
        "— can't exercise the load path without a genuine 'not loaded' precondition"
    )

    remote = RemoteEmbeddingModel(live_client, remote_model_name, load_client=load_client)

    # ensure_model_loaded() must see the model isn't loaded and issue a
    # real POST /api/v1/models/load to bring it up.
    assert remote.ensure_model_loaded() is True

    assert _is_model_loaded(load_client, remote_model_name), (
        f"{remote_model_name!r} was not loaded via POST /api/v1/models/load after "
        "ensure_model_loaded() found it missing"
    )

    # And the full encode() path should now work end-to-end on the freshly loaded model.
    vectors = remote.encode(["Title: load-from-cold smoke check"])
    assert len(vectors[0]) == _SUMMARY_TITLE_CONFIG.dim


def test_live_gpu_entrypoint_only_routes_remote_for_models_with_remote_entry(monkeypatch):
    """Regression test for the "some models present, some not" health-loop issue:
    the GPU entrypoint must only send a model through the remote endpoint when
    EMBED_CONFIGS declares an explicit `remote_model` for it. Models without one
    (e.g. body/title's bge models) are not guaranteed to be loaded on the remote
    box, and routing them there anyway means every encode() for them fails its
    model-load check and its /embeddings call, on every single request.
    """
    monkeypatch.setattr(settings, "embed_remote_url", _REMOTE_URL)

    qwen_model = gpu_entrypoint._load_model(_SUMMARY_TITLE_CONFIG.model)
    assert isinstance(qwen_model, FallbackEmbeddingModel)

    vectors = qwen_model.encode(["Title: entrypoint smoke check"])
    assert len(vectors[0]) == _SUMMARY_TITLE_CONFIG.dim

    bge_body_config = EMBED_CONFIGS["body"]
    assert bge_body_config.remote_model is None, (
        "This test assumes the body config has no remote_model configured — "
        "update the test if that changes"
    )
    # Mock the local leg so this stays a fast check of the remote-routing
    # decision, not a real (slow/network-dependent) SentenceTransformer load.
    monkeypatch.setenv("MOCK_EMBEDDINGS", "1")
    bge_model = gpu_entrypoint._load_model(bge_body_config.model)
    assert not isinstance(bge_model, FallbackEmbeddingModel), (
        "A model with no `remote_model` entry must never be routed through the "
        "remote endpoint, even when EMBED_REMOTE_URL is set — the remote server "
        "isn't guaranteed to have it loaded"
    )
