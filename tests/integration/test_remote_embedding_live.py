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

import httpx
import pytest

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.worker.remote_embedding import (
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
        health_interval_s=9999,  # keep the background thread quiet for the test's lifetime
    )

    vectors = model.encode(["Title: quick smoke check"])

    assert len(vectors) == 1
    assert len(vectors[0]) == _SUMMARY_TITLE_CONFIG.dim
    assert model.name == _SUMMARY_TITLE_CONFIG.model
