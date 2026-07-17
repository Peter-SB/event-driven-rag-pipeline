"""Regression/smoke tests for _load_model in worker/entrypoints/gpu.py.

MOCK_EMBEDDINGS=1 keeps these fast and GPU-free (matches the convention in
tests/integration/test_gpu_worker.py).

Tested behaviours
------------------
- Regression: EMBED_REMOTE_URL unset → _load_model returns the local model
  directly, unchanged from pre-remote-embedding behaviour.
- Opt-in: EMBED_REMOTE_URL set but the requested model has no ``remote_model``
  entry in EMBED_CONFIGS (e.g. BAAI/bge-base-en-v1.5) → _load_model still
  returns the local model directly. Only models that explicitly declare a
  ``remote_model`` (e.g. the qwen3 gguf used for summary_title/analysis) are
  routed through the remote endpoint — otherwise every encode() for a model
  the remote server doesn't have would fail its model-load/verify call and
  its /embeddings call, every time.
- Smoke: EMBED_REMOTE_URL set for a model *with* a ``remote_model`` entry →
  _load_model wraps the local model in FallbackEmbeddingModel, and encode()
  still produces a usable vector even when nothing is listening at the
  remote URL.
"""
from __future__ import annotations

import os

import pytest

os.environ["MOCK_EMBEDDINGS"] = "1"

from event_driven_rag_service.config.settings import settings
from event_driven_rag_service.worker.entrypoints import gpu as gpu_entrypoint
from event_driven_rag_service.worker.remote_embedding import FallbackEmbeddingModel


@pytest.fixture(autouse=True)
def _reset_remote_url():
    original = settings.embed_remote_url
    yield
    settings.embed_remote_url = original


def test_load_model_returns_local_model_when_remote_url_unset():
    settings.embed_remote_url = ""
    model = gpu_entrypoint._load_model("BAAI/bge-base-en-v1.5")
    assert isinstance(model, gpu_entrypoint._MockEmbeddingModel)
    assert model.name == "BAAI/bge-base-en-v1.5"


def test_load_model_stays_local_when_model_has_no_remote_entry():
    """bge-base-en-v1.5 (body) has no `remote_model` in EMBED_CONFIGS — it should
    never be routed through the remote endpoint, even when one is configured."""
    settings.embed_remote_url = "http://127.0.0.1:1"  # nothing listens here
    model = gpu_entrypoint._load_model("BAAI/bge-base-en-v1.5")
    assert isinstance(model, gpu_entrypoint._MockEmbeddingModel)
    assert model.name == "BAAI/bge-base-en-v1.5"


def test_load_model_wraps_in_fallback_when_model_has_remote_entry():
    """Qwen3 (summary_title/analysis) declares `remote_model` — it should be
    routed through FallbackEmbeddingModel when EMBED_REMOTE_URL is set."""
    settings.embed_remote_url = "http://127.0.0.1:1"  # nothing listens here
    model = gpu_entrypoint._load_model("Qwen3-Embedding-0.6B-Q8_0.gguf")
    assert isinstance(model, FallbackEmbeddingModel)
    assert model.name == "Qwen3-Embedding-0.6B-Q8_0.gguf"


def test_fallback_wrapped_model_still_embeds_when_remote_unreachable():
    settings.embed_remote_url = "http://127.0.0.1:1"
    model = gpu_entrypoint._load_model("Qwen3-Embedding-0.6B-Q8_0.gguf")

    vectors = model.encode(["hello world"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 1024  # qwen3 dim, from _MockEmbeddingModel
