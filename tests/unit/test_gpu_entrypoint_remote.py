"""Regression/smoke tests for _load_model in worker/entrypoints/gpu.py.

MOCK_EMBEDDINGS=1 keeps these fast and GPU-free (matches the convention in
tests/integration/test_gpu_worker.py).

Tested behaviours
------------------
- Regression: EMBED_REMOTE_URL unset → _load_model returns the local model
  directly, unchanged from pre-remote-embedding behaviour.
- Opt-out: a config with ``allow_remote=False`` stays local even when an
  endpoint is configured.
- Smoke: EMBED_REMOTE_URL set for a model with ``allow_remote`` → _load_model
  wraps the local model in FallbackEmbeddingModel, and encode() still produces
  a usable vector even when nothing is listening at the remote URL.
- ``allow_local=False`` gives a remote-only model: it still wraps, but the
  fallback leg raises instead of quietly embedding on the CPU, and the
  combination of both flags off is a startup error rather than a runtime one.
- ``quantization`` from the config reaches the remote client, so precision is
  declared in one place rather than separately on each side.
- Coverage: the routing decision is checked for *every* EMBED_CONFIGS entry,
  not just one, so changing a flag can't silently change which models reach
  the remote embedder service.
"""
from __future__ import annotations

import os
from dataclasses import replace

import pytest

os.environ["MOCK_EMBEDDINGS"] = "1"

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.config.settings import settings
from event_driven_rag_service.worker.entrypoints import gpu as gpu_entrypoint
from event_driven_rag_service.worker.remote_embedding import FallbackEmbeddingModel

# "analysis" still uses Qwen3-Embedding-0.6B with allow_local=True, so it's the
# right stand-in for generic remote+local-fallback behaviour. "summary_title"
# is now Qwen3-Embedding-4B with allow_local=False (remote-only, too large for
# this box's local fallback leg) — its own tests live under "allow_local".
QWEN = EMBED_CONFIGS["analysis"].primary
BGE_BASE = EMBED_CONFIGS["body"].primary
BGE_SMALL = EMBED_CONFIGS["title"].primary

DEAD_URL = "http://127.0.0.1:1"  # nothing listens here


@pytest.fixture(autouse=True)
def _reset_remote_url():
    original = settings.embed_remote_url
    yield
    settings.embed_remote_url = original


@pytest.fixture
def patched_configs(monkeypatch):
    """Swap EMBED_CONFIGS for a modified copy, for testing flag combinations
    that the real config doesn't currently exercise."""

    def _apply(task_type: str, **changes):
        patched = dict(EMBED_CONFIGS)
        patched[task_type] = replace(EMBED_CONFIGS[task_type], **changes)
        monkeypatch.setattr(gpu_entrypoint, "EMBED_CONFIGS", patched)
        return patched[task_type]

    return _apply


def test_load_model_returns_local_model_when_remote_url_unset():
    settings.embed_remote_url = ""
    model = gpu_entrypoint._load_model(BGE_BASE)
    assert isinstance(model, gpu_entrypoint._MockEmbeddingModel)
    assert model.name == BGE_BASE


def test_load_model_stays_local_when_allow_remote_is_false(patched_configs):
    """`allow_remote=False` pins a model to this box even with an endpoint
    configured — the opt-out that used to be expressed as "no remote_model"."""
    cfg = patched_configs("title", allow_remote=False)
    settings.embed_remote_url = DEAD_URL

    model = gpu_entrypoint._load_model(cfg.primary)

    assert isinstance(model, gpu_entrypoint._MockEmbeddingModel)
    assert model.name == BGE_SMALL


def test_load_model_wraps_in_fallback_when_remote_allowed():
    settings.embed_remote_url = DEAD_URL
    model = gpu_entrypoint._load_model(QWEN)
    assert isinstance(model, FallbackEmbeddingModel)
    assert model.name == QWEN


def test_fallback_wrapped_model_still_embeds_when_remote_unreachable():
    settings.embed_remote_url = DEAD_URL
    model = gpu_entrypoint._load_model(QWEN)

    vectors = model.encode(["hello world"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 1024  # qwen3 dim, from _MockEmbeddingModel


def test_config_quantization_is_passed_to_the_remote_client():
    """The precision lives in EmbedConfig; the remote must be told about it,
    otherwise the two sides decide independently and can silently disagree."""
    settings.embed_remote_url = DEAD_URL

    model = gpu_entrypoint._load_model(QWEN)

    assert model._remote._quantization == EMBED_CONFIGS["analysis"].quantization
    assert model._remote._quantization == "int8"


# --- allow_local ------------------------------------------------------------


def test_remote_only_model_raises_rather_than_embedding_locally(patched_configs):
    """`allow_local=False` marks a model this box can't run. When the remote is
    down the batch must fail loudly and be redelivered, not silently produce
    vectors from some other model — those would land in the same pgvector
    column and quietly poison every future search against it."""
    cfg = patched_configs("summary_title", allow_local=False)
    settings.embed_remote_url = DEAD_URL

    model = gpu_entrypoint._load_model(cfg.primary)

    assert isinstance(model, FallbackEmbeddingModel)
    with pytest.raises(RuntimeError, match="local embedding is disabled"):
        model.encode(["hello world"])


def test_model_with_neither_leg_allowed_fails_at_load_time(patched_configs):
    """A config that permits nothing is a config error. Raising during startup
    beats discovering it on the first task hours later."""
    cfg = patched_configs("summary_title", allow_local=False, allow_remote=False)
    settings.embed_remote_url = DEAD_URL

    with pytest.raises(RuntimeError, match="allow_local=False"):
        gpu_entrypoint._load_model(cfg.primary)


def test_remote_only_model_also_fails_at_load_time_without_an_endpoint(patched_configs):
    """Same, reached the other way: allow_remote is meaningless with no URL."""
    cfg = patched_configs("summary_title", allow_local=False)
    settings.embed_remote_url = ""

    with pytest.raises(RuntimeError, match="allow_local=False"):
        gpu_entrypoint._load_model(cfg.primary)


# --- whole-config coverage --------------------------------------------------


@pytest.mark.parametrize(
    "task_type", sorted(EMBED_CONFIGS), ids=sorted(EMBED_CONFIGS)
)
def test_routing_decision_matches_config_for_every_task_type(task_type):
    """Every EMBED_CONFIGS entry is routed according to its own `allow_remote`.

    Parametrised over the whole config rather than a hand-picked model so that
    adding a config, or flipping a flag on an existing one, can't quietly
    change which models the remote embedder service is expected to serve. Its
    counterpart lives in remote_embedder_service/tests/test_registry.py, which
    checks the service actually serves every id declared here.
    """
    cfg = EMBED_CONFIGS[task_type]
    settings.embed_remote_url = DEAD_URL

    model = gpu_entrypoint._load_model(cfg.primary)

    if cfg.allow_remote:
        assert isinstance(model, FallbackEmbeddingModel)
    else:
        assert not isinstance(model, FallbackEmbeddingModel)
    # Either way the model reports its primary name — that's what chunk rows,
    # queue routing and the query-prefix lookup are all keyed on, so it must
    # not vary with which leg happens to serve the batch.
    assert model.name == cfg.primary


@pytest.mark.parametrize(
    "task_type", sorted(EMBED_CONFIGS), ids=sorted(EMBED_CONFIGS)
)
def test_every_config_stays_local_when_remote_url_unset(task_type):
    """With EMBED_REMOTE_URL unset nothing is routed remotely, regardless of
    what `allow_remote` says -- except a remote-only model (`allow_local=False`),
    which has nowhere left to run and must fail loudly at load time instead."""
    settings.embed_remote_url = ""
    cfg = EMBED_CONFIGS[task_type]

    if not cfg.allow_local:
        with pytest.raises(RuntimeError, match="allow_local=False"):
            gpu_entrypoint._load_model(cfg.primary)
        return

    model = gpu_entrypoint._load_model(cfg.primary)
    assert not isinstance(model, FallbackEmbeddingModel)
