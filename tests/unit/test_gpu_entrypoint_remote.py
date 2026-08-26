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
  fallback leg raises ``EmbeddingUnavailableError`` instead of quietly
  embedding on the CPU (``EmbedHandler`` turns that into a skip for the
  backfill endpoint), and the combination of both flags off is a startup error
  rather than a runtime one.
- The local leg honours ``EmbedConfig.quantization``: bitsandbytes on CUDA,
  PyTorch dynamic quantization on CPU. Same config field the remote leg uses,
  so a batch that falls back mid-run is not silently full precision.
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
from event_driven_rag_service.exceptions import EmbeddingUnavailableError
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
    down the batch must raise rather than silently produce vectors from some
    other model — those would land in the same pgvector column and quietly
    poison every future search against it. The raise is how the batch gets
    skipped and left for the backfill endpoint."""
    cfg = patched_configs("summary_title", allow_local=False)
    settings.embed_remote_url = DEAD_URL

    model = gpu_entrypoint._load_model(cfg.primary)

    assert isinstance(model, FallbackEmbeddingModel)
    # The typed error is the contract EmbedHandler keys on to skip the batch
    # rather than dead-letter it — a bare RuntimeError would be DLQ'd instead.
    with pytest.raises(EmbeddingUnavailableError, match="local embedding is disabled"):
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


# --- local leg: same quantisation mechanism as the remote -------------------


class _FakeSentenceTransformer:
    """Records the kwargs it was constructed with, so the tests can assert on
    how the local leg asked for the model rather than actually loading one."""

    last_kwargs: dict | None = None

    def __init__(self, model_name, **kwargs):
        type(self).last_kwargs = {"model_name": model_name, **kwargs}
        self.model_name = model_name

    def get_embedding_dimension(self):
        return 1024

    def encode(self, texts):
        return [[0.0] * 1024 for _ in texts]


@pytest.fixture
def fake_sentence_transformer(monkeypatch):
    """Patch SentenceTransformer at its import site inside sentence_transformers,
    since _load_sentence_transformer imports it lazily on each call."""
    import sentence_transformers

    _FakeSentenceTransformer.last_kwargs = None
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer
    )
    return _FakeSentenceTransformer


def test_local_load_applies_int8_via_bitsandbytes_on_cuda(
    fake_sentence_transformer, monkeypatch
):
    """`quantization="int8"` must reach the local load as a BitsAndBytesConfig.

    This is the whole point of the config field being shared: the remote sends
    it on /api/v1/models/load, the local leg passes it to transformers, and
    both produce int8 vectors from the same checkpoint. If the local leg
    quietly loaded full precision, a fallback batch would sit in the same
    pgvector column at a different precision.
    """
    monkeypatch.setattr(gpu_entrypoint, "_resolve_device", lambda: "cuda")

    gpu_entrypoint._load_sentence_transformer("Qwen/Qwen3-Embedding-0.6B", "int8")

    kwargs = fake_sentence_transformer.last_kwargs
    assert "quantization_config" in kwargs["model_kwargs"]
    assert kwargs["model_kwargs"]["quantization_config"].load_in_8bit is True
    # accelerate places quantised weights; passing `device=` alongside errors.
    assert kwargs["model_kwargs"]["device_map"] == "auto"
    assert "device" not in kwargs


def test_local_load_applies_dynamic_quantization_without_cuda(
    fake_sentence_transformer, monkeypatch
):
    """int8 via bitsandbytes is a CUDA path. On a CPU-only box `quantization="int8"`
    must still produce an int8 model — via PyTorch dynamic quantization — rather
    than silently loading full precision, since a CPU box is exactly where the
    memory/throughput win matters most."""
    monkeypatch.setattr(gpu_entrypoint, "_resolve_device", lambda: "cpu")
    calls = []
    monkeypatch.setattr(
        gpu_entrypoint,
        "_quantize_dynamic_cpu",
        lambda model: calls.append(model) or model,
    )

    gpu_entrypoint._load_sentence_transformer("Qwen/Qwen3-Embedding-0.6B", "int8")

    kwargs = fake_sentence_transformer.last_kwargs
    assert "model_kwargs" not in kwargs
    assert kwargs["device"] == "cpu"
    assert len(calls) == 1, "dynamic quantization must be applied on the CPU leg"


def test_local_load_retries_at_full_precision_when_dynamic_quant_fails(
    fake_sentence_transformer, monkeypatch
):
    """PyTorch dynamic quantization can fail on an unusual module tree; a
    slower model beats a worker that cannot embed at all."""
    monkeypatch.setattr(gpu_entrypoint, "_resolve_device", lambda: "cpu")

    def _boom(model):
        raise RuntimeError("quantize_dynamic exploded")

    monkeypatch.setattr(gpu_entrypoint, "_quantize_dynamic_cpu", _boom)

    model = gpu_entrypoint._load_sentence_transformer("Qwen/Qwen3-Embedding-0.6B", "int8")

    assert model.name == "Qwen/Qwen3-Embedding-0.6B"


def test_local_load_retries_at_full_precision_when_int8_load_fails(monkeypatch):
    """bitsandbytes may be missing on a box that only ever expected to be the
    fallback leg. A slower model beats a worker that cannot embed at all."""
    import sentence_transformers

    attempts: list[bool] = []

    class _Flaky(_FakeSentenceTransformer):
        def __init__(self, model_name, **kwargs):
            quantised = "model_kwargs" in kwargs
            attempts.append(quantised)
            if quantised:
                raise ImportError("no bitsandbytes here")
            super().__init__(model_name, **kwargs)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _Flaky)
    monkeypatch.setattr(gpu_entrypoint, "_resolve_device", lambda: "cuda")

    model = gpu_entrypoint._load_sentence_transformer("Qwen/Qwen3-Embedding-0.6B", "int8")

    assert attempts == [True, False], "should try int8 first, then full precision"
    assert model.name == "Qwen/Qwen3-Embedding-0.6B"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("int8", True),
        ("8bit", True),
        ("Q8_0", True),
        (None, False),
        ("none", False),
        ("fp16", False),
        ("nonsense", False),  # unknown → full precision, never a failed load
    ],
)
def test_quantization_spellings_match_the_remote_services_vocabulary(value, expected):
    """The two legs read the same config string, so they must agree on what it
    means. Drift here would make one leg int8 and the other full precision off
    a single `quantization=` value."""
    assert gpu_entrypoint._wants_int8(value, "some/model") is expected


def test_local_load_rejects_a_model_whose_dim_disagrees_with_the_config(
    fake_sentence_transformer, monkeypatch
):
    """The remote verifies vector width after loading; the local leg must too.

    These vectors land in a fixed-width pgvector column, so a repo revision that
    changed shape has to fail at load rather than as opaque insert errors, one
    batch at a time, for as long as nobody looks.
    """
    monkeypatch.setattr(gpu_entrypoint, "_resolve_device", lambda: "cpu")

    # The fake reports 1024 dims; bge-small declares 384.
    with pytest.raises(RuntimeError, match="refusing to embed"):
        gpu_entrypoint._load_sentence_transformer(BGE_SMALL, None)
