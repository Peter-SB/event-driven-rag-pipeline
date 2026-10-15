"""ModelManager: residency, batching and idle unload.

Tested behaviours
------------------
- encode() splits at MAX_BATCH_SIZE and reassembles sub-batches in input order.
- Batch size of 1 and an exact multiple of the batch size both behave.
- load() is idempotent and evicts any other resident model.
- encode() loads on demand when nothing is resident.
- A model whose vectors don't match its registry `dim` is refused.
- reap_once() unloads past the idle threshold, spares a recently used model,
  and is disabled by idle_unload_s=0.
- Unknown model ids raise UnknownModelError rather than blowing up later.
- default_loader warns when a `quantize` model can't actually be quantised
  because the resolved device isn't CUDA.
"""
from __future__ import annotations

import pytest

from remote_embedder.manager import (
    ModelLoadError,
    ModelManager,
    UnknownModelError,
    default_loader,
    resolve_quantization,
)
from remote_embedder.registry import MODEL_REGISTRY, ModelSpec

from conftest import FakeClock, RecordingLoader

BGE_BASE = "text-embedding-bge-base-en-v1.5"
BGE_SMALL = "text-embedding-bge-small-en-v1.5"
QWEN = "text-embedding-qwen3-embedding-0.6b"


# --- batching --------------------------------------------------------------


def test_encode_splits_into_sub_batches(settings, loader):
    settings.max_batch_size = 4
    mgr = ModelManager(settings, loader=loader)
    texts = [f"text-{i}" for i in range(10)]

    vectors = mgr.encode(BGE_BASE, texts)

    assert len(vectors) == 10
    encoder = mgr._model  # the FakeEncoder that was handed out
    # First call is the dimension probe at load time; the rest are the batches.
    batch_calls = encoder.calls[1:]
    assert [len(c) for c in batch_calls] == [4, 4, 2]


def test_sub_batches_are_reassembled_in_input_order(settings, loader):
    """A text repeated at the first and last position must produce identical
    vectors. If sub-batches were concatenated out of order — or a batch's
    results were zipped against the wrong slice — this is what catches it,
    and downstream it would mean chunks silently getting each other's embeddings.
    """
    settings.max_batch_size = 3
    mgr = ModelManager(settings, loader=loader)
    texts = ["repeated", "b", "c", "d", "e", "f", "repeated"]

    vectors = mgr.encode(BGE_BASE, texts)

    assert len(vectors) == 7
    assert vectors[0] == vectors[6]
    assert vectors[0] != vectors[1]


def test_encode_with_exact_multiple_of_batch_size(settings, loader):
    settings.max_batch_size = 5
    mgr = ModelManager(settings, loader=loader)

    vectors = mgr.encode(BGE_BASE, [f"t{i}" for i in range(10)])

    assert len(vectors) == 10
    assert [len(c) for c in mgr._model.calls[1:]] == [5, 5]


def test_encode_with_batch_size_one(settings, loader):
    settings.max_batch_size = 1
    mgr = ModelManager(settings, loader=loader)

    vectors = mgr.encode(BGE_BASE, ["a", "b", "c"])

    assert len(vectors) == 3
    assert [len(c) for c in mgr._model.calls[1:]] == [1, 1, 1]


def test_encode_empty_list_returns_empty_without_loading(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    assert mgr.encode(BGE_BASE, []) == []
    assert loader.loaded == []


def test_encode_vector_width_matches_registry_dim(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    vectors = mgr.encode(QWEN, ["hello"])
    assert len(vectors[0]) == MODEL_REGISTRY[QWEN].dim


# --- residency -------------------------------------------------------------


def test_load_is_idempotent(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    mgr.load(BGE_BASE)
    mgr.load(BGE_BASE)
    assert loader.loaded == [BGE_BASE]


def test_loading_a_second_model_evicts_the_first(settings, loader):
    """Only one model is resident at a time — VRAM on a homelab box is the
    binding constraint, and the single GPU worker never needs two at once."""
    mgr = ModelManager(settings, loader=loader)
    mgr.load(BGE_BASE)
    mgr.load(QWEN)

    assert mgr.loaded_model_id == QWEN
    assert loader.loaded == [BGE_BASE, QWEN]


def test_encode_loads_on_demand(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    assert mgr.loaded_model_id is None

    mgr.encode(BGE_SMALL, ["x"])

    assert mgr.loaded_model_id == BGE_SMALL


def test_encode_switches_model_when_a_different_one_is_requested(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    mgr.encode(BGE_BASE, ["x"])
    vectors = mgr.encode(QWEN, ["x"])

    assert mgr.loaded_model_id == QWEN
    assert len(vectors[0]) == MODEL_REGISTRY[QWEN].dim


def test_unload_frees_the_model(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    mgr.load(BGE_BASE)

    assert mgr.unload(BGE_BASE) is True
    assert mgr.loaded_model_id is None
    assert mgr.unload(BGE_BASE) is False


def test_unload_of_a_different_model_is_a_no_op(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    mgr.load(BGE_BASE)

    assert mgr.unload(QWEN) is False
    assert mgr.loaded_model_id == BGE_BASE


def test_unknown_model_raises(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    with pytest.raises(UnknownModelError):
        mgr.load("no-such-model")
    with pytest.raises(UnknownModelError):
        mgr.encode("no-such-model", ["x"])


def test_load_failure_is_wrapped(settings):
    def boom(spec, settings):
        raise OSError("network is down")

    mgr = ModelManager(settings, loader=boom)
    with pytest.raises(ModelLoadError):
        mgr.load(BGE_BASE)
    assert mgr.loaded_model_id is None


def test_dim_mismatch_refuses_the_load(settings):
    """A model producing the wrong vector width must be rejected at load time.

    These vectors go into a fixed-width pgvector column. Serving a 512-dim
    vector for a model the pipeline believes is 768-dim surfaces as an opaque
    insert error three services downstream, long after the cause.
    """
    mgr = ModelManager(settings, loader=RecordingLoader(dim_override=99))

    with pytest.raises(ModelLoadError, match="refusing to serve"):
        mgr.load(BGE_BASE)
    assert mgr.loaded_model_id is None


# --- states ----------------------------------------------------------------


def test_states_lists_every_model_with_load_state(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    mgr.load(QWEN)

    states = {s["id"]: s["state"] for s in mgr.states()}

    assert set(states) == set(MODEL_REGISTRY)
    assert states[QWEN] == "loaded"
    assert states[BGE_BASE] == "not-loaded"


def test_states_reports_quantization_flag(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    by_id = {s["id"]: s for s in mgr.states()}
    assert by_id[QWEN]["quantization"] == "int8"
    assert by_id[BGE_BASE]["quantization"] is None


# --- idle unload -----------------------------------------------------------


def test_reap_unloads_after_idle_threshold(settings, loader):
    settings.idle_unload_s = 100
    clock = FakeClock()
    mgr = ModelManager(settings, loader=loader, clock=clock)
    mgr.load(BGE_BASE)

    clock.advance(101)

    assert mgr.reap_once() is True
    assert mgr.loaded_model_id is None


def test_reap_spares_a_recently_used_model(settings, loader):
    settings.idle_unload_s = 100
    clock = FakeClock()
    mgr = ModelManager(settings, loader=loader, clock=clock)
    mgr.load(BGE_BASE)

    clock.advance(99)

    assert mgr.reap_once() is False
    assert mgr.loaded_model_id == BGE_BASE


def test_encode_resets_the_idle_timer(settings, loader):
    settings.idle_unload_s = 100
    clock = FakeClock()
    mgr = ModelManager(settings, loader=loader, clock=clock)
    mgr.load(BGE_BASE)

    clock.advance(90)
    mgr.encode(BGE_BASE, ["still working"])
    clock.advance(90)

    assert mgr.reap_once() is False
    assert mgr.loaded_model_id == BGE_BASE


def test_reap_is_a_no_op_when_nothing_is_loaded(settings, loader):
    clock = FakeClock()
    mgr = ModelManager(settings, loader=loader, clock=clock)
    clock.advance(10_000)
    assert mgr.reap_once() is False


# --- device warnings --------------------------------------------------------


def test_default_loader_warns_when_quantize_requested_without_cuda(
    settings, monkeypatch, caplog
):
    """A registry entry that asks for int8 silently loads full-precision
    instead when there's no CUDA device — silent is the wrong word for it:
    it must say so, since the alternative is a surprisingly slow first
    request with no clue why."""

    class _FakeSentenceTransformer:
        def __init__(self, repo_id, **kwargs):
            self.repo_id = repo_id
            self.kwargs = kwargs
            self.max_seq_length = None

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", _FakeSentenceTransformer
    )
    settings.device = "cpu"
    settings.enable_quantization = True
    spec = ModelSpec(id="fake-model", repo_id="fake/repo", dim=8, quantize=True)

    with caplog.at_level("WARNING"):
        model = default_loader(spec, settings)

    assert isinstance(model, _FakeSentenceTransformer)
    assert "quantization_config" not in model.kwargs.get("model_kwargs", {})
    assert model.kwargs.get("device") == "cpu"  # full precision, plain device kwarg
    assert any(
        "cpu" in r.message and "fake-model" in r.message for r in caplog.records
    )


def test_default_loader_does_not_warn_for_non_quantized_models_on_cpu(
    settings, monkeypatch, caplog
):
    """Only registry entries that actually asked for int8 trigger the warning
    — bge-base/bge-small run on CPU without complaint."""

    class _FakeSentenceTransformer:
        def __init__(self, repo_id, **kwargs):
            pass

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", _FakeSentenceTransformer
    )
    settings.device = "cpu"
    spec = ModelSpec(id="fake-model", repo_id="fake/repo", dim=8, quantize=False)

    with caplog.at_level("WARNING"):
        default_loader(spec, settings)

    assert caplog.records == []


def test_idle_unload_disabled_by_zero(settings, loader):
    settings.idle_unload_s = 0
    clock = FakeClock()
    mgr = ModelManager(settings, loader=loader, clock=clock)
    mgr.load(BGE_BASE)

    clock.advance(10_000)

    assert mgr.reap_once() is False
    assert mgr.loaded_model_id == BGE_BASE


# --- quantisation override --------------------------------------------------


def test_resolve_quantization_defaults_to_the_registry():
    assert resolve_quantization(MODEL_REGISTRY[QWEN], None) is True
    assert resolve_quantization(MODEL_REGISTRY[BGE_BASE], None) is False


@pytest.mark.parametrize("spelling", ["int8", "INT8", " 8bit ", "8-bit", "q8_0"])
def test_resolve_quantization_accepts_int8_spellings(spelling):
    assert resolve_quantization(MODEL_REGISTRY[BGE_BASE], spelling) is True


@pytest.mark.parametrize("spelling", ["none", "off", "fp16", "FULL"])
def test_resolve_quantization_accepts_disabling_spellings(spelling):
    assert resolve_quantization(MODEL_REGISTRY[QWEN], spelling) is False


def test_unrecognised_quantization_falls_back_to_the_default_with_a_warning(caplog):
    """Refusing the load would be worse than ignoring the value: the client
    reads any non-2xx as "embed locally for this batch", so a typo would cost
    remote embedding entirely rather than just the requested precision."""
    with caplog.at_level("WARNING"):
        assert resolve_quantization(MODEL_REGISTRY[QWEN], "int4") is True

    assert any("int4" in r.message for r in caplog.records)


def test_load_honours_a_quantization_override(settings, loader):
    mgr = ModelManager(settings, loader=loader)

    mgr.load(BGE_BASE, "int8")

    assert mgr.applied_quantization == "int8"
    assert {s["id"]: s["quantization"] for s in mgr.states()}[BGE_BASE] == "int8"


def test_load_can_disable_quantization_for_a_quantized_model(settings, loader):
    mgr = ModelManager(settings, loader=loader)

    mgr.load(QWEN, "none")

    assert mgr.applied_quantization is None


def test_reloading_at_a_different_precision_replaces_the_resident_model(settings, loader):
    """Same id, different precision, must not short-circuit on "already loaded"
    — the caller asked for int8 and would otherwise silently keep fp16."""
    mgr = ModelManager(settings, loader=loader)
    mgr.load(BGE_BASE)
    assert mgr.applied_quantization is None

    mgr.load(BGE_BASE, "int8")

    assert mgr.applied_quantization == "int8"
    assert loader.loaded == [BGE_BASE, BGE_BASE]  # genuinely reloaded


def test_reloading_at_the_same_precision_is_still_idempotent(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    mgr.load(QWEN, "int8")
    mgr.load(QWEN, "int8")
    assert loader.loaded == [QWEN]


def test_unload_clears_the_applied_precision(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    mgr.load(BGE_BASE, "int8")
    mgr.unload(BGE_BASE)
    assert mgr.applied_quantization is None


# --- reranking ---------------------------------------------------------------

RERANK_0_6B = "text-rerank-qwen3-reranker-0.6b"
RERANK_4B = "text-rerank-qwen3-reranker-4b"


def test_rerank_loads_on_demand(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    assert mgr.loaded_model_id is None

    mgr.rerank(RERANK_0_6B, "query", ["doc"])

    assert mgr.loaded_model_id == RERANK_0_6B


def test_rerank_returns_one_score_per_document_in_order(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    scores = mgr.rerank(RERANK_0_6B, "hello world", ["hello there", "world peace", "unrelated"])
    assert len(scores) == 3


def test_rerank_empty_documents_returns_empty_without_loading(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    assert mgr.rerank(RERANK_0_6B, "query", []) == []
    assert loader.loaded == []


def test_rerank_evicts_a_resident_embedder_and_vice_versa(settings, loader):
    """Only one model is resident regardless of kind — an embedder and a
    reranker share the same VRAM budget on a homelab box."""
    mgr = ModelManager(settings, loader=loader)
    mgr.encode(BGE_BASE, ["x"])
    mgr.rerank(RERANK_0_6B, "q", ["d"])

    assert mgr.loaded_model_id == RERANK_0_6B
    assert loader.loaded == [BGE_BASE, RERANK_0_6B]


def test_encode_refuses_a_rerank_model_id(settings, loader):
    """POST /v1/embeddings must not silently score with a reranker — the
    kind mismatch has to surface as an error, not a shape-mangled response."""
    mgr = ModelManager(settings, loader=loader)
    with pytest.raises(UnknownModelError):
        mgr.encode(RERANK_0_6B, ["x"])


def test_rerank_refuses_an_embed_model_id(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    with pytest.raises(UnknownModelError):
        mgr.rerank(BGE_BASE, "q", ["d"])


def test_reranker_quantization_is_labelled_int4_not_int8(settings, loader):
    mgr = ModelManager(settings, loader=loader)
    mgr.load(RERANK_0_6B, "int4")
    assert mgr.applied_quantization == "int4"
    assert {s["id"]: s["quantization"] for s in mgr.states()}[RERANK_0_6B] == "int4"


def test_reranker_dim_verification_is_skipped(settings, loader):
    """A reranker's scalar score has no width to verify — load() must not
    call _verify_dim (which would call .encode(), a method rerankers don't have)."""
    mgr = ModelManager(settings, loader=loader)
    mgr.load(RERANK_0_6B)  # must not raise AttributeError from a missing .encode()
    assert mgr.loaded_model_id == RERANK_0_6B
