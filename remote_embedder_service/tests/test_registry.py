"""Registry sanity, and agreement with the pipeline's EMBED_CONFIGS.

The cross-check is the point of this file. The pipeline and this service are
separately installed packages that agree only by convention on two strings per
model — the wire id and the vector width. Nothing at runtime notices when they
drift: a renamed `remote_model` just makes the GPU worker fall back to local
embedding silently, and a changed `dim` corrupts a pgvector column.

EMBED_CONFIGS is read by parsing the pipeline's source with `ast`, not by
importing it — this service must not depend on `event_driven_rag_service`, which
would drag asyncpg, aio-pika and the whole pipeline onto the GPU box.

Tested behaviours
------------------
- Registry keys match their ModelSpec.id.
- Repo ids are safetensors repos, never GGUF.
- Every `remote_model` declared in the pipeline's EMBED_CONFIGS is servable here.
- Declared dims match the pipeline's `dim` for the same model.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from remote_embedder.manager import _QUANT_INT8, _QUANT_OFF, resolve_quantization
from remote_embedder.registry import MODEL_REGISTRY, all_specs, embed_specs, get_spec

PIPELINE_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "event_driven_rag_service"
    / "config"
    / "embedding_config.py"
)

PIPELINE_RERANK_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "event_driven_rag_service"
    / "config"
    / "reranking_config.py"
)


def _pipeline_embed_configs() -> list[dict]:
    """Extract every EmbedConfig(...) literal from the pipeline's config module.

    Static parse only — keyword arguments whose values aren't literals (none are
    today) come back as None and are simply not asserted on.
    """
    tree = ast.parse(PIPELINE_CONFIG.read_text(encoding="utf-8"))
    configs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "EmbedConfig"):
            continue
        entry = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            try:
                entry[kw.arg] = ast.literal_eval(kw.value)
            except ValueError:
                entry[kw.arg] = None
        configs.append(entry)
    return configs


def _pipeline_rerank_configs() -> list[dict]:
    """Extract every RerankConfig(...) literal from the pipeline's config
    module. Same static-parse approach and rationale as _pipeline_embed_configs."""
    tree = ast.parse(PIPELINE_RERANK_CONFIG.read_text(encoding="utf-8"))
    configs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "RerankConfig"):
            continue
        entry = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            try:
                entry[kw.arg] = ast.literal_eval(kw.value)
            except ValueError:
                entry[kw.arg] = None
        configs.append(entry)
    return configs


# --- registry internals ----------------------------------------------------


def test_registry_keys_match_spec_ids():
    for key, spec in MODEL_REGISTRY.items():
        assert key == spec.id


def test_embed_dims_are_positive():
    """dim is only meaningful for embedders — a reranker's scalar score has
    no width, so rerank specs are exempt (see ModelSpec.dim's docstring)."""
    assert all(spec.dim > 0 for spec in embed_specs())


def test_no_gguf_repos():
    """Specs load through transformers, which cannot read GGUF. A GGUF repo id
    here would only fail at download time on the remote box."""
    for spec in all_specs():
        assert "gguf" not in spec.repo_id.lower(), spec.id


def test_get_spec_returns_none_for_unknown():
    assert get_spec("no-such-model") is None


# --- agreement with the pipeline ------------------------------------------


def test_pipeline_config_is_readable():
    assert PIPELINE_CONFIG.exists(), f"expected pipeline config at {PIPELINE_CONFIG}"
    assert _pipeline_embed_configs(), "no EmbedConfig(...) literals parsed"


def test_every_pipeline_remote_model_is_servable():
    """A `remote_model` the registry doesn't serve means the GPU worker asks for
    a model this service will 400/404 on, and silently embeds locally forever."""
    missing = [
        cfg["remote_model"]
        for cfg in _pipeline_embed_configs()
        if cfg.get("remote_model") and cfg["remote_model"] not in MODEL_REGISTRY
    ]
    assert not missing, f"pipeline requests models this service does not serve: {missing}"


@pytest.mark.parametrize(
    "cfg",
    [c for c in _pipeline_embed_configs() if c.get("remote_model")],
    ids=lambda c: c["remote_model"],
)
def test_dims_agree_with_pipeline(cfg):
    """Vector width is the one number both sides must agree on — the pipeline
    declares the pgvector column from its `dim`, and we fill it from ours."""
    spec = MODEL_REGISTRY[cfg["remote_model"]]
    assert spec.dim == cfg["dim"], (
        f"{cfg['remote_model']}: pipeline expects dim={cfg['dim']}, "
        f"registry declares {spec.dim}"
    )


@pytest.mark.parametrize(
    "cfg",
    [c for c in _pipeline_embed_configs() if c.get("remote_model")],
    ids=lambda c: c["remote_model"],
)
def test_quantization_requests_are_understood(cfg):
    """Whatever the pipeline puts in `EmbedConfig.quantization` is sent verbatim
    on the load call, so this service has to recognise the spelling.

    An unknown value isn't fatal — resolve_quantization falls back to the
    registry default — but it would mean the pipeline silently not getting the
    precision it asked for, which is exactly the kind of quiet drift this file
    exists to catch.
    """
    requested = cfg.get("quantization")
    if requested is None:
        pytest.skip("no quantization declared for this model")

    spec = MODEL_REGISTRY[cfg["remote_model"]]
    # Recognised in one direction or the other; the fallback branch warns.
    assert requested.strip().lower() in (_QUANT_INT8 | _QUANT_OFF), (
        f"{cfg['remote_model']}: pipeline requests quantization={requested!r}, "
        f"which manager.resolve_quantization does not recognise"
    )
    assert resolve_quantization(spec, requested) is (
        requested.strip().lower() in _QUANT_INT8
    )


# --- agreement with the pipeline's RERANK_CONFIGS --------------------------


def test_pipeline_rerank_config_is_readable():
    assert PIPELINE_RERANK_CONFIG.exists(), f"expected pipeline config at {PIPELINE_RERANK_CONFIG}"
    assert _pipeline_rerank_configs(), "no RerankConfig(...) literals parsed"


def test_every_pipeline_rerank_remote_model_is_servable():
    """A `remote_model` the registry doesn't serve means the GPU worker asks
    for a reranker this service will 400/404 on, and silently reranks
    locally forever."""
    missing = [
        cfg["remote_model"]
        for cfg in _pipeline_rerank_configs()
        if cfg.get("remote_model") and cfg["remote_model"] not in MODEL_REGISTRY
    ]
    assert not missing, f"pipeline requests rerankers this service does not serve: {missing}"


def test_pipeline_rerank_remote_models_are_registered_as_rerank_kind():
    """A pipeline RerankConfig.remote_model that resolves to a kind="embed"
    entry would get routed through /v1/embeddings by mistake if the wire ids
    ever collided — assert they're on the rerank side of the registry."""
    for cfg in _pipeline_rerank_configs():
        remote_model = cfg.get("remote_model")
        if not remote_model:
            continue
        spec = MODEL_REGISTRY.get(remote_model)
        assert spec is not None and spec.kind == "rerank", (
            f"{remote_model}: expected a kind='rerank' registry entry, got {spec!r}"
        )
