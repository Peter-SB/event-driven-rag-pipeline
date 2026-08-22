"""Model registry — the set of models this service is willing to serve.

Keyed by *wire id*: the string clients send in ``{"model": ...}``. These ids are
deliberately the same slugs LM Studio used, so the pipeline's
``EMBED_CONFIGS[...].remote_model`` values keep working unchanged when
``EMBED_REMOTE_URL`` is repointed at this service.

Repo ids are safetensors repos, never GGUF — this service runs the models through
transformers/sentence-transformers on CUDA, not llama.cpp.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """One servable model."""

    # The id clients send. Matches EMBED_CONFIGS[...].remote_model in the pipeline.
    id: str
    # HuggingFace repo to load from (safetensors).
    repo_id: str
    # Expected embedding dimension. Verified against the first vector a freshly
    # loaded model produces: downstream this lands in a fixed-width pgvector
    # column, so a repo revision silently changing shape must fail loudly here
    # rather than corrupt the table.
    dim: int
    # Load in 8-bit via bitsandbytes. Worth it for the 0.6B model; for the small
    # BGE encoders LLM.int8's per-layer outlier handling costs more time than the
    # saved VRAM is worth, so they stay fp16/fp32.
    quantize: bool = False
    # Override the model's own max_seq_length. None keeps the checkpoint default.
    max_seq_length: int | None = None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "text-embedding-bge-base-en-v1.5": ModelSpec(
        id="text-embedding-bge-base-en-v1.5",
        repo_id="BAAI/bge-base-en-v1.5",
        dim=768,
        quantize=False,
    ),
    "text-embedding-bge-small-en-v1.5": ModelSpec(
        id="text-embedding-bge-small-en-v1.5",
        repo_id="BAAI/bge-small-en-v1.5",
        dim=384,
        quantize=False,
    ),
    "text-embedding-qwen3-embedding-0.6b": ModelSpec(
        id="text-embedding-qwen3-embedding-0.6b",
        # NOT the -GGUF repo: that one is for the pipeline's local llama-cpp
        # fallback leg. This service needs the safetensors weights.
        repo_id="Qwen/Qwen3-Embedding-0.6B",
        dim=1024,
        quantize=True,
    ),
    "text-embedding-qwen3-embedding-4b": ModelSpec(
        id="text-embedding-qwen3-embedding-4b",
        repo_id="Qwen/Qwen3-Embedding-4B",
        dim=2560,
        # int8 measured at ~4.2GB resident (weights) on an 8GB card, leaving
        # enough headroom for a small batch — full precision would not fit
        # alongside any activations at all.
        quantize=True,
    ),
}


def get_spec(model_id: str) -> ModelSpec | None:
    """The spec for ``model_id``, or None if this service does not serve it."""
    return MODEL_REGISTRY.get(model_id)


def all_specs() -> list[ModelSpec]:
    return list(MODEL_REGISTRY.values())
