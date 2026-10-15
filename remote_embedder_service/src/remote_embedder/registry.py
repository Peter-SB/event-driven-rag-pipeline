"""Model registry — the set of models this service is willing to serve.

Keyed by *wire id*: the string clients send in ``{"model": ...}``. These ids are
deliberately the same slugs LM Studio used, so the pipeline's
``EMBED_CONFIGS[...].remote_model`` values keep working unchanged when
``EMBED_REMOTE_URL`` is repointed at this service.

Repo ids are safetensors repos — the same checkpoints the pipeline's local
fallback leg loads, run here through transformers/sentence-transformers on CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """One servable model — either an embedder (kind="embed") or a reranker
    (kind="rerank"). The two kinds load through different transformers
    entrypoints (SentenceTransformer vs AutoModelForCausalLM) and expose
    different work methods (encode() vs score()) — see manager.py."""

    # The id clients send. Matches EMBED_CONFIGS[...].remote_model /
    # RERANK_CONFIGS[...].remote_model in the pipeline.
    id: str
    # HuggingFace repo to load from (safetensors).
    repo_id: str
    # "embed" or "rerank" — selects the loader and the served method.
    kind: str = "embed"
    # Expected embedding dimension. Verified against the first vector a freshly
    # loaded model produces: downstream this lands in a fixed-width pgvector
    # column, so a repo revision silently changing shape must fail loudly here
    # rather than corrupt the table. Unused (0) for kind="rerank" — a scalar
    # relevance score has no "width" to verify.
    dim: int = 0
    # Load quantised. For kind="embed" this means int8 via bitsandbytes
    # (LLM.int8 — worth it for the 0.6B/4B Qwen3 models; the small BGE
    # encoders' per-layer outlier handling costs more time than the saved
    # VRAM is worth, so they stay fp16/fp32). For kind="rerank" this means
    # int4 (nf4) — a reranker only has to preserve relative ordering within
    # one candidate pool, so it tolerates lower precision than an embedding
    # space compared across pool boundaries via ANN.
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
    "text-rerank-qwen3-reranker-0.6b": ModelSpec(
        id="text-rerank-qwen3-reranker-0.6b",
        repo_id="Qwen/Qwen3-Reranker-0.6B",
        kind="rerank",
        quantize=True,
    ),
    "text-rerank-qwen3-reranker-4b": ModelSpec(
        id="text-rerank-qwen3-reranker-4b",
        repo_id="Qwen/Qwen3-Reranker-4B",
        kind="rerank",
        quantize=True,
    ),
}


def embed_specs() -> list[ModelSpec]:
    return [s for s in MODEL_REGISTRY.values() if s.kind == "embed"]


def rerank_specs() -> list[ModelSpec]:
    return [s for s in MODEL_REGISTRY.values() if s.kind == "rerank"]


def get_spec(model_id: str) -> ModelSpec | None:
    """The spec for ``model_id``, or None if this service does not serve it."""
    return MODEL_REGISTRY.get(model_id)


def all_specs() -> list[ModelSpec]:
    return list(MODEL_REGISTRY.values())
