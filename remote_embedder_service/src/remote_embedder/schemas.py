"""Request/response schemas.

Response shapes are dictated by the client in
``src/event_driven_rag_service/worker/remote_embedding.py`` — see README for the
field-by-field contract. Do not rename fields here without changing that client.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# --- POST /v1/embeddings ---------------------------------------------------


class EmbeddingsRequest(BaseModel):
    model: str
    # OpenAI accepts a bare string or a list; the pipeline always sends a list,
    # but curl-by-hand debugging is much nicer with the string form allowed.
    input: str | list[str]
    encoding_format: str | None = None  # accepted and ignored ("float" only)


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    model: str
    data: list[EmbeddingItem]
    usage: Usage = Field(default_factory=Usage)


# --- GET /v1/models (OpenAI-compatible) ------------------------------------


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "remote-embedder"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]


# --- GET /api/v0/models (load-state aware) ---------------------------------


class NativeModel(BaseModel):
    id: str
    object: str = "model"
    type: str = "embeddings"
    # "loaded" | "not-loaded" — the field the client keys its load decision off.
    state: str
    dim: int
    quantization: str | None = None
    max_context_length: int | None = None


class NativeModelList(BaseModel):
    object: str = "list"
    data: list[NativeModel]


# --- POST /api/v1/models/{load,unload} -------------------------------------


class LoadRequest(BaseModel):
    model: str
    # Optional precision override, e.g. "int8". Absent means "use the registry
    # default for this model". The pipeline sends EmbedConfig.quantization here
    # so the precision lives in one place — its config — rather than being
    # decided independently on each side. Unrecognised values fall back to the
    # default rather than failing; see manager.resolve_quantization.
    quantization: str | None = None


class UnloadRequest(BaseModel):
    # LM Studio's unload takes `instance_id`; `model` is accepted as an alias so
    # either spelling works from curl.
    instance_id: str | None = None
    model: str | None = None

    def target(self) -> str | None:
        return self.instance_id or self.model


class ModelActionResponse(BaseModel):
    model: str
    state: str
    # What was actually applied, which is not always what was asked for (no CUDA
    # device, bitsandbytes missing, unrecognised value). The client logs a warning
    # when this disagrees with its request.
    quantization: str | None = None
    detail: str | None = None
