from dataclasses import dataclass


@dataclass
class ChunkConfig:
    strategy: str
    target_words: int       # desired chunk size in words
    hard_limit_words: int   # absolute max words per chunk before a forced split
    chunk_overlap: float    # fraction of previous chunk to prepend as overlap (e.g. 0.10)


@dataclass(frozen=True)
class GgufSpec:
    """Where to get a GGUF build of a model, for the *local* embedding leg only.

    SentenceTransformer cannot load GGUF, so a config carrying one of these
    loads locally through llama-cpp-python instead. The remote leg never uses
    this — `remote_embedder_service` runs safetensors through transformers, so
    the same logical model is packaged differently on each side.
    """

    repo_id: str    # HF repo hosting the file, e.g. "Qwen/Qwen3-Embedding-0.6B-GGUF"
    filename: str   # the file within it, e.g. "Qwen3-Embedding-0.6B-Q8_0.gguf"


@dataclass
class EmbedConfig:
    # Canonical identity of the model, in HF `org/name` form. This one string is
    # the config's primary key in every sense: it selects the GPU queue, rides on
    # ChunkTask.embed_model and SearchJobCreatedEvent.embedding_profile, is the
    # SentenceTransformer id for a plain local load, and (sanitised) becomes the
    # model segment of the chunk table name. Changing it renames tables — see
    # docs/RemoteEmbedding.md.
    primary: str
    queue: str    # RabbitMQ routing key: "gpu.embed.{model}"
    dim: int      # embedding vector dimension (required for pgvector column declaration)

    # --- where the work may happen -----------------------------------------
    # Both legs are opt-in-able independently. `allow_remote` without a reachable
    # endpoint is harmless (it just falls through to local); `allow_local=False`
    # is the meaningful one — it marks a model this box genuinely cannot run,
    # so a failed remote batch raises instead of silently loading 1.2GB of
    # weights onto a CPU and taking minutes per batch.
    allow_local: bool = True
    allow_remote: bool = True

    # Model id to send to the remote embedding service, when it differs from
    # `primary`. The service keys its registry on these wire ids (they're the
    # slugs LM Studio used, kept for continuity). None means "same as primary".
    remote_model: str | None = None

    # Optional GGUF packaging for the local leg. None means `primary` is loaded
    # directly as a SentenceTransformer.
    gguf: GgufSpec | None = None

    # Requested weight precision, e.g. "int8". Declarative and shared across both
    # legs, but satisfied differently by each: the remote service applies it via
    # bitsandbytes at load time (it's sent on POST /api/v1/models/load), while
    # locally it's already baked into the GGUF build referenced above. None means
    # full precision. A model whose GGUF is Q8_0 should say "int8" here — the two
    # express the same intent through different machinery.
    quantization: str | None = None

    # Text prepended to the raw query string before embedding (never applied to
    # document/chunk text). These are asymmetric retrieval models — trained so
    # queries and passages get different input formatting. Without this prefix the
    # query embedding stays generic and barely discriminates between unrelated
    # queries, so search results end up dominated by whichever passage sits
    # closest to that generic region regardless of the query.
    # None means the model doesn't need one (symmetric encoder).
    query_prefix: str | None = None

    # Per-model override for GpuEmbedWorker's messages-per-poll. None falls back
    # to the worker's global default (32). Needed when a model's VRAM footprint
    # can't sustain that default — e.g. a 4B model on an 8GB card OOMs well
    # before 32 docs stack up (see docs/RemoteEmbedding.md).
    max_batch: int | None = None

    @property
    def remote_id(self) -> str:
        """The id to send to the remote service. Only meaningful when
        `allow_remote` — callers should check that first."""
        return self.remote_model or self.primary



# Keyed by chunk type — controls chunk sizing per type. Body/title/analysis stay small
# (aim ~400 words, hard cap 512) since they embed with short-context models; summary_title
# uses Qwen3's much larger context window, so it can hold a full summary in one chunk (up to 8k words).
CHUNK_CONFIGS: dict[str, ChunkConfig] = {
    "body":          ChunkConfig(strategy="boundary", target_words=400,  hard_limit_words=480,  chunk_overlap=0.10),
    "title":         ChunkConfig(strategy="boundary", target_words=400,  hard_limit_words=480,  chunk_overlap=0.0),
    "summary_title": ChunkConfig(strategy="boundary", target_words=18000, hard_limit_words=32000, chunk_overlap=0.0),
    "analysis":      ChunkConfig(strategy="boundary", target_words=400,  hard_limit_words=512,  chunk_overlap=0.10),
}

# Fallback for any task_type not listed in CHUNK_CONFIGS above.
DEFAULT_CHUNK_CONFIG = ChunkConfig(
    strategy="boundary",
    target_words=500,
    hard_limit_words=750,
    chunk_overlap=0.10,
)

# Keyed by chunk type — determines which model embeds each chunk type.
# Search queries use the config for the chunk type being searched (no separate query entry).

qwen3_embed_cfg = EmbedConfig(
    primary="Qwen/Qwen3-Embedding-0.6B",
    queue="gpu.embed.qwen3-0.6b",
    dim=1024,
    remote_model="text-embedding-qwen3-embedding-0.6b",
    # 8-bit both ways: bitsandbytes on the remote, the Q8_0 build locally.
    quantization="int8",
    gguf=GgufSpec(
        repo_id="Qwen/Qwen3-Embedding-0.6B-GGUF",
        filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
    ),
    query_prefix=(
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query: "
    ),
)

qwen3_4b_embed_cfg = EmbedConfig(
    primary="Qwen/Qwen3-Embedding-4B",
    queue="gpu.embed.qwen3-4b",
    dim=2560,
    remote_model="text-embedding-qwen3-embedding-4b",
    quantization="int8",
    # This box's 8GB card OOMs above ~4 docs of full-length summary text at
    # once (measured), and has no local GGUF leg configured for this model —
    # a failed remote batch should raise, not fall back to a multi-minute CPU
    # load.
    allow_local=False,
    max_batch=2,
    query_prefix=(
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query: "
    ),
)

bge_base_embed_cfg = EmbedConfig(
    primary="BAAI/bge-base-en-v1.5",
    queue="gpu.embed.bge-base-en-v1.5",
    dim=768,
    remote_model="text-embedding-bge-base-en-v1.5",
    # No quantisation: LLM.int8's per-layer outlier handling costs more time on a
    # 110M-param encoder than the saved VRAM is worth.
    query_prefix="Represent this sentence for searching relevant passages: ",
)

bge_small_embed_cfg = EmbedConfig(
    primary="BAAI/bge-small-en-v1.5",
    queue="gpu.embed.bge-small-en-v1.5",
    dim=384,
    remote_model="text-embedding-bge-small-en-v1.5",
    query_prefix="Represent this sentence for searching relevant passages: ",
)

EMBED_CONFIGS: dict[str, EmbedConfig] = {
    "body":          bge_base_embed_cfg,
    "title":         bge_small_embed_cfg,
    "summary_title": qwen3_4b_embed_cfg,
    "analysis":      qwen3_embed_cfg,
}
