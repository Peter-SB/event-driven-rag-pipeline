from dataclasses import dataclass


@dataclass
class ChunkConfig:
    strategy: str
    target_words: int       # desired chunk size in words
    hard_limit_words: int   # absolute max words per chunk before a forced split
    chunk_overlap: float    # fraction of previous chunk to prepend as overlap (e.g. 0.10)


@dataclass
class EmbedConfig:
    model: str    # determines the GPU queue to route to; also the local SentenceTransformer name
    queue: str    # RabbitMQ routing key: "gpu.embed.{model}"
    dim: int      # embedding vector dimension (required for pgvector column declaration)
    # Model id to send to a remote OpenAI-compatible endpoint (e.g. LM Studio), if different
    # from `model`. LM Studio assigns its own slug per loaded model rather than reusing the
    # HF/gguf filename, so this can't always be derived from `model`. Defaults to `model`.
    remote_model: str | None = None
    # HF repo id to load `model` (a .gguf filename) from via llama-cpp-python instead of
    # SentenceTransformer. Set only for models distributed as GGUF — SentenceTransformer
    # cannot load those directly. None means `model` is a regular SentenceTransformer id.
    local_repo_id: str | None = None


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
    model="Qwen3-Embedding-0.6B-Q8_0.gguf",
    queue="gpu.embed.qwen3-0.6b",
    dim=1024,
    remote_model="text-embedding-qwen3-embedding-0.6b",  # LM Studio's id for this model
    local_repo_id="Qwen/Qwen3-0.6B",       # loaded via llama-cpp-python
)

EMBED_CONFIGS: dict[str, EmbedConfig] = {
    "body":          EmbedConfig(model="BAAI/bge-base-en-v1.5",     queue="gpu.embed.bge-base-en-v1.5",     dim=768),
    "title":         EmbedConfig(model="BAAI/bge-small-en-v1.5",    queue="gpu.embed.bge-small-en-v1.5",    dim=384),
    "summary_title": qwen3_embed_cfg,
    "analysis":      qwen3_embed_cfg,
}